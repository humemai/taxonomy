#!/usr/bin/env python
"""
Generate a hierarchical taxonomy tree using a GPT-2 language model.

This script loads a custom GPT-2 model and tokenizer along with a set of phrases,
builds a trie and recursively expands a tree using parallel batch generation.
The resulting tree is printed to the console and saved as a JSON file.
Optionally, the generated taxonomy can be visualized using NetworkX and PyVis/Dash Cytoscape.

Example usage:
    python generate_taxonomy.py --tokenizer_path "./custom_tokenizer" --num_classes 10 --force_device cuda \
        --top_p 0.95 --max_depth 32 --max_width 4 --max_tokens_per_phrase 20 --temperature 1.5
"""

from __future__ import annotations
import argparse
import json
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from tqdm.auto import tqdm
import webbrowser
import networkx as nx
from pyvis.network import Network
import concurrent.futures
import contextlib
import random
import os
import threading

# Additional imports for Dash Cytoscape visualization.
from jupyter_dash import JupyterDash
import dash_cytoscape as cyto
from dash import html
from IPython.display import clear_output

cyto.load_extra_layouts()


###############################################
# Utility Functions: Loading Model, Tokenizer & Phrases
###############################################


def get_latest_checkpoint_dir(base_dir):
    # List all subdirectories that match the pattern "checkpoint-XXX"
    checkpoint_dirs = [
        d
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("checkpoint-")
    ]
    if not checkpoint_dirs:
        return None
    # Sort the directories by the numeric portion after "checkpoint-"
    checkpoint_dirs.sort(key=lambda d: int(d.split("-")[-1]))
    latest_checkpoint = checkpoint_dirs[-1]
    return os.path.join(base_dir, latest_checkpoint)


def load_tokenizer(tokenizer_path: str) -> GPT2Tokenizer:
    """
    Load a GPT-2 tokenizer from the specified checkpoint path.

    Args:
        tokenizer_path (str): The file system path to the tokenizer checkpoint directory.

    Returns:
        GPT2Tokenizer: The loaded tokenizer instance.
    """
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    print("Tokenizer loaded from checkpoint.")
    print(f"Special tokens: {tokenizer.special_tokens_map}")
    return tokenizer


def load_model(
    checkpoint_dir: str, tokenizer: GPT2Tokenizer, force_device="cpu"
) -> tuple:
    """
    Load a GPT-2 language model from a checkpoint and prepare it for inference.

    Args:
        checkpoint_dir (str): The path to the model checkpoint directory.
        tokenizer (GPT2Tokenizer): The tokenizer corresponding to the model.
        force_device (str, optional): Device to force the model onto (e.g., "cpu" or "cuda"). Defaults to "cpu".

    Returns:
        tuple: A tuple (model, device) where:
            - model (GPT2LMHeadModel): The loaded and configured language model.
            - device (torch.device): The device on which the model is running.
    """
    config = GPT2Config.from_pretrained(checkpoint_dir)
    model = GPT2LMHeadModel.from_pretrained(checkpoint_dir, config=config)
    model.resize_token_embeddings(len(tokenizer))
    device = torch.device(force_device)
    model.to(device)
    model.eval()
    print(f"Model loaded from {checkpoint_dir}")
    return model, device


def load_phrases(num_classes: int, allowed_threshold: float) -> dict:
    """
    Load vocabulary phrases and their counts from JSON files and combine them into a dictionary.

    Args:
        num_classes (int): The number of classes (or top phrases) to load.
        allowed_threshold (float): The threshold for allowed phrases.

    Returns:
        dict: A dictionary mapping each phrase (str) to another dictionary with keys:
              - "count": The count of the phrase.
              - "id": The original ID.
    """
    with open(
        f"./process_paths/allowed_threshold_{allowed_threshold:.1f}/vocab_top_{num_classes}.json",
        "r",
        encoding="utf-8",
    ) as f:
        phrases_ = json.load(f)
    with open(
        f"./process_paths/allowed_threshold_{allowed_threshold:.1f}/counts_top_{num_classes}.json",
        "r",
        encoding="utf-8",
    ) as f:
        counts = json.load(f)
    phrases = {}
    for key, val in phrases_.items():
        phrases[val] = {"count": counts[key], "id": key}
    return phrases


###############################################
# Build a Trie for the Phrase Vocabulary
###############################################


class TrieNode:
    """
    A node in a trie (prefix tree) used to store token sequences representing phrases.

    Attributes:
        children (dict): Mapping from token IDs to child TrieNode objects.
        is_end_of_phrase (bool): Indicates whether this node marks the end of a complete phrase.
    """

    def __init__(self):
        self.children = {}
        self.is_end_of_phrase = False


class PhraseTrie:
    """
    A trie (prefix tree) data structure for storing and searching tokenized phrases.

    Attributes:
        root (TrieNode): The root node of the trie.
    """

    def __init__(self):
        self.root = TrieNode()

    def insert(self, token_ids: list) -> None:
        """
        Insert a sequence of token IDs into the trie.

        Args:
            token_ids (list): A list of token IDs representing a phrase.
        """
        node = self.root
        for token_id in token_ids:
            if token_id not in node.children:
                node.children[token_id] = TrieNode()
            node = node.children[token_id]
        node.is_end_of_phrase = True

    def starts_with(self, token_ids: list) -> TrieNode:
        """
        Search for a prefix in the trie.

        Args:
            token_ids (list): A list of token IDs representing the prefix.

        Returns:
            TrieNode or None: The node corresponding to the end of the prefix if found; otherwise, None.
        """
        node = self.root
        for token_id in token_ids:
            if token_id not in node.children:
                return None
            node = node.children[token_id]
        return node


def build_phrase_trie(phrases: dict, tokenizer: GPT2Tokenizer) -> PhraseTrie:
    """
    Build a phrase trie from a collection of phrases.

    Args:
        phrases (dict): A dictionary where keys are phrases to be added to the trie.
        tokenizer (GPT2Tokenizer): The tokenizer used to convert phrases into token IDs.

    Returns:
        PhraseTrie: A trie containing all the tokenized phrases.
    """
    trie = PhraseTrie()
    for phrase in phrases.keys():
        token_ids = tokenizer.encode(phrase, add_special_tokens=False)
        if token_ids:
            trie.insert(token_ids)
    print("Phrase trie built with {} phrases.".format(len(phrases)))
    return trie


###############################################
# Tree Node Definition
###############################################


class TreeNode:
    """
    Represents a node in a hierarchical tree structure.

    Attributes:
        phrase (str): The phrase associated with this node.
        parent (TreeNode or None): The parent node in the tree.
        children (list): A list of child TreeNode objects.
    """

    def __init__(self, phrase: str, parent: TreeNode = None):
        """
        Initialize a TreeNode with a given phrase and an optional parent.

        Args:
            phrase (str): The phrase for this node.
            parent (TreeNode, optional): The parent node. Defaults to None.
        """
        self.phrase = phrase
        self.parent = parent
        self.children = []

    def add_child(self, child_node: TreeNode) -> None:
        """
        Add a child node to this node.

        Args:
            child_node (TreeNode): The TreeNode to be added as a child.
        """
        self.children.append(child_node)

    def get_branch(self) -> list:
        """
        Retrieve the branch (path) from the root to this node.

        Returns:
            list: A list of phrases representing the path from the root to this node.
        """
        branch = []
        node = self
        while node is not None:
            branch.append(node.phrase)
            node = node.parent
        return list(reversed(branch))


###############################################
# Top-p Filtering Function (without top_k)
###############################################


def top_p_filtering(
    logits: torch.Tensor, top_p: float = 0.0, filter_value=-float("Inf")
) -> torch.Tensor:
    """
    Filter a distribution of logits using nucleus (top-p) filtering.

    Args:
        logits (torch.Tensor): A 2D tensor of shape (batch_size, vocab_size) with logits.
        top_p (float, optional): Cumulative probability threshold for nucleus sampling.
        filter_value (float, optional): Value assigned to filtered-out logits.

    Returns:
        torch.Tensor: The filtered logits tensor.
    """
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the mask right to keep the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[0, indices_to_remove] = filter_value
    return logits


###############################################
# Batch Generation Function for Candidate Phrases
###############################################


def generate_batch_child_phrases(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    prompts: list,
    max_tokens_per_phrase: int = 20,
    top_p: float = 0.95,
    temperature: float = 1.0,
) -> list:
    """
    Generate candidate phrases for a list of prompts in batch, stopping generation
    as soon as we see <DOWNWARD> or <EOS> for each individual sequence.
    """

    device = model.device

    # Encode all prompts
    encoded_prompts = [
        tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompts
    ]
    batch_size = len(encoded_prompts)

    # Pad all prompt encodings to the same length
    max_prompt_len = max(len(ids) for ids in encoded_prompts)
    padded_prompts = []
    for ids in encoded_prompts:
        needed = max_prompt_len - len(ids)
        padded = ids + [tokenizer.eos_token_id] * needed
        padded_prompts.append(padded)

    input_ids = torch.tensor(padded_prompts, device=device, dtype=torch.long)
    generated = input_ids.clone()

    # Identify boundary tokens
    downward_ids = tokenizer.encode("<DOWNWARD>", add_special_tokens=False)
    eos_ids = tokenizer.encode("<EOS>", add_special_tokens=False)

    boundary_ids = set()
    if downward_ids:
        boundary_ids.add(downward_ids[0])
    if eos_ids:
        boundary_ids.add(eos_ids[0])

    # Keep track of which sequences are "done" (already produced boundary)
    done_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)

    amp_context = (
        torch.cuda.amp.autocast() if device.type == "cuda" else contextlib.nullcontext()
    )

    with torch.inference_mode(), amp_context:
        for step in range(max_tokens_per_phrase):
            # Forward pass
            outputs = model(generated)
            logits = outputs.logits[:, -1, :]  # shape: (batch_size, vocab_size)

            # Apply top-p filtering only for those *not* done
            for i in range(batch_size):
                if not done_mask[i]:
                    logits[i : i + 1] = top_p_filtering(
                        logits[i : i + 1].clone(), top_p=top_p
                    )
                else:
                    # Force done sequences to produce only EOS (or pad),
                    # ensuring they remain "stalled"
                    logits[i] = float("-inf")
                    logits[i, tokenizer.eos_token_id] = 0.0

            # Sample with temperature
            probs = torch.softmax(logits / temperature, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            # shape: (batch_size,)

            # Append new tokens
            generated = torch.cat((generated, next_tokens.unsqueeze(-1)), dim=1)

            # Mark sequences done if they produce a boundary
            for i in range(batch_size):
                if (not done_mask[i]) and (next_tokens[i].item() in boundary_ids):
                    done_mask[i] = True

            # If everyone is done, break early
            if done_mask.all():
                break

    # Now decode. We only want the portion from end-of-prompt up to the first boundary.
    candidate_texts = []
    for prompt_ids, full_gen_ids in zip(encoded_prompts, generated.tolist()):
        start_index = len(prompt_ids)

        # By default, take everything after the prompt
        candidate_ids = full_gen_ids[start_index:]

        # Truncate at the earliest boundary (if any)
        earliest_boundary_idx = None
        for i in range(start_index, len(full_gen_ids)):
            if full_gen_ids[i] in boundary_ids:
                earliest_boundary_idx = i
                break
        if earliest_boundary_idx is not None:
            candidate_ids = full_gen_ids[start_index:earliest_boundary_idx]

        # Decode
        candidate = tokenizer.decode(candidate_ids, skip_special_tokens=True).strip()
        candidate = " ".join(candidate.split())
        candidate_texts.append(candidate)

    return candidate_texts


###############################################
# Parallelized Recursive Tree Expansion Using Batching
###############################################


def expand_node_parallel(
    node: TreeNode,
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    phrase_trie: PhraseTrie,
    max_depth: int,
    max_width: int,
    phrases: dict,
    current_depth: int = 0,
    top_p: float = 0.95,
    max_tokens_per_phrase: int = 20,
    temperature: float = 1.0,
    current_width: int = None,
    width_decay_factor: float = 0.8,  # New parameter
) -> None:
    """
    Recursively expand a tree node by generating child phrases. The width used at
    each level starts at max_width and is halved at each subsequent level, down
    to a minimum of 1.
    """
    if current_depth >= max_depth:
        return

    if current_width is None:
        current_width = max_width

    # Construct the branch prompt.
    branch = node.get_branch()
    branch_clean = [phrase.strip() for phrase in branch if phrase.strip() != ""]

    # -- Simplified prompt construction. Always join by <DOWNWARD> --
    prompt = "<DOWNWARD>".join(branch_clean)

    print(f"DEBUG: Expanding node at depth {current_depth} with branch: {branch_clean}")
    print(f"DEBUG: Using prompt: '{prompt}'")
    print(f"DEBUG: Using current_width = {current_width}")

    # Create a batch of prompts equal to the current_width.
    batch_prompts = []
    for _ in range(current_width):
        current_prompt = prompt + "<DOWNWARD>"
        batch_prompts.append(current_prompt)

    # Generate candidate phrases in batch.
    candidates = generate_batch_child_phrases(
        model,
        tokenizer,
        batch_prompts,
        max_tokens_per_phrase=max_tokens_per_phrase,
        top_p=top_p,
        temperature=temperature,
    )

    children_generated = 0
    for candidate in list(set(candidates)):
        if not candidate:
            print("DEBUG: Candidate empty, skipping.")
            continue
        if candidate not in phrases:
            print(f"DEBUG: Candidate '{candidate}' is not a valid phrase, skipping.")
            continue
        if candidate in [child.phrase for child in node.children]:
            print(
                f"DEBUG: Candidate '{candidate}' already exists as a child, skipping."
            )
            continue
        if candidate in branch_clean:
            print(f"DEBUG: Candidate '{candidate}' already in branch, skipping.")
            continue

        print(f"DEBUG: Accepted candidate: '{candidate}'")
        child_node = TreeNode(candidate, parent=node)
        node.add_child(child_node)
        children_generated += 1

        # Recursively expand this child, halving the width for the next level.
        next_width = max(1, int(current_width * width_decay_factor))
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.submit(
                expand_node_parallel,
                child_node,
                model,
                tokenizer,
                phrase_trie,
                max_depth,
                max_width,
                phrases,
                current_depth + 1,
                top_p,
                max_tokens_per_phrase,
                temperature,
                next_width,
                width_decay_factor,
            )

        # If we've created all possible children for this node, stop.
        if children_generated >= current_width:
            break


###############################################
# Visualization Functions: Tree to NetworkX Graph and Save/Visualize
###############################################


def tree_to_nx_graph(root: TreeNode, phrases: dict) -> nx.DiGraph:
    """
    Convert a tree of TreeNode objects into a NetworkX directed graph.

    Args:
        root (TreeNode): The root node of the tree.
        phrases (dict): The dictionary mapping phrases to their metadata (including 'id').

    Returns:
        nx.DiGraph: A directed graph representation of the tree.
    """
    G = nx.DiGraph()

    def add_edges(node):
        # If this is the special root phrase <BOS>entity, display it as "entity".
        if node.phrase == "<BOS>entity":
            label = "entity"
        elif node.phrase in phrases:
            label = f"{node.phrase}\n({phrases[node.phrase]['id']})"
        else:
            label = node.phrase

        G.add_node(id(node), label=label)
        for child in node.children:
            add_edges(child)
            G.add_edge(id(node), id(child))

    add_edges(root)
    return G


def save_taxonomy_json(
    root: TreeNode, phrases: dict, output_filename: str = "taxonomy.json"
) -> None:
    """
    Convert the tree into a NetworkX directed graph, then save it in node-link JSON format.

    Args:
        root (TreeNode): The root node of the tree.
        phrases (dict): The dictionary mapping phrases to their metadata.
        output_filename (str, optional): The filename for the output JSON file.

    Returns:
        None
    """
    G = tree_to_nx_graph(root, phrases)
    data = nx.node_link_data(G, edges="edges")
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Taxonomy saved as JSON to {output_filename}")


def load_taxonomy_json(filename: str = "taxonomy.json") -> nx.DiGraph:
    """
    Load a taxonomy (graph) from a JSON file in node-link format.

    Args:
        filename (str, optional): The filename of the JSON file to load.

    Returns:
        nx.DiGraph: The reconstructed directed graph.
    """
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    G = nx.node_link_graph(data, edges="edges")
    return G


def visualize_graph(
    G: nx.DiGraph, output_filename: str = "taxonomy.html", height: str = "1080px"
) -> None:
    """
    Visualize a NetworkX graph using PyVis and open the visualization in a web browser.

    Args:
        G (nx.DiGraph): The directed graph to visualize.
        output_filename (str, optional): The output HTML filename.
        height (str, optional): The height for the visualization canvas.

    Returns:
        None
    """
    net = Network(height=height, width="100%", directed=True, notebook=False)
    net.from_nx(G)

    net.set_options(
        """
    {
      "layout": {
        "hierarchical": {
          "enabled": true,
          "direction": "LR",
          "sortMethod": "directed"
        }
      },
      "edges": {
        "smooth": false
      },
      "physics": {
        "hierarchicalRepulsion": {
          "centralGravity": 0.0,
          "springLength": 100,
          "springConstant": 0.01,
          "nodeDistance": 120,
          "damping": 0.09
        },
        "minVelocity": 0.75
      }
    }
    """
    )
    net.write_html(output_filename)
    webbrowser.open(output_filename)
    print(f"Graph visualization saved to {output_filename}")


def visualize_graph_dash(G: nx.DiGraph, host="127.0.0.1", port=8050) -> None:
    """
    Visualize a NetworkX graph using Dash Cytoscape in an external web browser.

    Args:
        G (nx.DiGraph): The directed graph to visualize.
        host (str, optional): The hostname for the Dash server.
        port (int, optional): The port for the Dash server.

    Returns:
        None
    """
    # Convert nodes for Cytoscape.
    nodes = []
    for node_id, data in G.nodes(data=True):
        label = data.get("label", str(node_id))
        nodes.append({"data": {"id": str(node_id), "label": label}})

    # Convert edges.
    edges = []
    for source, target in G.edges():
        edges.append({"data": {"source": str(source), "target": str(target)}})

    app = JupyterDash(__name__)
    app.layout = html.Div(
        [
            cyto.Cytoscape(
                id="cytoscape-graph",
                elements=nodes + edges,
                layout={
                    "name": "dagre",
                    "rankDir": "LR",
                    "nodeSep": 30,
                    "edgeSep": 10,
                    "rankSep": 70,
                    "padding": 10,
                },
                style={"width": "100%", "height": "100vh"},
                stylesheet=[
                    {
                        "selector": "node",
                        "style": {
                            "label": "data(label)",
                            "background-color": "#87CEFA",
                            "color": "#000",
                            "font-size": "14px",
                            "text-valign": "bottom",
                            "text-halign": "center",
                            "text-margin-y": "10px",
                            "width": "40px",
                            "height": "40px",
                            "text-wrap": "wrap",
                            "text-max-width": "200px",
                        },
                    },
                    {"selector": "edge", "style": {"line-color": "#B3B3B3"}},
                ],
            )
        ]
    )

    def open_browser():
        webbrowser.open(f"http://{host}:{port}")

    threading.Timer(1, open_browser).start()
    clear_output(wait=True)
    app.run_server(mode="external", debug=True, host=host, port=port)


###############################################
# Main Execution: Build and Visualize the Hierarchical Tree
###############################################


def main():
    parser = argparse.ArgumentParser(
        description="Generate a hierarchical taxonomy tree using a GPT-2 language model."
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="./custom_tokenizer",
        help="Path to the tokenizer checkpoint directory.",
    )
    parser.add_argument(
        "--num_classes", type=int, default=10, help="Number of classes/phrases to load."
    )
    parser.add_argument(
        "--force_device",
        type=str,
        default="cuda",
        help="Device to run the model on ('cpu' or 'cuda').",
    )
    parser.add_argument(
        "--top_p", type=float, default=0.95, help="Nucleus (top-p) sampling threshold."
    )
    parser.add_argument(
        "--max_depth", type=int, default=32, help="Maximum depth to expand the tree."
    )
    parser.add_argument(
        "--max_width", type=int, default=4, help="Maximum number of children per node."
    )
    parser.add_argument(
        "--max_tokens_per_phrase",
        type=int,
        default=20,
        help="Maximum tokens to generate per candidate phrase.",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.5, help="Sampling temperature."
    )
    parser.add_argument(
        "--allowed_threshold",
        type=float,
        help="Allowed threshold",
    )
    parser.add_argument(
        "--loss_threshold",
        type=float,
        default=0.1,
        help="Training stops if the loss goes below this value (default: 0.1)",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        choices=["tiny", "small", "medium", "large"],
        default="small",
        help="Model size (default: small)",
    )
    parser.add_argument(
        "--width_decay_factor",
        type=float,
        default=0.8,
        help="Multiplicative decay factor for reducing node width at each level (e.g., 0.8 means a 20% reduction per level).",
    )
    args = parser.parse_args()

    # Load resources.
    phrases = load_phrases(args.num_classes, args.allowed_threshold)
    tokenizer = load_tokenizer(args.tokenizer_path)
    phrase_trie = build_phrase_trie(phrases, tokenizer)

    # Build the base checkpoint directory path
    checkpoint_dir = (
        f"model_output/allowed_threshold_{args.allowed_threshold:.1f}/"
        f"model_size_{args.model_size}/loss_threshold_{args.loss_threshold:.1f}/"
        f"num_classes_{args.num_classes}/"
        f"sample_first_batch_True/"
        f"sampling_mode_class_aware/"
    )

    # Find the latest checkpoint subdirectory
    latest_checkpoint_dir = get_latest_checkpoint_dir(checkpoint_dir)
    if latest_checkpoint_dir is not None:
        model, device = load_model(
            latest_checkpoint_dir, tokenizer, force_device=args.force_device
        )
    else:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}.")

    # Build tree starting from <BOS>entity (label it as "entity" in visualization)
    root = TreeNode("<BOS>entity")

    # We can start from depth=1 so that the root is effectively "entity" at depth 1.
    expand_node_parallel(
        root,
        model,
        tokenizer,
        phrase_trie,
        max_depth=args.max_depth,
        max_width=args.max_width,
        phrases=phrases,
        top_p=args.top_p,
        max_tokens_per_phrase=args.max_tokens_per_phrase,
        temperature=args.temperature,
        current_depth=0,  # start from "entity" as depth 0
        width_decay_factor=args.width_decay_factor,
    )

    # Optionally print the hierarchical tree to the console.
    def print_tree(node, phrases, indent=0):
        if node.phrase == "<BOS>entity":
            label = "entity"
        elif node.phrase in phrases:
            label = f"{node.phrase} ({phrases[node.phrase]['id']})"
        else:
            label = node.phrase
        print("  " * indent + label)
        for child in node.children:
            print_tree(child, phrases, indent + 1)

    print("Hierarchical Tree:")
    print_tree(root, phrases)

    # Save the taxonomy.
    os.makedirs("trees", exist_ok=True)
    filename = (
        f"trees/"
        f"model_size_{args.model_size}"
        f"_num_classes_{args.num_classes}"
        f"_allowed_threshold_{args.allowed_threshold:.1f}"
        f"_loss_threshold_{args.loss_threshold:.1f}"
        f"_top_p_{args.top_p}"
        f"_max_depth_{args.max_depth}"
        f"_max_width_{args.max_width}"
        f"_temperature_{args.temperature}"
        f"_width_decay_factor_{args.width_decay_factor}"
        f".json"
    )

    save_taxonomy_json(root, phrases, output_filename=filename)


if __name__ == "__main__":
    main()
