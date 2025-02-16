#!/usr/bin/env python3
"""
get_paths.py

Generates all possible combinations of unique upward and/or downward paths for the top N classes
and exports them as batched TSV files into class-specific directories. This script supports DFS-based
path generation, filtering of paths by an allowed node threshold, and optional removal of the last
downward path. It also logs detailed statistics for each class while managing memory via explicit
garbage collection.

Usage:
    python get_paths.py [--num-classes NUM_CLASSES]
        [--max-depth MAX_DEPTH] [--max-paths-per-class MAX_PATHS_PER_CLASS]
        [--allowed-threshold ALLOWED_THRESHOLD]
        [--batch-size BATCH_SIZE] --direction {upward,downward,both}
        [--output-dir OUTPUT_DIR]

Example:
    python get_paths.py --num_classes 20 --max_depth 5 \
        --max_paths_per_class 1000 --allowed_threshold 0.5 \
        --batch_size 50000 --direction both --output_dir ./extracted_paths
"""


import os
import json
import random
import time
import argparse
from itertools import islice
from tqdm.auto import tqdm
from collections import defaultdict
from typing import Optional
import psutil
import gc
import csv
import sys

# Set a random seed for reproducibility (optional)
# random.seed(42)


class TrieNode:
    """
    Represents a node in a Trie (prefix tree) data structure.

    Attributes:
        children (dict[str, TrieNode]): A dictionary mapping entities to their child TrieNodes.
        is_end_of_path (bool): Indicates whether the current node marks the end of a valid path.
    """

    def __init__(self) -> None:
        self.children: dict[str, "TrieNode"] = {}
        self.is_end_of_path: bool = False


class PathTrie:
    """
    Trie (prefix tree) data structure for efficiently storing and retrieving unique paths.

    Attributes:
        root (TrieNode): The root node of the Trie.
    """

    def __init__(self) -> None:
        self.root = TrieNode()

    def insert(self, path: list[str]) -> None:
        """
        Inserts a path into the Trie.

        Args:
            path (list[str]): A list of entities representing the path to be inserted.
        """
        node = self.root
        for entity in path:
            if entity not in node.children:
                node.children[entity] = TrieNode()
            node = node.children[entity]
        node.is_end_of_path = True

    def traverse(
        self,
        node: Optional[TrieNode] = None,
        path: Optional[list[str]] = None,
        all_paths: Optional[list[list[str]]] = None,
    ) -> list[list[str]]:
        """
        Traverses the Trie and retrieves all unique complete paths.

        Args:
            node (Optional[TrieNode]): The current node in the Trie during traversal. Defaults to root.
            path (Optional[list[str]]): The current path being traversed. Defaults to an empty list.
            all_paths (Optional[list[list[str]]]): The list accumulating all unique paths. Defaults to an empty list.

        Returns:
            list[list[str]]: A list of all unique complete paths stored in the Trie.
        """
        if node is None:
            node = self.root
        if path is None:
            path = []
        if all_paths is None:
            all_paths = []

        # A complete path is one that ends here and has no further children
        if node.is_end_of_path and not node.children:
            all_paths.append(path.copy())

        for entity, child_node in node.children.items():
            path.append(entity)
            self.traverse(child_node, path, all_paths)
            path.pop()

        return all_paths


def get_memory_usage() -> float:
    """
    Retrieves the current memory usage of the process in megabytes.

    Returns:
        float: Memory usage in megabytes.
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024**2)  # Convert bytes to MB


def generate_paths_dfs(
    node: str,
    mapping: dict[str, list[str]],
    min_depth: int,
    max_depth: Optional[int] = None,
    max_paths: Optional[int] = None,
    allowed_nodes: Optional[set[str]] = None,
    allowed_threshold: Optional[float] = None,
) -> list[list[str]]:
    """
    Generates all complete paths from the given node using Depth-First Search (DFS),
    stopping early if the maximum number of paths is reached.

    Args:
        node (str): The starting node for path generation.
        mapping (dict[str, list[str]]): A mapping from each node to its adjacent nodes
            (either parents or children).
        min_depth (int): The minimum allowed depth for path generation.
        max_depth (Optional[int]): The maximum allowed depth for path generation.
            If None, no upper bound.
        max_paths (Optional[int]): The maximum number of paths to generate. If None, no limit.
        allowed_nodes (Optional[set[str]]): The set of allowed nodes (e.g., top classes).
        allowed_threshold (Optional[float]): The minimum fraction of allowed nodes that must be in a path.

    Yields:
        list[str]: A list of entities representing a complete path from the starting node.
    """
    stack: list[tuple[str, list[str]]] = [(node, [node])]
    paths_generated = 0  # Counter to track the number of generated paths

    while stack:
        if max_paths and paths_generated >= max_paths:
            break  # Stop if we've reached the maximum number of paths

        current, path = stack.pop()

        # Skip if path exceeds max_depth
        if max_depth and len(path) > max_depth:
            continue

        if current not in mapping or not mapping[current]:
            if len(path) >= min_depth:
                # If threshold checking is requested, check the fraction here.
                if allowed_nodes is not None and allowed_threshold is not None:
                    allowed_count = sum(1 for n in path if n in allowed_nodes)
                    if (allowed_count / len(path)) < allowed_threshold:
                        continue  # Skip yielding this path.
                yield path
                paths_generated += 1
            continue

        adjacent_nodes = mapping.get(current, [])
        if max_paths is not None:
            random.shuffle(adjacent_nodes)
        for adjacent in adjacent_nodes:
            if adjacent not in path:  # Prevent cycles
                stack.append((adjacent, path + [adjacent]))


def invert_mapping(child_to_parents: dict[str, list[str]]) -> dict[str, list[str]]:
    """
    Inverts a child_to_parents mapping to create a parent_to_children mapping,
    ensuring that there are no duplicate children in the lists.

    Args:
        child_to_parents (dict[str, list[str]]): Mapping from child nodes to parent nodes.

    Returns:
        dict[str, list[str]]: Mapping from parent nodes to unique child nodes.
    """
    parent_to_children: defaultdict[str, set[str]] = defaultdict(set)

    for child, parents in child_to_parents.items():
        for parent in parents:
            parent_to_children[parent].add(child)  # Using set to prevent duplicates

    # Convert sets back to sorted lists for consistency
    return {parent: sorted(children) for parent, children in parent_to_children.items()}


def format_time(seconds: float) -> str:
    """
    Format time duration into days, hours, minutes, and seconds.

    Args:
        seconds (float): Duration in seconds.

    Returns:
        str: Formatted time string.
    """
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    parts = []
    if days > 0:
        parts.append(f"{int(days)} day(s)")
    if hours > 0:
        parts.append(f"{int(hours)} hour(s)")
    if minutes > 0:
        parts.append(f"{int(minutes)} minute(s)")
    parts.append(f"{seconds:.2f} second(s)")
    return ", ".join(parts)


def unique_list_of_lists(lists: list[list]):
    """
    Returns the unique lists from a list of lists of strings.

    Args:
        lists (List[List[str]]): A list containing lists of strings.

    Returns:
        List[List[str]]: A list of unique lists of strings.
    """
    unique = []
    seen = set()

    for lst in lists:
        # Convert the list to a tuple (which is hashable) for set membership check
        key = tuple(lst)
        if key not in seen:
            seen.add(key)
            unique.append(lst)

    return unique


def sample_and_combine_paths(
    num_classes: int,
    class_counts: dict[str, int],
    child_to_parents: dict[str, list[str]],
    parent_to_children: dict[str, list[str]],
    output_dir: str,
    direction: str,
    max_depth: Optional[int] = None,
    max_paths_per_class: Optional[int] = None,
    batch_size: int = 50000,
    allowed_threshold: Optional[float] = None,
    remove_the_last_downward_path: bool = False,
    nodes_to_avoid: list[str] = [],
) -> None:
    """
    Samples upward and/or downward paths from the top N classes, combines them in a shuffled order,
    batches them, and inserts them as TSV files. Logs important statistics for each class.

    Args:
        num_classes (int): Number of top classes to process.
        class_counts (dict[str, int]): Mapping of classes to their counts.
        child_to_parents (dict[str, list[str]]): Child to parents mapping.
        parent_to_children (dict[str, list[str]]): Parent to children mapping.
        output_dir (str): Directory where TSV and log files will be saved.
        direction (str): Direction of paths to include ('upward', 'downward', 'both').
        max_depth (Optional[int]): Maximum depth for path generation.
        max_paths_per_class (Optional[int]): Maximum number of paths per class for each direction.
        batch_size (int): Number of combined paths per batch TSV file.
        allowed_threshold (Optional[float]): Minimum fraction of allowed nodes in a path.
        remove_the_last_downward_path (bool): Remove the last downward path.
        nodes_to_avoid (list[str]): List of nodes to avoid due to excessive paths.
    """
    print(f"Starting path sampling and combination for top {num_classes} classes.")
    print(f"nodes to avoid: {nodes_to_avoid}")

    # Compute allowed nodes (top num_classes) from class_counts keys
    allowed_nodes = set(node for node, _ in islice(class_counts.items(), num_classes))

    with tqdm(total=num_classes, desc="Processing Classes") as pbar:
        for idx, (node, count) in enumerate(
            islice(class_counts.items(), num_classes), start=1
        ):
            if node in nodes_to_avoid:
                continue

            class_start_time = time.time()
            print(
                f"\n--- Processing Class {idx}/{num_classes}: '{node}' (Count: {count}) ---"
            )

            class_output_dir = os.path.join(output_dir, node)
            os.makedirs(class_output_dir, exist_ok=True)
            print(f"Created directory '{class_output_dir}' for class '{node}'.")

            # Generate all complete upward paths if required
            upward_paths = []
            unique_upward_paths = []
            if direction in ("upward", "both"):
                print(f"Generating upward paths for '{node}'...")

                upward_paths_gen = generate_paths_dfs(
                    node,
                    child_to_parents,
                    min_depth=2,
                    max_depth=max_depth,
                    max_paths=max_paths_per_class,
                    allowed_nodes=allowed_nodes,
                    allowed_threshold=allowed_threshold,
                )
                upward_paths = list(upward_paths_gen)

                print(f"Found {len(upward_paths)} upward paths for '{node}'.")

                # Initialize PathTrie for upward paths to ensure uniqueness
                upward_trie = PathTrie()
                for path in upward_paths:
                    upward_trie.insert(path)
                unique_upward_paths = upward_trie.traverse()
                print(
                    f"Unique upward paths after Trie insertion: {len(unique_upward_paths)}"
                )
            else:
                print(f"Skipping upward paths for '{node}' as per direction selection.")

            # Generate all complete downward paths if required
            downward_paths = []
            unique_downward_paths = []
            if direction in ("downward", "both"):
                print(f"Generating downward paths for '{node}'...")
                downward_paths_gen = generate_paths_dfs(
                    node,
                    parent_to_children,
                    min_depth=3,
                    max_depth=max_depth,
                    max_paths=max_paths_per_class,
                    allowed_nodes=allowed_nodes,
                    allowed_threshold=allowed_threshold,
                )
                downward_paths = list(downward_paths_gen)

                # remove the last one since it's the instance level
                if remove_the_last_downward_path:
                    downward_paths = unique_list_of_lists(
                        [path[:-1] for path in downward_paths if len(path[:-1]) > 0]
                    )

                print(f"Found {len(downward_paths)} downward paths for '{node}'.")

                # Initialize PathTrie for downward paths to ensure uniqueness
                downward_trie = PathTrie()
                for path in downward_paths:
                    downward_trie.insert(path)
                unique_downward_paths = downward_trie.traverse()
                print(
                    f"Unique downward paths after Trie insertion: {len(unique_downward_paths)}"
                )
            else:
                print(
                    f"Skipping downward paths for '{node}' as per direction selection."
                )

            # Determine the combination logic based on direction
            if direction == "both":
                # Shuffle the upward and downward paths separately
                print(f"Shuffling upward and downward paths for '{node}'...")
                random.shuffle(unique_upward_paths)
                random.shuffle(unique_downward_paths)
                print(f"Shuffled upward and downward paths.")

                # Combine upward and downward paths on-the-fly and batch them
                print(f"Combining and batching paths for '{node}'...")
                combined_paths_count = 0
                num_batches = 0
                batch_paths = []

                for up_path in unique_upward_paths:
                    # Reverse and remove last element (so we don't double-count the node)
                    reversed_up_path = up_path[::-1][:-1]
                    for down_path in unique_downward_paths:
                        combined_path = reversed_up_path + down_path
                        batch_paths.append(combined_path)
                        combined_paths_count += 1

                        # If batch size is reached, write to TSV
                        if len(batch_paths) == batch_size:
                            num_batches += 1
                            tsv_filename = f"batch_{num_batches}.tsv"
                            tsv_filepath = os.path.join(class_output_dir, tsv_filename)
                            try:
                                with open(
                                    tsv_filepath, "w", encoding="utf-8", newline=""
                                ) as tsv_file:
                                    writer = csv.writer(tsv_file, delimiter="\t")
                                    writer.writerows(batch_paths)
                                print(
                                    f"Exported batch {num_batches} with {len(batch_paths)} paths to '{tsv_filepath}'."
                                )
                            except Exception as e:
                                print(
                                    f"Error writing to TSV file '{tsv_filepath}': {e}"
                                )
                                # Continue processing other batches
                            batch_paths = []  # Reset batch

                # Write any remaining paths that didn't fill a full batch
                if batch_paths:
                    num_batches += 1
                    tsv_filename = f"batch_{num_batches}.tsv"
                    tsv_filepath = os.path.join(class_output_dir, tsv_filename)
                    try:
                        with open(
                            tsv_filepath, "w", encoding="utf-8", newline=""
                        ) as tsv_file:
                            writer = csv.writer(tsv_file, delimiter="\t")
                            writer.writerows(batch_paths)
                        print(
                            f"Exported batch {num_batches} with {len(batch_paths)} paths to '{tsv_filepath}'."
                        )
                    except Exception as e:
                        print(f"Error writing to TSV file '{tsv_filepath}': {e}")
                        # Continue processing

                print(
                    f"Exported {combined_paths_count} combined paths as {num_batches} TSV batch file(s) to '{class_output_dir}'."
                )
            elif direction == "upward":
                # Shuffle the upward paths
                print(f"Shuffling upward paths for '{node}'...")
                random.shuffle(unique_upward_paths)
                print(f"Shuffled upward paths.")

                # Batch the upward paths
                print(f"Batching upward paths for '{node}'...")
                combined_paths_count = len(unique_upward_paths)
                num_batches = 0
                batch_paths = []

                for up_path in unique_upward_paths:
                    reversed_up_path = up_path[::-1]  # Reverse to put 'node' at front
                    batch_paths.append(reversed_up_path)

                    # If batch size is reached, write to TSV
                    if len(batch_paths) == batch_size:
                        num_batches += 1
                        tsv_filename = f"batch_{num_batches}.tsv"
                        tsv_filepath = os.path.join(class_output_dir, tsv_filename)
                        try:
                            with open(
                                tsv_filepath, "w", encoding="utf-8", newline=""
                            ) as tsv_file:
                                writer = csv.writer(tsv_file, delimiter="\t")
                                writer.writerows(batch_paths)
                            print(
                                f"Exported batch {num_batches} with {len(batch_paths)} paths to '{tsv_filepath}'."
                            )
                        except Exception as e:
                            print(f"Error writing to TSV file '{tsv_filepath}': {e}")
                            # Continue processing other batches
                        batch_paths = []  # Reset batch

                # Write any remaining paths that didn't fill a full batch
                if batch_paths:
                    num_batches += 1
                    tsv_filename = f"batch_{num_batches}.tsv"
                    tsv_filepath = os.path.join(class_output_dir, tsv_filename)
                    try:
                        with open(
                            tsv_filepath, "w", encoding="utf-8", newline=""
                        ) as tsv_file:
                            writer = csv.writer(tsv_file, delimiter="\t")
                            writer.writerows(batch_paths)
                        print(
                            f"Exported batch {num_batches} with {len(batch_paths)} paths to '{tsv_filepath}'."
                        )
                    except Exception as e:
                        print(f"Error writing to TSV file '{tsv_filepath}': {e}")
                        # Continue processing

                print(
                    f"Exported {combined_paths_count} upward paths as {num_batches} TSV batch file(s) to '{class_output_dir}'."
                )
            elif direction == "downward":
                # Shuffle the downward paths
                print(f"Shuffling downward paths for '{node}'...")
                random.shuffle(unique_downward_paths)
                print(f"Shuffled downward paths.")

                # Batch the downward paths
                print(f"Batching downward paths for '{node}'...")
                combined_paths_count = len(unique_downward_paths)
                num_batches = 0
                batch_paths = []

                for down_path in unique_downward_paths:
                    batch_paths.append(down_path)

                    # If batch size is reached, write to TSV
                    if len(batch_paths) == batch_size:
                        num_batches += 1
                        tsv_filename = f"batch_{num_batches}.tsv"
                        tsv_filepath = os.path.join(class_output_dir, tsv_filename)
                        try:
                            with open(
                                tsv_filepath, "w", encoding="utf-8", newline=""
                            ) as tsv_file:
                                writer = csv.writer(tsv_file, delimiter="\t")
                                writer.writerows(batch_paths)
                            print(
                                f"Exported batch {num_batches} with {len(batch_paths)} paths to '{tsv_filepath}'."
                            )
                        except Exception as e:
                            print(f"Error writing to TSV file '{tsv_filepath}': {e}")
                            # Continue processing other batches
                        batch_paths = []  # Reset batch

                # Write any remaining paths that didn't fill a full batch
                if batch_paths:
                    num_batches += 1
                    tsv_filename = f"batch_{num_batches}.tsv"
                    tsv_filepath = os.path.join(class_output_dir, tsv_filename)
                    try:
                        with open(
                            tsv_filepath, "w", encoding="utf-8", newline=""
                        ) as tsv_file:
                            writer = csv.writer(tsv_file, delimiter="\t")
                            writer.writerows(batch_paths)
                        print(
                            f"Exported batch {num_batches} with {len(batch_paths)} paths to '{tsv_filepath}'."
                        )
                    except Exception as e:
                        print(f"Error writing to TSV file '{tsv_filepath}': {e}")
                        # Continue processing

                print(
                    f"Exported {combined_paths_count} downward paths as {num_batches} TSV batch file(s) to '{class_output_dir}'."
                )
            else:
                print(
                    f"Invalid direction '{direction}' specified. Skipping combination."
                )
                combined_paths_count = 0
                num_batches = 0

            # Collect statistics
            class_end_time = time.time()
            elapsed_time = class_end_time - class_start_time
            current_memory = get_memory_usage()

            # Prepare log data for the current class
            log_data = {
                "class": node,
                "initial_upward_paths": len(upward_paths),
                "unique_upward_paths": len(unique_upward_paths),
                "initial_downward_paths": len(downward_paths),
                "unique_downward_paths": len(unique_downward_paths),
                "combined_paths": combined_paths_count,
                "num_batches": num_batches,
                "batch_size": batch_size,
                "direction": direction,
                "time_taken_seconds": elapsed_time,
                "memory_usage_mb": current_memory,
            }

            # Define the log file path
            log_file = os.path.join(class_output_dir, f"{node}.log")

            # Write log statistics to the class-specific log file
            try:
                with open(log_file, "w", encoding="utf-8") as log:
                    log.write(f"Class: {log_data['class']}\n")
                    log.write(
                        f"Initial Upward Paths: {log_data['initial_upward_paths']}\n"
                    )
                    log.write(
                        f"Unique Upward Paths: {log_data['unique_upward_paths']}\n"
                    )
                    log.write(
                        f"Initial Downward Paths: {log_data['initial_downward_paths']}\n"
                    )
                    log.write(
                        f"Unique Downward Paths: {log_data['unique_downward_paths']}\n"
                    )
                    log.write(f"Combined Paths: {log_data['combined_paths']}\n")
                    log.write(f"Number of Batches: {log_data['num_batches']}\n")
                    log.write(f"Batch Size: {log_data['batch_size']}\n")
                    log.write(f"Direction: {log_data['direction']}\n")
                    log.write(
                        f"Time Taken: {format_time(log_data['time_taken_seconds'])}\n"
                    )
                    log.write(f"Memory Usage: {log_data['memory_usage_mb']:.2f} MB\n")
                print(f"Logged statistics to '{log_file}'.")
            except Exception as e:
                print(f"Error writing to log file '{log_file}': {e}")

            print(f"Time taken for '{node}': {format_time(elapsed_time)}")
            print(f"Current Memory Usage: {current_memory:.2f} MB")

            # *** Insert Garbage Collection Here ***
            # Conditionally delete variables based on direction
            if direction in ("upward", "both"):
                del upward_paths, unique_upward_paths
                if upward_trie:
                    del upward_trie
            if direction in ("downward", "both"):
                del downward_paths, unique_downward_paths
                if downward_trie:
                    del downward_trie
            del batch_paths, log_data
            gc.collect()  # Force garbage collection

            pbar.update(1)


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Generate, combine, batch, and export unique upward and/or downward paths for top N classes."
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=10,
        help="Number of top classes to process (default: 10)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Maximum depth for path generation (default: None)",
    )
    parser.add_argument(
        "--max-paths-per-class",
        type=int,
        default=None,
        help="Maximum number of paths per class for each direction (default: None)",
    )
    parser.add_argument(
        "--allowed-threshold",
        type=float,
        default=None,
        help="Allowed threshold for the number of top classes in the generated paths (default: None)",
    )
    parser.add_argument(
        "--no-remove-the-last-downward-path",
        action="store_false",
        dest="remove_the_last_downward_path",
        help="Do not remove the last downward path (default: remove the last downward path is True)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50000,
        help="Number of combined paths per batch TSV file (default: 50000)",
    )
    parser.add_argument(
        "--direction",
        type=str,
        required=True,
        choices=["upward", "downward", "both"],
        help="Direction of paths to include: 'upward', 'downward', or 'both' (required)",
    )
    parser.add_argument(
        "--class-counts-json",
        type=str,
        default="./process_p31_p279/class_counts.json",
        help="Path to class_counts.json, which contains class counts (default: './process_p31_p279/class_counts.json')",
    )
    parser.add_argument(
        "--child-to-parents-json",
        type=str,
        default="./process_p31_p279/child_to_parents.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./extracted_paths",
        help="Directory to store TSV and log files (default: './extracted_paths')",
    )
    parser.add_argument(
        "--nodes-to-avoid",
        type=str,
        nargs="+",
        default=[],
        help="List of nodes to avoid due to excessive paths (default: [])",
    )
    args = parser.parse_args()

    num_classes = args.num_classes
    MAX_DEPTH = args.max_depth
    MAX_PATHS_PER_CLASS = args.max_paths_per_class
    ALLOWED_THRESHOLD = args.allowed_threshold
    REMOVE_THE_LAST_DOWNWARD_PATH = args.remove_the_last_downward_path
    BATCH_SIZE = args.batch_size
    DIRECTION = args.direction
    output_dir = args.output_dir
    class_counts_path = args.class_counts_json
    child_to_parents_path = args.child_to_parents_json
    nodes_to_avoid = args.nodes_to_avoid

    # Validate direction argument
    if DIRECTION not in ("upward", "downward", "both"):
        print(
            f"Invalid direction '{DIRECTION}'. Must be 'upward', 'downward', or 'both'."
        )
        sys.exit(1)

    # Start total timer
    total_start_time = time.time()

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory '{output_dir}' is ready.")

    with open(class_counts_path, "r", encoding="utf-8") as f:
        class_counts = json.load(f)

    with open(child_to_parents_path, "r", encoding="utf-8") as f:
        child_to_parents = json.load(f)

    # Automatically generate parent_to_children mapping
    parent_to_children = invert_mapping(child_to_parents)
    print("Inverted child_to_parents to parent_to_children mapping.")

    # Perform path sampling, combination, batching, and insertion
    sample_and_combine_paths(
        num_classes=num_classes,
        class_counts=class_counts,
        child_to_parents=child_to_parents,
        parent_to_children=parent_to_children,
        output_dir=output_dir,
        direction=DIRECTION,
        max_depth=MAX_DEPTH,
        max_paths_per_class=MAX_PATHS_PER_CLASS,
        batch_size=BATCH_SIZE,
        allowed_threshold=ALLOWED_THRESHOLD,
        remove_the_last_downward_path=REMOVE_THE_LAST_DOWNWARD_PATH,
        nodes_to_avoid=nodes_to_avoid,
    )

    # End total timer
    total_end_time = time.time()
    total_elapsed_time = total_end_time - total_start_time

    print(f"\nTotal time taken: {format_time(total_elapsed_time)}")
    print("Script execution completed successfully.")


if __name__ == "__main__":
    main()
