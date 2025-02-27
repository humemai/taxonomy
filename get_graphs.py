#!/usr/bin/env python3

import os
import json
import argparse
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import logging
from glob import glob
from datetime import datetime


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Aggregate paths for top-K Wikidata classes and produce a single histogram + stats."
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        nargs="+",
        default=[10, 100, 1000, 10000],
        help="List of top-K classes to process. E.g. --num_classes 10 100 1000 10000",
    )
    parser.add_argument(
        "--class_counts_json",
        type=str,
        default="./process_p31_p279/class_counts.json",
        help="Path to class_counts.json.",
    )
    parser.add_argument(
        "--extracted_paths_dir",
        type=str,
        default="./extracted_paths",
        help="Directory containing extracted_paths subfolders (one per class).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./extracted_graphs",
        help="Directory to store outputs (graphs, JSON files, logs).",
    )
    parser.add_argument(
        "--sample_first_batch",
        action="store_true",
        help="If set, only sample the first batch file for each class",
    )
    return parser.parse_args()


def setup_logging(log_file):
    """Set up logging to specific log file and console."""
    # Reset logging handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",  # Simple format without timestamps
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return log_file


def count_cycles(G):
    """Count the number of cycles in a directed graph."""
    try:
        # This might be computationally expensive for large graphs
        cycles = list(nx.simple_cycles(G))
        return len(cycles)
    except Exception as e:
        # The algorithm might not work for large graphs due to recursion limit
        return "Unable to compute - graph too large"


def get_tsv_paths(
    class_counts_json, extracted_paths_dir, num_classes, sample_first_batch=False
):
    """Load TSV file paths based on class counts."""
    logging.info(f"Loading class counts from {class_counts_json}")
    with open(class_counts_json, "r", encoding="utf-8") as f:
        class_counts = json.load(f)

    starting_entities = set(list(class_counts.keys())[:num_classes])
    logging.info(f"Selected {len(starting_entities)} classes to process")

    tsv_paths_by_class = {}
    for path in glob(f"{extracted_paths_dir}/*/*.tsv"):
        class_dir = os.path.basename(os.path.dirname(path))
        if class_dir in starting_entities:
            tsv_paths_by_class.setdefault(class_dir, []).append(path)

    tsv_paths = []
    if sample_first_batch:
        for class_dir, paths in tsv_paths_by_class.items():
            batch1_files = [p for p in paths if "batch_1" in os.path.basename(p)]
            if batch1_files:
                tsv_paths.append(batch1_files[0])
            else:
                tsv_paths.append(paths[0])
    else:
        for paths in tsv_paths_by_class.values():
            tsv_paths.extend(paths)

    logging.info(f"Found {len(tsv_paths)} TSV files to process")
    return tsv_paths


def read_tsv(path):
    """Read paths from a TSV file."""
    with open(path, "r", encoding="utf-8") as tf:
        for line in tf:
            path_entities = line.strip().split("\t")
            yield path_entities


def load_paths(class_counts_json, extracted_paths_dir, num_classes, sample_first_batch):
    """Load all paths from TSV files."""
    tsv_paths = get_tsv_paths(
        class_counts_json, extracted_paths_dir, num_classes, sample_first_batch
    )
    paths = []

    for i, tsv_path in enumerate(tsv_paths):
        for path in read_tsv(tsv_path):
            paths.append(path)

        if (i + 1) % 10 == 0 or i == len(tsv_paths) - 1:
            logging.info(
                f"Processed {i+1}/{len(tsv_paths)} TSV files, found {len(paths)} paths so far"
            )

    logging.info(f"Loaded {len(paths)} total paths from {len(tsv_paths)} files")
    return paths


def create_graph_from_paths(paths):
    """Create a directed graph from paths."""
    logging.info("Creating graph from paths")
    G = nx.DiGraph()

    # Add edges from all paths
    for path in paths:
        for i in range(len(path) - 1):
            G.add_edge(path[i], path[i + 1])

    logging.info(
        f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges"
    )
    return G


def analyze_graphs(G):
    """Find all connected components and analyze them."""
    logging.info("Analyzing graph components")

    # Find weakly connected components
    components = list(nx.weakly_connected_components(G))
    component_sizes = [len(comp) for comp in components]

    logging.info(f"Number of components detected: {len(components)}")

    # Sort components by size
    sorted_components = sorted(components, key=len, reverse=True)

    # Extract component subgraphs for further analysis
    subgraphs = []
    for i, component in enumerate(sorted_components):
        subgraph = G.subgraph(component).copy()
        subgraphs.append(subgraph)

    return subgraphs


def save_graph_to_json(G, filename):
    """Save a NetworkX graph as a JSON file in node-link format."""
    logging.info(f"Saving graph to {filename}")

    # Add labels to nodes if they don't have them
    for node in G.nodes():
        if "label" not in G.nodes[node]:
            G.nodes[node]["label"] = str(node)

    # Convert the graph to node-link format
    data = nx.node_link_data(G)

    # Save to file
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main():
    args = parse_arguments()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process each specified num_classes value
    for num_class in args.num_classes:
        # Set up specific log file for this run
        log_file = os.path.join(args.output_dir, f"graph_{num_class}.log")
        setup_logging(log_file)

        logging.info(f"Processing top {num_class} classes")

        # Read paths
        paths = load_paths(
            args.class_counts_json,
            args.extracted_paths_dir,
            num_class,
            args.sample_first_batch,
        )

        if not paths:
            logging.info(f"No paths found for top {num_class} classes, skipping")
            continue

        # Create graph from paths
        G = create_graph_from_paths(paths)

        # Find connected components and analyze them
        subgraphs = analyze_graphs(G)

        if not subgraphs:
            logging.info(f"No subgraphs found for top {num_class} classes, skipping")
            continue

        # Log statistics for all components
        logging.info(f"Statistics for all components:")
        for i, subgraph in enumerate(subgraphs):
            logging.info(f"Component {i+1}:")
            logging.info(f"  - Nodes: {subgraph.number_of_nodes()}")
            logging.info(f"  - Edges: {subgraph.number_of_edges()}")

            # Count cycles in the component
            cycle_count = count_cycles(subgraph)
            logging.info(f"  - Cycles: {cycle_count}")

        # Save only the largest component as JSON
        if subgraphs:
            largest = subgraphs[
                0
            ]  # First one is the largest due to sorting in analyze_graphs
            graph_file = os.path.join(args.output_dir, f"graph_{num_class}.json")
            logging.info(f"Saving only the largest component to {graph_file}")
            save_graph_to_json(largest, graph_file)


if __name__ == "__main__":
    main()
