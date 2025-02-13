#!/usr/bin/env python3
"""
Script to process paths for top-K classes in Wikidata:

1. Reads a mapping of entity IDs to labels (entityid2label.json).
2. Reads class_counts.json to determine the top-K classes (based on their counts).
3. Aggregates all paths from those top-K classes:
   - Keeps path-length counts instead of storing all paths in memory.
   - Gathers entity frequencies.
4. Builds a single histogram of path lengths for all combined paths.
5. Saves:
   - A histogram image named: `hist_path_lengths_top_{num_classes}.png`
   - `counts_{num_classes}.json`: entity frequencies sorted descending
   - `vocab_{num_classes}.json`: labels in the same order as counts
   - `stats_{num_classes}.json`: overall path-length statistics (min, max, average, median, mode)
"""

import os
import json
import argparse
import statistics
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from collections import defaultdict


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Aggregate paths for top-K Wikidata classes and produce a single histogram + stats."
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        nargs="+",
        default=[10, 100, 1000, 10000],
        help="List of top-K classes to process. E.g. --num-classes 10 100 1000 10000",
    )
    parser.add_argument(
        "--entityid2label-json",
        type=str,
        default="./entityid2label.json",
        help="Path to the entityid2label.json file.",
    )
    parser.add_argument(
        "--class-counts-json",
        type=str,
        default="./process_p31_p279/class_counts.json",
        help="Path to class_counts.json.",
    )
    parser.add_argument(
        "--extracted-paths-dir",
        type=str,
        default="./extracted_paths",
        help="Directory containing extracted_paths subfolders (one per class).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./process_paths",
        help="Directory to store outputs (histograms, JSON files).",
    )
    return parser.parse_args()


def plot_histogram_from_frequency(
    freq_dict,
    title="Histogram of Path Lengths",
    xlabel="Path Length",
    ylabel="Frequency",
    save_path=None,
):
    """
    Create and save a histogram from a dictionary of {path_length: frequency}.
    We'll build a small list of lengths and counts for Seaborn.
    """
    # Prepare data for plotting.
    # Two ways:
    # (A) expand all lengths (can still be huge for extremely large data)
    # (B) pass in x=keys, weights=values

    lengths = []
    frequencies = []
    for length, freq in sorted(freq_dict.items()):
        lengths.append(length)
        frequencies.append(freq)

    plt.figure(figsize=(8, 5))
    # Seaborn allows us to pass 'weights=' for each x sample.
    sns.histplot(
        x=lengths, weights=frequencies, bins=30, color="skyblue", edgecolor="black"
    )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis="y", alpha=0.75)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"[INFO] Histogram saved to {save_path}")

    plt.close()


def main():
    args = parse_arguments()

    # 1. Create output directory if not exists
    os.makedirs(args.output_dir, exist_ok=True)

    # 2. Load entityid2label
    with open(args.entityid2label_json, "r", encoding="utf-8") as f:
        entityid2label = json.load(f)

    # 3. Load and sort class_counts in descending order
    with open(args.class_counts_json, "r", encoding="utf-8") as f:
        class_counts_full = json.load(f)

    class_counts_sorted = dict(
        sorted(class_counts_full.items(), key=lambda x: x[1], reverse=True)
    )

    # 4. For each requested 'num_classes'
    for num_classes in args.num_classes:
        print(f"\n[INFO] Processing top {num_classes} classes...")

        # Restrict to top-K classes
        filtered_class_counts = {
            k: v
            for i, (k, v) in enumerate(class_counts_sorted.items())
            if i < num_classes
        }

        # We'll aggregate path-length frequencies rather than storing all paths
        length_counter = defaultdict(int)  # path_length -> frequency
        total_paths = 0
        sum_of_lengths = 0
        min_len = float("inf")
        max_len = 0

        # For entity frequencies
        counts = {}  # entity -> frequency
        vocab = {}  # entity -> label

        # Identify folders corresponding to the top-K classes
        class_folders = [
            path
            for path in glob(os.path.join(args.extracted_paths_dir, "*"))
            if os.path.basename(path) in filtered_class_counts
        ]

        # 5. Gather path data from each folder
        for folder in tqdm(
            class_folders, desc=f"Reading TSV files for top-{num_classes}"
        ):
            tsv_paths = glob(os.path.join(folder, "*.tsv"))
            for tsv_file in tsv_paths:
                with open(tsv_file, "r", encoding="utf-8") as tf:
                    for line in tf:
                        path_entities = line.strip().split("\t")
                        path_len = len(path_entities)

                        # Update length stats
                        length_counter[path_len] += 1
                        total_paths += 1
                        sum_of_lengths += path_len
                        if path_len < min_len:
                            min_len = path_len
                        if path_len > max_len:
                            max_len = path_len

                        # Update entity frequency and vocab
                        for ent in path_entities:
                            label = entityid2label.get(ent)
                            vocab[ent] = label
                            counts[ent] = counts.get(ent, 0) + 1

        print(
            f"[INFO]   -> Found {total_paths} total paths from top-{num_classes} classes."
        )

        # 6. Compute path-length stats
        if total_paths > 0:
            # average length
            avg_len = sum_of_lengths / total_paths

            # We can compute median and mode from the frequency distribution in length_counter.
            # Convert freq dict to a list of (length, freq), sorted by length.
            length_freq_pairs = sorted(length_counter.items(), key=lambda x: x[0])
            # For median, we need the "middle" path in sorted order.
            # We'll do a cumulative sum of frequencies to find the middle index.
            cumulative = 0
            middle = (total_paths + 1) // 2  # works for odd/even
            median_len = None
            for length_val, freq_val in length_freq_pairs:
                cumulative += freq_val
                if cumulative >= middle:
                    median_len = length_val
                    break

            # For mode, we just need the length with the highest frequency
            # If there are multiple, we'll store them all.
            max_freq = max(length_counter.values())
            mode_len_candidates = [
                length_ for length_, f_ in length_counter.items() if f_ == max_freq
            ]
            mode_len = (
                mode_len_candidates[0]
                if len(mode_len_candidates) == 1
                else mode_len_candidates
            )
        else:
            # If there's no path, set default empty stats
            min_len = None
            max_len = None
            avg_len = None
            median_len = None
            mode_len = None

        # 7. Plot a single histogram for all path lengths in top-K
        if total_paths > 0:
            histogram_path = os.path.join(
                args.output_dir, f"hist_path_lengths_top_{num_classes}.png"
            )
            plot_histogram_from_frequency(
                length_counter,
                title=f"Histogram of Path Lengths (Top-{num_classes} Classes)",
                xlabel="Path Length",
                ylabel="Frequency",
                save_path=histogram_path,
            )

        # 8. Sort counts by frequency (descending)
        counts_sorted = dict(
            sorted(counts.items(), key=lambda item: item[1], reverse=True)
        )
        # Reorder vocab to match the ordering of counts
        vocab_sorted = {k: vocab[k] for k in counts_sorted if k in vocab}

        # 9. Save counts and vocab
        counts_path = os.path.join(args.output_dir, f"counts_{num_classes}.json")
        vocab_path = os.path.join(args.output_dir, f"vocab_{num_classes}.json")

        with open(counts_path, "w", encoding="utf-8") as cf:
            json.dump(counts_sorted, cf, ensure_ascii=False, indent=4)

        with open(vocab_path, "w", encoding="utf-8") as vf:
            json.dump(vocab_sorted, vf, ensure_ascii=False, indent=4)

        print(f"[INFO]   -> Saved {counts_path} and {vocab_path}")

        # 10. Save stats
        stats_dict = {
            "num_classes": len(filtered_class_counts),
            "total_paths": total_paths,
            "min_path_length": min_len,
            "max_path_length": max_len,
            "average_path_length": avg_len,
            "median_path_length": median_len,
            "mode_path_length": mode_len,
        }

        stats_path = os.path.join(args.output_dir, f"stats_{num_classes}.json")
        with open(stats_path, "w", encoding="utf-8") as sf:
            json.dump(stats_dict, sf, ensure_ascii=False, indent=4)

        print(f"[INFO]   -> Saved {stats_path}")


if __name__ == "__main__":
    main()
