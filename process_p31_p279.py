"""
process_p31_and_p279.py

This script processes subclass (P279) and instance (P31) relationships from TSV files
to analyze class distributions within a knowledge graph. Additionally, it processes
property usage statistics. It performs the following tasks:

1. **Property Processing:**
   - Loads property usage statistics from `property_stats.json`.
   - Loads property labels from `properties.json`.
   - Computes cumulative distribution of property usage.
   - Identifies the number of properties required to cover specified thresholds (80%, 90%, 95%, 99%).
   - Plots and saves the cumulative distribution of properties.
   - Prints details of the top 100 properties.

2. **P31/P279 Processing:**
   - Loads entity labels from `entityid2label.json`.
   - Loads subclass relationships from TSV files in the `./P279/` directory.
   - Loads instance relationships from TSV files in the `./P31/` directory.
   - Filters out non-English entities based on the loaded labels.
   - Builds parent-to-children and child-to-parents mappings.
   - Counts the frequency of each class based on instance data.
   - Computes cumulative distribution of class counts.
   - Identifies the number of classes required to cover specified thresholds (80%, 90%, 95%, 99%).
   - Plots and saves the cumulative distribution of classes.
   - Prints details of the top 100 classes.

3. **Data Saving:**
   - Saves all processed data structures (`class_counts`, `child_to_parents`)
         as JSON files within the `process_p31_p279` directory.
   - Saves all generated figures within the `process_p31_p279` directory.

4. **Logging and Statistics:**
   - Measures and logs the total processing time.
   - Logs the number of entities processed, classes counted, and properties analyzed.
   - Records any errors encountered during processing.

**Usage:**
    Ensure that the following files and directories are present relative to the script's location:
    - `./entityid2label.json`
    - `./P279/*.tsv`
    - `./P31/*.tsv`
    - `./property_stats.json`
    - `./properties.json`

    Execute the script using Python 3.10:
        python process_p31_and_p279.py

**Dependencies:**
    - Python 3.10+
    - matplotlib
    - tqdm

    Install any missing packages using pip:
        pip install matplotlib tqdm
"""

import os
import json
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from itertools import islice
from tqdm.auto import tqdm
from glob import glob
import csv
import time
from typing import Union


# ------------------------------------------------------------------------------
# 1. DIRECTORY CREATION
# ------------------------------------------------------------------------------


def create_output_directory(directory: str) -> None:
    """
    Creates the specified directory if it doesn't exist.

    Args:
        directory (str): The path to the directory to create.
    """
    try:
        os.makedirs(directory, exist_ok=True)
        print(f"Directory '{directory}' is ready.\n")
    except Exception as e:
        print(f"Error creating directory '{directory}': {e}")


# ------------------------------------------------------------------------------
# 2. PROPERTY PROCESSING FUNCTIONS
# ------------------------------------------------------------------------------


def load_properties_used(filepath: str) -> dict[str, int]:
    """
    Loads property usage statistics from a JSON file.

    Args:
        filepath (str): Path to the property_stats.json file.

    Returns:
        dict[str, int]: Mapping of property IDs to their usage counts.
    """
    print(f"Loading property usage statistics from '{filepath}'...")
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            properties_used = json.load(f)
        print(f"Loaded {len(properties_used)} properties.\n")
        return properties_used
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: File '{filepath}' is not a valid JSON.")
        return {}


def load_properties_labels(filepath: str) -> dict[str, dict[str, any]]:
    """
    Loads property labels from a JSON file.

    Args:
        filepath (str): Path to the properties.json file.

    Returns:
        dict[str, dict[str, any]]: Mapping of property IDs to their details.
    """
    print(f"Loading property labels from '{filepath}'...")
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            properties = json.load(f)
        print(f"Loaded labels for {len(properties)} properties.\n")
        return properties
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: File '{filepath}' is not a valid JSON.")
        return {}


def compute_cumulative_distribution_counts(counts: list[int]) -> list[int]:
    """
    Computes the cumulative counts from a list of counts.

    Args:
        counts (list[int]): List of counts sorted in descending order.

    Returns:
        list[int]: Cumulative counts.
    """
    cumulative_counts = []
    running_sum = 0
    for cnt in counts:
        running_sum += cnt
        cumulative_counts.append(running_sum)
    return cumulative_counts


def compute_cumulative_percentage(cumulative_counts: list[int]) -> list[float]:
    """
    Computes the cumulative percentage from cumulative counts.

    Args:
        cumulative_counts (list[int]): Cumulative counts.

    Returns:
        list[float]: Cumulative percentage distribution.
    """
    total_count = cumulative_counts[-1] if cumulative_counts else 0
    if total_count == 0:
        return []
    return [count / total_count * 100 for count in cumulative_counts]


def find_thresholds_in_distribution(
    cumulative_percentage: list[float], thresholds: list[int]
) -> dict[int, int]:
    """
    Finds the number of properties needed to reach specified cumulative percentage
    thresholds.

    Args:
        cumulative_percentage (list[float]): Cumulative percentage distribution.
        thresholds (list[int]): List of percentage thresholds to find.

    Returns:
        dict[int, int]: Mapping from threshold percentage to the number of properties
        needed.
    """
    properties_for_thresholds: dict[int, int] = {}

    print("Finding the number of properties needed to cover specified thresholds...")
    for threshold in thresholds:
        for i, pct in enumerate(cumulative_percentage):
            if pct >= threshold:
                properties_for_thresholds[threshold] = (
                    i + 1
                )  # +1 because index is 0-based
                break
    return properties_for_thresholds


def plot_cumulative_distribution_properties(
    cumulative_percentage: list[float],
    thresholds: list[int],
    properties_for_thresholds: dict[int, int],
    output_directory: str,
    output_filename: str = "properties_cumulative_distribution.pdf",
) -> None:
    """
    Plots the cumulative distribution of property usage and marks specified thresholds.

    Args:
        cumulative_percentage (list[float]): Cumulative percentage distribution.
        thresholds (list[int]): List of percentage thresholds to mark on the plot.
        properties_for_thresholds (dict[int, int]): Mapping from threshold to number of
            properties needed.
        output_directory (str): Directory to save the plot.
        output_filename (str): Filename for the saved plot.
    """
    print("\nPlotting the cumulative distribution of properties...")
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, len(cumulative_percentage) + 1),
        cumulative_percentage,
        marker="o",
        linestyle="-",
        color="b",
        markersize=2,
    )
    plt.xlabel("Number of Properties")
    plt.ylabel("Cumulative Percentage (%)")
    plt.title("Cumulative Distribution of Properties Used")

    for threshold in thresholds:
        x_val = properties_for_thresholds.get(threshold)
        if x_val:
            plt.axvline(
                x=x_val, linestyle="--", label=f"{threshold}% at {x_val} properties"
            )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(output_directory, output_filename)
    try:
        plt.savefig(save_path)
        print(f"Plot saved as '{save_path}'.")
    except Exception as e:
        print(f"Error saving plot '{save_path}': {e}")
    finally:
        plt.close()


def print_top_properties(
    properties_used: dict[str, int],
    properties_labels: dict[str, dict[str, any]],
    top_n: int = 100,
) -> None:
    """
    Prints details for the top N properties based on their usage counts.

    Args:
        properties_used (dict[str, int]): Mapping of property IDs to their usage counts.
        properties_labels (dict[str, dict[str, any]]): Mapping of property IDs to their
            labels.
        top_n (int): Number of top properties to print.
    """
    print(f"\nPrinting details for the top {top_n} properties...")
    top_properties = islice(properties_used.items(), top_n)
    for prop_id, count in tqdm(
        top_properties, desc=f"Printing Top {top_n} Properties", total=top_n
    ):
        label = properties_labels.get(prop_id, {}).get("label", "N/A")
        print(f"{prop_id} | {label} | {count}")

    print(f"\nTop {top_n} properties printed successfully.")


# ------------------------------------------------------------------------------
# 3. P31/P279 PROCESSING FUNCTIONS
# ------------------------------------------------------------------------------


def load_entity_labels(filepath: str) -> dict[str, str]:
    """
    Loads entity labels from a JSON file.

    Args:
        filepath (str): Path to the entityid2label.json file.

    Returns:
        dict[str, str]: Mapping of entity IDs to their labels.
    """
    print(f"Loading entity labels from '{filepath}'...")
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            entityid2label = json.load(f)
        print(f"Loaded {len(entityid2label)} entity labels.\n")
        return entityid2label
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: File '{filepath}' is not a valid JSON.")
        return {}


def load_relationships_p31_p279(
    glob_path: str,
    relationship_type: str = "subclass",
    entityid2label: dict[str, str] | None = None,
) -> dict[str, list[str]]:
    """
    Loads relationships from TSV files and populates the mapping.

    Args:
        glob_path (str): Glob pattern to match TSV files.
        relationship_type (str): Type of relationship ('subclass' or 'instance').
        entityid2label (dict[str, str] | None): Mapping of entity IDs to labels for
            filtering.

    Returns:
        dict[str, list[str]]: Mapping from child to parents or entities to instances.
    """
    if relationship_type not in {"subclass", "instance"}:
        raise ValueError("relationship_type must be either 'subclass' or 'instance'")

    mapping: dict[str, list[str]] = {}
    print(f"Loading {relationship_type} information from '{glob_path}'...")
    for file_path in tqdm(glob(glob_path), desc=f"Loading {relationship_type} data"):
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                tsv_reader = csv.reader(file, delimiter="\t")
                for row in tsv_reader:
                    if not row or "entity_id" in row[0]:
                        continue  # Skip header or malformed rows
                    if len(row) < 3:
                        continue  # Ensure there are enough columns
                    child_id, _, parent_id = row
                    if child_id not in mapping:
                        mapping[child_id] = []
                    mapping[child_id].append(parent_id)
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found.")
            continue
        except Exception as e:
            print(f"Error processing file '{file_path}': {e}")
            continue

    print(f"{len(mapping)} entities have {relationship_type} information")

    if entityid2label:
        print(f"Removing non-English {relationship_type} entries...")
        original_length = len(mapping)
        mapping = {
            key: val
            for key, val in mapping.items()
            if key in entityid2label and all(v in entityid2label for v in val)
        }
        print(
            f"{len(mapping)} entities have English {relationship_type} information "
            f"(removed {original_length - len(mapping)} entries)."
        )

    return mapping


def count_classes_p31_p279(entity_instance_of: dict[str, list[str]]) -> dict[str, int]:
    """
    Counts the frequency of each class based on instance data and returns a sorted
    dictionary.

    Args:
        entity_instance_of (dict[str, list[str]]): Mapping from entities to their
            classes.

    Returns:
        dict[str, int]: A sorted dictionary mapping class IDs to their occurrence counts
            in descending order.
    """
    print("Counting classes...")
    class_counts = Counter()
    for entity_id, classes in tqdm(
        entity_instance_of.items(), desc="Processing Entities"
    ):
        class_counts.update(classes)
    print(f"Found {len(class_counts)} distinct classes.\n")

    # Sort the class_counts by count descendingly and convert to a regular dict
    sorted_class_counts = dict(class_counts.most_common())

    print("class_counts is sorted in descending order.")

    return sorted_class_counts


def compute_cumulative_distribution_classes(
    class_counts: dict[str, int]
) -> list[float]:
    """
    Computes the cumulative distribution percentage of class counts sorted in descending
    order.

    Args:
        class_counts (dict[str, int]): Mapping from class IDs to their occurrence
            counts.

    Returns:
        list[float]: Cumulative percentage distribution.
    """
    print("Calculating cumulative distribution in a single pass (O(n))...")
    cumulative_counts: list[int] = []
    running_sum = 0
    for cnt in class_counts.values():
        running_sum += cnt
        cumulative_counts.append(running_sum)

    total_count = running_sum  # sum of all occurrences
    if total_count == 0:
        cumulative_percentage: list[float] = []
    else:
        cumulative_percentage: list[float] = [
            count / total_count * 100 for count in cumulative_counts
        ]

    return cumulative_percentage


def find_thresholds_classes(
    cumulative_percentage: list[float], thresholds: list[int]
) -> dict[int, int]:
    """
    Finds the number of classes needed to reach specified cumulative percentage
    thresholds.

    Args:
        cumulative_percentage (list[float]): Cumulative percentage distribution of class
           counts.
        thresholds (list[int]): List of percentage thresholds to find.

    Returns:
        dict[int, int]: Mapping from threshold percentage to the number of classes
           needed.
    """
    classes_for_thresholds: dict[int, int] = {}

    print("Finding the number of classes needed to cover specified thresholds...")
    for threshold in thresholds:
        for i, pct in enumerate(cumulative_percentage):
            if pct >= threshold:
                classes_for_thresholds[threshold] = i + 1  # +1 because index is 0-based
                break

    return classes_for_thresholds


def plot_cumulative_distribution_classes(
    cumulative_percentage: list[float],
    thresholds: list[int],
    classes_for_thresholds: dict[int, int],
    output_directory: str,
    output_filename: str = "classes_cumulative_distribution.pdf",
) -> None:
    """
    Plots the cumulative distribution of class counts and marks specified thresholds.

    Args:
        cumulative_percentage (list[float]): Cumulative percentage distribution.
        thresholds (list[int]): List of percentage thresholds to mark on the plot.
        classes_for_thresholds (dict[int, int]): Mapping from threshold to number of
            classes needed.
        output_directory (str): Directory to save the plot.
        output_filename (str): Filename for the saved plot.
    """
    print("\nPlotting the cumulative distribution of classes...")
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, len(cumulative_percentage) + 1),
        cumulative_percentage,
        marker="o",
        linestyle="-",
        color="g",
        markersize=2,
    )
    plt.xlabel("Number of Classes (sorted by frequency)")
    plt.ylabel("Cumulative Percentage (%)")
    plt.title("Cumulative Distribution of Classes (P31)")

    for threshold in thresholds:
        x_val = classes_for_thresholds.get(threshold)
        if x_val:
            plt.axvline(
                x=x_val, linestyle="--", label=f"{threshold}% at {x_val} classes"
            )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(output_directory, output_filename)
    try:
        plt.savefig(save_path)
        print(f"Plot saved as '{save_path}'.")
    except Exception as e:
        print(f"Error saving plot '{save_path}': {e}")
    finally:
        plt.close()


def print_top_classes(
    class_counts: dict[str, int], entityid2label: dict[str, str], top_n: int = 100
) -> None:
    """
    Prints details for the top N classes based on their counts.

    Args:
        class_counts (dict[str, int]): Mapping from class IDs to their occurrence
            counts.
        entityid2label (dict[str, str]): Mapping from entity IDs to their labels.
        top_n (int): Number of top classes to print.
    """
    print(f"\nPrinting details for the top {top_n} classes...")
    top_classes = islice(class_counts.items(), top_n)
    for class_id, cnt in tqdm(
        top_classes, desc=f"Printing Top {top_n} Classes", total=top_n
    ):
        label = entityid2label.get(class_id, "N/A")
        print(f"{class_id} | {label} | {cnt}")

    print(f"\nTop {top_n} classes printed successfully.")


# ------------------------------------------------------------------------------
# 4. DATA SAVING FUNCTIONS
# ------------------------------------------------------------------------------


def save_to_json(data: Union[dict, list], filename: str) -> None:
    """
    Saves a Python object to a JSON file.

    Args:
        data (dict | list): The data to serialize.
        filename (str): The target JSON file path.
    """
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Saved data to '{filename}'.")
    except Exception as e:
        print(f"Error saving data to '{filename}': {e}")


# ------------------------------------------------------------------------------
# 5. TIME FORMATTING AND LOGGING FUNCTIONS
# ------------------------------------------------------------------------------


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


def log_statistics(
    log_file: str,
    elapsed_time: float,
    total_entities: int,
    total_properties: int,
    total_classes: int,
    num_p279_entries: int,
    num_p31_entries: int,
    num_errors: int,
) -> None:
    """
    Logs the processing statistics to a log file.

    Args:
        log_file (str): Path to the log file.
        elapsed_time (float): Total elapsed time in seconds.
        total_entities (int): Total number of entities processed.
        total_properties (int): Total number of properties processed.
        total_classes (int): Total number of classes counted.
        num_p279_entries (int): Number of P279 entries processed.
        num_p31_entries (int): Number of P31 entries processed.
        num_errors (int): Number of errors encountered.
    """
    try:
        with open(log_file, "w", encoding="utf-8") as log:
            log.write(f"Processing completed in {format_time(elapsed_time)}\n")
            log.write(f"Total properties loaded: {total_properties}\n")
            log.write(f"Total classes counted: {total_classes}\n")
            log.write(f"Total entities processed: {total_entities}\n")
            log.write(f"P279 entries processed: {num_p279_entries}\n")
            log.write(f"P31 entries processed: {num_p31_entries}\n")
            log.write(f"Total errors encountered: {num_errors}\n")
        print(f"Logged statistics to '{log_file}'.")
    except Exception as e:
        print(f"Error writing to log file '{log_file}': {e}")


# ------------------------------------------------------------------------------
# 6. MAIN FUNCTION
# ------------------------------------------------------------------------------


def main() -> None:
    """
    Main function to execute the data processing pipeline.
    """
    # Define the output directory
    output_directory = "process_p31_p279"

    # Create the output directory
    create_output_directory(output_directory)

    # Initialize statistics
    start_time = time.perf_counter()
    total_errors = 0

    # ------------------------------------------------------------------------------
    # 1. PROPERTY PROCESSING
    # ------------------------------------------------------------------------------

    # Load property usage statistics
    properties_used = load_properties_used("./property_stats.json")
    total_properties = len(properties_used)

    if properties_used:
        # Load property labels
        properties_labels = load_properties_labels("./properties.json")

        # Extract counts and sort in descending order
        property_ids, property_counts = zip(*properties_used.items())
        sorted_counts = sorted(property_counts, reverse=True)

        # Compute cumulative counts and percentages
        cumulative_counts_properties = compute_cumulative_distribution_counts(
            list(sorted_counts)
        )
        cumulative_percentage_properties = compute_cumulative_percentage(
            cumulative_counts_properties
        )

        # Define thresholds
        thresholds_properties = [80, 90, 95, 99]

        # Find thresholds
        properties_for_thresholds = find_thresholds_in_distribution(
            cumulative_percentage_properties, thresholds_properties
        )

        # Print threshold results
        print("\nResults for property thresholds (80%, 90%, 95%, 99%):")
        num_total_properties = len(properties_used)
        for threshold in thresholds_properties:
            num_properties_needed = properties_for_thresholds.get(threshold, 0)
            proportion = (
                (num_properties_needed / num_total_properties * 100)
                if num_total_properties
                else 0
            )
            print(
                f"  - To reach {threshold}% coverage, need {num_properties_needed} properties "
                f"({proportion:.2f}% of total {num_total_properties})."
            )

        # Plot and save cumulative distribution of properties
        plot_cumulative_distribution_properties(
            cumulative_percentage=cumulative_percentage_properties,
            thresholds=thresholds_properties,
            properties_for_thresholds=properties_for_thresholds,
            output_directory=output_directory,
            output_filename="properties_cumulative_distribution.pdf",
        )

        # Print top 100 properties
        print_top_properties(
            properties_used=properties_used,
            properties_labels=properties_labels,
            top_n=100,
        )
    else:
        print("No property usage data to process.\n")

    # ------------------------------------------------------------------------------
    # 2. P31/P279 PROCESSING
    # ------------------------------------------------------------------------------

    # Load entity labels
    entityid2label = load_entity_labels("./entityid2label.json")
    total_entities = len(entityid2label)

    if not entityid2label:
        print("Entity labels are missing. Exiting P31/P279 processing.")
    else:
        # Load subclass information (P279)
        child_to_parents = load_relationships_p31_p279(
            glob_path="./P279/*.tsv",
            relationship_type="subclass",
            entityid2label=entityid2label,
        )
        num_p279_entries = len(child_to_parents)

        # Load instance information (P31)
        entity_instance_of = load_relationships_p31_p279(
            glob_path="./P31/*.tsv",
            relationship_type="instance",
            entityid2label=entityid2label,
        )
        num_p31_entries = len(entity_instance_of)

        # Count classes
        class_counts = count_classes_p31_p279(entity_instance_of)
        total_classes = len(class_counts)

        # Compute cumulative distribution of classes
        cumulative_percentage_classes = compute_cumulative_distribution_classes(
            class_counts
        )

        # Define thresholds
        thresholds_classes = [80, 90, 95, 99]

        # Find thresholds
        classes_for_thresholds = find_thresholds_classes(
            cumulative_percentage=cumulative_percentage_classes,
            thresholds=thresholds_classes,
        )

        # Print threshold results
        print("\nResults for class thresholds (80%, 90%, 95%, 99%):")
        total_classes_count = len(class_counts)
        for threshold in thresholds_classes:
            num_classes_needed = classes_for_thresholds.get(threshold, 0)
            proportion = (
                (num_classes_needed / total_classes_count * 100)
                if total_classes_count
                else 0
            )
            print(
                f"  - To reach {threshold}% coverage, need {num_classes_needed} classes "
                f"({proportion:.2f}% of total {total_classes_count})."
            )

        # Plot and save cumulative distribution of classes
        plot_cumulative_distribution_classes(
            cumulative_percentage=cumulative_percentage_classes,
            thresholds=thresholds_classes,
            classes_for_thresholds=classes_for_thresholds,
            output_directory=output_directory,
            output_filename="classes_cumulative_distribution.pdf",
        )

        # Print top 100 classes
        print_top_classes(
            class_counts=class_counts, entityid2label=entityid2label, top_n=100
        )

        # ------------------------------------------------------------------------------
        # 3. SAVE DATA STRUCTURES AND FIGURES
        # ------------------------------------------------------------------------------

        print("\nSaving data structures and figures...")

        # Save class_counts
        class_counts_dict = dict(class_counts)
        save_to_json(
            class_counts_dict, os.path.join(output_directory, "class_counts.json")
        )

        # Save child_to_parents
        save_to_json(
            child_to_parents, os.path.join(output_directory, "child_to_parents.json")
        )

    # ------------------------------------------------------------------------------
    # 4. LOGGING AND STATISTICS
    # ------------------------------------------------------------------------------

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    # Define the log file path
    log_file = os.path.join(output_directory, "run_p31_p279.log")

    # Log statistics
    log_statistics(
        log_file=log_file,
        elapsed_time=elapsed_time,
        total_entities=total_entities,
        total_properties=total_properties,
        total_classes=total_classes if "total_classes" in locals() else 0,
        num_p279_entries=num_p279_entries if "num_p279_entries" in locals() else 0,
        num_p31_entries=num_p31_entries if "num_p31_entries" in locals() else 0,
        num_errors=total_errors,
    )

    # Also, print statistics to the console
    print(f"\nProcessing completed in {format_time(elapsed_time)}")
    print(f"Total properties loaded: {total_properties}")
    print(
        f"Total classes counted: {total_classes if 'total_classes' in locals() else 0}"
    )
    print(f"Total entities processed: {total_entities}")
    print(
        f"P279 entries processed: {num_p279_entries if 'num_p279_entries' in locals() else 0}"
    )
    print(
        f"P31 entries processed: {num_p31_entries if 'num_p31_entries' in locals() else 0}"
    )
    print(f"Total errors encountered: {total_errors}")


# ------------------------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------------------------


if __name__ == "__main__":
    main()
