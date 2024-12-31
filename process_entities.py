"""
This script processes batches of Wikidata entity JSON files in parallel.
It extracts relevant information, counts missing data, and saves the results in various JSON files.
The script supports parallel processing using the specified number of processes.

The following JSON files are generated:
- entityid2label.json: Maps entity IDs to their English labels.
- entity_instance_of.json: Contains the "instance of" (P31) claims for each entity.
- entity_subclass_of.json: Contains the "subclass of" (P279) claims for each entity.
- properties_used.json: A count of all properties used, ordered by their occurrences.
- stats.json: A summary of missing data (e.g., missing labels or properties).
"""

import json
import os
from glob import glob
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
import argparse


def process_entity_batch(batch_json: str, include_qualifiers: bool = False) -> tuple[
    dict[str, str],
    dict[str, list[str]],
    dict[str, list[str]],
    list[str],
    dict[str, int],
]:
    """
    Processes a batch of entity data from a given JSON file.

    Args:
        batch_json (str): The path to the JSON file containing entities.
        include_qualifiers (bool): Whether to include qualifiers in the claims.

    Returns:
        tuple: A tuple containing processed data:
            - A dictionary of entity IDs to labels (English).
            - A dictionary of entity IDs to "instance of" (P31) claims.
            - A dictionary of entity IDs to "subclass of" (P279) claims.
            - A list of used properties in the batch.
            - A dictionary of counts for missing data (e.g., missing labels or properties).
    """
    local_entityid2label = {}
    local_entity_instance_of = {}
    local_entity_subclass_of = {}
    local_properties_used = []
    local_count = {"no_english_label": 0, "no_P31": 0, "no_P279": 0, "no_properties": 0}

    print(f"Processing {batch_json} ...")
    with open(batch_json, encoding="utf-8") as f:
        entities = json.load(f)
    print(f"{batch_json} has {len(entities)} entities ...")

    for entity in entities:
        try:
            local_entityid2label[entity["id"]] = entity["labels"]["en"]
        except KeyError:
            local_count["no_english_label"] += 1

        try:
            if include_qualifiers:
                local_entity_instance_of[entity["id"]] = entity["claims"]["P31"]
            else:
                local_entity_instance_of[entity["id"]] = [
                    foo["value"] for foo in entity["claims"]["P31"]
                ]
        except KeyError:
            local_count["no_P31"] += 1

        try:
            if include_qualifiers:
                local_entity_subclass_of[entity["id"]] = entity["claims"]["P279"]
            else:
                local_entity_subclass_of[entity["id"]] = [
                    foo["value"] for foo in entity["claims"]["P279"]
                ]
        except KeyError:
            local_count["no_P279"] += 1

        try:
            local_properties_used.extend(list(entity["claims"].keys()))
        except KeyError:
            local_count["no_properties"] += 1

    return (
        local_entityid2label,
        local_entity_instance_of,
        local_entity_subclass_of,
        local_properties_used,
        local_count,
    )


def main(num_processes: int, include_qualifiers: bool, dummy: bool) -> None:
    """
    Main function to process entity batches in parallel, count missing data,
    and save results to JSON files.

    Args:
        num_processes (int): The number of processes to use for parallel execution.
        include_qualifiers (bool): Whether to include qualifiers in the claims.
        dummy (bool): Whether to process only 10 files for testing purposes.
    """
    # Get list of all JSON files in the "./entities" directory
    batches = glob("./entities/*.json")

    # If --dummy is set, only process 10 files
    if dummy:
        batches = batches[:10]

    # Data containers
    entityid2label = {}
    entity_instance_of = {}
    entity_subclass_of = {}
    properties_used = []
    count = {"no_english_label": 0, "no_P31": 0, "no_P279": 0, "no_properties": 0}

    # Create a ProcessPoolExecutor to parallelize the processing of batches
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = []
        for batch_json in tqdm(sorted(batches)):
            futures.append(
                executor.submit(process_entity_batch, batch_json, include_qualifiers)
            )

        for future in as_completed(futures):
            (
                local_entityid2label,
                local_entity_instance_of,
                local_entity_subclass_of,
                local_properties_used,
                local_count,
            ) = future.result()

            # Combine results from each process
            entityid2label.update(local_entityid2label)
            entity_instance_of.update(local_entity_instance_of)
            entity_subclass_of.update(local_entity_subclass_of)
            properties_used.extend(local_properties_used)
            for key, value in local_count.items():
                count[key] += value

    # Save the results to JSON files
    with open("entityid2label.json", "w", encoding="utf-8") as f:
        json.dump(entityid2label, f, ensure_ascii=False, indent=4)

    with open("entity_instance_of.json", "w", encoding="utf-8") as f:
        json.dump(entity_instance_of, f, ensure_ascii=False, indent=4)

    with open("entity_subclass_of.json", "w", encoding="utf-8") as f:
        json.dump(entity_subclass_of, f, ensure_ascii=False, indent=4)

    # Count and save properties used, ordered by occurrence
    properties_count = dict(Counter(properties_used))
    properties_count = dict(
        sorted(properties_count.items(), key=lambda item: item[1], reverse=True)
    )
    with open("properties_used.json", "w", encoding="utf-8") as f:
        json.dump(properties_count, f, ensure_ascii=False, indent=4)

    # Save the stats as a separate JSON file
    with open("stats.json", "w", encoding="utf-8") as f:
        json.dump(count, f, ensure_ascii=False, indent=4)

    # Print the summary count
    print(count)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process entity batches in parallel.")
    parser.add_argument(
        "--num_processes", type=int, default=1, help="Number of processes to use."
    )
    parser.add_argument(
        "--include_qualifiers",
        action="store_true",
        help="Include qualifiers in the claims.",
    )
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Only process 10 files (for testing purposes).",
    )
    args = parser.parse_args()

    main(args.num_processes, args.include_qualifiers, args.dummy)
