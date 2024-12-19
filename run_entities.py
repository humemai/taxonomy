"""
This script processes a Wikidata JSON dump into simplified JSON batches.
It reads the compressed JSON file in chunks, simplifies the entity data by
extracting essential fields and relationships, and saves the processed
data into batch files. The script supports multiprocessing for efficiency.

Usage:
    python run.py <file_path> <output_dir> [--num_workers <int>]
    [--num_entities_per_batch <int>] [--lines_per_chunk <int>]

Arguments:
    file_path: Path to the gzipped Wikidata JSON file (e.g., `latest-all.json.gz`).
    output_dir: Directory to store the output batch files.
    --num_workers: Number of parallel worker processes (default: 4).
    --num_entities_per_batch: Number of entities per batch file (default: 10000).
    --lines_per_chunk: Number of lines to read per chunk (default: 100000).
"""

import gzip
import ijson
import os
import json
import argparse
from decimal import Decimal
from multiprocessing import Pool


def custom_serializer(obj) -> None:
    """
    Serialize non-serializable objects like Decimal.

    Args:
        obj: The object to serialize.

    Returns:
        float: The serialized object as a float.

    Raises:
        TypeError: If the object type is not serializable.
    """
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError(f"Type {type(obj)} not serializable")


def extract_claims(claims: dict) -> dict:
    """
    Extract simplified claims and their qualifiers from an entity's claims.

    Args:
        claims (dict): Claims dictionary from a Wikidata entity.

    Returns:
        dict: A simplified representation of claims with qualifiers.
    """
    simplified_claims = {}
    for property_id, statements in claims.items():
        simplified_claims[property_id] = []
        for statement in statements:
            value = statement.get("mainsnak", {}).get("datavalue", {}).get("value")
            if isinstance(value, dict) and "entity-type" in value:
                value = value.get("id")
            qualifiers = {
                qualifier_id: [
                    q.get("datavalue", {}).get("value") for q in qualifier_statements
                ]
                for qualifier_id, qualifier_statements in statement.get(
                    "qualifiers", {}
                ).items()
            }
            simplified_claims[property_id].append(
                {"value": value, "qualifiers": qualifiers}
            )
    return simplified_claims


def process_batch(batch: list, output_dir: str, batch_idx: int) -> None:
    """
    Process a batch of entities and save them as a JSON file.

    Args:
        batch (list): List of simplified entities.
        output_dir (str): Directory to store the batch file.
        batch_idx (int): Index of the batch for file naming.
    """
    batch_file = os.path.join(output_dir, f"batch_{batch_idx}.json")
    with open(batch_file, "w", encoding="utf-8") as f:
        json.dump(batch, f, ensure_ascii=False, indent=4, default=custom_serializer)


def worker(task: tuple) -> None:
    """
    Worker function to process a single batch task.

    Args:
        task (tuple): A tuple containing the batch, output directory, and batch index.
    """
    batch, output_dir, batch_idx = task
    process_batch(batch, output_dir, batch_idx)


def process_in_chunks(
    file_path: str,
    output_dir: str,
    num_workers: int,
    num_entities_per_batch: int,
    lines_per_chunk: int,
) -> None:
    """
    Process the Wikidata JSON dump in manageable chunks.

    Args:
        file_path (str): Path to the gzipped Wikidata JSON file.
        output_dir (str): Directory to store the output batch files.
        num_workers (int): Number of worker processes for multiprocessing.
        num_entities_per_batch (int): Number of entities per batch file.
        lines_per_chunk (int): Number of lines to read into memory per chunk.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Processing {file_path} with {num_workers} workers...")

    with gzip.open(file_path, "rt", encoding="utf-8") as file:
        entity_count = 0
        batch_idx = 0
        batch = []
        chunk = []

        try:
            # Attempt to parse as JSON array using ijson
            for item in ijson.items(file, "item"):
                chunk.append(item)
                if len(chunk) >= lines_per_chunk:
                    for entity in chunk:
                        filtered_entity = {
                            "id": entity.get("id"),
                            "type": entity.get("type"),
                            "labels": (
                                {"en": entity["labels"]["en"]["value"]}
                                if "labels" in entity and "en" in entity["labels"]
                                else {}
                            ),
                            "descriptions": (
                                {"en": entity["descriptions"]["en"]["value"]}
                                if "descriptions" in entity
                                and "en" in entity["descriptions"]
                                else {}
                            ),
                            "aliases": (
                                {
                                    "en": [
                                        alias["value"]
                                        for alias in entity["aliases"]["en"]
                                    ]
                                }
                                if "aliases" in entity and "en" in entity["aliases"]
                                else {}
                            ),
                            "claims": extract_claims(entity.get("claims", {})),
                            "modified": entity.get("modified"),
                        }
                        batch.append(filtered_entity)
                        entity_count += 1

                        if len(batch) >= num_entities_per_batch:
                            task = (batch, output_dir, batch_idx)
                            with Pool(num_workers) as pool:
                                pool.map(worker, [task])
                            batch = []
                            batch_idx += 1

                            if entity_count % 10000 == 0:
                                print(f"Processed {entity_count} entities...")

                    chunk = []

        except ijson.common.IncompleteJSONError:
            # Fallback to reading as newline-delimited JSON
            print("Falling back to line-by-line parsing (ndjson detected)...")
            file.seek(0)
            for line in file:
                entity = json.loads(line.strip())
                filtered_entity = {
                    "id": entity.get("id"),
                    "type": entity.get("type"),
                    "labels": (
                        {"en": entity["labels"]["en"]["value"]}
                        if "labels" in entity and "en" in entity["labels"]
                        else {}
                    ),
                    "descriptions": (
                        {"en": entity["descriptions"]["en"]["value"]}
                        if "descriptions" in entity and "en" in entity["descriptions"]
                        else {}
                    ),
                    "aliases": (
                        {"en": [alias["value"] for alias in entity["aliases"]["en"]]}
                        if "aliases" in entity and "en" in entity["aliases"]
                        else {}
                    ),
                    "claims": extract_claims(entity.get("claims", {})),
                    "modified": entity.get("modified"),
                }
                batch.append(filtered_entity)
                entity_count += 1

                if len(batch) >= num_entities_per_batch:
                    task = (batch, output_dir, batch_idx)
                    with Pool(num_workers) as pool:
                        pool.map(worker, [task])
                    batch = []
                    batch_idx += 1

                    if entity_count % 10000 == 0:
                        print(f"Processed {entity_count} entities...")

        if batch:
            task = (batch, output_dir, batch_idx)
            with Pool(num_workers) as pool:
                pool.map(worker, [task])

    print(f"Processing completed. Total entities processed: {entity_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process Wikidata JSON dump into simplified JSON batches."
    )
    parser.add_argument(
        "file_path", type=str, help="Path to the gzipped Wikidata JSON file."
    )
    parser.add_argument(
        "output_dir", type=str, help="Directory to store the output batches."
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of parallel workers."
    )
    parser.add_argument(
        "--num_entities_per_batch",
        type=int,
        default=10000,
        help="Entities per batch file.",
    )
    parser.add_argument(
        "--lines_per_chunk", type=int, default=100000, help="Lines to read per chunk."
    )

    args = parser.parse_args()
    process_in_chunks(
        args.file_path,
        args.output_dir,
        args.num_workers,
        args.num_entities_per_batch,
        args.lines_per_chunk,
    )
