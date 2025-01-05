"""
This script processes a Wikidata JSON dump into simplified JSON batches.
It reads the compressed JSON file in chunks, simplifies the entity data by
extracting essential fields and relationships, and saves the processed
data into batch files. The script supports multiprocessing for efficiency.

Features:
- Handles both array-based and newline-delimited JSON formats.
- Strips beginning (`[`) and ending (`]`) brackets for JSON arrays.
- Supports multiprocessing for faster batch processing.
- Efficiently writes valid JSON output batches.
- Logs processing time, output details, and statistics.
- Includes a `dummy run` mode for quick testing.

Usage:
    python run.py <file_path> <output_dir> [--num_workers <int>]
    [--num_entities_per_batch <int>] [--dummy]

Arguments:
    file_path: Path to the gzipped Wikidata JSON file (e.g., `latest-all.json.gz`).
    output_dir: Directory to store the output batch files.
    --num_workers: Number of parallel worker processes (default: 4).
    --num_entities_per_batch: Number of entities per batch file (default: 10000).
    --dummy: Run in dummy mode (process 100 entities with preset parameters).
"""

import gzip
import ijson
import os
import json
import argparse
import time
from decimal import Decimal
from multiprocessing import Pool


def custom_serializer(obj: any) -> float:
    """
    Serialize non-serializable objects like Decimal.

    Args:
        obj (any): The object to serialize.

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


def process_entity(entity: dict) -> dict:
    """
    Simplify an individual entity for batch processing.

    Args:
        entity (dict): The original entity dictionary.

    Returns:
        dict: The simplified entity dictionary.
    """
    return {
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


def format_size(size: int) -> str:
    """
    Format size in bytes to a human-readable string (e.g., KiB, MiB, GiB, TiB).

    Args:
        size (int): Size in bytes.

    Returns:
        str: Human-readable size string.
    """
    for unit in ["bytes", "KiB", "MiB", "GiB", "TiB"]:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0


def write_batch(batch: list, output_dir: str, batch_idx: int) -> None:
    """
    Write a batch of entities to a JSON file.

    Args:
        batch (list): List of simplified entities.
        output_dir (str): Directory to store the batch file.
        batch_idx (int): Index of the batch for file naming.
    """
    batch_file = os.path.join(output_dir, f"batch_{batch_idx}.json")
    with open(batch_file, "w", encoding="utf-8") as f:
        json.dump(batch, f, ensure_ascii=False, indent=4, default=custom_serializer)


def format_time(seconds: float) -> str:
    """
    Format time duration into days, hours, minutes, and seconds.

    Args:
        seconds (float): Time duration in seconds.

    Returns:
        str: Formatted time duration string.
    """
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    parts = []
    if days > 0:
        parts.append(f"{int(days)}d")
    if hours > 0 or days > 0:
        parts.append(f"{int(hours)}h")
    if minutes > 0 or hours > 0 or days > 0:
        parts.append(f"{int(minutes)}m")
    parts.append(f"{seconds:.2f}s")
    return " ".join(parts)


def process_in_chunks(
    file_path: str,
    output_dir: str,
    num_workers: int,
    num_entities_per_batch: int,
    dummy: bool = False,
) -> None:
    """
    Process the Wikidata JSON dump in manageable chunks with parallel entity processing.

    Args:
        file_path (str): Path to the gzipped Wikidata JSON file.
        output_dir (str): Directory to store the output batch files.
        num_workers (int): Number of worker processes for multiprocessing.
        num_entities_per_batch (int): Number of entities per batch file.
        dummy (bool): Whether to run in dummy mode.
    """
    if dummy:
        print("Running in dummy mode...")
        output_dir = "./dummy"
        num_workers = 4
        num_entities_per_batch = 123

    os.makedirs(output_dir, exist_ok=True)
    print(f"Processing {file_path} with {num_workers} workers...")

    start_time = time.time()
    entity_count = 0
    batch_idx = 0
    entity_buffer = []

    with gzip.open(file_path, "rt", encoding="utf-8") as file:
        # Initialize the pool of workers
        with Pool(num_workers) as pool:
            # Use ijson to parse individual JSON objects (entities) from the array
            for entity in ijson.items(file, "item"):
                entity_buffer.append(entity)
                entity_count += 1

                # If dummy mode and limit reached, break early
                if dummy and entity_count >= 987:
                    break

                # Once we have enough entities for a batch, process them
                if len(entity_buffer) >= num_entities_per_batch:
                    print(f"Processing batch {batch_idx}...")
                    processed_batch = pool.map(process_entity, entity_buffer)
                    write_batch(processed_batch, output_dir, batch_idx)

                    if entity_count % num_entities_per_batch == 0 and not dummy:
                        print(
                            f"Processed batch {batch_idx}!\n"
                            f"{entity_count} entities processed so far..."
                        )

                    batch_idx += 1
                    entity_buffer = []

            # Handle the last batch if any remain
            if entity_buffer:
                processed_batch = pool.map(process_entity, entity_buffer)
                write_batch(processed_batch, output_dir, batch_idx)

    end_time = time.time()

    # Log processing stats
    log_file = os.path.join(output_dir, "run_entities.log")
    if entity_count > 0:
        original_size = os.path.getsize(file_path)
        output_size = sum(
            os.path.getsize(os.path.join(output_dir, f))
            for f in os.listdir(output_dir)
            if f.endswith(".json")
        )
        avg_entity_size = output_size / entity_count if entity_count > 0 else 0
    else:
        original_size = 0
        output_size = 0
        avg_entity_size = 0

    total_time = end_time - start_time

    with open(log_file, "w") as log:
        log.write(f"Processing completed in {format_time(total_time)}\n")
        log.write(f"Total entities processed: {entity_count}\n")
        log.write(f"Original file ({file_path}) size : {format_size(original_size)}\n")
        log.write(f"Output directory size: {format_size(output_size)}\n")
        log.write(f"Average entity size: {avg_entity_size:.2f} bytes\n")

    print(f"Log written to {log_file}")


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
        default=50000,
        help="Entities per batch file.",
    )
    parser.add_argument("--dummy", action="store_true", help="Run in dummy mode.")

    args = parser.parse_args()

    process_in_chunks(
        args.file_path,
        args.output_dir,
        args.num_workers,
        args.num_entities_per_batch,
        dummy=args.dummy,
    )
