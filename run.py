import gzip
import ijson
import os
import json
import argparse
from decimal import Decimal
from multiprocessing import Pool, Manager
from math import ceil


# Custom JSON serializer to handle non-serializable types
def custom_serializer(obj):
    """
    Serialize non-serializable objects like Decimal.

    Args:
        obj: The object to serialize.

    Returns:
        The serialized object as float.

    Raises:
        TypeError: If the object type is not serializable.
    """
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError(f"Type {type(obj)} not serializable")


# Helper function to extract main triples and qualifiers
def extract_claims(claims):
    """
    Extract simplified claims and their qualifiers.

    Args:
        claims (dict): Claims dictionary from a Wikidata entity.

    Returns:
        dict: Simplified claims with qualifiers.
    """
    simplified_claims = {}
    for property_id, statements in claims.items():
        simplified_claims[property_id] = []
        for statement in statements:
            # Extract the main value
            value = statement.get("mainsnak", {}).get("datavalue", {}).get("value")
            # Convert entity-type items to IDs
            if isinstance(value, dict) and "entity-type" in value:
                value = value.get("id")

            # Extract qualifiers
            qualifiers = {}
            for qualifier_id, qualifier_statements in statement.get(
                "qualifiers", {}
            ).items():
                qualifiers[qualifier_id] = [
                    q.get("datavalue", {}).get("value") for q in qualifier_statements
                ]

            simplified_claims[property_id].append(
                {"value": value, "qualifiers": qualifiers}
            )
    return simplified_claims


# Function to process a batch of entities
def process_batch(batch, output_dir, batch_idx):
    """
    Process a batch of entities and save them as a JSON file.

    Args:
        batch (list): List of entities to process.
        output_dir (str): Directory to store the output file.
        batch_idx (int): Batch index for naming the file.
    """
    batch_file = os.path.join(output_dir, f"batch_{batch_idx}.json")
    with open(batch_file, "w", encoding="utf-8") as f:
        json.dump(batch, f, ensure_ascii=False, indent=4, default=custom_serializer)


# Worker function for multiprocessing
def worker(task):
    """
    Worker function for processing tasks.

    Args:
        task (tuple): Contains batch, output directory, and batch index.
    """
    batch, output_dir, batch_idx = task
    process_batch(batch, output_dir, batch_idx)


# Main function
def main(args):
    """
    Main function to process the Wikidata JSON dump.

    Args:
        args: Parsed command-line arguments.
    """
    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Processing {args.file_path} with {args.num_workers} workers...")

    # Open the gzipped file and initialize processing
    with gzip.open(args.file_path, "rt", encoding="utf-8") as file:
        entities = ijson.items(file, "item")
        manager = Manager()
        tasks = manager.Queue()

        batch = []
        batch_idx = 0
        entity_count = 0

        # Collect tasks for workers
        for entity in entities:
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

            if len(batch) >= args.num_entities_per_batch:
                tasks.put((batch, args.output_dir, batch_idx))
                batch = []
                batch_idx += 1

                if entity_count % 10000 == 0:
                    print(f"Queued {entity_count} entities for processing...")

        # Add any remaining entities
        if batch:
            tasks.put((batch, args.output_dir, batch_idx))

        # Start multiprocessing pool
        with Pool(args.num_workers) as pool:
            pool.map(worker, [tasks.get() for _ in range(ceil(tasks.qsize()))])

    print(f"Processing completed. Total entities processed: {entity_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process Wikidata JSON dump into simplified JSON batches."
    )
    parser.add_argument(
        "--file_path", type=str, help="Path to the gzipped Wikidata JSON file."
    )
    parser.add_argument(
        "--output_dir", type=str, help="Directory to store the output batches."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of parallel workers to process the data.",
    )
    parser.add_argument(
        "--num_entities_per_batch",
        type=int,
        default=10000,
        help="Number of entities to include in each batch file.",
    )

    args = parser.parse_args()
    main(args)
