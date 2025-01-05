"""
This script extracts `P279` (subclass of) claims
from a Wikidata JSON dump and saves them as TSV batch files in a specified directory.
It uses native `json` for parsing and supports a `dummy` mode for quick testing.

Usage:
    python extract_p279.py --dump_file <file> --p279_dir <directory>
                          [--num_entities_per_batch <int>] [--dummy]

Arguments:
    --dump_file (str): Path to the compressed Wikidata JSON dump file
                       (default: 'latest-all.json.gz').
    --p279_dir (str): Directory to save `P279` triples (default: 'P279').
    --num_entities_per_batch (int): Entities per batch file (default: 50000).
    --dummy: Optional flag to process only one batch.
"""

import gzip
import json
import os
import argparse
import time


def format_time(seconds: float) -> str:
    """
    Format time duration into days, hours, minutes, and seconds.
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


def extract_property_triples(entity: dict, property_id: str) -> list:
    """
    Extract triples for a specific property (`P279`) from an entity.
    """
    triples = []
    entity_id = entity.get("id")
    claims = entity.get("claims", {})

    if property_id in claims:
        for claim in claims[property_id]:
            mainsnak = claim.get("mainsnak", {})
            if mainsnak.get("snaktype") == "value":
                datavalue = mainsnak.get("datavalue", {})
                value = datavalue.get("value", {})

                # Ensure the value is a valid entity reference
                if isinstance(value, dict) and "id" in value:
                    triples.append((entity_id, property_id, value["id"]))

    return triples


def process_file(
    dump_file: str,
    p279_dir: str,
    num_entities_per_batch: int,
    dummy: bool,
) -> None:
    """
    Process the Wikidata dump file and extract P279 triples, saving results
    in the specified directory.
    """
    start_time = time.time()

    # Create directory for P279
    os.makedirs(p279_dir, exist_ok=True)

    entity_buffer = []
    batch_idx_p279 = 0
    total_entities = 0
    num_lines_error = 0

    with gzip.open(dump_file, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line in ("[", "]"):  # Skip JSON array brackets
                continue
            try:
                entity = json.loads(line.rstrip(","))
                entity_buffer.append(entity)
                total_entities += 1

                # Process batch when buffer is full
                if len(entity_buffer) >= num_entities_per_batch:
                    print(f"Processing batch {batch_idx_p279} for P279...")

                    # Process P279
                    triples_p279 = []
                    for entity in entity_buffer:
                        triples_p279.extend(extract_property_triples(entity, "P279"))
                    batch_file_p279 = os.path.join(
                        p279_dir, f"batch_{batch_idx_p279}.tsv"
                    )
                    with open(batch_file_p279, "w", encoding="utf-8") as out_f:
                        out_f.write("entity_id\tproperty_id\tvalue_id\n")
                        for triple in triples_p279:
                            out_f.write(f"{triple[0]}\t{triple[1]}\t{triple[2]}\n")
                    batch_idx_p279 += 1

                    entity_buffer = []

                    # Stop early in dummy mode
                    if dummy and batch_idx_p279 >= 1:
                        break

            except json.JSONDecodeError:
                print("Error decoding JSON line:", line)
                num_lines_error += 1
                continue

        # Process remaining entities in the buffer
        if entity_buffer:
            print(f"Processing final batch {batch_idx_p279} for P279...")

            triples_p279 = []
            for entity in entity_buffer:
                triples_p279.extend(extract_property_triples(entity, "P279"))
            batch_file_p279 = os.path.join(p279_dir, f"batch_{batch_idx_p279}.tsv")
            with open(batch_file_p279, "w", encoding="utf-8") as out_f:
                out_f.write("entity_id\tproperty_id\tvalue_id\n")
                for triple in triples_p279:
                    out_f.write(f"{triple[0]}\t{triple[1]}\t{triple[2]}\n")

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Log results
    log_file = os.path.join(os.path.dirname(p279_dir), "run_p279.log")
    with open(log_file, "w") as log:
        log.write(f"Processing completed in {format_time(elapsed_time)}\n")
        log.write(f"Total entities processed: {total_entities}\n")
        log.write(f"P279 output directory: {p279_dir}\n")
        log.write(f"Lines with JSON decoding errors: {num_lines_error}\n")

    print(f"Processing completed in {format_time(elapsed_time)}")
    print(f"Total entities processed: {total_entities}")
    print(f"P279 output directory: {p279_dir}")
    print(f"Lines with JSON decoding errors: {num_lines_error}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract P279 claims from a Wikidata JSON dump and save as TSV batch files."
    )
    parser.add_argument(
        "--dump_file",
        type=str,
        default="latest-all.json.gz",
        help="Path to the compressed Wikidata JSON dump file (default: latest-all.json.gz).",
    )
    parser.add_argument(
        "--p279_dir",
        type=str,
        default="P279",
        help="Directory to save `P279` triples (default: 'P279').",
    )
    parser.add_argument(
        "--num_entities_per_batch",
        type=int,
        default=50000,
        help="Entities per batch file (default: 50000).",
    )
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="If set, process only one batch.",
    )
    args = parser.parse_args()

    # Ensure the dump file exists
    if not os.path.exists(args.dump_file):
        print(f"Error: Dump file '{args.dump_file}' does not exist.")
        exit(1)

    process_file(
        args.dump_file,
        args.p279_dir,
        args.num_entities_per_batch,
        dummy=args.dummy,
    )
