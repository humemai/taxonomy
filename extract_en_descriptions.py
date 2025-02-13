"""
This script extracts the English descriptions of Wikidata entities
from a compressed JSON dump file and saves them into TSV batch files.
After creating en_description.json, the entire directory containing
the TSV files is deleted.

Usage:
    python extract_en_descriptions.py --dump_file <file> --desc_dir <directory>
                                      [--num_entities_per_batch <int>] [--dummy]

Arguments:
    --dump_file (str): Path to the compressed Wikidata JSON dump file
                       (default: 'latest-all.json.gz').
    --desc_dir (str): Directory to save extracted English descriptions
                      (default: 'en_description').
    --num_entities_per_batch (int): Entities per batch file (default: 50000).
    --dummy: Optional flag to process only one batch (useful for quick testing).
"""

import os
import sys
import json
import gzip
import time
import csv
import argparse
import shutil
from glob import glob
from tqdm.auto import tqdm


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


def extract_en_description(entity: dict) -> tuple:
    """
    Extract the English description from an entity, if present.

    Returns:
        (entity_id, description) if found, otherwise None.
    """
    entity_id = entity.get("id")
    descriptions = entity.get("descriptions", {})
    en_desc = descriptions.get("en", {})
    if "value" in en_desc:
        # Clean up any tabs or newlines that might break TSV formatting
        desc_cleaned = en_desc["value"].replace("\t", " ").replace("\n", " ")
        return (entity_id, desc_cleaned)
    return None


def write_batch(entity_list, desc_dir, batch_idx):
    """
    Given a list of entities, extract English descriptions and write them
    to a TSV file named 'batch_{batch_idx}.tsv' in desc_dir.
    """
    tsv_filename = os.path.join(desc_dir, f"batch_{batch_idx}.tsv")
    with open(tsv_filename, "w", encoding="utf-8") as out_f:
        # Write header
        out_f.write("entity_id\tdescription\n")
        # Write each English description
        for entity in entity_list:
            result = extract_en_description(entity)
            if result is not None:
                entity_id, en_desc = result
                out_f.write(f"{entity_id}\t{en_desc}\n")


def process_file(
    dump_file: str,
    desc_dir: str,
    num_entities_per_batch: int,
    dummy: bool,
) -> None:
    """
    Process the Wikidata dump file and extract English descriptions,
    saving the results in TSV batch files. After creating en_description.json,
    the entire directory containing the TSV files is deleted.
    """
    start_time = time.time()

    # Create directory for English descriptions
    os.makedirs(desc_dir, exist_ok=True)

    entity_buffer = []
    batch_idx = 0
    total_entities = 0
    num_lines_error = 0
    processed_dummy_batch = False

    with gzip.open(dump_file, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Skip the JSON array opening/closing brackets
            if line in ("[", "]"):
                continue

            try:
                # Each line is one JSON entity (with a trailing comma except the last)
                entity = json.loads(line.rstrip(","))
                entity_buffer.append(entity)
                total_entities += 1

                # Process a batch when the buffer is full
                if len(entity_buffer) >= num_entities_per_batch:
                    print(f"Processing batch {batch_idx} for English descriptions...")
                    write_batch(entity_buffer, desc_dir, batch_idx)
                    batch_idx += 1
                    entity_buffer = []

                    # In dummy mode, stop after writing the first batch
                    if dummy:
                        processed_dummy_batch = True
                        break

            except json.JSONDecodeError:
                print("Error decoding JSON line:", line)
                num_lines_error += 1
                continue

    # Process remaining entities in the buffer, unless dummy mode already completed one batch
    if entity_buffer and not (dummy and processed_dummy_batch):
        print(f"Processing final batch {batch_idx} for English descriptions...")
        write_batch(entity_buffer, desc_dir, batch_idx)

    # Increase CSV field size limit
    csv.field_size_limit(sys.maxsize)

    en_description = {}

    # Read all TSV files and collect paths
    for tsv_filepath in tqdm(glob(os.path.join(desc_dir, "*.tsv"))):
        with open(tsv_filepath, "r", encoding="utf-8", newline="") as tsv_file:
            tsv_reader = csv.reader(tsv_file, delimiter="\t")
            # Skip header row
            next(tsv_reader, None)
            for row in tsv_reader:
                if len(row) < 2:
                    continue  # Skip malformed rows
                en_description[row[0]] = row[1]

    # Safely remove a key if it exists
    en_description.pop("entity_id", None)

    # Create en_description.json
    with open("en_description.json", "w", encoding="utf-8") as f:
        json.dump(en_description, f, ensure_ascii=False, indent=4)

    # Remove the entire directory containing the TSV files
    shutil.rmtree(desc_dir)
    print(f"Directory '{desc_dir}' and all its contents have been deleted.")

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Always write a log file, regardless of dummy mode
    log_file = os.path.join(os.path.dirname(desc_dir), "run_en_desc.log")
    with open(log_file, "w", encoding="utf-8") as log:
        log.write(f"Processing completed in {format_time(elapsed_time)}\n")
        log.write(f"Total entities processed: {total_entities}\n")
        log.write(f"English descriptions output directory (deleted): {desc_dir}\n")
        log.write(f"Lines with JSON decoding errors: {num_lines_error}\n")
        log.write("en_description.json created\n")
        log.write(f"Directory '{desc_dir}' deleted\n")

    print(f"Processing completed in {format_time(elapsed_time)}")
    print(f"Total entities processed: {total_entities}")
    print(f"English descriptions output directory (deleted): {desc_dir}")
    print(f"Lines with JSON decoding errors: {num_lines_error}")
    print("en_description.json created")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract English descriptions from a Wikidata JSON dump and save as TSV batch files."
    )
    parser.add_argument(
        "--dump_file",
        type=str,
        default="latest-all.json.gz",
        help="Path to the compressed Wikidata JSON dump file (default: latest-all.json.gz).",
    )
    parser.add_argument(
        "--desc_dir",
        type=str,
        default="en_description",
        help="Directory to save the extracted English descriptions (default: 'en_description').",
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
        help="If set, process only one batch for quick testing.",
    )
    args = parser.parse_args()

    # Ensure the dump file exists
    if not os.path.exists(args.dump_file):
        print(f"Error: Dump file '{args.dump_file}' does not exist.")
        exit(1)

    process_file(
        args.dump_file,
        args.desc_dir,
        args.num_entities_per_batch,
        dummy=args.dummy,
    )
