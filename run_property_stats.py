"""
This script calculates the usage statistics of properties in Wikidata entity JSON files
and saves the results as `property_stats.json`.

It also generates a log file (`run_property_stats.log`) with details about the
processing.

Usage:
    python run_property_stats.py --dump_file <file> --output_file <file> [--dummy]

Arguments:
    --dump_file (str): Path to the compressed Wikidata JSON dump file
                       (default: 'latest-all.json.gz').
    --output_file (str): Path to save the output JSON file
                         (default: 'property_stats.json').
    --dummy: Optional flag to process only the first 10,000 entities.
"""

import gzip
import json
import argparse
import time
from collections import Counter


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


def calculate_property_stats(
    dump_file: str, output_file: str, dummy: bool = False
) -> None:
    """
    Calculate usage statistics of properties from the Wikidata JSON dump.

    Args:
        dump_file (str): Path to the compressed Wikidata JSON dump file.
        output_file (str): Path to save the output JSON file.
        dummy (bool): Whether to process only the first 10,000 entities.
    """
    properties_count = Counter()
    processed_count = 0
    decoding_errors = 0
    max_entities = 10_000 if dummy else float("inf")

    start_time = time.time()
    print(f"Processing file: {dump_file}")

    with gzip.open(dump_file, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line in ("[", "]"):  # Skip JSON array brackets
                continue
            try:
                entity = json.loads(line.rstrip(","))
                # Extract properties and update counts
                claims = entity.get("claims", {})
                properties_count.update(claims.keys())

                processed_count += 1
                if processed_count >= max_entities:
                    print("Dummy mode: Reached 10,000 entities. Stopping early.")
                    break
                if processed_count % 50_000 == 0:
                    print(f"Processed {processed_count} entities.")

            except json.JSONDecodeError:
                decoding_errors += 1
                continue

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Sort properties by occurrence
    properties_count = dict(
        sorted(properties_count.items(), key=lambda item: item[1], reverse=True)
    )

    # Save the results to a JSON file
    with open(output_file, "w", encoding="utf-8") as out_f:
        json.dump(properties_count, out_f, ensure_ascii=False, indent=4)

    # Log the results
    log_file = "run_property_stats.log"
    with open(log_file, "w", encoding="utf-8") as log_f:
        log_f.write(f"Processing completed in: {format_time(elapsed_time)}\n")
        log_f.write(f"Total entities processed: {processed_count}\n")
        log_f.write(f"Decoding errors: {decoding_errors}\n")
        log_f.write(f"Output file: {output_file}\n")
        log_f.write(f"Top 5 properties used:\n")
        for prop, count in list(properties_count.items())[:5]:
            log_f.write(f"  {prop}: {count} times\n")

    print(f"Processing completed in: {format_time(elapsed_time)}")
    print(f"Total entities processed: {processed_count}")
    print(f"Decoding errors: {decoding_errors}")
    print(f"Output file: {output_file}")
    print(f"Log file: {log_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate usage statistics of properties from a Wikidata JSON dump."
    )
    parser.add_argument(
        "--dump_file",
        type=str,
        default="latest-all.json.gz",
        help="Path to the compressed Wikidata JSON dump file (default: 'latest-all.json.gz').",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="property_stats.json",
        help="Path to save the output JSON file (default: 'property_stats.json').",
    )
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Process only the first 10,000 entities for testing.",
    )
    args = parser.parse_args()

    # Run the calculation
    calculate_property_stats(args.dump_file, args.output_file, args.dummy)
