"""
Script to fetch properties from Wikidata and save them as a JSON file.

This script fetches all properties from Wikidata, including their labels, aliases, 
and descriptions, and saves them in a JSON file. If the '--dummy' argument is passed, 
only 100 properties will be fetched for testing purposes.

Usage:
    python run_properties.py [--dummy]
    
    --dummy  Only fetches 100 properties for testing purposes.

The JSON output file ('properties.json') will contain an array of property objects
with the following fields:
    - property_id: The unique identifier for the property (e.g., P6)
    - label: The name or label of the property (e.g., "head of government")
    - aliases: A list of alternative labels for the property (e.g., ["leader", "chief of
          state"])
    - description: A description of the property (e.g., "The principal leader of a
          government.")
"""

import requests
import json
from SPARQLWrapper import SPARQLWrapper, JSON
import time
import argparse
from tqdm.auto import tqdm


def fetch_wikidata_properties(sparql_query_url: str) -> dict:
    """
    Fetch the JSON data from the given SPARQL query URL.

    Args:
        sparql_query_url (str): The URL of the SPARQL query to fetch data from.

    Returns:
        dict: The JSON data parsed into a Python dictionary.

    Raises:
        Exception: If the HTTP request fails.
    """
    response = requests.get(sparql_query_url)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to fetch data. Status code: {response.status_code}")


def fetch_all_property_ids(is_dummy: bool = False) -> list:
    """
    Fetch all property IDs and labels from Wikidata using SPARQL.

    Args:
        is_dummy (bool): If True, fetch only 100 properties for testing purposes.

    Returns:
        list: A list of tuples (property_id, property_label).
    """
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

    all_properties = []
    offset = 0
    max_retries = 3  # Retry limit in case of failure

    print("Fetching property IDs and labels...")
    while True:
        sparql.setQuery(
            f"""
            SELECT DISTINCT ?property ?propertyLabel
            WHERE {{
              ?property rdf:type wikibase:Property.
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
            }}
            LIMIT {100 if is_dummy else 10000}  # Fetch a large number of properties
            OFFSET {offset}
        """
        )
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()

        chunk = [
            (
                result["property"]["value"].split("/")[-1],
                result["propertyLabel"]["value"],
            )
            for result in results["results"]["bindings"]
        ]

        if not chunk:
            print(f"No more properties found. Stopping...")
            break  # Stop if no more results are returned

        all_properties.extend(chunk)
        offset += len(chunk)  # Move to the next chunk

        print(f"Fetched {offset} properties so far...")
        if is_dummy and offset >= 100:
            break

    print(f"Fetched a total of {len(all_properties)} properties.")
    return all_properties


def fetch_property_details(property_id: str) -> tuple[list, str]:
    """
    Fetch aliases and descriptions for a given property ID.

    Args:
        property_id (str): The property ID to fetch details for.

    Returns:
        tuple: A tuple containing a list of aliases and a description (or None if not available).
    """
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

    # Retry logic for fetching aliases and descriptions
    max_retries = 100
    retry_count = 0
    while retry_count < max_retries:
        try:
            # Query 1: Get all aliases
            sparql.setQuery(
                f"""
                SELECT DISTINCT ?alias
                WHERE {{
                  wd:{property_id} skos:altLabel ?alias.
                  FILTER (LANG(?alias) = "en")
                }}
            """
            )
            sparql.setReturnFormat(JSON)
            aliases_results = sparql.query().convert()

            aliases = [
                result["alias"]["value"]
                for result in aliases_results["results"]["bindings"]
            ]

            # Query 2: Get the description
            sparql.setQuery(
                f"""
                SELECT DISTINCT ?description
                WHERE {{
                  wd:{property_id} schema:description ?description.
                  FILTER (LANG(?description) = "en")
                }}
            """
            )
            description_results = sparql.query().convert()
            description = (
                description_results["results"]["bindings"][0]["description"]["value"]
                if description_results["results"]["bindings"]
                else None
            )

            return aliases, description

        except Exception as e:
            retry_count += 1
            print(
                f"Error fetching details for {property_id} (attempt {retry_count}/{max_retries}): {e}"
            )
            if retry_count >= max_retries:
                raise Exception(
                    f"Failed to fetch details for {property_id} after {max_retries} attempts."
                )
            time.sleep(5)  # Wait for 5 seconds before retrying


def save_properties(properties_dict: dict, output_file: str) -> None:
    """
    Save the properties dictionary to a JSON file.

    Args:
        properties_dict (dict): The dictionary containing property details.
        output_file (str): The path to the output JSON file.
    """
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(properties_dict, f, ensure_ascii=False, indent=4)
    print(f"Properties saved to {output_file}")


def main() -> None:
    """
    Main function to fetch, process, and save Wikidata properties.
    """
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Fetch Wikidata properties and save as JSON."
    )
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Fetch only 100 properties for testing purposes.",
    )
    args = parser.parse_args()

    try:
        # Fetch property IDs and labels (limit to 100 if dummy argument is passed)
        property_ids = fetch_all_property_ids(is_dummy=args.dummy)
        total_properties = len(property_ids)
        print(f"Total number of properties fetched: {total_properties}")

        properties_dict: dict[str, dict[str, str | list]] = {}  # Initialize the dictionary to store properties

        # Iterate over property IDs and fetch aliases and descriptions
        print("Fetching details for each property...")
        for index, (property_id, property_label) in tqdm(
            enumerate(property_ids, start=1)
        ):
            if index % 50 == 0:  # Print progress every 50 properties
                percentage = (index / total_properties) * 100
                print(f"Processed {index} properties ({percentage:.2f}% complete)...")

            aliases, description = fetch_property_details(property_id)

            # Store the property details in the dictionary
            properties_dict[property_id] = {
                "label": property_label,
                "aliases": aliases,
                "description": description,
            }

        # Save the properties to a JSON file
        save_properties(properties_dict, "properties.json")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
