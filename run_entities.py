"""
Script to fetch Wikidata properties via a SPARQL query and save them as a JSON file.

This script downloads the list of properties from Wikidata using a SPARQL query,
processes the data to extract property IDs, labels, aliases, and descriptions, 
and saves the result as a JSON file named 'properties.json'.

Usage:
    python fetch_wikidata_properties.py
"""

import requests
import json
from SPARQLWrapper import SPARQLWrapper, JSON


def fetch_wikidata_properties(sparql_query_url):
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


def fetch_all_property_ids():
    """
    Fetch all property IDs and labels from Wikidata using SPARQL.

    Returns:
        list: A list of tuples (property_id, property_label).
    """
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

    all_properties = []
    offset = 0
    limit = 500  # Fetch 500 properties per query to avoid timeouts

    print("Fetching property IDs and labels...")
    while True:
        sparql.setQuery(
            f"""
            SELECT DISTINCT ?property ?propertyLabel
            WHERE {{
              ?property rdf:type wikibase:Property.
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
            }}
            LIMIT {limit}
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
            break  # Stop if no more results are returned

        all_properties.extend(chunk)
        offset += limit  # Move to the next chunk

    print(f"Fetched {len(all_properties)} properties.")
    return all_properties


def fetch_property_details(property_id):
    """
    Fetch aliases and descriptions for a given property ID.

    Args:
        property_id (str): The property ID to fetch details for.

    Returns:
        tuple: A tuple containing a list of aliases and a description (or None if not available).
    """
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

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
        result["alias"]["value"] for result in aliases_results["results"]["bindings"]
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


def save_properties(properties_dict, output_file):
    """
    Save the properties dictionary to a JSON file.

    Args:
        properties_dict (dict): The dictionary containing property details.
        output_file (str): The path to the output JSON file.
    """
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(properties_dict, f, ensure_ascii=False, indent=4)
    print(f"Properties saved to {output_file}")


def main():
    """
    Main function to fetch, process, and save Wikidata properties.
    """
    try:
        # Fetch all property IDs and labels
        property_ids = fetch_all_property_ids()
        print(f"Total number of properties fetched: {len(property_ids)}")

        properties = []

        # Iterate over property IDs and fetch aliases and descriptions
        print("Fetching details for each property...")
        for index, (property_id, property_label) in enumerate(property_ids, start=1):
            if index % 50 == 0:  # Print progress every 50 properties
                print(f"Processed {index} properties...")

            aliases, description = fetch_property_details(property_id)
            property_data = {
                "property_id": property_id,
                "label": property_label,
                "aliases": aliases,
                "description": description,
            }
            properties.append(property_data)

        # Save the properties to a JSON file
        save_properties(properties, "properties.json")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
