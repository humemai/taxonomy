"""
Script to fetch Wikidata properties via a SPARQL query and save them as a JSON file.

This script downloads the list of properties from Wikidata using a SPARQL query,
processes the data to extract property IDs and labels, and saves the result
as a JSON file named 'properties.json'.

Usage:
    python fetch_wikidata_properties.py
"""

import requests
import json


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


def process_properties(json_data):
    """
    Process the JSON data to extract property IDs and labels.

    Args:
        json_data (dict): The JSON data containing the properties.

    Returns:
        dict: A dictionary with property IDs as keys and property labels as values.
    """
    properties = json_data["results"]["bindings"]
    processed_properties = {
        prop["property"]["value"].split("/")[-1]: prop["propertyLabel"]["value"]
        for prop in properties
    }

    print(f"{len(processed_properties)} properties processed")
    return processed_properties


def save_properties(properties_dict, output_file):
    """
    Save the properties dictionary to a JSON file.

    Args:
        properties_dict (dict): The dictionary containing property IDs and labels.
        output_file (str): The path to the output JSON file.
    """
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(properties_dict, f, ensure_ascii=False, indent=2)
    print(f"Properties saved to {output_file}")


def main():
    """
    Main function to fetch, process, and save Wikidata properties.
    """
    sparql_query_url = (
        "https://query.wikidata.org/sparql"
        "?format=json"
        "&query=SELECT%20%3Fproperty%20%3FpropertyLabel%20WHERE%20%7B%0A"
        "%20%20%20%20%3Fproperty%20a%20wikibase%3AProperty%20.%0A"
        "%20%20%20%20SERVICE%20wikibase%3Alabel%20%7B%0A"
        "%20%20%20%20%20%20bd%3AserviceParam%20wikibase%3Alanguage%20%22en%22%20.%0A"
        "%20%20%20%7D%0A%20%7D%0A%0A"
    )
    try:
        json_data = fetch_wikidata_properties(sparql_query_url)
        properties_dict = process_properties(json_data)
        save_properties(properties_dict, "properties.json")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
