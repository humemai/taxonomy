import argparse
from rdflib import Graph


def convert_ttl_to_jsonld(ttl_file_path, output_file_path):
    """
    Converts a TTL file to JSON-LD format with indentation.

    Args:
        ttl_file_path (str): The path to the input TTL file.
        output_file_path (str): The path to the output JSON-LD file.
    """
    try:
        # Load TTL file into an RDF graph
        g = Graph()
        g.parse(ttl_file_path, format="turtle")

        # Serialize as JSON-LD with indentation
        jsonld_data = g.serialize(format="json-ld", indent=4)

        # Save JSON-LD data to a file with UTF-8 encoding
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(jsonld_data)

        print(f"Successfully converted {ttl_file_path} to {output_file_path}")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description="Convert TTL file to JSON-LD format.")
    parser.add_argument("--ttl_file", help="Path to the input TTL file")
    parser.add_argument("--output_file", help="Path to the output JSON-LD file")

    # Parse arguments
    args = parser.parse_args()

    # Convert TTL to JSON-LD
    convert_ttl_to_jsonld(args.ttl_file, args.output_file)
