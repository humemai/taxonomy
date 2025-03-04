"""
This script parses a TTL file containing RDF data, extracts subclass relationships (rdfs:subClassOf),
and converts the extracted taxonomy into a directed graph represented in JSON node-link format.

The script uses the `rdflib` library to parse the RDF triples from the TTL file and `networkx` to
construct a directed graph representing the taxonomy. Each node in the graph represents a class,
and edges represent the subclass relationships. The script identifies root nodes (nodes with no
incoming edges) and includes them in the output JSON.  Node labels are extracted using the
`extract_label` function.

The resulting JSON file contains a node-link representation of the graph, suitable for visualization
or further processing.  It includes a list of nodes with their labels, a list of edges representing
the subclass relationships, and a list of root nodes.

The script takes two command-line arguments: the path to the input TTL file and the path to the
output JSON file.

Example usage:
    python yago-to-graph.py --ttl_file yago_taxonomy.ttl --output_json yago_taxonomy.json
"""
import rdflib
import networkx as nx
import json


def parse_ttl_to_json(ttl_file_path: str, output_json_path: str) -> None:
    """
    Parses a TTL file, extracts subclass relationships, and saves the graph in JSON
    node-link format.

    Args:
        ttl_file_path (str): Path to the input TTL file.
        output_json_path (str): Path to the output
    Returns:
        None
    """
    g = rdflib.Graph()
    try:
        g.parse(ttl_file_path, format="turtle")
    except Exception as e:
        raise ValueError(f"Error parsing Turtle file: {e}")

    G = nx.DiGraph()
    subclass_of = rdflib.RDFS.subClassOf

    for s, p, o in g.triples((None, subclass_of, None)):
        s_str = str(s)
        o_str = str(o)
        G.add_edge(o_str, s_str)
        G.add_node(s_str, label=extract_label(g, s))
        G.add_node(o_str, label=extract_label(g, o))

    # Find root nodes (nodes with no incoming edges)
    root_nodes = [node for node, in_degree in G.in_degree() if in_degree == 0]

    data = nx.node_link_data(G, edges="edges")
    data["root_nodes"] = root_nodes

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def extract_label(graph: rdflib.Graph, node: rdflib.URIRef) -> str:
    """
    Extract the label of a node from the RDF graph.
    """
    label = None
    # Try rdfs:label
    for _, _, o in graph.triples((node, rdflib.RDFS.label, None)):
        label = str(o)
        break
    # If no rdfs:label, use the node's URI fragment
    if label is None:
        label = node.split("#")[-1].split("/")[
            -1
        ]  # Extract fragment or last part of URI
    return label


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert TTL taxonomy to JSON.")
    parser.add_argument("--ttl_file", required=True, help="Path to the TTL file.")
    parser.add_argument(
        "--output_json", required=True, help="Path to the output JSON file."
    )
    args = parser.parse_args()

    parse_ttl_to_json(args.ttl_file, args.output_json)
    print(f"Successfully converted {args.ttl_file} to {args.output_json}")
