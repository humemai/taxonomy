"""
graph_viewer.py

This script loads a taxonomy graph from a JSON file in node-link format and visualizes it
using Dash Cytoscape. The graph is rendered in a web browser using the dagre layout for a balanced
hierarchical tree.

Configuration parameters (with default values):
    num_classes   : 10000
    top_p         : 0.9
    max_depth     : 6    (for debugging, try a shallow tree first)
    max_width     : 16   (maximum children per node)
    max_attempts  : 32
    temperature   : 1.5
    port          : 8056

The filename is constructed as:
    trees/num_classes_{num_classes}_top_p_{top_p}_max_depth_{max_depth}_max_width_{max_width}_max_attempts_{max_attempts}_temperature_{temperature}.json

Usage:
    python graph_viewer.py
    [--num_classes NUM_CLASSES] [--top_p TOP_P] [--max_depth MAX_DEPTH]
    [--max_width MAX_WIDTH] [--max_attempts MAX_ATTEMPTS]
    [--temperature TEMPERATURE] [--port PORT]
    [--host HOST]

Example:
    python graph_viewer.py --num_classes 10000 --top_p 0.9 --max_depth 6 --max_width 16 \
                           --max_attempts 32 --temperature 1.5 --port 8056 --host 0.0.0.0
"""

import argparse
import json
import threading
import webbrowser

import networkx as nx
from dash import Dash, html
import dash_cytoscape as cyto
from IPython.display import clear_output

# Load extra layouts for Dash Cytoscape.
cyto.load_extra_layouts()


def load_taxonomy_json(filename: str) -> nx.DiGraph:
    """
    Load a taxonomy (graph) from a JSON file in node-link format and convert it into a NetworkX graph.

    Args:
        filename (str): The filename of the JSON file to load.

    Returns:
        nx.DiGraph: The reconstructed directed graph.
    """
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    G = nx.node_link_graph(data, edges="edges")
    return G


def visualize_graph_dash(
    G: nx.DiGraph, host: str = "127.0.0.1", port: int = 8050
) -> None:
    """
    Visualize a NetworkX graph using Dash Cytoscape in an external web browser,
    using the dagre layout for a balanced hierarchical tree.

    Args:
        G (nx.DiGraph): The NetworkX graph to visualize.
        host (str, optional): The host address to bind the server. Use "0.0.0.0" for remote access.
                              Defaults to "127.0.0.1".
        port (int, optional): The port to run the Dash server on. Defaults to 8050.
    """
    # Convert nodes for Cytoscape.
    nodes = []
    for node_id, data in G.nodes(data=True):
        label = data.get("label", str(node_id))
        nodes.append({"data": {"id": str(node_id), "label": label}})

    # Convert edges.
    edges = []
    for source, target in G.edges():
        edges.append({"data": {"source": str(source), "target": str(target)}})

    # Create the Dash app using Dash.
    app = Dash(__name__)
    app.layout = html.Div(
        [
            cyto.Cytoscape(
                id="cytoscape-graph",
                elements=nodes + edges,
                layout={
                    "name": "dagre",
                    "rankDir": "LR",  # orient the tree from left to right
                    "nodeSep": 30,
                    "edgeSep": 10,
                    "rankSep": 70,
                    "padding": 10,
                },
                style={"width": "100%", "height": "100vh"},
                stylesheet=[
                    {
                        "selector": "node",
                        "style": {
                            "label": "data(label)",
                            "background-color": "#87CEFA",  # light sky blue
                            "color": "#000",  # black font color
                            "font-size": "14px",
                            "text-valign": "bottom",
                            "text-halign": "center",
                            "text-margin-y": "10px",
                            "width": "40px",  # bigger node size
                            "height": "40px",
                            "text-wrap": "wrap",  # allow text to wrap
                            "text-max-width": "200px",  # increase maximum text width
                        },
                    },
                    {"selector": "edge", "style": {"line-color": "#B3B3B3"}},
                ],
            )
        ]
    )

    def open_browser():
        webbrowser.open(f"http://{host}:{port}")

    # Launch the browser after a short delay.
    threading.Timer(1, open_browser).start()

    # Clear inline output (useful when running in a notebook)
    clear_output(wait=True)

    # Run the Dash app.
    app.run_server(debug=True, host=host, port=port)


def main():
    """
    Main entry point for the graph viewer script.
    """
    parser = argparse.ArgumentParser(
        description="Visualize a taxonomy graph using Dash Cytoscape."
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=10000,
        help="Number of classes (default: 10000)",
    )
    parser.add_argument(
        "--top_p", type=float, default=0.9, help="Top p value (default: 0.9)"
    )
    parser.add_argument(
        "--max_depth", type=int, default=6, help="Maximum depth (default: 6)"
    )
    parser.add_argument(
        "--max_width",
        type=int,
        default=16,
        help="Maximum children per node (default: 16)",
    )
    parser.add_argument(
        "--max_attempts", type=int, default=32, help="Maximum attempts (default: 32)"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.5, help="Temperature (default: 1.5)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8056,
        help="Port to run the server on (default: 8056)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server (default: 0.0.0.0 for remote access)",
    )
    args = parser.parse_args()

    # Construct the filename using the configuration parameters.
    filename = (
        f"trees/num_classes_{args.num_classes}_top_p_{args.top_p}_max_depth_{args.max_depth}"
        f"_max_width_{args.max_width}_max_attempts_{args.max_attempts}"
        f"_temperature_{args.temperature}.json"
    )

    # Load the graph from the JSON file.
    try:
        G = load_taxonomy_json(filename=filename)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return

    # Visualize the graph.
    visualize_graph_dash(G, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
