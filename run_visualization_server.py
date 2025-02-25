#!/usr/bin/env python
"""
run_visualization_server.py

A drill-down Dash Cytoscape viewer using dash-cytoscape==1.0.2.
This version converts node IDs to strings for display but converts them back to integers for graph lookups.
It now toggles children: when you click an expanded node, all of its descendant nodes are removed (collapsed).

Usage:
    python run_visualization_server.py --filename /path/to/your.json [--host 0.0.0.1] [--port 8050]
"""

import argparse
import json
import threading
import webbrowser

import networkx as nx
from dash import Dash, html, dcc, Input, Output, State
import dash_cytoscape as cyto

# Load extra layouts (e.g. dagre)
cyto.load_extra_layouts()


def load_taxonomy_json(filename: str) -> nx.DiGraph:
    """
    Load a taxonomy graph from a JSON file in node-link format.
    The JSON is expected to have the edge list under the key "edges".
    """
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Explicitly tell NetworkX that edges are under "edges"
    G = nx.node_link_graph(data, edges="edges", directed=True)
    return G


def create_dash_app(G: nx.DiGraph, host: str, port: int) -> Dash:
    app = Dash(__name__)

    # --- DEBUG: Print graph info ---
    print(
        "Graph loaded with",
        G.number_of_nodes(),
        "nodes and",
        G.number_of_edges(),
        "edges.",
    )

    # --- Find the root node: the node whose "label" is "entity" ---
    try:
        root_node = next(n for n, d in G.nodes(data=True) if d.get("label") == "entity")
        print("Found root node (raw):", root_node)
    except StopIteration:
        raise ValueError("No node with label 'entity' found in the graph.")

    # Convert the root node ID to a string for display
    root_node_str = str(root_node)

    # --- Prepare initial elements as a Python list ---
    init_elements = [
        {
            "data": {
                "id": root_node_str,
                "label": G.nodes[root_node].get("label", root_node_str),
            }
        }
    ]
    print("Initial elements (Python list):", init_elements)

    # Pass the elements as a Python list directly.
    app.layout = html.Div(
        [
            # dcc.Store holds the elements as a Python list.
            dcc.Store(id="store-elements", data=init_elements),
            cyto.Cytoscape(
                id="cytoscape-graph",
                elements=init_elements,
                layout={"name": "dagre", "rankDir": "LR"},
                style={"width": "100%", "height": "100vh"},
                stylesheet=[
                    {
                        "selector": "node",
                        "style": {
                            "label": "data(label)",
                            "background-color": "#87CEFA",
                            "color": "#000",
                            "font-size": "14px",
                            "text-valign": "bottom",
                            "text-halign": "center",
                            "text-margin-y": "10px",
                            "width": "40px",
                            "height": "40px",
                            "text-wrap": "wrap",
                            "text-max-width": "200px",
                        },
                    },
                    {"selector": "edge", "style": {"line-color": "#B3B3B3"}},
                ],
            ),
        ]
    )

    @app.callback(
        Output("store-elements", "data"),
        Input("cytoscape-graph", "tapNodeData"),
        State("store-elements", "data"),
    )
    def toggle_children(tapped_node, current_elements):
        """
        When a node is clicked:
          - If its immediate children are visible, remove all descendant nodes (collapse).
          - Otherwise, add its immediate children (expand).
        """
        print("\n--- Callback: toggle_children triggered ---")
        print("Tapped node data:", tapped_node)
        print("Current stored elements (list):", current_elements)
        if not tapped_node:
            return current_elements

        # The tapped node's id comes in as a string, but the graph uses integers.
        node_id_str = tapped_node["id"]
        node_id = int(node_id_str)

        # Get immediate children of this node.
        immediate_children = list(G.successors(node_id))
        immediate_children_str = {str(child) for child in immediate_children}

        # Check if any immediate child is already visible.
        expanded = any(
            el["data"]["id"] in immediate_children_str
            for el in current_elements
            if "source" not in el["data"]
        )

        if expanded:
            # Collapse: remove all descendant nodes (and associated edges) of the tapped node.
            all_descendants = nx.descendants(G, node_id)
            all_descendants_str = {str(x) for x in all_descendants}
            print("Collapsing descendants:", all_descendants_str)

            # Remove node elements whose id is in all_descendants_str.
            new_elements = [
                el
                for el in current_elements
                if not (
                    "source" not in el["data"]
                    and el["data"]["id"] in all_descendants_str
                )
            ]
            # Also remove edges that connect to any of these nodes.
            new_elements = [
                el
                for el in new_elements
                if "source" not in el["data"]
                or (
                    el["data"]["source"] not in all_descendants_str
                    and el["data"]["target"] not in all_descendants_str
                )
            ]
            print("New elements after collapse:", new_elements)
            return new_elements
        else:
            # Expand: add immediate children (if not already present).
            existing_node_ids = {
                el["data"]["id"]
                for el in current_elements
                if "source" not in el["data"]
            }
            existing_edges = {
                (el["data"]["source"], el["data"]["target"])
                for el in current_elements
                if "source" in el["data"]
            }
            new_nodes = []
            new_edges = []
            for child in immediate_children:
                child_str = str(child)
                if child_str not in existing_node_ids:
                    label = G.nodes[child].get("label", child_str)
                    new_nodes.append({"data": {"id": child_str, "label": label}})
                if (node_id_str, child_str) not in existing_edges:
                    new_edges.append(
                        {"data": {"source": node_id_str, "target": child_str}}
                    )
            updated_elements = current_elements + new_nodes + new_edges
            print("New nodes added:", new_nodes)
            print("New edges added:", new_edges)
            print("Updated elements after expansion:", updated_elements)
            return updated_elements

    @app.callback(
        Output("cytoscape-graph", "elements"), Input("store-elements", "data")
    )
    def update_graph_elements(stored_elements):
        print("\n--- Callback: update_graph_elements triggered ---")
        print("Stored elements (list):", stored_elements)
        return stored_elements

    # Optional: auto-open the browser after a short delay.
    def open_browser():
        webbrowser.open(f"http://{host}:{port}")

    threading.Timer(1, open_browser).start()

    return app


def main():
    parser = argparse.ArgumentParser(
        description="Drill-down Dash Cytoscape viewer using dash-cytoscape==1.0.2."
    )
    parser.add_argument(
        "--filename",
        required=True,
        help="Path to the taxonomy JSON in node-link format.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host for the Dash server.")
    parser.add_argument(
        "--port", type=int, default=8050, help="Port for the Dash server."
    )
    args = parser.parse_args()

    G = load_taxonomy_json(args.filename)
    app = create_dash_app(G, host=args.host, port=args.port)
    app.run_server(debug=True, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
