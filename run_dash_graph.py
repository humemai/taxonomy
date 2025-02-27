#!/usr/bin/env python
"""
visualize_graph.py

A Dash-based graph visualization using dash-cytoscape.
Handles general graphs rather than just trees, with interactive exploration functionality.

Usage:
    python visualize_graph.py --filename /path/to/your.json [--host 0.0.0.0] [--port 8050]
"""

import argparse
import json
import threading
import webbrowser
from typing import Dict, List, Set

import networkx as nx
from dash import Dash, html, dcc, Input, Output, State, callback
import dash_cytoscape as cyto

# Load extra layouts
cyto.load_extra_layouts()

def load_graph_json(filename: str) -> nx.Graph:
    """Load a graph from a JSON file in node-link format."""
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    G = nx.node_link_graph(data)
    return G

def create_dash_app(G: nx.Graph, host: str, port: int) -> Dash:
    app = Dash(__name__)
    
    # Print graph info
    print(f"Graph loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    
    # Create elements for ALL nodes and edges (instead of just highest degree node)
    elements = []
    
    # Check if graph is large and warn about performance
    if G.number_of_nodes() > 1000:
        print("Warning: Visualizing a large graph with over 1000 nodes may cause performance issues.")
    
    # Add all nodes
    for node in G.nodes():
        node_str = str(node)
        elements.append({
            'data': {
                'id': node_str,
                'label': G.nodes[node].get('label', node_str)
            }
        })
    
    # Add all edges
    for source, target in G.edges():
        source_str = str(source)
        target_str = str(target)
        elements.append({
            'data': {
                'source': source_str,
                'target': target_str
            }
        })
    
    print(f"Created visualization with {len([e for e in elements if 'source' not in e['data']])} nodes and {len([e for e in elements if 'source' in e['data']])} edges")
    
    # Track all nodes as already expanded since we're showing everything
    expanded_nodes = [str(node) for node in G.nodes()]
    
    # Create the layout
    app.layout = html.Div([
        html.H1('Graph Visualization'),
        html.P('Showing all nodes and edges in the graph.'),
        
        # Store for elements
        dcc.Store(id='store-elements', data=elements),
        
        # Store for expanded nodes to track what's already shown
        dcc.Store(id='store-expanded-nodes', data=expanded_nodes),
        
        # Controls
        html.Div([
            html.Label('Layout:'),
            dcc.Dropdown(
                id='layout-dropdown',
                options=[
                    {'label': 'Concentric', 'value': 'concentric'},
                    {'label': 'Cola', 'value': 'cola'},
                    {'label': 'Cose', 'value': 'cose'},
                    {'label': 'Circle', 'value': 'circle'},
                    {'label': 'Grid', 'value': 'grid'},
                    {'label': 'Breadthfirst', 'value': 'breadthfirst'},
                    {'label': 'Dagre', 'value': 'dagre'}
                ],
                value='cose'  # Changed default to cose which often works better for full graphs
            ),
        ], style={'width': '300px', 'margin': '10px'}),
        
        # Cytoscape component
        cyto.Cytoscape(
            id='cytoscape-graph',
            elements=elements,
            layout={
                'name': 'cose',
                'fit': True,
                'animate': True,
                'nodeDimensionsIncludeLabels': True,
                'idealEdgeLength': 100,
                'nodeRepulsion': 5000,  # Increase this for more spacing
                'edgeElasticity': 100,
                'nodeOverlap': 20,
            },
            style={'width': '100%', 'height': '85vh'},
            stylesheet=[
                {
                    'selector': 'node',
                    'style': {
                        'label': 'data(label)',
                        'background-color': '#87CEFA',
                        'color': '#000',
                        'font-size': '12px',
                        'text-valign': 'center',
                        'text-halign': 'center',
                        'width': '35px',
                        'height': '35px',
                        'text-wrap': 'wrap',
                        'text-max-width': '80px',
                    }
                },
                {
                    'selector': 'edge',
                    'style': {
                        'width': 1.5,
                        'line-color': '#B3B3B3',
                        'curve-style': 'bezier',
                        'target-arrow-shape': 'triangle',
                        'target-arrow-color': '#B3B3B3',
                        'arrow-scale': 1.5
                    }
                },
                {
                    'selector': '.highlighted',
                    'style': {
                        'background-color': '#FF4500',
                        'line-color': '#FF4500',
                        'width': 3
                    }
                }
            ]
        )
    ])
    
    @app.callback(
        [Output('store-elements', 'data'),
         Output('store-expanded-nodes', 'data')],
        [Input('cytoscape-graph', 'tapNodeData')],
        [State('store-elements', 'data'),
         State('store-expanded-nodes', 'data')]
    )
    def update_elements(tap_node, elements, expanded_nodes):
        # Use dash.callback_context instead of callback.ctx
        from dash import callback_context
        ctx = callback_context
        
        if not tap_node:
            return elements, expanded_nodes
        
        node_id_str = tap_node['id']
        node_id = node_id_str  # Keep as string for id matching
        
        # Try to convert to the original data type if needed (int, etc.)
        try:
            node_id_converted = int(node_id_str)
            if node_id_converted in G:
                node_id = node_id_converted
        except ValueError:
            pass
        
        # If node is already expanded, remove its neighbors
        if node_id_str in expanded_nodes:
            # Instead, add its neighbors if not already added
            neighbors = list(G.neighbors(node_id))
            neighbors_str = [str(n) for n in neighbors]
            
            # Filter out nodes that are already displayed
            existing_node_ids = {el['data']['id'] for el in elements if 'source' not in el['data']}
            new_neighbors = [n for n in neighbors_str if n not in existing_node_ids]
            
            # Add new neighbor nodes
            for neighbor in new_neighbors:
                try:
                    neighbor_converted = int(neighbor) if neighbor.isdigit() else neighbor
                    if neighbor_converted in G.nodes:
                        label = G.nodes[neighbor_converted].get('label', neighbor)
                        elements.append({'data': {'id': neighbor, 'label': label}})
                except (ValueError, KeyError):
                    elements.append({'data': {'id': neighbor, 'label': neighbor}})
            
            # Add edges to new neighbors
            for neighbor in new_neighbors:
                elements.append({'data': {'source': node_id_str, 'target': neighbor}})
            
            # Add this node to expanded nodes
            if node_id_str not in expanded_nodes:
                expanded_nodes.append(node_id_str)
        
        return elements, expanded_nodes
    
    @app.callback(
        Output('cytoscape-graph', 'layout'),
        Input('layout-dropdown', 'value')
    )
    def update_layout(layout):
        return {
            'name': layout,
            'fit': True,
            'animate': True
        }
    
    @app.callback(
        Output('cytoscape-graph', 'elements'),
        Input('store-elements', 'data')
    )
    def update_cytoscape(elements):
        return elements

    # Auto-open browser
    def open_browser():
        webbrowser.open(f"http://{host}:{port}")
    
    threading.Timer(1, open_browser).start()
    
    return app

def main():
    parser = argparse.ArgumentParser(description="Interactive graph visualization using Dash and Cytoscape")
    parser.add_argument('--filename', required=True, help="Path to the graph JSON file")
    parser.add_argument('--host', default='127.0.0.1', help="Host for the Dash server")
    parser.add_argument('--port', type=int, default=8050, help="Port for the Dash server")
    args = parser.parse_args()
    
    G = load_graph_json(args.filename)
    app = create_dash_app(G, args.host, args.port)
    app.run_server(debug=True, host=args.host, port=args.port)

if __name__ == '__main__':
    main()