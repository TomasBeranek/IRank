import os
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import pandas as pd
from collections import defaultdict
from pprint import pprint


def add_edge(start, end, incoming_edges, outgoing_edges):
    incoming_edges[end].add(f'{start}->{end}')
    outgoing_edges[start].add(f'{start}->{end}')


def read_csv_files(directory):
    valid_nodes = set()
    node_data = {}
    edge_data = {}
    incoming_edges = defaultdict(set)
    outgoing_edges = defaultdict(set)

    for filename in os.listdir(directory):
        if filename.endswith('_cypher.csv') or filename.endswith('_header.csv'):
            continue

        # Skip REACHING_DEF for now
        if 'REACHING_DEF' in filename:
            continue

        if filename.startswith("nodes_"):
            type_name = '_'.join(filename.split("_")[1:-1])
            path_data = os.path.join(directory, filename)
            path_header = os.path.join(directory, f"nodes_{type_name}_header.csv")
            headers = pd.read_csv(path_header, header=None).iloc[0]
            df = pd.read_csv(path_data, header=None, names=headers)

            # Keep only useful features (see model/schemes/latent_nodes/basic_cpg.pbtxt for more info)
            if 'BLOCK' in filename:
                df = df[[':ID', 'ORDER:int']]

            node_data[type_name] = df

        elif filename.startswith("edges_"):
            # Only AST for now
            # if 'AST' not in filename:
            #     continue

            type_name = '_'.join(filename.split("_")[1:-1])
            path = os.path.join(directory, filename)
            edge_data[type_name] = pd.read_csv(path, header=None, names=["start", "end", 'type'])

        # Add nodes with attributes
        for node_type, data in node_data.items():
            for _, row in data.iterrows():
                valid_nodes.add(row[':ID'])

        all_nodes = set(valid_nodes)

        # Add edges
        for edge_type, data in edge_data.items():
            # We are interested in removing AST edges
            if edge_type == 'AST':
                for _, row in data.iterrows():
                    # Some edges will be saved multiple times, but since
                    # incoming_edges[NODE] is set, it doesn't matter and the code
                    # is much simpler
                    add_edge(row['start'], row['end'], incoming_edges, outgoing_edges)
                    all_nodes.add(row['start'])
                    all_nodes.add(row['end'])

    return node_data, edge_data, valid_nodes, all_nodes, incoming_edges, outgoing_edges


def create_graph(node_data, edge_data):
    G = nx.DiGraph()

    # Add nodes with attributes
    for node_type, data in node_data.items():
        for _, row in data.iterrows():
            G.add_node(row[':ID'], **row[1:].to_dict(), type=node_type)

    # Add edges
    for edge_type, data in edge_data.items():
        for _, row in data.iterrows():
            # if (row['start'] in G.nodes) and (row['end'] in G.nodes):
            G.add_edge(row['start'], row['end'], type=edge_type)

    return G


def plot_graph(G):
    plt.figure(figsize=(12, 8))

    # Define a color mapping
    node_color_map = {
        'BLOCK': 'blue',
        'CALL': 'green',
        'FIELD_IDENTIFIER': 'red',
        'IDENTIFIER': 'cyan',
        'LITERAL': 'magenta',
        'LOCAL': 'black',
        'METHOD_REF': 'orange',
        'RETURN': 'yellow',
        'UNKNOWN': 'purple',
        'default_type': 'grey',
    }

    edge_color_map = {
        'AST': 'red',
        'CALL': 'purple',
        'CDG': 'green',
        'CFG': 'blue',
        'REACHING_DEF': 'grey',
        'RECEIVER': 'cyan',
        'ARGUMENT': 'yellow',
    }

    # Differentiate colors for nodes
    node_types = [G.nodes[n].get('type', 'default_type') for n in G.nodes]
    # # Differentiate colors for edges
    edge_types = [G.edges[e]['type'] for e in G.edges]

    # Apply color mapping
    node_colors = [node_color_map[node_type] for node_type in node_types]
    edge_colors = [edge_color_map[edge_type] for edge_type in edge_types]

    # Create legend patches for nodes
    node_legend_patches = [mpatches.Patch(color=color, label=node_type) for node_type, color in node_color_map.items()]

    # Create legend patches for edges
    edge_legend_patches = [mpatches.Patch(color=color, label=edge_type) for edge_type, color in edge_color_map.items()]

    G_undirected = nx.Graph(G)  # Create an equivalent undirected graph
    pos = nx.spring_layout(G_undirected)  # Compute layout

    # Draw nodes with type-based colors
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, alpha=0.8)

    # Draw directed edges with arrows
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, alpha=0.5, arrows=True)

    # Draw labels for nodes
    node_labels = {n: n for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=node_labels)

    # Create the first legend (for nodes) and add it to the plot
    first_legend = plt.legend(handles=node_legend_patches, loc='upper right', title='Node Types')
    ax = plt.gca()  # Get the current axes
    ax.add_artist(first_legend)  # Add the first legend back to the plot

    # Create the second legend (for edges)
    plt.legend(handles=edge_legend_patches, loc='lower right', title='Edge Types')

    plt.title('Graph Visualization')
    plt.show()


def remove_edge(edge, incoming_edges, outgoing_edges):
    start = int(edge.split('->')[0])
    end   = int(edge.split('->')[1])
    outgoing_edges[start].remove(edge)
    incoming_edges[end].remove(edge)


def remove_leaf(node, all_nodes, incoming_edges, outgoing_edges):
    edge_to_remove = next(iter(incoming_edges[node]))
    remove_edge(edge_to_remove, incoming_edges, outgoing_edges)
    all_nodes.remove(node)


def remove_inner(node, all_nodes, incoming_edges, outgoing_edges):
    parent_edge = next(iter(incoming_edges[node]))
    parent_node = int(parent_edge.split('->')[0])
    child_nodes = [int(child_edge.split('->')[1]) for child_edge in outgoing_edges[node]]

    # Remove all edges related to this node
    remove_edge(parent_edge, incoming_edges, outgoing_edges)
    child_edges = outgoing_edges[node].copy() # We donn't want change the set we are iterating on
    for child_edge in child_edges:
        remove_edge(child_edge, incoming_edges, outgoing_edges)

    # Make new edges from parent to child nodes (bypassing current node)
    for child_node in child_nodes:
        add_edge(parent_node, child_node, incoming_edges, outgoing_edges)

    all_nodes.remove(node)


def is_leaf(node, incoming_edges, outgoing_edges):
    return len(outgoing_edges[node]) == 0


def is_root(node, incoming_edges):
    return len(incoming_edges[node]) == 0


def is_inner(node, incoming_edges, outgoing_edges):
    return (len(incoming_edges[node]) == 1) and (len(outgoing_edges[node]) > 0)


def remove_invalid_nodes(node_data, edge_data, valid_nodes, all_nodes, incoming_edges, outgoing_edges):
    invalid_nodes = all_nodes - valid_nodes

    # Keep removing until all invalid nodes are gone
    while invalid_nodes:
        node = invalid_nodes.pop()

        if is_leaf(node, incoming_edges, outgoing_edges):
            remove_leaf(node, all_nodes, incoming_edges, outgoing_edges)
        elif is_root(node, incoming_edges):
            # Make this node a valid BLOCK node
            node_data['BLOCK'].loc[len(node_data['BLOCK'])] = {':ID': node, 'ORDER:int': 0}
        elif is_inner(node, incoming_edges, outgoing_edges):
            remove_inner(node, all_nodes, incoming_edges, outgoing_edges)
        else:
            # This shouldn't happen, because it would mean that the AST is not a tree
            print(f'ERROR: visualize_graph.py: current graph has a node which is not root, inner or leaf - which is not possible in a tree!')
            exit(1)

    # TODO: Remove leaf BLOCK nodes (and possibly others)

    # Filter out removed edges/nodes from original data
    all_edges = set()
    for edges in incoming_edges.values():
        all_edges.update(edges)

    # Keep only AST edges which weren't removed
    edges_list = [(int(edge.split('->')[0]), int(edge.split('->')[1]), 'AST') for edge in all_edges]
    edge_data['AST'] = pd.DataFrame(edges_list, columns=['start', 'end', 'type'])

    # TODO: Create nx graph

    # TODO: Check if the number of WCC (Weakly Connected Components) is the same
    # as before the removal of invalid nodes (it should be the same)

    # TODO: Remove WCCs which are composed only from BLOCK nodes

    # TODO: Add rest of the edges 'CFG', 'CALL', ...

    # TODO: Remove edges of other types which are related to invalid nodes

    # TODO: Check if the graph is a single WCC

    # TODO: Print compression of the graph after removal
    exit()


if __name__ == "__main__":
    directory = sys.argv[1]  # Get directory from the first argument
    node_data, edge_data, valid_nodes, all_nodes, incoming_edges, outgoing_edges = read_csv_files(directory)
    remove_invalid_nodes(node_data, edge_data, valid_nodes, all_nodes, incoming_edges, outgoing_edges)
    G = create_graph(node_data, edge_data)
    plot_graph(G)
