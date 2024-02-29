import os
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import pandas as pd
import concurrent.futures


headers_cache = {}


def load_sample_data(directory):
    valid_nodes = set()
    node_data = {}
    edge_data = {}

    for filename in os.listdir(directory):
        if filename.endswith('_cypher.csv') or filename.endswith('_header.csv'):
            continue

        # Skip REACHING_DEF for now
        if 'REACHING_DEF' in filename:
            continue

        # Load nodes
        if filename.startswith("nodes_"):
            type_name = '_'.join(filename.split("_")[1:-1])
            path_data = os.path.join(directory, filename)
            df = pd.read_csv(path_data, header=None, names=headers_cache[type_name])

            # Keep only useful features (see model/schemes/latent_nodes/basic_cpg.pbtxt for more info)
            if 'BLOCK' in filename:
                df = df[[':ID', 'ORDER:int']]

            node_data[type_name] = df

        # Load edges
        elif filename.startswith("edges_"):
            type_name = '_'.join(filename.split("_")[1:-1])
            path = os.path.join(directory, filename)
            df = pd.read_csv(path, header=None, names=["start", "end", 'type'])
            edge_data[type_name] = df[['start', 'end']] # Drop edge type - we already know it

        # Valid nodes - nodes that are actually stored in CSV files, any other
        # node is considered invalid
        for node_type, data in node_data.items():
            valid_nodes.update(set(data[':ID']))

    return node_data, edge_data, valid_nodes


def create_directional_graph(node_data, edge_data):
    G = nx.MultiDiGraph()

    # Add nodes with attributes
    for node_type, data in node_data.items():
        nodes = data.to_dict('records')
        for node in nodes:
            G.add_node(node[':ID'], **{k: v for k, v in node.items() if k != ':ID'}, type=node_type)

    # Add edges with types
    for edge_type, data in edge_data.items():
        edges = data.to_dict('records')
        for edge in edges:
            G.add_edge(edge['start'], edge['end'], type=edge_type)

    return G


def plot_graph(G):
    # Define a node color mapping
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

    # Define a edge color mapping
    edge_color_map = {
        'AST': 'red',
        'CALL': 'purple',
        'CDG': 'green',
        'CFG': 'blue',
        'REACHING_DEF': 'grey',
        'RECEIVER': 'cyan',
        'ARGUMENT': 'yellow',
    }

    # Get types of nodes/edges
    node_types = [G.nodes[n].get('type', 'default_type') for n in G.nodes]
    edge_types = [G.edges[e]['type'] for e in G.edges]

    # Apply color mapping
    node_colors = [node_color_map[node_type] for node_type in node_types]
    edge_colors = [edge_color_map[edge_type] for edge_type in edge_types]

    # Create legend patches for nodes
    node_legend_patches = [mpatches.Patch(color=color, label=node_type) for node_type, color in node_color_map.items()]

    # Create legend patches for edges
    edge_legend_patches = [mpatches.Patch(color=color, label=edge_type) for edge_type, color in edge_color_map.items()]

    # Create an equivalent undirected graph (since we want its layout)
    G_undirected = nx.Graph(G)

    # Compute layout
    pos = nx.spring_layout(G_undirected)

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

    # Show the graph
    plt.title('Graph Visualization')
    plt.show()


def remove_leaf(node, G):
    # This will remove the node and all related edges
    G.remove_node(node)


def remove_inner(node, G):
    parent_node = list(G.predecessors(node))[0] # There should be only one
    child_nodes = list(G.successors(node))

    # Remove the node and all its related edges
    G.remove_node(node)

    # Make new edges from parent to child nodes (bypassing current node)
    for child_node in child_nodes:
        G.add_edge(parent_node, child_node, type='AST')


def is_leaf(node, G):
    return G.out_degree(node) == 0


def is_root(node, G):
    return G.in_degree(node) == 0


def is_inner(node, G):
    return (G.in_degree(node) == 1) and (G.out_degree(node) > 0)


def remove_all_wcc_nodes(wcc, G):
    for node in wcc:
        G.remove_node(node)


def remove_invalid_nodes(sample_id, node_data, edge_data, valid_nodes):
    # Get some stats for compression info
    original_edge_count = 0
    for data in edge_data.values():
        original_edge_count += len(data)

    # Add only AST edges, since they form a tree and the following alg works for trees only
    G = create_directional_graph(node_data, {'AST': edge_data.pop('AST')})
    wccs_num_before = len(list(nx.weakly_connected_components(G)))

    invalid_nodes = set(G.nodes) - valid_nodes

    # Keep removing until all invalid nodes are gone
    while invalid_nodes:
        node = invalid_nodes.pop()

        if is_leaf(node, G):
            remove_leaf(node, G)
        elif is_root(node, G):
            # Make this node a valid BLOCK node
            G.nodes[node].update({'ORDER:int': 0, 'type': 'BLOCK'})
        elif is_inner(node, G):
            remove_inner(node, G)
        else:
            # This shouldn't happen, because it would mean that the AST is not a tree
            print(f'ERROR: visualize_graph.py: current graph has a node which is not root, inner or leaf - which is not possible in a tree!')
            exit(1)

    wccs_after = list(nx.weakly_connected_components(G))
    wccs_num_after = len(wccs_after)

    # The number of WCCs shouldn't change after removal of invalid nodes
    if wccs_num_before != wccs_num_after:
        print('ERROR: visualize_graph.py: By removing the invalid nodes, the graph was split into multiple WCCs!')

    # Remove WCCs which are composed only from BLOCK nodes
    for wcc in wccs_after:
        remove_wcc = True
        for node in wcc:
            if G.nodes[node]['type'] != 'BLOCK':
                remove_wcc = False
                break

        if remove_wcc:
            remove_all_wcc_nodes(wcc, G)

    # Remove leaf BLOCK nodes (and possibly others)
    all_nodes_copy = list(G.nodes)
    for node in all_nodes_copy:
        if is_leaf(node, G) and G.nodes[node]['type'] == 'BLOCK':
            G.remove_node(node)

    # All current nodes are now considered valid
    valid_nodes = set(G.nodes)

    # Add rest of the edges 'CFG', 'CALL', ...
    for edge_type, data in edge_data.items():
        edges = data.to_dict('records')
        for edge in edges:
            G.add_edge(edge['start'], edge['end'], type=edge_type)

    # Newly added edges introduced new invalid nodes
    all_nodes = set(G.nodes)

    # Remove edges of other types which are related to invalid nodes
    for node in all_nodes:
        if node not in valid_nodes:
            G.remove_node(node)

    # Check if the graph is a single WCC
    if len(list(nx.weakly_connected_components(G))) != 1:
        print('ERROR: visualize_graph.py: The graph consists of more than one WCC!')

    # Print compression of the graph after removal
    after_edge_count = G.number_of_edges()
    compression_percentage = ((original_edge_count - after_edge_count) / original_edge_count) * 100
    print(f'Note: visualize_graph.py: Graph \'{sample_id}\' compressed by {compression_percentage:.1f}%.')

    return G


def process_sample(directory):
    sample_id = directory.split('/')[-1]
    node_data, edge_data, valid_nodes = load_sample_data(directory)
    remove_invalid_nodes(sample_id, node_data, edge_data, valid_nodes)
    return


def cache_headers(headers_dir):
    global headers_cache

    if not os.path.isdir(headers_dir):
        print(f"Error: visualize_graph.py: Directory with CSV headers doesn't exist! Try running 'make header-cache'.")

    for filename in os.listdir(headers_dir):
        # Skip everything other than a file (there shouldn't be anything else though)
        if not os.path.isfile(os.path.join(headers_dir, filename)):
            continue

        # Skip edge files headers - they all have the same header (START,END,TYPE)
        if 'nodes' not in filename:
            continue

        # Load header
        path_header = os.path.join(headers_dir, filename)
        header = pd.read_csv(path_header, header=None).iloc[0]

        # Save header
        header_type = filename[6:-11] # Remove 'nodes_' and '_header.csv'
        headers_cache[header_type] = header


if __name__ == "__main__":
    # Cache CSV headers
    cache_headers('header_cache/')

    if sys.stdin.isatty():
        # If stdin is empty - run in single-threaded mode (only one graph)
        directory = sys.argv[1]
        sample_id = directory.split('/')[-1]
        node_data, edge_data, valid_nodes = load_sample_data(directory)
        G = remove_invalid_nodes(sample_id, node_data, edge_data, valid_nodes)
        plot_graph(G)
    else:
        # Stdin has data, process each line in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
            for line in sys.stdin:
                directory = line.strip()
                executor.submit(process_sample, directory)
