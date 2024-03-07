import os
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import pandas as pd
import concurrent.futures


headers_cache = {}

def drop_unwanted_attributes(df, type_name):
    if type_name == 'METHOD':
        df = df[[':ID', 'ORDER:int', 'FULL_NAME:string', 'IS_EXTERNAL:boolean']]
        df = df.rename(columns={'ORDER:int': 'ORDER', 'FULL_NAME:string': 'FULL_NAME', 'IS_EXTERNAL:boolean': 'IS_EXTERNAL'})
        df['ARGUMENT_INDEX'] = 0 # Since 'METHOD' doesn't have ARGUMENT_INDEX, we use invalid value 0
    elif type_name in ['METHOD_PARAMETER_IN', 'METHOD_RETURN', 'MEMBER', 'LOCAL']:
        df = df[[':ID', 'ORDER:int']]
        df = df.rename(columns={'ORDER:int': 'ORDER'})
        df['ARGUMENT_INDEX'] = 0 # Since these nodes also don't have ARGUMENT_INDEX, we use invalid value 0
    elif type_name == 'TYPE':
        df = df[[':ID', 'FULL_NAME:string']]
        df = df.rename(columns={'FULL_NAME:string': 'FULL_NAME'})
    elif type_name == 'TYPE_DECL':
        df = df[[':ID']] # All TYPE_DECL nodes will be removed, but in a special way from other nodes
    elif type_name == 'LITERAL':
        df = df[[':ID', 'ARGUMENT_INDEX:int', 'ORDER:int', 'CODE:string']]
        df = df.rename(columns={'ARGUMENT_INDEX:int': 'ARGUMENT_INDEX', 'ORDER:int': 'ORDER', 'CODE:string': 'CODE'})
    else:
        # BLOCK, CALL, FIELD_IDENTIFIER, IDENTIFIER, METHOD_REF, RETURN, UNKNOWN
        df = df[[':ID', 'ARGUMENT_INDEX:int', 'ORDER:int']]
        df = df.rename(columns={'ARGUMENT_INDEX:int': 'ARGUMENT_INDEX', 'ORDER:int': 'ORDER'})

    df = df.rename(columns={':ID': 'ID'})
    return df


def load_sample_data(directory):
    valid_nodes = set()
    node_data = {}
    edge_data = {}
    required_nodes = {
        # 'META_DATA',
        # 'FILE',
        # 'NAMESPACE',
        # 'NAMESPACE_BLOCK',
        'METHOD',
        'METHOD_PARAMETER_IN',
        # 'METHOD_PARAMETER_OUT',
        'METHOD_RETURN',
        'MEMBER',
        'TYPE',
        # 'TYPE_ARGUMENT',
        'TYPE_DECL', # Needs to be removed carefully
        # 'TYPE_PARAMETER',
        # 'AST_NODE',
        'BLOCK',
        'CALL',
        # 'CALL_REPR',
        # 'CONTROL_STRUCTURE',
        # 'EXPRESSION',
        'FIELD_IDENTIFIER',
        'IDENTIFIER',
        # 'JUMP_LABEL',
        # 'JUMP_TARGET',
        'LITERAL',
        'LOCAL',
        'METHOD_REF',
        # 'MODIFIER',
        'RETURN',
        # 'TYPE_REF',
        'UNKNOWN',
        # 'CFG_NODE',
        # 'COMMENT',
        # 'FINDING',
        # 'KEY_VALUE_PAIR',
        # 'LOCATION',
        # 'TAG',
        # 'TAG_NODE_PAIR',
        # 'CONFIG_FILE',
        # 'BINDING',
        # 'ANNOTATION',
        # 'ANNOTATION_LITERAL',
        # 'ANNOTATION_PARAMETER',
        # 'ANNOTATION_PARAMETER_ASSIGN',
        # 'ARRAY_INITIALIZER',
        # 'DECLARATION',
    }

    required_edges = {
        # 'SOURCE_FILE',
        # 'ALIAS_OF',
        # 'BINDS_TO',
        # 'INHERITS_FROM',
        'AST',
        # 'CONDITION',
        'ARGUMENT',
        'CALL',
        # 'RECEIVER',
        'CFG',
        # 'DOMINATE',
        # 'POST_DOMINATE',
        'CDG',
        # 'REACHING_DEF',
        # 'CONTAINS',
        'EVAL_TYPE',
        # 'PARAMETER_LINK',
        # 'TAGGED_BY',
        # 'BINDS',
        'REF',
    }

    for filename in os.listdir(directory):
        if not filename.endswith('_data.csv'):
            continue

        type_name = '_'.join(filename.split("_")[1:-1])

        # Load nodes
        if filename.startswith("nodes_"):
            if type_name not in required_nodes:
                continue

            path_data = os.path.join(directory, filename)
            df = pd.read_csv(path_data, header=None, names=headers_cache[type_name])

            node_data[type_name] = drop_unwanted_attributes(df, type_name)

        # Load edges
        elif filename.startswith("edges_"):
            if type_name not in required_edges:
                continue

            path = os.path.join(directory, filename)
            df = pd.read_csv(path, header=None, names=["start", "end", 'type'])
            edge_data[type_name] = df[['start', 'end']] # Drop edge type - we already know it

        # Valid nodes - nodes that are actually stored in CSV files, any other
        # node is considered invalid
        for node_type, data in node_data.items():
            valid_nodes.update(set(data['ID']))

    return node_data, edge_data, valid_nodes


def create_directional_graph(node_data, edge_data):
    G = nx.MultiDiGraph()

    # Add nodes with attributes
    for node_type, data in node_data.items():
        nodes = data.to_dict('records')
        for node in nodes:
            G.add_node(node['ID'], **{k: v for k, v in node.items() if k != 'ID'}, type=node_type)

    # Add edges with types
    for edge_type, data in edge_data.items():
        edges = data.to_dict('records')
        for edge in edges:
            G.add_edge(edge['start'], edge['end'], type=edge_type)

    return G


def plot_graph(G):
    # Define a node color mapping
    node_color_map = {
            'METHOD': 'red',
            'METHOD_PARAMETER_IN': 'blue',
            'METHOD_RETURN': 'blue',
            'MEMBER': 'blue',
            'TYPE': 'blue',
            'TYPE_DECL': 'blue', # Needs to be removed carefully
            'BLOCK': 'blue',
            'CALL': 'blue',
            'FIELD_IDENTIFIER': 'blue',
            'IDENTIFIER': 'blue',
            'LITERAL': 'green',
            'LOCAL': 'blue',
            'METHOD_REF': 'blue',
            'RETURN': 'blue',
            'UNKNOWN': 'blue',
            # Splitted nodes
            'METHOD_INFO': 'orange',
            'LITERAL_VALUE': 'purple',
    }

    # Define a edge color mapping
    edge_color_map = {
        # 'AST': 'red',
        # 'CALL': 'purple',
        # 'CDG': 'green',
        # 'CFG': 'blue',
        # 'REACHING_DEF': 'grey',
        # 'RECEIVER': 'cyan',
        # 'ARGUMENT': 'yellow',
        'METHOD_INFO_LINK': 'orange',
        'LITERAL_VALUE_LINK': 'purple',
    }

    # Get types of nodes/edges
    node_types = [G.nodes[n]['type'] for n in G.nodes]
    edge_types = [G.edges[e]['type'] for e in G.edges]

    # Apply color mapping
    node_colors = [node_color_map.get(node_type, 'grey') for node_type in node_types]
    edge_colors = [edge_color_map.get(edge_type, 'grey') for edge_type in edge_types]

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


def filter_eval_type_edges(G):
    edges_to_remove = []

    # Find EVAL_TYPE edges which are connected to nodes for which we don't need types
    for start, end, key, data in G.edges(keys=True, data=True):
        if data['type'] != 'EVAL_TYPE':
            continue

        if G.nodes[start]['type'] in ['METHOD', 'BLOCK', 'METHOD_REF']:
            edges_to_remove.append((start, end, key))

    # Remove the edges
    for start, end, key in edges_to_remove:
        G.remove_edge(start, end, key)

    return G


def is_AST_child_of_CALL(G, node):
    for start, _, edge_data in G.in_edges(node, data=True):
        # Find AST edge which starts in node's parent (they should max one - its a tree)
        if edge_data['type'] == 'AST':
            return G.nodes[start]['type'] == 'CALL'

    # There won't be any incoming AST edge for the root
    return False


def set_argument_index(G):
    for node, node_data in G.nodes(data=True):
        # Skip nodes which don't have the ARGUMENT_INDEX property
        if 'ARGUMENT_INDEX' not in node_data:
            continue

        # If node is AST child of CALL we keep it's ARGUMENT_INDEX value, otherwise the value is set to 0
        if not is_AST_child_of_CALL(G, node):
            G.nodes[node]['ARGUMENT_INDEX'] = 0

    return G


def num_of_incoming_type_edges(G, node, edge_type):
    cnt = 0

    for _, _, edge_data in G.in_edges(node, data=True):
        if edge_data['type'] == edge_type:
            cnt += 1

    return cnt


def get_member_children(G, node):
    member_children = []

    for _, end, edge_data in G.out_edges(node, data=True):
        if edge_data['type'] != 'AST' and edge_data['type'] != 'CONSISTS_OF':
            continue

        if G.nodes[end]['type'] == 'MEMBER':
            member_children.append(end)

    return member_children


# This filtering should also remove recursive structures - because they can only contain reference (pointer)
# to itself, so there won't be any selfloop of EVAL_TYPE (even more complex ones), which would case the TYPE
# node to be considered as "USEFUL"
def filter_type_nodes(G):
    type_nodes = [ node for node, node_data in G.nodes(data=True) if node_data['type'] == 'TYPE' ]

    # Iterate until all TYPE nodes without incoming EVAL_TYPE edges are removed
    while True:
        nodes_without_incoming_edges = [ node for node in type_nodes if not num_of_incoming_type_edges(G, node, 'EVAL_TYPE')]

        if not nodes_without_incoming_edges:
            # All TYPE nodes now have incoming EVAL_TYPE edges (or there are no TYPE nodes at all - very unlikely...)
            break

        for node in nodes_without_incoming_edges:
            member_children = get_member_children(G, node)
            G.remove_node(node)
            type_nodes.remove(node)
            for member_node in member_children:
                G.remove_node(member_node)

    return G


def get_type_parent(G, node, node_type):
    for start, _, edge_data in G.in_edges(node, data=True):
        # Find AST edge which starts in node's parent (they should max one - its a tree)
        if edge_data['type'] == node_type:
            return start


def remove_type_decl_nodes(G):
    nodes_to_remove = []
    edges_to_add = []

    for node, data in G.nodes(data=True):
        # We only want to remove TYPE_DECL nodes
        if data['type'] != 'TYPE_DECL':
            continue

        member_children = get_member_children(G, node)

        if member_children:
            # There is at least one MEMBER child -> we need to connect it to node's parent
            # When node is root, it will crash later on, but nodes TYPE shouldn't be roots
            parent = get_type_parent(G, node, 'REF')
            for member_child in member_children:
                edges_to_add.append((parent, member_child))

        nodes_to_remove.append(node)

    # Remove the nodes
    for node in nodes_to_remove:
        G.remove_node(node)

    # Add back the edges
    for start, end in edges_to_add:
        G.add_edge(start, end, type='CONSISTS_OF')

    return G


def get_ast_children(G, node):
    ast_children = []

    for _, end, edge_data in G.out_edges(node, data=True):
        if edge_data['type'] != 'AST':
            continue

        ast_children.append(end)

    return ast_children


def have_incoming_call_edges(G, node):
    for _, _, edge_data in G.in_edges(node, data=True):
        if edge_data['type'] == 'CALL':
            return True

    return False


def remove_external_method_children(G):
    nodes_to_remove = []

    for node, data in G.nodes(data=True):
        if data['type'] != 'METHOD' or not data['IS_EXTERNAL']:
            continue

        if not have_incoming_call_edges(G, node):
            # If the method is not used, there is no need to keep it in a graph
            nodes_to_remove.append(node)

        nodes_to_remove += get_ast_children(G, node)

    for node in nodes_to_remove:
        G.remove_node(node)

    return G


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

    # Remove AST children (METHOD_PARAMETER_IN and METHOD_RETURN) of external methods
    G = remove_external_method_children(G)

    # Remove EVAL_TYPE edges from some types of nodes
    G = filter_eval_type_edges(G)

    # Carefully remove TYPE_DECL nodes
    G = remove_type_decl_nodes(G)

    # Remove TYPE nodes (and it's child MEMBER nodes) without incoming EVAL_TYPE edges
    G = filter_type_nodes(G)

    # If node is not AST children of CALL node, it's ARGUMENT_INDEX will be set to 0
    G = set_argument_index(G)

    # Check if the graph is a single WCC
    if len(list(nx.weakly_connected_components(G))) != 1:
        print('ERROR: visualize_graph.py: The graph consists of more than one WCC!')

    # Print compression of the graph after removal
    after_edge_count = G.number_of_edges()
    compression_percentage = ((original_edge_count - after_edge_count) / original_edge_count) * 100
    print(f'Note: visualize_graph.py: Graph \'{sample_id}\' compressed by {compression_percentage:.1f}%.')

    return G


def split_method_node(G, node, new_node_id):
    full_name = G.nodes[node].pop('FULL_NAME')
    is_external = G.nodes[node].pop('IS_EXTERNAL')

    # Add new METHOD_INFO node
    G.add_node(new_node_id, FULL_NAME=full_name, IS_EXTERNAL=is_external, type='METHOD_INFO')

    # Connect it to the original METHOD node with METHOD_INFO_LINK edge
    G.add_edge(new_node_id, node, type='METHOD_INFO_LINK')

    return G


def split_literal_node(G, node, new_node_id):
    code = G.nodes[node].pop('CODE')

    # Add new LITERAL_VALUE node
    G.add_node(new_node_id, CODE=code, type='LITERAL_VALUE')

    # Connect it to the original LITERAL node with LITERAL_VALUE_LINK edge
    G.add_edge(new_node_id, node, type='LITERAL_VALUE_LINK')

    return G


def split_nodes(G):
    node_list = list(G.nodes(data=True))
    new_node_id = max(G.nodes) + 1

    for node, data in node_list:
        if data['type'] == 'METHOD':
            G = split_method_node(G, node, new_node_id)
            new_node_id += 1
        elif data['type'] == 'LITERAL':
            G = split_literal_node(G, node, new_node_id)
            new_node_id += 1

    return G


def process_sample(directory):
    sample_id = directory.split('/')[-1]
    node_data, edge_data, valid_nodes = load_sample_data(directory)
    G = remove_invalid_nodes(sample_id, node_data, edge_data, valid_nodes)
    G = split_nodes(G)
    return G


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
        G = process_sample(directory)
        plot_graph(G)
    else:
        # Stdin has data, process each line in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
            for line in sys.stdin:
                directory = line.strip()
                executor.submit(process_sample, directory)
