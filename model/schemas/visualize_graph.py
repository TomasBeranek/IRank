import os
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import pandas as pd
import concurrent.futures
import re
import hashlib
import numpy as np
import math
import threading
import tensorflow_gnn as tfgnn
import tensorflow as tf


TFRecord_writing_lock = threading.Lock()

FP_data_types = {'void': 0, # For code simplicity (although it isn't FP type)
                 'half': 1,
                 'float': 2,
                 'double': 3,
                 'fp128': 4}

norm_coeffs = {'libtiff': {
                             'ARGUMENT_INDEX': 18,
                             'LEN': 8192,
                             'MEMBER_ORDER': 82,
                             'OPERATORS': {'<operator>.addition': 1,
                                           '<operator>.addressOf': 2,
                                           '<operator>.and': 3,
                                           '<operator>.arithmeticShiftRight': 4,
                                           '<operator>.assignment': 5,
                                           '<operator>.cast': 6,
                                           '<operator>.division': 7,
                                           '<operator>.equals': 8,
                                           '<operator>.fneg': 9,
                                           '<operator>.getElementPtr': 10,
                                           '<operator>.greaterEqualsThan': 11,
                                           '<operator>.greaterThan': 12,
                                           '<operator>.indexAccess': 13,
                                           '<operator>.indirection': 14,
                                           '<operator>.lessEqualsThan': 15,
                                           '<operator>.lessThan': 1,
                                           '<operator>.logicalShiftRight': 16,
                                           '<operator>.modulo': 17,
                                           '<operator>.multiplication': 18,
                                           '<operator>.notEquals': 19,
                                           '<operator>.or': 20,
                                           '<operator>.pointerShift': 21,
                                           '<operator>.select': 22,
                                           '<operator>.shiftLeft': 23,
                                           '<operator>.subtraction': 24,
                                           '<operator>.xor': 25},
                             'ORDER': 1009,
                             'PTR': 4
                          }
              }

LABEL = None


def drop_unwanted_attributes(df, type_name):
    df['nodeset'] = 'AST_NODE'

    if type_name == 'METHOD':
        df = df[[':ID', 'ORDER:int', 'FULL_NAME:string', 'IS_EXTERNAL:boolean', 'nodeset']]
        df = df.rename(columns={'ORDER:int': 'ORDER', 'FULL_NAME:string': 'FULL_NAME', 'IS_EXTERNAL:boolean': 'IS_EXTERNAL'})
    elif type_name in ['METHOD_PARAMETER_IN', 'METHOD_RETURN', 'MEMBER', 'LOCAL']:
        df = df[[':ID', 'ORDER:int', 'nodeset']]
        df = df.rename(columns={'ORDER:int': 'ORDER'})
    elif type_name == 'TYPE':
        df = df[[':ID', 'FULL_NAME:string']]
        df = df.rename(columns={'FULL_NAME:string': 'FULL_NAME'})
        df['nodeset'] = 'TYPE'
    elif type_name == 'TYPE_DECL':
        df = df[[':ID']] # All TYPE_DECL nodes will be removed, but in a special way from other nodes
    elif type_name == 'LITERAL':
        df = df[[':ID', 'ARGUMENT_INDEX:int', 'ORDER:int', 'CODE:string', 'nodeset']]
        df = df.rename(columns={'ARGUMENT_INDEX:int': 'ARGUMENT_INDEX', 'ORDER:int': 'ORDER', 'CODE:string': 'CODE'})
    else:
        # BLOCK, CALL, FIELD_IDENTIFIER, IDENTIFIER, METHOD_REF, RETURN, UNKNOWN
        df = df[[':ID', 'ARGUMENT_INDEX:int', 'ORDER:int', 'nodeset']]
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
            path_header = os.path.join(directory, f'nodes_{type_name}_header.csv')

            header = pd.read_csv(path_header, header=None).iloc[0]
            df = pd.read_csv(path_data, header=None, names=header)

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
        'REF': 'red',
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

    # Add back the edges, but add them to as new CONSISTS_OF edgeset
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
    # original_edge_count = 0
    # for data in edge_data.values():
    #     original_edge_count += len(data)

    # Add only AST edges, since they form a tree and the following alg works for trees only
    G = create_directional_graph(node_data, {'AST': edge_data.pop('AST')})
    # wccs_num_before = len(list(nx.weakly_connected_components(G)))

    invalid_nodes = set(G.nodes) - valid_nodes

    # Keep removing until all invalid nodes are gone
    while invalid_nodes:
        node = invalid_nodes.pop()

        if is_leaf(node, G):
            remove_leaf(node, G)
        elif is_root(node, G):
            # Make this node a valid BLOCK node
            G.nodes[node].update({'ARGUMENT_INDEX': 0, 'ORDER': 0, 'nodeset': 'AST_NODE', 'type': 'BLOCK'})
        elif is_inner(node, G):
            remove_inner(node, G)
        else:
            # This shouldn't happen, because it would mean that the AST is not a tree
            print(f'ERROR: visualize_graph.py: current graph has a node which is not root, inner or leaf - which is not possible in a tree!', file=sys.stderr)
            exit(1)

    wccs_after = list(nx.weakly_connected_components(G))
    # wccs_num_after = len(wccs_after)

    # The number of WCCs shouldn't change after removal of invalid nodes
    # if wccs_num_before != wccs_num_after:
    #     print('ERROR: visualize_graph.py: By removing the invalid nodes, the graph was split into multiple WCCs!')

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
        if edge_type == 'ARGUMENT':
            # For arguments we add an ARGUMENT attribute for each edge
            for edge in edges:
                # We want keep only ARGUMENT edges between CALL and its arguments
                if G.nodes[edge['start']]['type'] == 'CALL':
                    argument_index = G.nodes[edge['end']].get('ARGUMENT_INDEX', 0) # If its missing use invalid value - 0
                    G.add_edge(edge['start'], edge['end'], type=edge_type, ARGUMENT_INDEX=argument_index)
        else:
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
    # Since we are moving ARGUMENT_INDEX to ARGUMENT edge this is no longer needed
    # G = set_argument_index(G)

    # Check if the graph is a single WCC
    if len(list(nx.weakly_connected_components(G))) != 1:
        print('ERROR: visualize_graph.py: The graph consists of more than one WCC!')

    # Print compression of the graph after removal
    # after_edge_count = G.number_of_edges()
    # compression_percentage = ((original_edge_count - after_edge_count) / original_edge_count) * 100
    # print(f'Note: visualize_graph.py: Graph \'{sample_id}\' compressed by {compression_percentage:.1f}%.')

    return G


def split_method_node(G, node, new_node_id):
    full_name = G.nodes[node].pop('FULL_NAME')
    is_external = G.nodes[node].pop('IS_EXTERNAL')

    # Add new METHOD_INFO node
    G.add_node(new_node_id, FULL_NAME=full_name, IS_EXTERNAL=is_external, nodeset='METHOD_INFO', type='METHOD_INFO')

    # Connect it to the original METHOD node with METHOD_INFO_LINK edge
    G.add_edge(new_node_id, node, type='METHOD_INFO_LINK')

    return G


def get_node_full_type(G, node):
    for _, end, edge_data in G.out_edges(node, data=True):
        if edge_data['type'] == 'EVAL_TYPE':
            return G.nodes[end]['FULL_NAME']


def split_literal_node(G, node, new_node_id):
    code = G.nodes[node].pop('CODE')

    # Get LITERAL type
    type_full_name = get_node_full_type(G, node)

    # Add new LITERAL_VALUE node
    G.add_node(new_node_id, value_type_pair=(code, type_full_name), nodeset='LITERAL_VALUE', type='LITERAL_VALUE')

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


def count_pointers(full_name):
    stripped_name = full_name.rstrip('*')
    pointers_cnt = len(full_name) - len(stripped_name)
    return stripped_name, pointers_cnt


def get_array_len(type_name):
    # Check if type is array
    if type_name[0] == '[' and type_name[-1] == ']':
        string_parts = type_name.split(' x ', 1)
        len_string = string_parts[0][1:] # Remove initial '['
        type_name_string = string_parts[1][:-1] # Remove trailing ']'
        return type_name_string, int(len_string)
    else:
        return type_name, 0


def hash_string_to_int24(str):
    # Use MD5 to hash str
    hash_obj = hashlib.md5(str.encode())  # Encode the string to bytes
    hash_int = int.from_bytes(hash_obj.digest()[:4], 'little', signed=False)  # Convert first 4 bytes to int

    # Extract 24 bits so this int can be correctly represented by float32 (which has mantissa == 23 + 1 implicit bit)
    extracted_bits = hash_int >> 8
    return extracted_bits


def split_type_full_name(full_name):
    INT = FP = HASH = 0

    # Remove and count the number of trailing '*' (pointers)
    name_without_pointers, PTR = count_pointers(full_name)

    # Remove and get LEN if type is array
    type_name, LEN = get_array_len(name_without_pointers)

    if re.match(r'^i\d+$', type_name):
        # Type is a integer
        INT = int(type_name[1:])
    elif type_name in FP_data_types:
        # Type is FP (floating point)
        FP = FP_data_types[type_name]
    # elif PTR or LEN or (type_name[0] == '{' and type_name[-1] == '}') or (type_name in ['data']):
    else:
        # These are either user defined types, structs or arrays
        HASH = hash_string_to_int24(type_name)

    return INT, FP, LEN, PTR, HASH


def transform_type_data(df):
    # Split FULL_NAME to INT, FP, LEN, PTR and HASH
    df[['INT', 'FP', 'LEN', 'PTR', 'HASH']] = df['FULL_NAME'].apply(lambda x: pd.Series(split_type_full_name(x)))

    # Keep only columns specified in TFGNN schema
    df = df.drop(['nodeset', 'type', 'FULL_NAME'], axis=1)

    # Normalize the values and retype to float32 (DT_FLOAT)
    df['INT'] = (df['INT'] / 128).astype('float32')
    df['FP'] = (df['FP'] / 4).astype('float32')
    df['LEN'] = (df['LEN'] / norm_coeffs['LEN']).astype('float32')
    df['PTR'] = (df['PTR'] / norm_coeffs['PTR']).astype('float32')
    df['HASH'] = (df['HASH'] / (2**24 - 1)).astype('float32')  # MAX_UINT for 24 bits

    return df


def split_method_full_name(full_name):
    HASH = OPERATOR = 0

    if full_name.startswith('<operator>.'):
        # LLVM IR operator
        OPERATOR = norm_coeffs['OPERATORS'][full_name]
    else:
        HASH = hash_string_to_int24(full_name)

    return HASH, OPERATOR


def transform_method_info_data(df):
    # Split FULL_NAME to INT, FP, LEN, PTR and HASH
    df[['HASH', 'OPERATOR']] = df['FULL_NAME'].apply(lambda x: pd.Series(split_method_full_name(x)))

    # Keep only columns specified in TFGNN schema
    df = df.drop(['type', 'nodeset', 'FULL_NAME'], axis=1)

    # Normalize the values and retype to float32 (DT_FLOAT)
    df['HASH'] = (df['HASH'] / (2**24 - 1)).astype('float32') # MAX_UINT for 24 bits
    df['OPERATOR'] = (df['OPERATOR'] / len(norm_coeffs['OPERATORS'])).astype('float32') # Number of found operators
    df['IS_EXTERNAL'] = df['IS_EXTERNAL'].astype('float32')

    return df


def transform_ast_node_data(df):
    labels = {  'UNKNOWN': 0,
                'METHOD': 1,
                'METHOD_PARAMETER_IN': 2,
                'METHOD_RETURN': 3,
                'BLOCK': 4,
                'CALL': 5,
                'FIELD_IDENTIFIER': 6,
                'IDENTIFIER': 7,
                'LITERAL': 8,
                'LOCAL': 9,
                'METHOD_REF': 10,
                'RETURN': 11}

    # Keep only columns specified in TFGNN schema
    df = df.drop(['ARGUMENT_INDEX', 'nodeset'], axis=1)

    # Rename type to LABEL
    df = df.rename(columns={'type': 'LABEL'})

    # Normalize the values and retype to float32 (DT_FLOAT)
    df['ORDER'] = (df['ORDER'] / norm_coeffs['ORDER']).astype('float32')
    df['LABEL'] = (df['LABEL'].map(labels) / len(labels)).astype('float32')

    return df


def encode_literal_value(value_type_pair):
    value, type = value_type_pair

    INT = FP_MANTISSA = FP_EXPONENT = HASH = INVALID_POINTER = ZERO_INITIALIZED = UNDEF = 0

    if value == 'undef':
        UNDEF = 1.0
    elif not type:
        # Type is missing
        HASH = hash_string_to_int24(str(value))
    elif type.endswith('*'):
        # Pointer
        if value == 'nullptr':
            INVALID_POINTER = 1.0
        else:
            # Literal's CODE property can contain function code for funciton pointers
            HASH = hash_string_to_int24(str(value))
    elif re.match(r'^i\d+$', type):
        # Integer
        if type == 'i1':
            # Bool is special case of integer - might be benefical to create a separate flag for it
            if value == 'false':
                value = 0
            elif value == 'true':
                value = 1

        # int_size = int(type[1:]) # In bits
        int_size = 16

        # To keep enough precision for lower values we sacrifice precision in higher values
        # We basically crop it to unsigned int16 so it can be represented by float32
        value = min(int(value), (2**(int_size - 1)) - 1)
        value = max(value, -(2**(int_size - 1)))
        value += 2**15

        # Convert it to "unsigned" and normalize it according to its size, we lose information this way, but we need to encode it to float32
        INT = value / (2**int_size - 1)
    elif type in ['half', 'float', 'double', 'fp128']:
        # Floating point
        value = -1
        value = np.float32(value)
        FP_MANTISSA, FP_EXPONENT = math.frexp(value)

        if value < 0:
            # Shift negative mantissa from (-1, -0.5> to (0, 0.5> where 0.5 is reserved for 0 (yes, we lose highest negative number and lowest postive number...)
            # This will break the equation value=FP_MANTISSA*2^FP_EXPONENT (but we don't need it)
            FP_MANTISSA += 1
        elif value == 0:
            # If value == 0 math.frexp will display it as 0,0 - we need to move it to 0.5
            FP_MANTISSA = 0.5

        FP_EXPONENT = np.float32(((FP_EXPONENT + 148) / (148 + 128)))
    else:
        # Arrays, structs, custom types (should be only 'data')
        if value == 'zero initialized':
            ZERO_INITIALIZED = 1.0
        else:
            HASH = hash_string_to_int24(str(value))

    return INT, FP_MANTISSA, FP_EXPONENT, HASH, INVALID_POINTER, ZERO_INITIALIZED, UNDEF


def transform_literal_value_node_data(df):
    df[['INT', 'FP_MANTISSA', 'FP_EXPONENT', 'HASH', 'INVALID_POINTER', 'ZERO_INITIALIZED', 'UNDEF']] = df['value_type_pair'].apply(lambda x: pd.Series(encode_literal_value(x)))

    df = df.drop(['nodeset', 'type', 'value_type_pair'], axis=1)

    # Normalize values
    df['HASH'] = (df['HASH'] / (2**24 - 1)).astype('float32') # MAX_UINT for 24 bits

    return df


def transform_member_data(df):
    df = df.drop(['nodeset', 'type', 'ARGUMENT_INDEX'], axis=1)

    # Normalize values
    df['ORDER'] = (df['ORDER'] / norm_coeffs['MEMBER_ORDER']).astype('float32')

    return df


def transform_data_types(G):
    # Export nodes and edges from NX graph to Pandas df
    nodes_df = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient='index')
    edges_list = [{'source': u, 'target': v, **data} for u, v, data in G.edges(data=True)]
    edges_df = pd.DataFrame(edges_list)

    # Split nodes and edges df to dfs according to their nodeset
    grouped_nodes = nodes_df.groupby('nodeset')
    grouped_edges = edges_df.groupby('type')
    grouped_nodes_dfs = {name: group for name, group in grouped_nodes}
    grouped_edges_dfs = {name: group.drop(columns=['type']) for name, group in grouped_edges}

    # Remove NaN columns
    graph_dfs = {name: df.dropna(axis=1, how='all') for name, df in {**grouped_nodes_dfs, **grouped_edges_dfs}.items()}

    # Separate MEMBER nodes from AST_NODE nodes
    graph_dfs['MEMBER'] = graph_dfs['AST_NODE'].loc[graph_dfs['AST_NODE']['type'] == 'MEMBER']
    graph_dfs['AST_NODE'] = graph_dfs['AST_NODE'].loc[graph_dfs['AST_NODE']['type'] != 'MEMBER']

    # Separate EVAL_TYPE edges which connect MEMBER and TYPE nodes
    graph_dfs['EVAL_MEMBER_TYPE'] = graph_dfs['EVAL_TYPE'].loc[graph_dfs['EVAL_TYPE']['source'].isin(graph_dfs['MEMBER'].index)]
    graph_dfs['EVAL_TYPE'] = graph_dfs['EVAL_TYPE'].loc[~graph_dfs['EVAL_TYPE']['source'].isin(graph_dfs['MEMBER'].index)]

    # Transform text features to normalized numeric form
    graph_dfs['MEMBER'] = transform_member_data(graph_dfs['MEMBER'])
    graph_dfs['AST_NODE'] = transform_ast_node_data(graph_dfs['AST_NODE'])
    graph_dfs['TYPE'] = transform_type_data(graph_dfs['TYPE'])
    graph_dfs['LITERAL_VALUE'] = transform_literal_value_node_data(graph_dfs['LITERAL_VALUE'])
    graph_dfs['METHOD_INFO'] = transform_method_info_data(graph_dfs['METHOD_INFO'])

    # Normalize ARGUMENT_INDEX value of edge ARGUMENT
    graph_dfs['ARGUMENT']['ARGUMENT_INDEX'] = (graph_dfs['ARGUMENT']['ARGUMENT_INDEX'] / norm_coeffs['ARGUMENT_INDEX']).astype('float32')

    return graph_dfs


def convert_ids_to_tfgnn_format(graph_in_dfs):
    edgeset_info = {
        'METHOD_INFO_LINK':   {'SOURCE': 'METHOD_INFO',   'TARGET': 'AST_NODE'},
        'CONSISTS_OF':        {'SOURCE': 'TYPE',          'TARGET': 'MEMBER'},
        'AST':                {'SOURCE': 'AST_NODE',      'TARGET': 'AST_NODE'},
        'LITERAL_VALUE_LINK': {'SOURCE': 'LITERAL_VALUE', 'TARGET': 'AST_NODE'},
        'ARGUMENT':           {'SOURCE': 'AST_NODE',      'TARGET': 'AST_NODE'},
        'CALL':               {'SOURCE': 'AST_NODE',      'TARGET': 'AST_NODE'},
        'CFG':                {'SOURCE': 'AST_NODE',      'TARGET': 'AST_NODE'},
        'CDG':                {'SOURCE': 'AST_NODE',      'TARGET': 'AST_NODE'},
        'EVAL_TYPE':          {'SOURCE': 'AST_NODE',      'TARGET': 'TYPE'},
        'EVAL_MEMBER_TYPE':   {'SOURCE': 'MEMBER',        'TARGET': 'TYPE'},
        'REF':                {'SOURCE': 'AST_NODE',      'TARGET': 'AST_NODE'}
    }

    # We need to convert source and target columns of all edge sets
    for edge_set_name, val in edgeset_info.items():
        if edge_set_name not in graph_in_dfs.keys():
            continue

        source_nodeset = val['SOURCE']
        target_nodeset = val['TARGET']

        # Transform source and target node ID to their location index in their dataframe
        graph_in_dfs[edge_set_name]['source'] = graph_in_dfs[edge_set_name]['source'].apply(lambda id: graph_in_dfs[source_nodeset].index.get_loc(id))
        graph_in_dfs[edge_set_name]['target'] = graph_in_dfs[edge_set_name]['target'].apply(lambda id: graph_in_dfs[target_nodeset].index.get_loc(id))

    return graph_in_dfs


def reverse_edge_sets(graph_in_dfs, edge_sets):
    for edge_set in edge_sets:
        graph_in_dfs[edge_set].rename(columns={'source': 'target', 'target': 'source'}, inplace=True)

    return graph_in_dfs


def process_sample(directory):
    sample_id = directory.split('/')[-1]
    node_data, edge_data, valid_nodes = load_sample_data(directory)
    G = remove_invalid_nodes(sample_id, node_data, edge_data, valid_nodes)
    G = split_nodes(G)
    graph_in_dfs = transform_data_types(G)
    graph_in_dfs = convert_ids_to_tfgnn_format(graph_in_dfs)
    graph_in_dfs = reverse_edge_sets(graph_in_dfs, ['ARGUMENT', 'EVAL_TYPE', 'EVAL_MEMBER_TYPE', 'CONSISTS_OF', 'REF'])

    return graph_in_dfs


def save_sample(directory, graph_spec, output_file, splits):
    sample_id = directory.split('/')[-1]
    graph_in_dfs = process_sample(directory)

    # If one of CONSISTS_OF, MEMBER or EVAL_MEMBER_TYPE dfs is missing, others should be missing as well
    # If CONSISTS_OF edge set is missing add it empty
    if 'CONSISTS_OF' not in graph_in_dfs.keys():
        edgeset_CONSISTS_OF = tfgnn.EdgeSet.from_fields(
            sizes=tf.constant([0]),
            adjacency=tfgnn.Adjacency.from_indices(
                source=('MEMBER', tf.constant([], dtype=tf.int64)),
                target=('TYPE',   tf.constant([], dtype=tf.int64))
        ))
    else:
        edgeset_CONSISTS_OF = tfgnn.EdgeSet.from_fields(
            sizes=tf.constant([len(graph_in_dfs['CONSISTS_OF'])]),
            adjacency=tfgnn.Adjacency.from_indices(
                source=('MEMBER', tf.constant(graph_in_dfs['CONSISTS_OF']['source'].tolist())),
                target=('TYPE',   tf.constant(graph_in_dfs['CONSISTS_OF']['target'].tolist()))
        ))

    # If MEMBER node set is missing add it empty
    if 'MEMBER' not in graph_in_dfs.keys():
        nodeset_MEMBER = tfgnn.NodeSet.from_fields(
                sizes=tf.constant([0]),
                features={
                    'ORDER': tf.constant([], dtype=tf.float32)
        })
    else:
        nodeset_MEMBER = tfgnn.NodeSet.from_fields(
                sizes=tf.constant([len(graph_in_dfs['MEMBER'])]),
                features={
                    'ORDER': tf.constant(graph_in_dfs['MEMBER']['ORDER'].tolist())
        })

    # If EVAL_MEMBER_TYPE edge set is missing add it empty
    if 'EVAL_MEMBER_TYPE' not in graph_in_dfs.keys():
        edgeset_EVAL_MEMBER_TYPE = tfgnn.EdgeSet.from_fields(
                sizes=tf.constant([0]),
                adjacency=tfgnn.Adjacency.from_indices(
                    source=('TYPE',   tf.constant([], dtype=tf.int64)),
                    target=('MEMBER', tf.constant([], dtype=tf.int64))
        ))
    else:
        edgeset_EVAL_MEMBER_TYPE = tfgnn.EdgeSet.from_fields(
                sizes=tf.constant([len(graph_in_dfs['EVAL_MEMBER_TYPE'])]),
                adjacency=tfgnn.Adjacency.from_indices(
                    source=('TYPE',   tf.constant(graph_in_dfs['EVAL_MEMBER_TYPE']['source'].tolist())),
                    target=('MEMBER', tf.constant(graph_in_dfs['EVAL_MEMBER_TYPE']['target'].tolist()))
        ))

    # If CDG edge set is missing add it empty
    if 'CDG' not in graph_in_dfs.keys():
        edgeset_CDG = tfgnn.EdgeSet.from_fields(
            sizes=tf.constant([0]),
            adjacency=tfgnn.Adjacency.from_indices(
                source=('AST_NODE', tf.constant([], dtype=tf.int64)),
                target=('AST_NODE', tf.constant([], dtype=tf.int64))
        ))
    else:
        edgeset_CDG = tfgnn.EdgeSet.from_fields(
            sizes=tf.constant([len(graph_in_dfs['CDG'])]),
            adjacency=tfgnn.Adjacency.from_indices(
                source=('AST_NODE', tf.constant(graph_in_dfs['CDG']['source'].tolist())),
                target=('AST_NODE', tf.constant(graph_in_dfs['CDG']['target'].tolist()))
        ))

    # Transform graph to TFGNN GraphTensor
    graph = tfgnn.GraphTensor.from_pieces(
        context=tfgnn.Context.from_fields(features={'label': tf.constant([LABEL])}),
        node_sets={
            'METHOD_INFO': tfgnn.NodeSet.from_fields(
                sizes=tf.constant([len(graph_in_dfs['METHOD_INFO'])]),
                features={
                    'IS_EXTERNAL': tf.constant(graph_in_dfs['METHOD_INFO']['IS_EXTERNAL'].tolist()),
                    'HASH':        tf.constant(graph_in_dfs['METHOD_INFO']['HASH'].tolist()),
                    'OPERATOR':    tf.constant(graph_in_dfs['METHOD_INFO']['OPERATOR'].tolist())
            }),
            'TYPE': tfgnn.NodeSet.from_fields(
                sizes=tf.constant([len(graph_in_dfs['TYPE'])]),
                features={
                    'INT':  tf.constant(graph_in_dfs['TYPE']['INT'].tolist()),
                    'FP':   tf.constant(graph_in_dfs['TYPE']['FP'].tolist()),
                    'HASH': tf.constant(graph_in_dfs['TYPE']['HASH'].tolist()),
                    'PTR':  tf.constant(graph_in_dfs['TYPE']['PTR'].tolist()),
                    'LEN':  tf.constant(graph_in_dfs['TYPE']['LEN'].tolist())
            }),
            'MEMBER': nodeset_MEMBER,
            'AST_NODE': tfgnn.NodeSet.from_fields(
                sizes=tf.constant([len(graph_in_dfs['AST_NODE'])]),
                features={
                    'LABEL': tf.constant(graph_in_dfs['AST_NODE']['LABEL'].tolist()),
                    'ORDER': tf.constant(graph_in_dfs['AST_NODE']['ORDER'].tolist()),
            }),
            'LITERAL_VALUE': tfgnn.NodeSet.from_fields(
                sizes=tf.constant([len(graph_in_dfs['LITERAL_VALUE'])]),
                features={
                    'INT':              tf.constant(graph_in_dfs['LITERAL_VALUE']['INT'].tolist()),
                    'FP_MANTISSA':      tf.constant(graph_in_dfs['LITERAL_VALUE']['FP_MANTISSA'].tolist()),
                    'FP_EXPONENT':      tf.constant(graph_in_dfs['LITERAL_VALUE']['FP_EXPONENT'].tolist()),
                    'HASH':             tf.constant(graph_in_dfs['LITERAL_VALUE']['HASH'].tolist()),
                    'INVALID_POINTER':  tf.constant(graph_in_dfs['LITERAL_VALUE']['INVALID_POINTER'].tolist()),
                    'ZERO_INITIALIZED': tf.constant(graph_in_dfs['LITERAL_VALUE']['ZERO_INITIALIZED'].tolist()),
                    'UNDEF':            tf.constant(graph_in_dfs['LITERAL_VALUE']['UNDEF'].tolist())
            }),
        },
        edge_sets={
            'METHOD_INFO_LINK': tfgnn.EdgeSet.from_fields(
                sizes=tf.constant([len(graph_in_dfs['METHOD_INFO_LINK'])]),
                adjacency=tfgnn.Adjacency.from_indices(
                    source=('METHOD_INFO', tf.constant(graph_in_dfs['METHOD_INFO_LINK']['source'].tolist())),
                    target=('AST_NODE',    tf.constant(graph_in_dfs['METHOD_INFO_LINK']['target'].tolist()))
            )),
            'CONSISTS_OF': edgeset_CONSISTS_OF,
            'AST': tfgnn.EdgeSet.from_fields(
                sizes=tf.constant([len(graph_in_dfs['AST'])]),
                adjacency=tfgnn.Adjacency.from_indices(
                    source=('AST_NODE', tf.constant(graph_in_dfs['AST']['source'].tolist())),
                    target=('AST_NODE',    tf.constant(graph_in_dfs['AST']['target'].tolist()))
            )),
            'LITERAL_VALUE_LINK': tfgnn.EdgeSet.from_fields(
                sizes=tf.constant([len(graph_in_dfs['LITERAL_VALUE_LINK'])]),
                adjacency=tfgnn.Adjacency.from_indices(
                    source=('LITERAL_VALUE', tf.constant(graph_in_dfs['LITERAL_VALUE_LINK']['source'].tolist())),
                    target=('AST_NODE',    tf.constant(graph_in_dfs['LITERAL_VALUE_LINK']['target'].tolist()))
            )),
            'ARGUMENT': tfgnn.EdgeSet.from_fields(
                sizes=tf.constant([len(graph_in_dfs['ARGUMENT'])]),
                features={ 'ARGUMENT_INDEX': tf.constant(graph_in_dfs['ARGUMENT']['ARGUMENT_INDEX'].tolist()) },
                adjacency=tfgnn.Adjacency.from_indices(
                    source=('AST_NODE', tf.constant(graph_in_dfs['ARGUMENT']['source'].tolist())),
                    target=('AST_NODE',    tf.constant(graph_in_dfs['ARGUMENT']['target'].tolist()))
            )),
            'CALL': tfgnn.EdgeSet.from_fields(
                sizes=tf.constant([len(graph_in_dfs['CALL'])]),
                adjacency=tfgnn.Adjacency.from_indices(
                    source=('AST_NODE', tf.constant(graph_in_dfs['CALL']['source'].tolist())),
                    target=('AST_NODE',    tf.constant(graph_in_dfs['CALL']['target'].tolist()))
            )),
            'CFG': tfgnn.EdgeSet.from_fields(
                sizes=tf.constant([len(graph_in_dfs['CFG'])]),
                adjacency=tfgnn.Adjacency.from_indices(
                    source=('AST_NODE', tf.constant(graph_in_dfs['CFG']['source'].tolist())),
                    target=('AST_NODE',    tf.constant(graph_in_dfs['CFG']['target'].tolist()))
            )),
            'CDG': edgeset_CDG,
            'EVAL_TYPE': tfgnn.EdgeSet.from_fields(
                sizes=tf.constant([len(graph_in_dfs['EVAL_TYPE'])]),
                adjacency=tfgnn.Adjacency.from_indices(
                    source=('TYPE', tf.constant(graph_in_dfs['EVAL_TYPE']['source'].tolist())),
                    target=('AST_NODE',    tf.constant(graph_in_dfs['EVAL_TYPE']['target'].tolist()))
            )),
            'EVAL_MEMBER_TYPE': edgeset_EVAL_MEMBER_TYPE,
            'REF': tfgnn.EdgeSet.from_fields(
                sizes=tf.constant([len(graph_in_dfs['REF'])]),
                adjacency=tfgnn.Adjacency.from_indices(
                    source=('AST_NODE', tf.constant(graph_in_dfs['REF']['source'].tolist())),
                    target=('AST_NODE',    tf.constant(graph_in_dfs['REF']['target'].tolist()))
            ))
        }
    )

    # Save TFGNN GraphTensor to TFRecords file
    # with TFRecord_writing_lock:
    if sample_id in splits['train'][0]:
        example = tfgnn.write_example(graph)
        splits['train'][1].write(example.SerializeToString())
    elif sample_id in splits['val'][0]:
        example = tfgnn.write_example(graph)
        splits['val'][1].write(example.SerializeToString())
    elif sample_id in splits['test'][0]:
        example = tfgnn.write_example(graph)
        splits['test'][1].write(example.SerializeToString())
    else:
        print(f'ERROR: visualize_graph.py: Sample ID \'{sample_id}\' is not in train, val nor test data!', file=sys.stderr)
        exit(1)

    print(f'Note: visualize_graph.py: Graph Tensor \'{sample_id}\' successfully saved!')


if __name__ == '__main__':
    if sys.stdin.isatty():
        # If stdin is empty - run in single-threaded mode (only one graph)
        directory = sys.argv[1]
        sample_id = directory.split('/')[-1]
        node_data, edge_data, valid_nodes = load_sample_data(directory)
        G = remove_invalid_nodes(sample_id, node_data, edge_data, valid_nodes)
        G = split_nodes(G)
        plot_graph(G)
    else:
        # Load TFGNN Schema
        schema = tfgnn.read_schema(sys.argv[1])
        graph_spec = tfgnn.create_graph_spec_from_schema_pb(schema)

        output_file = sys.argv[2]
        project = sys.argv[3] # libtiff, ffmpeg, ...
        label = int(sys.argv[4]) # 0 or 1

        # Remove previous results, otherwise we only append the new ones
        for file in [output_file + '.train', output_file + '.val', output_file + '.test']:
            if os.path.exists(file):
                os.remove(file)

        # Set normalization_coefficients - we do it globally because it is used across many function
        norm_coeffs = norm_coeffs[project]

        # Same for label
        LABEL = label

        # Load splits and determine which data are for training, validation (development) and testing
        df = pd.read_csv(sys.argv[5], header=None)
        project_df = df[df[0].apply(lambda x: x.startswith(f'{project}_') and x.endswith(f'_{LABEL}'))] # Filter only current project and GT
        train_ids = set(project_df[project_df[1] == 'train'][0]) # Filter train data and keep only IDs
        val_ids   = set(project_df[project_df[1] == 'dev'][0])
        test_ids  = set(project_df[project_df[1] == 'test'][0])

        # Open tfrecords files now, because appending samples after closing the files is not supported
        with tf.io.TFRecordWriter(output_file + '.train') as writer_train, \
             tf.io.TFRecordWriter(output_file + '.val') as writer_val, \
             tf.io.TFRecordWriter(output_file + '.test') as writer_test:

            splits = {'train': (train_ids, writer_train), 'val': (val_ids, writer_val), 'test': (test_ids, writer_test)}

            # Stdin has data, process each line (parallel seems to be slower + some tf calls seem to have trouble in parallel)
            for line in sys.stdin:
                directory = line.strip()
                save_sample(directory, graph_spec, output_file, splits)

        # Stdin has data, process each line in parallel
        # with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        #     for line in sys.stdin:
        #         directory = line.strip()
        #         executor.submit(save_sample, directory, graph_spec, output_file)
