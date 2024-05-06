import os
import sys
import csv
from collections import defaultdict
import pandas as pd
from pprint import pprint
from tqdm import tqdm


AST_NODES = ['nodes_UNKNOWN_header.csv',
             'nodes_METHOD_header.csv',
             'nodes_METHOD_PARAMETER_IN_header.csv',
             'nodes_METHOD_RETURN_header.csv',
             'nodes_MEMBER_header.csv',
             'nodes_BLOCK_header.csv',
             'nodes_CALL_header.csv',
             'nodes_FIELD_IDENTIFIER_header.csv',
             'nodes_IDENTIFIER_header.csv',
             'nodes_LITERAL_header.csv',
             'nodes_LOCAL_header.csv',
             'nodes_METHOD_REF_header.csv',
             'nodes_RETURN_header.csv']

def load_sample_header(path):
    with open(path, mode='r') as file:
        reader = csv.reader(file)
        header = next(reader)
    return header


def count_pointers(full_name):
    stripped_name = full_name.rstrip('*')
    pointers_cnt = len(full_name) - len(stripped_name)
    return stripped_name, pointers_cnt


def get_array_len(type_name):
    # Check if type is array
    if type_name[0] == '[' and type_name[-1] == ']':
        string_parts = type_name.split(' x ', 1)
        len_string = string_parts[0][1:] # Remove initial '['
        return int(len_string)
    else:
        return 0


def extract_LEN_and_PTR(full_name_list):
    MAX_LEN = MAX_PTR = 0

    for full_name in full_name_list:
        name_without_pointers, PTR = count_pointers(full_name)
        LEN = get_array_len(name_without_pointers)
        MAX_LEN = max(MAX_LEN, LEN)
        MAX_PTR = max(MAX_PTR, PTR)

    return MAX_LEN, MAX_PTR


def extract_operators(full_name_list):
    OPERATORS = set()

    for full_name in full_name_list:
        if full_name.startswith('<operator>.'):
            OPERATORS.add(full_name)

    return OPERATORS


def find_normalization_coefficients(directory, num_samples, normalization_coefficients, train_ids):
    first_dir = True

    # Iterate over samples
    for subdir, _, files in tqdm(os.walk(directory), total=num_samples):
        # Skipt first dir - its the parent dir itself
        if first_dir:
            first_dir = False
            continue

        # Skip val (dev) and test samples
        if subdir.split('/')[-1] not in train_ids:
            continue

        df_argument_index = pd.DataFrame(columns=[':ID', 'ARGUMENT_INDEX:int'])

        for header_file in files:
            if not header_file.endswith('_header.csv'):
                continue

            if (header_file not in AST_NODES) and (header_file != 'nodes_TYPE_header.csv'):
                continue

            data_file = header_file[:-10] + 'data.csv'
            header = load_sample_header(os.path.join(subdir, header_file))

            # Read data and update normalization coefficients
            df = pd.read_csv(os.path.join(subdir, data_file), header=None)
            df.columns = header

            # TYPE: LEN (full_name)
            # TYPE: PTR (full_name)
            if header_file == 'nodes_TYPE_header.csv':
                LEN, PTR = extract_LEN_and_PTR(df['FULL_NAME:string'].tolist())
                normalization_coefficients['LEN'] = max(normalization_coefficients['LEN'], LEN)
                normalization_coefficients['PTR'] = max(normalization_coefficients['PTR'], PTR)
                continue

            # Only AST files are left
            # AST: ORDER
            if header_file == 'nodes_MEMBER_header.csv':
                MEMBER_ORDER = df['ORDER:int'].max()
                normalization_coefficients['MEMBER_ORDER'] = max(normalization_coefficients['MEMBER_ORDER'], MEMBER_ORDER)
            else:
                ORDER = df['ORDER:int'].max()
                normalization_coefficients['ORDER'] = max(normalization_coefficients['ORDER'], ORDER)

            # METHOD: operators (full_name)
            if header_file == 'nodes_METHOD_header.csv':
                OPERATORS = extract_operators(df['FULL_NAME:string'].tolist())
                normalization_coefficients['OPERATORS'].update(OPERATORS)

            # Append current df if it has ARGUMENT_INDEX
            if 'ARGUMENT_INDEX:int' in df.columns:
                df = df[[':ID', 'ARGUMENT_INDEX:int']]
                df_argument_index = pd.concat([df_argument_index, df], ignore_index=True)

        # After all AST nodes are loaded to df we extract max ARGUMENT_INDEX from the relevant ones
        # AST: ARGUMENT_INDEX
        df = pd.read_csv(os.path.join(subdir, 'edges_ARGUMENT_data.csv'), header=None)
        relevant_node_ids = df[1].tolist()
        df_filtered = df_argument_index[df_argument_index[':ID'].isin(relevant_node_ids)]
        ARGUMENT_INDEX = df_filtered['ARGUMENT_INDEX:int'].max()
        normalization_coefficients['ARGUMENT_INDEX'] = max(normalization_coefficients['ARGUMENT_INDEX'], ARGUMENT_INDEX)

    return normalization_coefficients


def find_context_normalization_coefficients(train_ids, slicing_info_TP_file, slicing_info_FP_file, normalization_coefficients):
    columns = ['status', 'bug_id', 'entry', 'file', 'fun', 'line', 'variable']
    df_1 = pd.read_csv(slicing_info_TP_file, header=None, names=columns)
    df_0 = pd.read_csv(slicing_info_FP_file, header=None, names=columns)
    df = df_1.append(df_0, ignore_index=True)

    # Keep only train samples
    df = df[df['bug_id'].isin(train_ids)]

    normalization_coefficients['LINE'] = df['line'].max()

    return normalization_coefficients


if __name__ == '__main__':
    path_to_project_FP = sys.argv[1] # ../../D2A-CPG/libtiff_0
    path_to_project_TP = sys.argv[2] # ../../D2A-CPG/libtiff_1
    project = sys.argv[3] # libtiff, openssl, ...
    splits_path = sys.argv[4] # ../../dataset/d2a/splits.csv
    slicing_info_FP_file = sys.argv[5] # ../../dataset/slicing-info/libtiff_labeler_0.csv
    slicing_info_TP_file = sys.argv[6] # ../../dataset/slicing-info/libtiff_labeler_1.csv

    # We need to increase possible size of CSV cell - for array literals, array types etc.
    csv.field_size_limit(sys.maxsize)

    # For progress bar
    num_samples = {'libtiff_1': 459, 'libtiff_0': 9276}

    # Load splits and determine which data are for training (we don't want to extract coefficients from val nor test data)
    df = pd.read_csv(splits_path, header=None)
    project_df = df[df[0].apply(lambda x: x.startswith(project))] # Filter only current project
    train_df = project_df[project_df[1] == 'train'] # Remove val (dev) and test data
    train_ids = set(train_df[0]) # Keep only IDs

    # Set default values (can be manually set if e.g. ffmpeg_0 is too big to be computed in one run)
    normalization_coefficients = defaultdict(int)
    normalization_coefficients['OPERATORS'] = set()

    # FP
    normalization_coefficients = find_normalization_coefficients(path_to_project_FP, num_samples['libtiff_0'], normalization_coefficients, train_ids)

    # TP
    normalization_coefficients = find_normalization_coefficients(path_to_project_TP, num_samples['libtiff_1'], normalization_coefficients, train_ids)

    # Load D2A and extract context info - only LINE for now
    normalization_coefficients = find_context_normalization_coefficients(train_ids, slicing_info_TP_file, slicing_info_FP_file, normalization_coefficients)

    # Bug types are defined from the beginning, since we filtered D2A for only those types which have enough samples
    normalization_coefficients['BUG_TYPES'] = ['NULL_DEREFERENCE',
                                               'UNINITIALIZED_VALUE',
                                               'INFERBO_ALLOC_MAY_BE_BIG',
                                               'NULLPTR_DEREFERENCE',
                                               'BUFFER_OVERRUN_L1',
                                               'BUFFER_OVERRUN_L5',
                                               'BUFFER_OVERRUN_L4',
                                               'BUFFER_OVERRUN_U5',
                                               'BUFFER_OVERRUN_L3',
                                               'BUFFER_OVERRUN_L2'
                                               'INTEGER_OVERFLOW_L1',
                                               'INTEGER_OVERFLOW_L5',
                                               'INTEGER_OVERFLOW_U5',
                                               'INTEGER_OVERFLOW_L2']

    pprint(normalization_coefficients)
