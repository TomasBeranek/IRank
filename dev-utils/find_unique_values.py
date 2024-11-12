# ******************************************************************************
#  File:            find_unique_values.py
#  Master's Thesis: Evaluating Reliability of Static Analysis Results
#                   Using Machine Learning
#  Author:          Beranek Tomas (xberan46)
#  Date:            14.5.2024
#  Up2date sources: https://github.com/TomasBeranek/but-masters-thesis
#  Description:     Script for finding unique attribute values for node sets.
# ******************************************************************************

# Usage: python3.8 find_unique_values.py nodes_BLOCK
#        python3.8 find_unique_values.py nodes_METHOD_PARAMETER_IN

import os
import sys
import csv
import pickle
import re
from collections import defaultdict


TYPES_FILE = 'unique_type_names.pickle'
METHODS_FILE = 'unique_method_names.pickle'
VALUES_FILE = 'unique_literal_values.pickle'


def find_unique_values(directory, partial_filename):
    unique_values = defaultdict(set)
    samples = 0
    headers = None
    headers_already_loaded = False

    # Iterate over samples
    for subdir, _, files in os.walk(directory):
        samples += 1
        data_file = f'{partial_filename}_data.csv'
        header_file = f'{partial_filename}_header.csv'

        if data_file in files and header_file in files:
            if not headers_already_loaded:
                # Read header
                with open(os.path.join(subdir, header_file), mode='r') as file:
                    reader = csv.reader(file)
                    headers = next(reader)

                headers_already_loaded = True

            # Read data and collect unique values
            with open(os.path.join(subdir, data_file), mode='r') as file:
                reader = csv.DictReader(file, fieldnames=headers)
                for row in reader:
                    for header in headers:
                        if header == 'FULL_NAME:string':
                            if not unique_values[header]:
                                unique_values[header] = defaultdict(int)

                            full_name = row[header]
                            unique_values[header][full_name] += 1
                            continue

                        if len(unique_values[header]) < 30:
                            unique_values[header].add(row[header])

    return unique_values, samples


def save_names(names, file):
    sorted_types = sorted(names.items(), key=lambda item: item[1], reverse=True)

    with open(file, 'wb') as f:
        pickle.dump(sorted_types, f)

    for key, value in sorted_types:
        print(f'{key}: {value}')


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


def print_type_names(type_names):
    default_types = ['void',
                     'half',
                     'bfloat',
                     'float',
                     'double',
                     'fp128',
                     'x86_fp80',
                     'ppc_fp128',
                     'x86_amx',
                     'x86_mmx']

    array_cnt = normal_types_cnt = signatures_cnt = pointers_cnt = structs_cnt = 0
    max_LEN = max_PTR = 0

    for index, (name, cnt) in enumerate(type_names):
        if index >= 0:
            _, PTR = count_pointers(name)
            _, LEN = get_array_len(name)

            max_PTR = max(max_PTR, PTR)
            max_LEN = max(max_LEN, LEN)

            type_of_type = None
            if name[-1] == '*':
                pointers_cnt += 1
                type_of_type = 'pointer'
            elif name[0] == '{' and name[-1] == '}':
                structs_cnt += 1
                type_of_type = 'struct'
            elif name[0] == '[' and name[-1] == ']':
                array_cnt += 1
                type_of_type = 'array'
            elif '(' in name:
                signatures_cnt += 1
                type_of_type = 'signature'
            else:
                normal_types_cnt += 1
                type_of_type = 'normal'

                if re.match(r'^i\d+$', name) or name in default_types:
                    print(name)
                    continue
                else:
                    continue

    print(f"Pointers: {pointers_cnt}")
    print(f"Structs: {structs_cnt}")
    print(f"Arrays: {array_cnt}")
    print(f"Normal: {normal_types_cnt}")
    print(f"Signatures: {signatures_cnt}")
    print(f"All: {len(type_names)}")
    print(f"Max PTR: {max_PTR}")
    print(f"Max LEN: {max_LEN}")


def print_method_names(method_names):
    operator_cnt = 0
    for name, cnt in method_names:
        if '<operator>' in name:
            print(f"'{name}': {operator_cnt},")
            operator_cnt += 1


def save_literal_values(directory, partial_filename, save_to_file):
    unique_values = set()
    samples = 0
    headers = None
    headers_already_loaded = False

    # Iterate over samples
    for subdir, _, files in os.walk(directory):
        samples += 1
        data_file = f'{partial_filename}_data.csv'
        header_file = f'{partial_filename}_header.csv'

        if data_file in files and header_file in files:
            if not headers_already_loaded:
                # Read header
                with open(os.path.join(subdir, header_file), mode='r') as file:
                    reader = csv.reader(file)
                    headers = next(reader)

                headers_already_loaded = True

            # Read data and collect unique pairs of values and its types
            with open(os.path.join(subdir, data_file), mode='r') as file:
                reader = csv.DictReader(file, fieldnames=headers)
                for row in reader:
                    value_type_pair = (row['CODE:string'], row['TYPE_FULL_NAME:string'])
                    unique_values.add(value_type_pair)

    print(f'Samples: {samples}')

    # Save it
    with open(save_to_file, 'wb') as f:
        pickle.dump(unique_values, f)


def print_values_type_pairs(value_type_pairs):
    cnt = -1

    for value, type in value_type_pairs:
        cnt += 1

        # Missing type
        if not type:
            continue

        # Pointer
        if type.endswith('*'):
            if value in ['nullptr', 'undef']:
                continue

        # Array
        if type[0] == '[' and type[-1] == ']':
            continue

        # Struct
        if type[0] == '{' and type[-1] == '}':
            continue

        # Integer type iN
        if re.match(r'^i\d+$', type):
            continue

        # Floating point
        if type in ['half', 'float', 'double', 'fp128']:
            continue

        # Custom data types
        if type in ['ngx_module_s', 'module_struct', 'ngx_path_init_t', 'ngx_str_t', 'AVClass', 'module_struct_1', 'ngx_module_s_1', 'apr_bucket_type_t', 'ModeTab', 'AVClass_1', 'BlockNode', 'anon_2', 'ap_regex_t', 'AVFrac']:
            continue


        print('\n'*10)
        print(f'"{value}"')
        print(f'{cnt}/{len(value_type_pairs)}')
        print(f'TYPE: "{type}"')
        input()


if __name__ == '__main__':
    if len(sys.argv) < 2 or len(sys.argv) > 2:
        print('Usage: python find_unique_values.py <partial_filename>|--save-type-names|--print-type-names')
        sys.exit(1)

    directory = '../dataset/graph-d2a/'  # Script needs to be in dev-utils/ or change accordingly

    # We need to increase possible size of CSV cell - for array literals, array types etc.
    csv.field_size_limit(sys.maxsize)

    if sys.argv[1] == '--save-type-names':
        partial_filename = 'nodes_TYPE'
    elif sys.argv[1] == '--save-method-names':
        partial_filename = 'nodes_METHOD'
    elif sys.argv[1] == '--save-literal-values':
        partial_filename = 'nodes_LITERAL'
        save_literal_values(directory, partial_filename, VALUES_FILE)
        exit()
    elif sys.argv[1] == '--print-method-names':
        # Load saved type names
        with open(METHODS_FILE, 'rb') as f:
            method_names = pickle.load(f)

        print_method_names(method_names)
        exit()
    elif sys.argv[1] == '--print-type-names':
        # Load saved method names
        with open(TYPES_FILE, 'rb') as f:
            type_names = pickle.load(f)

        print_type_names(type_names)
        exit()
    elif sys.argv[1] == '--print-literal-values':
        # Load saved values type pairs
        with open(VALUES_FILE, 'rb') as f:
            value_type_pairs = pickle.load(f)

        print_values_type_pairs(value_type_pairs)
        exit()
    else:
        partial_filename = sys.argv[1]

    unique_values, samples = find_unique_values(directory, partial_filename)

    if sys.argv[1] == '--save-type-names':
        save_names(unique_values['FULL_NAME:string'], TYPES_FILE)
        exit()
    elif sys.argv[1] == '--save-method-names':
        save_names(unique_values['FULL_NAME:string'], METHODS_FILE)
        exit()

    # Print unique values for each column
    for header, values in unique_values.items():
        if header == ':ID':
            numeric_values = [int(value) for value in values]
            print(f'{header}: {min(numeric_values)} --> {max(numeric_values)}')
        else:
            print(f'{header}: {values}')
    print(f'Samples: {samples}')
