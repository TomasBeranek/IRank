# ******************************************************************************
#  File:            filter.py
#  Master's Thesis: Evaluating Reliability of Static Analysis Results
#                   Using Machine Learning
#  Author:          Beranek Tomas (xberan46)
#  Date:            14.5.2024
#  Up2date sources: https://github.com/TomasBeranek/but-masters-thesis
#  Description:     Script for filtering out unsupported error types and
#                   unnecessary attributes.
# ******************************************************************************

import pickle
import sys
import gzip
import os
import argparse
from tqdm import tqdm


# Colors for command line
OK = '\033[92m'
WARNING = '\033[93m'
ERROR = '\033[91m'
ENDC = '\033[0m'


SUPPORTED=[ 'INTEGER_OVERFLOW_L5',
            'BUFFER_OVERRUN_L5',
            'BUFFER_OVERRUN_L4',
            'INTEGER_OVERFLOW_U5',
            'BUFFER_OVERRUN_U5',
            'BUFFER_OVERRUN_L3',
            'NULLPTR_DEREFERENCE',
            'INTEGER_OVERFLOW_L2',
            'INFERBO_ALLOC_MAY_BE_BIG',
            'UNINITIALIZED_VALUE',
            'BUFFER_OVERRUN_L2',
            'NULL_DEREFERENCE',
            'BUFFER_OVERRUN_L1',
            'INTEGER_OVERFLOW_L1']


def init_parser():
    parser = argparse.ArgumentParser(description='Filter out non-supported bug types (defined in the script). It also removes information not needed for the pipeline.')

    parser.add_argument('-d', '--dir', required=True, type=str, help='directory with D2A files')
    parser.add_argument('-o', '--out-dir', required=True, type=str, help='output directory (if it does not exist, it will be created)')

    return parser


def filter_file(file, new_file):
    filtered_cnt = 0

    with gzip.open(file, mode = 'rb') as f_in, gzip.open(new_file, mode = 'wb') as f_out:
        while True:
            try:
                sample = pickle.load(f_in)
            except EOFError:
                break

            # Filter out unsupported bug types
            if sample['bug_type'] not in SUPPORTED:
                filtered_cnt += 1
                continue

            # Filter out information which is not needed by CPG construction pipeline
            sample.pop('label_source')
            sample.pop('bug_loc_trace_index')
            sample.pop('sample_type')
            sample['commit'].pop('changes')
            sample.pop('functions')
            sample.pop('zipped_bug_report')

            # Save sample to a new file
            pickle.dump(sample, f_out)

    return filtered_cnt


if __name__ == '__main__':
    parser = init_parser()
    args = parser.parse_args()

    # Get D2A files
    d2a_dir = os.path.abspath(args.dir)
    files = os.listdir(d2a_dir)

    # Take only samples with Infer's output
    files = [f'{d2a_dir}/{f}' for f in files if '_labeler_' in f]

    # Check if output directory exists
    if os.path.exists(args.out_dir):
        print(f'{WARNING}WARNING{ENDC}: {__file__}: directory "{args.out_dir}" already exists!', file=sys.stderr)
    else:
        os.makedirs(args.out_dir)

    filtered_cnt = 0

    # Filter each file
    for file in tqdm(files):
        original_file_name = file.split('/')[-1]
        out_dir = os.path.abspath(args.out_dir)
        new_file = out_dir + '/' + original_file_name
        filtered_cnt += filter_file(file, new_file)

    print(f'{OK}OK{ENDC}: {__file__}: Filtered out {filtered_cnt} samples!', file=sys.stderr)
