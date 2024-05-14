# ******************************************************************************
#  File:            stats.py
#  Master's Thesis: Evaluating Reliability of Static Analysis Results
#                   Using Machine Learning
#  Author:          Beranek Tomas (xberan46)
#  Date:            14.5.2024
#  Up2date sources: https://github.com/TomasBeranek/but-masters-thesis
#  Description:     Script for printing statistics about D2A files.
# ******************************************************************************

import pickle
import gzip
import os
import argparse
from tqdm import tqdm
from collections import Counter


negative_samples = 0
positive_samples = 0
total_samples = 0


def init_parser():
    parser = argparse.ArgumentParser(description='Print statistics about D2A files in given directory.')

    parser.add_argument('-d', '--dir', required=True, type=str, help='directory with D2A files')

    return parser


def load_stats(file):
    global total_samples, negative_samples, positive_samples

    bug_types = {}

    with gzip.open(file, mode = 'rb') as fp:
        while True:
            try:
                item = pickle.load(fp)
            except EOFError:
                break

            total_samples += 1
            if 'labeler_1.pickle.gz' in file:
                positive_samples += 1
            else:
                negative_samples += 1

            bug_name = item['bug_type']
            if bug_name in bug_types.keys():
                bug_types[bug_name] += 1
            else:
                bug_types[bug_name] = 1

    # Sort bug types according to their counts
    bug_names_sorted = sorted(bug_types, key=bug_types.get, reverse=True)

    # Print stats for current file
    file_name = file.split('/')[-1]
    print(f'File: {file_name}')
    for bug_name in bug_names_sorted:
        print(f"{bug_name}: {bug_types[bug_name]}")
    print('')

    return bug_types


if __name__ == '__main__':
    parser = init_parser()
    args = parser.parse_args()

    # Get D2A files
    d2a_dir = os.path.abspath(args.dir)
    files = os.listdir(d2a_dir)

    # Take only positive samples files, since we are only interested in project names
    files = [f for f in files if 'labeler_1.pickle.gz' in f]

    # Extract only file name
    project_names = [f.split('_')[0] for f in files]

    # Create dictionary of project names and labels
    stats = {}
    for name in project_names:
        stats[name] = { '0': {}, '1': {} }

    # Extract data from .pickle.gz files
    for name in tqdm(project_names):
        file_0 = f'{d2a_dir}/{name}_labeler_0.pickle.gz'
        file_1 = f'{d2a_dir}/{name}_labeler_1.pickle.gz'
        stats[name]['0'] = load_stats(file_0)
        stats[name]['1'] = load_stats(file_1)

    # Calculate '0' labels stats
    negative_dicts = [dict['0'] for dict in stats.values()]
    counters = [Counter(dict) for dict in negative_dicts]
    negative_stats = sum(counters, Counter())

    # Calculate '1' labels stats
    positive_dicts = [dict['1'] for dict in stats.values()]
    counters = [Counter(dict) for dict in positive_dicts]
    positive_stats = sum(counters, Counter())

    # Combine both labels
    all_stats = negative_stats + positive_stats

    print('#################################################################\n')

    # Print '0' labels stats
    bug_names_sorted = sorted(negative_stats, key=negative_stats.get, reverse=True)
    print(f'All files: \'0\' label')
    for bug_name in bug_names_sorted:
        print(f"{bug_name}: {negative_stats[bug_name]}")

    # Print '1' labels stats
    bug_names_sorted = sorted(positive_stats, key=positive_stats.get, reverse=True)
    print(f'\nAll files: \'1\' label')
    for bug_name in bug_names_sorted:
        print(f"{bug_name}: {positive_stats[bug_name]}")

    # Print both labels stats
    bug_names_sorted = sorted(all_stats, key=all_stats.get, reverse=True)
    print(f'\nAll files: combined labels')
    for bug_name in bug_names_sorted:
        print(f"{bug_name}: {all_stats[bug_name]}")

    print(f'\nNegative samples: {negative_samples}')
    print(f'\nPositive samples: {positive_samples}')
    print(f'\nTotal samples:    {total_samples}')
