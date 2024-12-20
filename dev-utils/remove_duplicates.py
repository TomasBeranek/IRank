# ******************************************************************************
#  File:            remove_duplicates.py
#  Master's Thesis: Evaluating Reliability of Static Analysis Results
#                   Using Machine Learning
#  Author:          Beranek Tomas (xberan46)
#  Date:            14.5.2024
#  Up2date sources: https://github.com/TomasBeranek/but-masters-thesis
#  Description:     Script for removal of LLVM bitcode for duplicit samples.
# ******************************************************************************

import argparse
import gzip
import os
import pickle
import json
import shutil
import copy
from tqdm import tqdm


# For faster work with symlinks
symlinks_cache = dict()


def init_parser():
    parser = argparse.ArgumentParser(description='Removes duplicit bitcode samples from D2A dataset.')

    parser.add_argument('--d2a-dir', metavar='DIR', required=True, type=str, help='directory with .pickle.gz D2A samples (dataset must NOT be filtered by filter.py!)')
    parser.add_argument('--bitcode-dir', metavar='DIR', required=True, type=str, help='directory with generated bitcode')

    return parser


def remove_url(x):
    if isinstance(x, dict):
        x.pop('url')
    elif isinstance(x, list):
        x = [remove_url(item) for item in x]
    else:
        # Shouldn't happen
        exit(1)
    return x


def remove_touched_by_commit(x):
    for item in x.values():
        item.pop('touched_by_commit')
    return x


def get_hash(sample):
    serialized_sample = json.dumps(sample, sort_keys=True)
    return hash(serialized_sample)


def construct_file_path(id):
    project = id.split('_')[0]
    label = id.split('_')[2]
    subdir = f'{project}_{label}'
    relative_path = f'{args.bitcode_dir}/{subdir}/{id}.bc'
    return relative_path


def calculate_hashes(dir):
    # Take only samples with Infer's output
    files = os.listdir(dir)
    files = [f'{dir}/{f}' for f in files if '_labeler_' in f]

    hashes = dict()
    for file in tqdm(files):
        with gzip.open(file, mode = 'rb') as fp:
            while True:
                try:
                    item = pickle.load(fp)
                except EOFError:
                    break

                # Load only samples for which there is a bitcode file
                bitcode_path = construct_file_path(item['id'])
                if not os.path.exists(bitcode_path):
                    continue

                # If bitcode is symlink, cache it
                if os.path.islink(bitcode_path):
                    symlink_target = os.readlink(bitcode_path)
                    if symlink_target in symlinks_cache:
                        symlinks_cache[symlink_target].append(bitcode_path)
                    else:
                        symlinks_cache[symlink_target] = [bitcode_path]

                # Keep only info which is needed to recognize functionally same
                # sample (e.g. ID will be DIFFERENT! - so we can't take that)
                sample = dict()
                sample['label'] = item['label']
                sample['bug_type'] = item['bug_type']
                sample['bug_info'] = remove_url(item['bug_info'])
                if item['adjusted_bug_loc'] != None:
                    sample['adjusted_bug_loc'] = remove_url(item['adjusted_bug_loc'])
                sample['bug_loc_trace_index'] = item['bug_loc_trace_index']
                sample['trace'] = remove_url(item['trace'])
                sample['functions'] = remove_touched_by_commit(item['functions'])

                sample_hash = get_hash(sample)

                # Safe hash to dict
                if sample_hash in hashes:
                    hashes[sample_hash].append(item['id'])
                else:
                    hashes[sample_hash] = [item['id']]

    return hashes


def load_samples(dir, ids):
    # Take only samples with Infer's output
    files = os.listdir(dir)
    files = [f'{dir}/{f}' for f in files if '_labeler_' in f]

    samples = dict()
    for file in tqdm(files):
        with gzip.open(file, mode = 'rb') as fp:
            while True:
                try:
                    item = pickle.load(fp)
                except EOFError:
                    break

                # Load only samples specified in set 'ids'
                if item['id'] not in ids:
                    continue

                # Keep only info which needed to recognize functionally same sample
                # (e.g. ID will be DIFFERENT! - so we can't that)
                sample = dict()
                sample['label'] = item['label']
                sample['bug_type'] = item['bug_type']
                sample['bug_info'] = remove_url(item['bug_info'])
                if item['adjusted_bug_loc'] != None:
                    sample['adjusted_bug_loc'] = remove_url(item['adjusted_bug_loc'])
                sample['bug_loc_trace_index'] = item['bug_loc_trace_index']
                sample['trace'] = remove_url(item['trace'])
                sample['functions'] = remove_touched_by_commit(item['functions'])
                samples[item['id']] = sample

    return samples


def all_items_are_same(x):
    return len(set(x)) == 1


def remove_symlinks_by_ids(ids):
    global symlinks_cache

    for id in ids:
        symlink_path = construct_file_path(id)
        target = os.readlink(symlink_path)
        os.remove(symlink_path)

        # Update symlinks cache
        symlinks_cache[target].remove(symlink_path)


def find_symlinks_to_file(file_path):
    symlink_target = file_path.split('/')[-1]
    if symlink_target in symlinks_cache:
        return copy.copy(symlinks_cache[symlink_target])
    else:
        return []


def remove_data_file(file_path):
    global symlinks_cache

    # Get list of symlinks pointing to file_path in format 'dir/subdir/id.bc'
    symlinks = find_symlinks_to_file(file_path)

    if len(symlinks) == 0:
        # If there is no symlink pointing to file_path, we can just remove it
        os.remove(file_path)
        return

    # One of the symlinks must now become new data file, since we are removing the original one
    new_data_file = symlinks.pop()
    shutil.move(file_path, new_data_file) # Overwrite chosen symlink with the data file
    new_data_file_link = new_data_file.split('/')[-1] # Take file name relative to symlink

    # Make all the remaining symlinks point to the new data file
    for symlink in symlinks:
        # Recreate the symlink with different target
        os.remove(symlink)
        os.symlink(new_data_file_link, symlink)

        # Remove old symlink from cache
        old_symlink_target = file_path.split('/')[-1]
        symlinks_cache[old_symlink_target].remove(symlink)

        # Add new symlink to cache
        if new_data_file_link in symlinks_cache:
            symlinks_cache[new_data_file_link].append(symlink)
        else:
            symlinks_cache[new_data_file_link] = [symlink]


def remove_data_files_by_ids(ids):
    for id in ids:
        file_path = construct_file_path(id)
        remove_data_file(file_path)


# Keeps only a single sample (bitcode file) from a list of IDs
def remove_bitcode(ids):
    # data files == not symlinks
    data_files_ids = [id for id in ids if not os.path.islink(construct_file_path(id))]

    if len(data_files_ids) > 1:
        # We have more data files - we will keep one and remove the others, all symlinks will be removed
        symlinks_ids = [id for id in ids if id not in data_files_ids]
        remove_symlinks_by_ids(symlinks_ids)

        # We keep one data file (doesn't matter which one) and remove the others
        data_files_ids.pop()
        remove_data_files_by_ids(data_files_ids)
    elif len(data_files_ids) == 1:
        # We have only one data file, others are symlinks - we can safely remove them
        ids.remove(data_files_ids[0])
        remove_symlinks_by_ids(ids)
    else:
        # All files are symlinks - we keep one (doesn't matter which one) and remove the others
        ids.pop()
        remove_symlinks_by_ids(ids)


if __name__ == '__main__':
    parser = init_parser()
    args = parser.parse_args()

    # Create dict of calculated hashes and sample IDs - for each hash (key) dict
    # contains a list of associated IDs
    hashes = calculate_hashes(args.d2a_dir)

    # Remove unique samples
    duplicit_hashes = {hash: ids for hash, ids in hashes.items() if len(ids) > 1}
    print(f'Number of duplicit sets: {len(duplicit_hashes)}')

    # Get set of ids from dict
    duplicit_ids = set()
    for hash, ids in duplicit_hashes.items():
        duplicit_ids |= set(ids)
    print(f'Number of duplicit samples: {len(duplicit_ids)}')

    # Load ONLY duplicit samples
    samples = load_samples(args.d2a_dir, duplicit_ids)

    # Remove bitcode of duplicit samples
    for _, ids in tqdm(duplicit_hashes.items()):
        serialized_samples = [json.dumps(samples[id], sort_keys=True) for id in ids]
        # Check if the samples are indeed duplicit (they can have the same hash but still be different)
        if not all_items_are_same(serialized_samples):
            # This should be extremely rare (NOTE: this won't happen in D2A)
            print('remove_duplicates.py: Found samples which have the same hash, but are different!')
            exit(1)

        remove_bitcode(ids)
