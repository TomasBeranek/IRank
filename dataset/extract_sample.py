import pickle
import sys
import json
import gzip
import os
import argparse
from collections import Counter


class BugTypeNotFound(Exception):
    pass


class NotEnoughSamples(Exception):
    pass


def init_parser():
    parser = argparse.ArgumentParser(description='Extract Nth sample of given bug type from D2A dataset.')

    parser.add_argument('-d', '--dir', required=True, type=str, help='directory with D2A files')
    parser.add_argument('-b', '--bug-type', required=True, type=str, help='Infer bug type')
    parser.add_argument('-n', required=True, type=int, help='Nth sample [1,2,...]')

    return parser

def load_sample(files, N, bug_type):
    cnt = 1

    for file in files:
        with gzip.open(file, mode = 'rb') as fp:
            while True:
                try:
                    item = pickle.load(fp)
                except EOFError:
                    break

                if item['bug_type'] == bug_type:
                    if cnt == N:
                        return item
                    else:
                        cnt += 1

    if cnt > 1:
        raise NotEnoughSamples()
    else:
        raise BugTypeNotFound()


if __name__ == '__main__':
    parser = init_parser()
    args = parser.parse_args()

    # Get D2A files
    d2a_dir = os.path.abspath(args.dir)
    files = os.listdir(d2a_dir)

    # Take only samples with Infer's output
    files = [f'{d2a_dir}/{f}' for f in files if '_labeler_' in f]

    # Extract data from .pickle.gz files
    try:
        sample = load_sample(files, args.n, args.bug_type)
    except NotEnoughSamples:
        print(f'Number of {args.bug_type} < {args.n}!', file=sys.stderr)
    except BugTypeNotFound:
        print(f'Bug type {args.bug_type} wasn\'t present!', file=sys.stderr)

    # Pretty print JSON
    json_formatted = json.dumps(sample, indent=4)
    print(json_formatted)
