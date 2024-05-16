# ******************************************************************************
#  File:            speed_test.py
#  Master's Thesis: Evaluating Reliability of Static Analysis Results
#                   Using Machine Learning
#  Author:          Beranek Tomas (xberan46)
#  Date:            14.5.2024
#  Up2date sources: https://github.com/TomasBeranek/but-masters-thesis
#  Description:     Part of the speed-test experiment.
# ******************************************************************************

import time
import os
import sys
from tqdm import tqdm


def find_bitcode_files():
    matches = set()
    for root, _, files in os.walk('.'):
        for file in files:
            if file.endswith('.bc'):
                matches.add(os.path.join(root, file))
    return matches

def find_bitcode_files_non_recursive():
    matches = set()
    for file in os.listdir():
        if file.endswith('.bc'):
            matches.add(file)
    return matches


os.chdir(sys.argv[1])

start = time.time()
for i in tqdm(range(10000)):
    find_bitcode_files()
end = time.time()

print(f"Time with recursive search: {(end-start) * 10**3} ms")

start = time.time()
for i in tqdm(range(10000)):
    find_bitcode_files_non_recursive()
end = time.time()

print(f"Time without recursive search: {(end-start) * 10**3} ms")
