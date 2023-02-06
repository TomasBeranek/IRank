#!/usr/bin/env python3.9

# The program takes as the only argument the name of the file with the Infer
# output in json format. The program processes and deletes the first report
# in the file.

# Meaning of error codes:
#   0 -- Ok.
#   1 -- Internal error.
#   5 -- Unsupported bug type.
#   6 -- The file is empty (not considered as an error).


import json
import sys
import os
import re


# Colors for command line
OK = '\033[92m'
WARNING = '\033[93m'
ERROR = '\033[91m'
ENDC = '\033[0m'


def extract_file_from_bug_trace(report):
    return report["bug_trace"][-1]["filename"]


def extract_line_from_bug_trace(report):
    return report["bug_trace"][-1]["line_number"]


def extract_variable_NULL_DEREFERENCE(report):
    # Get part of the qualifier which contains the variable name
    x = re.search(r'pointer `.*` last assigned', report["qualifier"])

    # If the given part wasn't found
    if not x:
        print(f'{ERROR}ERROR{ENDC}: slicing_criteria_extraction.py: unrecognized "NULL_DEREFERENCE" qualifier format: "{report["qualifier"]}"!', file=sys.stderr)
        exit(1)

    # The variable name is enclosed in `variable_name`
    x = x.group().split('`')

    if len(x) != 3:
        print(f'{ERROR}ERROR{ENDC}: slicing_criteria_extraction.py: unrecognized "NULL_DEREFERENCE" qualifier format: "{report["qualifier"]}"!', file=sys.stderr)
        exit(1)

    variable_name = x[1]

    return variable_name


def extract_variable_info(report):
    bug_type = report["bug_type"]

    if bug_type == "NULL_DEREFERENCE":
        info_file = extract_file_from_bug_trace(report)
        info_line = extract_line_from_bug_trace(report)
        info_variable = extract_variable_NULL_DEREFERENCE(report)
        return info_file, info_line, info_variable
    else:
        # Unsupported bug type
        print(f'{WARNING}WARNING{ENDC}: slicing_criteria_extraction.py: unsupported bug type "{bug_type}".', file=sys.stderr)
        exit(5)


def extract_slicing_info(report):
    # Entry function can be extracted in the same way for all bug types
    info_entry_function = report["procedure"]

    # Variable, line and file are information about the location of the error
    # Variable, line and file need to be extracted for each bug type individually
    info_file, info_line, info_variable = extract_variable_info(report)

    return info_entry_function, info_file, info_line, info_variable


if __name__ == "__main__":
    report_json_file = sys.argv[1]

    if not os.path.exists(report_json_file):
        # The Infer report file is missing
        print(f'{ERROR}ERROR{ENDC}: slicing_criteria_extraction.py: file "{report_json_file}" doesn\'t exist!', file=sys.stderr)
        exit(1)

    with open(report_json_file, "r") as f:
        try:
            reports = json.load(f)
        except json.decoder.JSONDecodeError:
            print(f'{ERROR}ERROR{ENDC}: slicing_criteria_extraction.py: file "{report_json_file}" in not in JSON format!', file=sys.stderr)
            exit(1)

    if not reports:
        # The report is empty, but in JSON format -- we processed all the reports
        exit(6)

    # Save the rest of the reports back to the original file
    with open(report_json_file, "w") as f:
        json.dump(reports[1:], f)

    slicing_info = extract_slicing_info(reports[0])
    print(f"{slicing_info[0]}\n{slicing_info[1]}\n{slicing_info[2]}\n{slicing_info[3]}")

    exit(0)
