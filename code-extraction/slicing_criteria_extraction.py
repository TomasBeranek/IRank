#!/usr/bin/env python3.9

# The program takes as the only argument the name of the file with the Infer
# output in json format. The program processes and deletes the first report
# in the file.

# Meaning of error codes:
#   0 -- Ok. The file might be empty, but in JSON format (not considered as an error).
#   1 -- Internal error.
#   7 -- The file is not in JSON format or is missing completely.


import json
import sys
import os
import re
import argparse
import gzip
import pickle


# Colors for command line
OK = '\033[92m'
WARNING = '\033[93m'
ERROR = '\033[91m'
ENDC = '\033[0m'


def init_parser():
    parser = argparse.ArgumentParser(description='Extract slicing criteria from Infer\'s output for LLVM slicer (DG) tool. The output is written to stdout in CSV format "status,bug_id,slicing_criteria". If a bug has an unsupported type, then the bug will be skipped (nothing is written to stdout).')

    parser.add_argument('file', metavar='FILE', type=str, help='file with reported bugs')
    parser.add_argument('--d2a', required=False, action='store_true', help='input file is in D2A format')
    parser.add_argument('--debug', required=False, metavar='N', type=int, help='print up to N reports in JSON format of each found bug format type (each bug type can be further subdivided by its format)')

    return parser


def location_from_bug_trace(report):
    file = report['bug_trace'][-1]['filename']
    fun = report['bug_trace'][-1]['func_name']
    line = report['bug_trace'][-1]['line_number']
    return file, fun, line


def is_header(file):
    return file.endswith('.h')

# Debug option - a number of reports to print (of each type)
DEBUG = 0
DEBUG_TYPES = {}

def init_debug():
    global DEBUG_TYPES
    bug_format_types = ['NULLPTR_DEREFERENCE',
                        'INTEGER_OVERFLOW_TYPE_1',
                        'INTEGER_OVERFLOW_TYPE_2',
                        'INFERBO_ALLOC_MAY_BE_BIG',
                        'UNINITIALIZED_VALUE_TYPE_1',
                        'UNINITIALIZED_VALUE_TYPE_2',
                        'BUFFER_OVERRUN_TYPE_1',
                        'BUFFER_OVERRUN_TYPE_2',
                        'NULL_DEREFERENCE_TYPE_1',
                        'NULL_DEREFERENCE_TYPE_2']

    # Initialize all format types to 0
    DEBUG_TYPES = dict(zip(bug_format_types, [0]*len(bug_format_types)))


def print_debug_report(report, bug_format_type):
    global DEBUG_TYPES

    if DEBUG and DEBUG_TYPES[bug_format_type] < DEBUG:
        # Pretty print JSON
        json_formatted = json.dumps(report, indent=4)
        print(f'FORMAT TYPE: {bug_format_type}:', file=sys.stderr)
        print(json_formatted, file=sys.stderr)
        DEBUG_TYPES[bug_format_type] += 1


# Numbers of true positives from D2A dataset > 200
#   INTEGER_OVERFLOW_L5: 10382
#   BUFFER_OVERRUN_L5: 3115
#   BUFFER_OVERRUN_L4: 1233
#   INTEGER_OVERFLOW_U5: 728
#   BUFFER_OVERRUN_U5: 640
#   BUFFER_OVERRUN_L3: 460
#   NULLPTR_DEREFERENCE: 422
#   INTEGER_OVERFLOW_L2: 369
#   INFERBO_ALLOC_MAY_BE_BIG: 360
#   UNINITIALIZED_VALUE: 240
#   BUFFER_OVERRUN_L2: 222
#   NULL_DEREFERENCE: 201

# Added because their are of similar type
# BUFFER_OVERRUN_L1: 28
# INTEGER_OVERFLOW_L1: 22

# IMPORTANT: When it is not possible to get the bug location from the basic
# information, there is an entry 'adjust_bug_loc' in D2A that points to the bug
# location (if it can be found e.g. in the bug trace). The correct behaviour
# has been verified for all supported bugs.


# NULLPTR_DEREFERENCE: three types were found:
#   1) "call to `put_bits()` eventually accesses memory that is the null pointer on line 543 indirectly during the call to `init_put_bits()`."
#       e.g. python3 extract_sample.py -d d2a/ -b NULLPTR_DEREFERENCE --bug-info-only -n 2
#           ID: ffmpeg_8e48b53d696b53cef2814548e4d0693387e875ea_1
#   2) "accessing memory that is the null pointer on line 3191 indirectly during the call to `av_malloc()`."
#       e.g. python3 extract_sample.py -d d2a/ -b NULLPTR_DEREFERENCE --bug-info-only -n 3
#           ID: ffmpeg_6a30264054cc320fe610c072c71d008f7e3c3efb_1
#   3) "accessing memory that is the null pointer on line 315."
#       e.g. python3 extract_sample.py -d d2a/ -b NULLPTR_DEREFERENCE --bug-info-only -n 30
#           ID: ffmpeg_9c908a4c99e0498dd26bd1de84ff085ac8e73e4a_1
#
# It is not possible to get the variable name from the descriptions, so it is
# possible to slice only by lines. The line can be obtained from the basic
# information in case 2) and 3). However, in case 1), only the line of the
# function call in which the NULL dereference occurs is given in the basic
# information (similar case to NULL_DEREFERENCE). However, in all three cases
# it is possible to get the NULL dereference line from the last step of the bug
# trace (description "invalid access occurs here").
def extract_NULLPTR_DEREFERENCE(report, id):
    if 'invalid access occurs here' != report['bug_trace'][-1]['description']:
        # Unexpected format of bug report --> try to slice by basic file/line
        report_msg = report['bug_trace'][-1]['description']
        print(f'{WARNING}WARNING{ENDC}: slicing_criteria_extraction.py: unrecognized "NULLPTR_DEREFERENCE" bug trace qualifier format: "{report_msg}" of bug #{id}! Slicing only by line.', file=sys.stderr)
        return 1, report['file'], report['procedure'], report['line'], ''
    else:
        file, fun, line = location_from_bug_trace(report)
        print_debug_report(report, 'NULLPTR_DEREFERENCE')
        return 0, file, fun, line, ''


INTEGER_OVERFLOW = [ 'INTEGER_OVERFLOW_L1',
                     'INTEGER_OVERFLOW_L5',
                     'INTEGER_OVERFLOW_U5',
                     'INTEGER_OVERFLOW_L2' ]
# INTEGER_OVERFLOW: two types were found:
#   1) "([0, 8] - [0, 8]):unsigned32."
#       e.g. python3 extract_sample.py -d d2a/ -b INTEGER_OVERFLOW_L2 --bug-info-only -n 1
#           ID: ffmpeg_1542087b54ddf682fb6177f999c6f9f79bd5613f_1
#   2) "([0, 1] - 1):unsigned32 by call to `avfilter_unref_buffer`."
#       e.g. python3 extract_sample.py -d d2a/ -b INTEGER_OVERFLOW_L2 --bug-info-only -n 3
#           ID: ffmpeg_ca5973f0bfac4560342605f8a52efc88b4f4dbd3_1
#
# This case is almost identical to BUFFER_OVERRUN_L2 -- there is no information
# about the exact position of the operation. It is therefore possible to slice
# only by line. In case 1) the line is extracted from the base information and
# in case 2) the line is extracted from the last step in the bug trace (e.g.
# description "Binary operation: ([0, 1] - 1):unsigned32 by call to `avfilter_unref_buffer` ").
def extract_INTEGER_OVERFLOW(report, id):
    file = report['file']
    fun = report['procedure']
    line = report['line']
    bug_type = report['bug_type']

    if 'by call to' in report['qualifier']:
        # Type 2) --> extract location from last step of bug trace

        if 'Binary operation:' not in report['bug_trace'][-1]['description']:
            # Unexpected format of bug report --> try to slice by basic file/line
            report_msg = report['bug_trace'][-1]['description']
            print(f'{WARNING}WARNING{ENDC}: slicing_criteria_extraction.py: unrecognized "{bug_type}" bug trace qualifier format: "{report_msg}" of bug #{id}! Slicing only by line.', file=sys.stderr)
            return 1, file, fun, line, ''

        # Expected format of bug report --> slice by the last step of bug trace
        file, fun, line = location_from_bug_trace(report)
        print_debug_report(report, 'INTEGER_OVERFLOW_TYPE_2')
        return 0, file, fun, line, ''
    else:
        # Type 1) --> extract location from basic info
        print_debug_report(report, 'INTEGER_OVERFLOW_TYPE_1')
        return 0, file, fun, line, ''


# INFERBO_ALLOC_MAY_BE_BIG: only one type found:
#   1) "Length: [0, 2147483631] by call to `av_dup_packet`."
#       e.g. python3 extract_sample.py -d d2a/ -b INFERBO_ALLOC_MAY_BE_BIG --bug-info-only -n 1
#           ID: ffmpeg_c36d9fb10c31c6835d01232fddff6932a3ce347f_1
#
# For this type of error, as with NULL_DEREFERENCE, only the location of the
# function call in which the misallocation occurs is specified in the basic
# information. However, in all cases viewed, the last step of the bug trace (with
# description e.g. "Allocation: Length: [-oo, 2147483641] by call to `av_frame_ref`")
# can be used to determine the exact location of the allocation. However, it is
# no longer possible to find out what variable or combination of variables may
# take on non-valid values. If we restrict ourselves to e.g. only realloc(ptr, size),
# then it would be possible to slice according to the 2nd parameter, which is
# always size (Infer is looking for an error here). However, the allocation
# function can be of any shape in general, and we don't know which parameter is
# the size of the allocation. Therefore, it will only be sliced by that line --
# plus the allocation function is generally not large, so it doesn't add as much
# unnecessary information. The entry function is taken from the basic information
# as with other errors.
def extract_INFERBO_ALLOC_MAY_BE_BIG(report, id):

    if 'Allocation:' not in report['bug_trace'][-1]['description']:
        # Unexpected format of bug report --> try to slice by basic file/line
        report_msg = report['bug_trace'][-1]['description']
        print(f'{WARNING}WARNING{ENDC}: slicing_criteria_extraction.py: unrecognized "INFERBO_ALLOC_MAY_BE_BIG" bug trace qualifier format: "{report_msg}" of bug #{id}! Slicing only by line.', file=sys.stderr)
        return 1, report['file'], report['procedure'], report['line'], ''
    else:
        file, fun, line = location_from_bug_trace(report)
        print_debug_report(report, 'INFERBO_ALLOC_MAY_BE_BIG')
        return 0, file, fun, line, ''


# UNINITIALIZED_VALUE: two types were found:
#   1) "The value read from ret was never initialized."
#       e.g. python3 extract_sample.py -d d2a/ -b UNINITIALIZED_VALUE --bug-info-only -n 1
#           ID: ffmpeg_ed80423e6bcfe18cca832b74dcc877427f8cf346_1
#   2) "The value read from pix[_] was never initialized."
#       e.g. python3 extract_sample.py -d d2a/ -b UNINITIALIZED_VALUE --bug-info-only -n 6
#       ID: ffmpeg_1f62bae77d6ced3b79deaa8ce5ba3381fd4a541d_1
#
# There is no additional information in the bug trace, everything can be
# extracted from the basic information. For 2) there is no information on what
# array item it is, but it could be read from the usage or it could be sliced
# by line (the information would be included), but since it isn't typically the
# case that a particular item is not initialized, but rather the whole array, I
# will just slice by array for now and leave room for future improvements
def extract_UNINITIALIZED_VALUE(report, id):
    file = report['file']
    fun = report['procedure']
    line = report['line']
    report_msg = report['qualifier']

    # Get part of the message which contains the variable name
    x = re.search(r'The value read from .* was never initialized.', report['qualifier'])

    # If the given part wasn't found
    if not x:
        print(f'{WARNING}WARNING{ENDC}: slicing_criteria_extraction.py: unrecognized "UNINITIALIZED_VALUE" qualifier format: "{report_msg}" of bug #{id}! Slicing only by line.', file=sys.stderr)
        return 1, file, fun, line, ''

    # Extract variable name
    variable = x.group().split(' ')[4]

    if r'[_]' in variable:
        # Type 2) --> variable == 'variable_name[_]' --> we want only variable_name
        variable = variable.split('[')[0]
        print_debug_report(report, 'UNINITIALIZED_VALUE_TYPE_2')
        return 0, file, fun, line, variable
    else:
        # Type 1)
        print_debug_report(report, 'UNINITIALIZED_VALUE_TYPE_1')
        return 0, file, fun, line, variable


BUFFER_OVERRUN = [ 'BUFFER_OVERRUN_L1',
                   'BUFFER_OVERRUN_L5',
                   'BUFFER_OVERRUN_L4',
                   'BUFFER_OVERRUN_U5',
                   'BUFFER_OVERRUN_L3',
                   'BUFFER_OVERRUN_L2' ]
# BUFFER_OVERRUN: two types were found (see NULL_DEREFERENCE):
#   1) "Offset: [0, 15] Size: 4."
#       e.g. python3 extract_sample.py -d d2a/ -b BUFFER_OVERRUN_L2 --bug-info-only -n 2
#           ID: ffmpeg_61d490455ade68a02dfdcfdb172ba3ded2fe0f9d_1
#   2) "Offset: [1, 4] Size: 4 by call to `filter_mb_mbaff_edgecv`."
#       e.g. python3 extract_sample.py -d d2a/ -b BUFFER_OVERRUN_L2 --bug-info-only -n 3
#       ID: ffmpeg_0f5e5ecc888af015015f2ce1211a066350fbe377_1
#
# For these types of errors, Infer does not list the names of the arrays or the
# names of the variables that are used to access the array. Thus, it is not
# possible to simply slice by variable name. Variable names could be obtained if
# Infer reported the exact location of the error, but Infer only reports the line.
# It only reports the column as the beginning of that line, so it would be
# necessary to do some additional analysis to determine which array (there may
# be more than one) on that line Infer is referring to. Or the Infer output would
# need to be modified, since the specific information about which array is meant
# must definitely be present in the analysis.
#
# For this reason, these types of errors can only be sliced by the line that
# can be extracted from the last step of bug trace (e.g. description
# "Array access: Offset: [1, 4] Size: 4 by call to `filter_mb_mbaff_edgecv` ").
def extract_BUFFER_OVERRUN(report, id):
    file = report['file']
    fun = report['procedure']
    line = report['line']
    bug_type = report['bug_type']

    if 'by call to' in report['qualifier']:
        # Type 2) --> extract location from last step of bug trace
        if 'Array access:' not in report['bug_trace'][-1]['description']:
            # Unexpected format of bug report --> try to slice by basic file/line
            report_msg = report['bug_trace'][-1]['description']
            print(f'{WARNING}WARNING{ENDC}: slicing_criteria_extraction.py: unrecognized "{bug_type}" bug trace qualifier format: "{report_msg}" of bug #{id}! Slicing only by line.', file=sys.stderr)
            return 1, file, fun, line, ''

        # Expected format of bug report --> slice by the last step of bug trace
        file, fun, line = location_from_bug_trace(report)
        print_debug_report(report, 'BUFFER_OVERRUN_TYPE_2')
        return 0, file, fun, line, ''
    else:
        # Type 1) --> extract location from basic info
        print_debug_report(report, 'BUFFER_OVERRUN_TYPE_1')
        return 0, file, fun, line, ''


# NULL_DEREFERENCE: two types were found:
#   1) "pointer `filter` last assigned on line 3191 could be null and is dereferenced at line 3194, column 9."
#       e.g. python3 extract_sample.py -d d2a/ -b NULL_DEREFERENCE --bug-info-only -n 1
#           ID: ffmpeg_15ae526d6763d8e21833feb78680ee3571080017_1
#   2) "pointer `null` is dereferenced by call to `ff_sdp_write_media()` at line 2538, column 5."
#       this happens in experiments/entry-function/scenario1.c type
#       e.g. python3 extract_sample.py -d d2a/ -b NULL_DEREFERENCE --bug-info-only -n 13
#           ID: ffmpeg_a94ada4250ff1d9e6101c910fe71dde6c3b5e485_1
#
# These types can be easily distinguished using the Infer message. In case 1)
# the variable name, line, file and entry function can be read from the basic
# information (no need to look in the bug trace). However, in case 2) only the
# entry function can be read from the basic information. The specific variable
# name in the referenced function cannot be found out! The line can be found out
# in some cases (e.g. scenario1.c) but sometimes it cannot
# (e.g. ffmpeg_a94ada4250ff1d9e6101c910fe71dde6c3b5e485_1). Also, the file may
# not be possible to detect because the dereference may not be directly in the
# function mentioned in the infer message ("by call to ...") but deeper in the
# call tree. Since it is not always possible in case 2) to know where the
# dereference occurs (nor is it possible to easily know when we are able to
# extract this information), in case 2) it will only be sliced by the line on
# which the function that dereferences the variable is called. In this way, all
# functions in the call tree that could dereference the variable should be
# included in the final CPG.
def extract_NULL_DEREFERENCE(report, id):
    file = report['file']
    fun = report['procedure']
    line = report['line']
    bug_type = report['bug_type']
    report_msg = report['qualifier']

    # If type 2) --> return '' --> slice only by line
    if 'by call to' in report_msg:
        print_debug_report(report, 'NULL_DEREFERENCE_TYPE_2')
        return 0, file, fun, line, ''

    # Get part of the message which contains the variable name
    x = re.search(r'pointer `.*` last assigned', report_msg)

    # If the given part wasn't found
    if not x:
        print(f'{WARNING}WARNING{ENDC}: slicing_criteria_extraction.py: unrecognized "{bug_type}" qualifier format: "{report_msg}" of bug #{id}! Slicing only by line.', file=sys.stderr)
        return 1, file, fun, line, ''

    # The variable name is enclosed in `variable_name`
    x = x.group().split('`')

    if len(x) != 3:
        print(f'{WARNING}WARNING{ENDC}: slicing_criteria_extraction.py: unrecognized "{bug_type}" qualifier format: "{report_msg}" of bug #{id}! Slicing only by line.', file=sys.stderr)
        return 1, file, fun, line, ''

    print_debug_report(report, 'NULL_DEREFERENCE_TYPE_1')
    return 0, file, fun, line, x[1]


# Variable, line and file are information about the location of the error
# and need to be extracted for each bug type individually
def extract_slicing_info(report, id):
    bug_type = report["bug_type"]

    if bug_type == 'NULL_DEREFERENCE':
        # File and line is taken from basic info. Although in some cases it might
        # be more precise to take it from bug trace e.g. expepriments/report1.json
        # but since we can't determine from which step --> it can't be done.
        return extract_NULL_DEREFERENCE(report, id)
    elif bug_type in BUFFER_OVERRUN:
        return extract_BUFFER_OVERRUN(report, id)
    elif bug_type == 'UNINITIALIZED_VALUE':
        return extract_UNINITIALIZED_VALUE(report, id)
    elif bug_type == 'INFERBO_ALLOC_MAY_BE_BIG':
        return extract_INFERBO_ALLOC_MAY_BE_BIG(report, id)
    elif bug_type in INTEGER_OVERFLOW:
        return extract_INTEGER_OVERFLOW(report, id)
    elif bug_type == 'NULLPTR_DEREFERENCE':
        return extract_NULLPTR_DEREFERENCE(report, id)
    else:
        # Unsupported bug type
        print(f'{WARNING}WARNING{ENDC}: slicing_criteria_extraction.py: unsupported bug type "{bug_type}" of bug #{id}.', file=sys.stderr)
        return 5, None, None, None, None


def transform_d2a_sample(report_d2a):
    report = report_d2a['bug_info']
    report['id'] = report_d2a['id']
    report['bug_type'] = report_d2a['bug_type']
    # We need only last step of bug trace
    report['bug_trace'] = [report_d2a['trace'][-1]]
    report['bug_trace'][-1]['filename'] = report['bug_trace'][-1]['file']
    # Extract line from location
    report['bug_trace'][-1]['line_number'] = report['bug_trace'][-1]['loc'].split(':')[0]
    report['adjusted_bug_loc'] = report_d2a['adjusted_bug_loc']

    return report


if __name__ == "__main__":
    parser = init_parser()
    args = parser.parse_args()

    if not os.path.exists(args.file):
        # The Infer report file is missing
        print(f'{ERROR}ERROR{ENDC}: slicing_criteria_extraction.py: file "{args.file}" doesn\'t exist!', file=sys.stderr)
        exit(7)

    if args.debug and args.debug > 0:
        DEBUG = args.debug
        init_debug()

    if args.d2a:
        # Input file is in D2A format
        reports = []
        with gzip.open(args.file, mode = 'rb') as fp:
            while True:
                try:
                    item = pickle.load(fp)
                except EOFError:
                    break

                reports.append(transform_d2a_sample(item))
    else:
        # Input file is in Infer format
        with open(args.file, "r") as f:
            try:
                reports = json.load(f)
            except json.decoder.JSONDecodeError:
                print(f'{ERROR}ERROR{ENDC}: slicing_criteria_extraction.py: file "{report_json_file}" in not in JSON format!', file=sys.stderr)
                exit(7)

    # Print output header
    # print(f'status,bug_id,entry_function,file,line,variable')

    if not reports:
        # The report is empty, but in JSON format
        exit(0)

    infer_bug_id = 0
    for report in reports:
        if args.d2a:
            bug_id = report['id']
        else:
            bug_id = infer_bug_id

        bug_type = report['bug_type']

        # Entry function can be extracted in the same way for all the bug types
        entry = report['procedure']

        # Extract bug location from report
        status, file, fun, line, variable = extract_slicing_info(report, bug_id)

        if status == 5:
            # 5 means an unsupported bug type --> don't print anything
            # print(f'{status},{bug_id},{bug_type}')
            infer_bug_id += 1
            continue

        # Extract only file name from possible relative path
        if file:
            file = os.path.basename(file)

        # If file is header, we have to slice differently - we need function
        # name instead of file name
        if is_header(file):
            file = ''
        else:
            fun = ''

        if status == 0:
            # 0 means everything is OK
            if variable:
                # Slicing by variable
                print(f'{status},{bug_id},{entry},{file},{fun},{line},&{variable}')
            else:
                # Slicing by line only
                print(f'{status},{bug_id},{entry},{file},{fun},{line},')
        else:
            # Status should be 1 which means internal error -- the script tries
            # to at least extract entry, file and line
            print(f'{status},{bug_id},{entry},{file},{fun},{line},')

        # 'adjusted_bug_loc' should be now the same as extracted bug location
        # Extracting slicing criteria works identically as in D2A -- this is needed for real-world programs
        # if report['adjusted_bug_loc']:
        #     if os.path.basename(report['adjusted_bug_loc']['file']) == file and report['adjusted_bug_loc']['line'] == int(line):
        #          print(f'{OK}OK{ENDC}', file=sys.stderr)
        #     else:
        #          print(f'{ERROR}ERROR{ENDC}', file=sys.stderr)
        #          exit(1)
        # else:
        #     if os.path.basename(report['file']) == file and report['line'] == int(line):
        #          print(f'{OK}OK{ENDC}', file=sys.stderr)
        #     else:
        #          print(f'{ERROR}ERROR{ENDC}', file=sys.stderr)
        #          exit(1)

        infer_bug_id += 1
