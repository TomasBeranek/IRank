#!/usr/bin/env python3.9

# The program takes as the only argument the name of the file with the Infer
# output in json format. The program processes and deletes the first report
# in the file.

# Meaning of error codes:
#   0 -- Ok.
#   1 -- Internal error.
#   5 -- Unsupported bug type.
#   6 -- The file is empty, but in JSON format (not considered as an error).
#   7 -- The file is not in JSON format or is missing completely.


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


# INTEGER_OVERFLOW_L5
# INTEGER_OVERFLOW_U5
# INTEGER_OVERFLOW_L2: two types were found:
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
# then it would be possible to prune according to the 2nd parameter, which is
# always size (Infer is looking for an error here). However, the allocation
# function can be of any shape in general, and we don't know which parameter is
# the size of the allocation. Therefore, it will only be pruned by that line --
# plus the allocation function is generally not large, so it doesn't add as much
# unnecessary information. The Entry function will take from the basic information
# as with other errors.


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


# BUFFER_OVERRUN_L5
# BUFFER_OVERRUN_L4
# BUFFER_OVERRUN_U5
# BUFFER_OVERRUN_L3
# BUFFER_OVERRUN_L2: two types were found (see NULL_DEREFERENCE):
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
        exit(7)

    with open(report_json_file, "r") as f:
        try:
            reports = json.load(f)
        except json.decoder.JSONDecodeError:
            print(f'{ERROR}ERROR{ENDC}: slicing_criteria_extraction.py: file "{report_json_file}" in not in JSON format!', file=sys.stderr)
            exit(7)

    if not reports:
        # The report is empty, but in JSON format -- we processed all the reports
        exit(6)

    # Save the rest of the reports back to the original file
    with open(report_json_file, "w") as f:
        json.dump(reports[1:], f)

    slicing_info = extract_slicing_info(reports[0])
    print(f"{slicing_info[0]}\n{slicing_info[1]}\n{slicing_info[2]}\n{slicing_info[3]}")
