#!/usr/bin/env python3.9

import os
import sqlite3
from sqlite3 import Error
import re
import json
import sys


def create_connection(db_file):
    connection = None
    try:
        connection = sqlite3.connect(db_file)
    except Error as e:
        print("ERROR: Failed to connect to the Infer's database: " + db_file)
        exit(1)
    return connection


def get_functions_from_BLOB(BLOB):
    #extract all C++/C (not Java!) function names from binary object
    return re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', BLOB.decode('cp437'))


def load_functions(db):
    # load all function names and their callees
    cursor = db.cursor()
    cursor.execute('SELECT p.proc_uid, p.callees FROM procedures p')
    rows = cursor.fetchall()

    # create dictionary of all functions and add their callees
    functions = {}
    for row in rows:
        function_name = row[0]
        function_callees = get_functions_from_BLOB(row[1])
        functions[function_name] = {}
        functions[function_name]['callees'] = function_callees
        functions[function_name]['callers'] = []

    # based on callees calculate direct callers (parents)
    for function_name in functions.keys():
        for callee_name in functions[function_name]['callees']:
            functions[callee_name]['callers'] += [function_name]

    return functions


def get_entry_point(functions, function_name):
    curr_entry_point = function_name

    while functions[curr_entry_point]['callers']:
        curr_entry_point = functions[curr_entry_point]['callers'][0]

    return curr_entry_point


if __name__ == "__main__":
    db_file = sys.argv[1] + '/results.db'

    backtrace_functions = 'main'
    # backtrace_functions = ['f']

    # establish database connection
    db = create_connection(db_file)
    if db == None:
        print("ERROR: Failed to connect to the Infer's database: " + db_file)
        exit(1)

    functions = load_functions(db)

    if isinstance(backtrace_functions, list):
        # since we have back trace across multiple functions, we want to use
        # the top level function as entry point
        print(f'Entry point: {backtrace_functions[-1]}')
    else:
        # we have only information where bug occured (without backtrace), so we
        # want to calculate any top-most entry point
        # TODO: we might want to consider whether bug is inter or intraprocedural
        #       since for intraprocedural bug (analysis) would current function
        #       suffice as entry point
        function_name = backtrace_functions
        entry_point_function = get_entry_point(functions, function_name)
        print(f'Entry point: {entry_point_function}')
