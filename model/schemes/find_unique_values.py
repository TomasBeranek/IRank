import os
import sys
import csv
from collections import defaultdict


def find_unique_values(directory, partial_filename):
    unique_values = defaultdict(set)
    samples = 0

    # Iterate over samples
    for subdir, _, files in os.walk(directory):
        samples += 1
        data_file = f'{partial_filename}_data.csv'
        header_file = f'{partial_filename}_header.csv'

        if data_file in files and header_file in files:
            # Read header
            with open(os.path.join(subdir, header_file), mode='r') as file:
                reader = csv.reader(file)
                headers = next(reader)

            # Read data and collect unique values
            with open(os.path.join(subdir, data_file), mode='r') as file:
                reader = csv.DictReader(file, fieldnames=headers)
                for row in reader:
                    for header in headers:
                        if len(unique_values[header]) < 30:
                            unique_values[header].add(row[header])

    return unique_values, samples


if __name__ == '__main__':
    if len(sys.argv) < 2 or len(sys.argv) > 2:
        print('Usage: python script.py <partial_filename>')
        sys.exit(1)

    partial_filename = sys.argv[1]
    directory = '../../D2A-CPG/'  # Script needs to be in model/schemes/ or change accordingly

    unique_values, samples = find_unique_values(directory, partial_filename)

    print(f'Samples: {samples}')

    # Print unique values for each column
    for header, values in unique_values.items():
        if header == ':ID':
            numeric_values = [int(value) for value in values]
            print(f'{header}: {min(numeric_values)} --> {max(numeric_values)}')
        else:
            print(f'{header}: {values}')
