#!/bin/bash

# Script for comparing old/new directories with generated graphs

# Check for the correct number of arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <original_dir> <new_dir>"
    exit 1
fi

original_dir="$1"
compare_dir="$2"

# Find subdirectories in the original directory
find "$original_dir" -type d | while read -r subdir; do
    # Skip the root the original dir
    if [[ "$subdir" == "$original_dir" ]]; then
      continue
    fi

    # Generate the subdirectory's name relative to the original directory
    relative_subdir="${subdir#$original_dir}"
    compare_subdir="$compare_dir$relative_subdir"

    if [ -d "$compare_subdir" ]; then
        # Iterate over files in the original subdirectory
        find "$subdir" -type f | while read -r file; do
            # Generate the file's name relative to the original subdirectory
            relative_file="${file#$subdir/}"

            # Construct the path of the corresponding file in the compare subdirectory
            compare_file="$compare_subdir/$relative_file"

            # Compare the file only if it exists in the compare subdirectory
            if [ -f "$compare_file" ]; then
                # Use diff to compare the two files
                diff -q "$file" "$compare_file"
            else
                # The file is missing in new version of graph
                echo "Miss: $file"
            fi
        done
    fi
done
