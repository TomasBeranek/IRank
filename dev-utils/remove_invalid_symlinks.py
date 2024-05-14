import os
import sys

# Script's 1st arg is name of the directory
directory_path = sys.argv[1]

invalid_symlinks_cnt = 0

# Recursively search for invalid symlinks in specified dir
for root, _, files in os.walk(directory_path):
    for file in files:
        file_path = os.path.join(root, file)

        # Check if the file is a symlink
        if os.path.islink(file_path):
            target_file = os.readlink(file_path)
            target_path = os.path.join(root, target_file)

            # Check if the target path exists
            if not os.path.exists(target_path):
                invalid_symlinks_cnt += 1
                os.remove(file_path)

print(f'Removed {invalid_symlinks_cnt} invalid symlinks!')
