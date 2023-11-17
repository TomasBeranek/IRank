import argparse
import subprocess
import os
import gzip
import pickle
import json
import sys
import shutil
from tqdm import tqdm


# colors
OK='\033[0;92m'
WARNING='\033[0;93m'
ERROR='\033[0;31m'
NOTE='\033[0;95m'
ENDC='\033[0m'


# HTTPD vars
HTTPD_TRACKED_FILES = ['include/ap_config_auto.h.in',
                       'include/ap_config_layout.h.in',
                       'modules/ssl/ssl_policies.h.in',
                       'buildconf']

# LIBTIFF vars
LIBTIFF_TRACKED_FILES = ['config/config.h.in',
                         'port/libport_config.h.in',
                         'libtiff/tiffvers.h.in',
                         'libtiff/tiffconf.h.in',
                         'libtiff/tif_config.h.in']

# Number of samples for which compilation command failed/succeded
failed_samples = 0
success_samples = 0


def init_parser():
    parser = argparse.ArgumentParser(description='Creates bitcode (.bc file) for each sample in D2A dataset.')

    parser.add_argument('-r', '--repository', metavar='DIR', required=True, type=str, help='local copy (git clone) of given repository')
    parser.add_argument('-f', '--file', metavar='FILE', required=True, type=str, help='.pickle.gz file with D2A samples')
    parser.add_argument('-o', '--output-dir', metavar='DIR', required=True, type=str, help='output directory with bitcode files')
    parser.add_argument('-c', '--commit', metavar='FILE', type=str, help='abbreviated hash of a commit to start with')
    parser.add_argument('--project', choices=['httpd', 'nginx', 'libtiff', 'ffmpeg', 'libav', 'openssl'], required=True, type=str, help="project name")
    # parser.add_argument('--include', required=True, type=str, help="path to 'include/' dir for specified project")

    return parser


# Latest hash is first
def get_git_commit_hashes(repository_path):
    command = ['git', 'log', '--all', '--format=%H']
    output = subprocess.check_output(command, cwd=args.repository).decode().strip()
    commit_hashes = output.split('\n')

    return commit_hashes


def get_d2a_hashes(file, output_dir):
    # Extract IDs of already proccessed samples
    files = [f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))]
    already_created_samples_IDs = set([f.split('.')[0] for f in files])

    hashes = dict()
    with gzip.open(file, mode = 'rb') as f:
        while True:
            try:
                sample = pickle.load(f)
            except EOFError:
                break

            # If there is already bitcode file for this sample - skip it
            if sample['id'] in already_created_samples_IDs:
                print(f"{NOTE}NOTE{ENDC}: construction_phase_d2a.py: Skipping sample '{sample['id']}' because bitcode already exists.", file=sys.stderr)
                continue

            # We need to take the 'before' version of each commit
            commit_hash = sample['versions']['before']

            if commit_hash not in hashes:
                # First sample with this hash
                hashes[commit_hash] = dict()

            # Associate current sample with its hash
            hashes[commit_hash][sample['id']] = {'label': sample['label'],
                                                 'compiler_args': sample['compiler_args'] }

    return hashes


# Returns all .bc files in current dir
def find_bitcode_files():
    matches = set()
    for root, _, files in os.walk('.'):
        for file in files:
            if file.endswith('.bc'):
                matches.add(os.path.join(root, file))
    return matches


def find_first_sys_idx(compiler_args):
    idx = -1

    for i, arg in enumerate(compiler_args):
        if '<$sys$>' in arg:
            idx = i
            break

    return idx


def run_compile_commands(compiler_args_dict):
    global failed_samples

    compilation_failed = False

    # For each file we generate a single .bc file
    for file, compiler_args in compiler_args_dict.items():
        # DEBUG info:
        print(f'\t{file}')

        # Set current repo path to compiler args
        compiler_args = compiler_args.replace('<$repo$>', os.path.abspath(os.getcwd()))

        # Split args to list
        compiler_args = compiler_args.split(' ')

        # Set library header files path
        if args.project == 'httpd':
            # Find index of first '-I<$sys$>...' arg
            idx = find_first_sys_idx(compiler_args)

            # Check there are any includes of external libraries
            if idx != -1:
                # Remove -I args which include external libraries
                compiler_args = [arg for arg in compiler_args if '<$sys$>' not in arg]

                # Add paths to external libraries instead of their first original occurence in args
                compiler_args = compiler_args[:idx] + ['-Isrclib/apr/include', '-Isrclib/apr-util/include'] + compiler_args[idx:]
        elif args.project == 'nginx':
            # Add paths to headers since in some D2A samples they seem to be missing
            if "-Isrc/core" not in compiler_args:
                compiler_args.append("-Isrc/core")
            if "-Iobjs" not in compiler_args:
                compiler_args.append("-Iobjs")
            if "-Isrc/os/unix" not in compiler_args:
                compiler_args.append("-Isrc/os/unix")
            if "-Isrc/event" not in compiler_args:
                compiler_args.append("-Isrc/event")
        elif args.project == 'openssl':
            # Add paths to headers since in some D2A samples they seem to be missing
            if "-Iinclude" not in compiler_args:
                compiler_args.append("-Iinclude")
            if "-Icrypto/include" not in compiler_args:
                compiler_args.append("-Icrypto/include")

        # Add args to generate bitcode
        compile_command = ['clang', '-emit-llvm', '-g', '-grecord-command-line', '-fno-inline-functions', '-fno-builtin'] + compiler_args + ['-c', file]

        # Run the compilation command
        completed_process = subprocess.run(compile_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Check for failure
        if completed_process.returncode != 0:
            print(" ".join(compile_command))
            print(completed_process.stdout.decode('utf-8'))
            print(completed_process.stderr.decode('utf-8'))
            print(f"{ERROR}ERROR{ENDC}: construction_phase_d2a.py: source files of bug '{id}' failed to compile!", file=sys.stderr)
            compilation_failed = True

    if compilation_failed:
        failed_samples += 1


def save_bitcode(id, bc_files, output_dir):
    # Run llvm-link to create a single bitcode file
    completed_process = subprocess.run(['llvm-link'] + list(bc_files) + ['-o', os.path.join(output_dir, f'{id}.bc')])
    if completed_process.returncode != 0:
        print(f"{ERROR}ERROR{ENDC}: construction_phase_d2a.py: failed to link bitcode files for smaple {id}!", file=sys.stderr)
        exit(1)


def files_updated(tracked_files, hash, prev_hash):
    if prev_hash is None:
        # We are at the first commit -> we need to create the headers for the first time
        return True

    # Just for testing
    # git diff --quiet 9fa142563c3bcddafcb892d59c458c87ec278a16 e86620ff6bb77998fde00f7033d283f23184b6fc -- include/ap_config_auto.h include/ap_config_layout.h modules/slotmem/mod_slotmem_shm.c

    # Check if any of tracked files changed between these two commits
    completed_process = subprocess.run(['git', 'diff', '--quiet', hash, prev_hash, '--'] + tracked_files)

    return completed_process.returncode != 0


if __name__ == '__main__':
    parser = init_parser()
    args = parser.parse_args()

    # Make output dir path absolute
    args.output_dir = os.path.abspath(args.output_dir)

    # Check if output dir exists
    if not os.path.exists(args.output_dir):
        print(f"{ERROR}ERROR{ENDC}: construction_phase_d2a.py: output directory '{args.output_dir}' doesn't exist!", file=sys.stderr)
        exit(1)

    # Get chronological list of hashes
    hashes = get_git_commit_hashes(args.repository)

    # Get set of hashes present in D2A dataset
    d2a_hashes = get_d2a_hashes(args.file, args.output_dir)

    # Remove hashes (commits) in which we are not interested in (not in D2A)
    hashes = [hash for hash in hashes if hash in d2a_hashes.keys()]

    # Navigate to given repository
    os.chdir(args.repository)

    # Latest commit is now last
    hashes = list(reversed(hashes))

    # Start from defined commit
    if args.commit:
        completed_process = subprocess.run(['git', 'rev-parse', args.commit], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        whole_hash = completed_process.stdout.decode('utf-8')[:-1] # drop last '\n'
        while hashes and hashes[0] != whole_hash:
            hashes.pop(0)

    # Check if we have dependencies for specified project
    if args.project == 'httpd':
        if not os.path.exists('../httpd-dependencies/srclib/apr') or not os.path.exists('../httpd-dependencies/srclib/apr-util'):
            print(f"{ERROR}ERROR{ENDC}: construction_phase_d2a.py: external libraries '../httpd-dependencies/srclib/apr' or '../httpd-dependencies/srclib/apr-util' for httpd project are missing!", file=sys.stderr)
            exit(1)

    # Iterate over hashes and switch repository to given commits
    prev_hash = None
    for hash in tqdm(hashes):
        # We want to restore all files to 'HASH' commit and delete any leftovers from previous compilation
        subprocess.run(['git', 'reset', '--hard', hash])
        subprocess.run(['git', 'clean', '-dfx'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if args.project == 'httpd':
            # We need copy back external libs and generated .h files (unless the template changed)
            # NOTE: Even if the template didn't change, the actual .h could change, because it takes
            #       values from various sources. But it would be too time consuming to find all the
            #       sources. We also can't generate the .h for each commit as that would be too
            #       computionally expensive. Therefore we have to do an APPROXIMATION, where we generate
            #       (update) all the .h files only when:
            #         1) any of the templates changed
            #         2) buildconf changed

            # Copy back external libs
            shutil.copytree('../httpd-dependencies/srclib/apr', 'srclib/apr')
            shutil.copytree('../httpd-dependencies/srclib/apr-util', 'srclib/apr-util')
            # This is from the newest version of HTTPD
            try:
                # We need to copy this, because sometimes it is missing
                shutil.copytree('../httpd-dependencies/srclib/pcre', 'srclib/pcre')
            except FileExistsError:
                # The pcre/ was present in the repo
                pass

            # In commit ff7722bc9a 'util_pcre.c' requires PCRE Version 6.7 or later,
            # but supplied version in this repo is lower (5.0) and compilation fails.
            # So we force the compilation to use the manually installed PCRE.
            # shutil.rmtree('srclib/pcre')

            # Header files from .h.in templates
            if files_updated(HTTPD_TRACKED_FILES, hash, prev_hash):
                # Generate new header files
                completed_process = subprocess.run(['./buildconf'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if completed_process.returncode != 0:
                    print(completed_process.stdout.decode('utf-8'))
                    print(completed_process.stderr.decode('utf-8'))
                    print(f"{WARNING}WARNING{ENDC}: construction_phase_d2a.py: command './buildconf' failed!", file=sys.stderr)

                completed_process = subprocess.run(['./configure'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if completed_process.returncode != 0:
                    print(completed_process.stdout.decode('utf-8'))
                    print(completed_process.stderr.decode('utf-8'))
                    print(f"{WARNING}WARNING{ENDC}: construction_phase_d2a.py: command './configure' failed!", file=sys.stderr)

                # Save generated files
                os.makedirs('/tmp/d2a_pipeline', exist_ok=True)
                shutil.copy('include/ap_config_auto.h', '/tmp/d2a_pipeline/')
                shutil.copy('include/ap_config_layout.h', '/tmp/d2a_pipeline/')
                try:
                    shutil.copy('modules/ssl/ssl_policies.h', '/tmp/d2a_pipeline/')
                except FileNotFoundError:
                    # The ssl_policies.h.in file is added in later versions of httpd
                    # TODO: We can dynamically get a list of .h.in files present and copy files based on this
                    #       find . -type f -name "*.h.in"
                    pass
            else:
                # Copy back old generated headers
                shutil.copy('/tmp/d2a_pipeline/ap_config_auto.h', 'include/')
                shutil.copy('/tmp/d2a_pipeline/ap_config_layout.h', 'include/')
                try:
                    shutil.copy('/tmp/d2a_pipeline/ssl_policies.h', 'modules/ssl/')
                except FileNotFoundError:
                    # The ssl_policies.h.in file is added in later versions of httpd
                    pass

            # Header files from .c templates
            # Some info: https://httpd.apache.org/docs/2.4/platform/netware.html
            if files_updated(['server/gen_test_char.c', 'srclib/pcre/dftables.c'], hash, prev_hash):
                # The 'test_char.h' is generated as stdout of 'server/gen_test_char.c'
                subprocess.run(['gcc', '-Isrclib/apr/include', '-Isrclib/apr-util/include', 'server/gen_test_char.c', '-o', 'gen_test_char'])
                with open('include/test_char.h', 'w') as f:
                    subprocess.run(['./gen_test_char'], stdout=f)
                shutil.copy('include/test_char.h', '/tmp/d2a_pipeline/')

                # The 'chartables.c' is generated using 'srclib/pcre/dftables.c'
                subprocess.run(['gcc', 'srclib/pcre/dftables.c', '-o', 'dftables'])
                subprocess.run(['./dftables', 'include/chartables.c'])
                shutil.copy('include/chartables.c', '/tmp/d2a_pipeline/')
            else:
                shutil.copy('/tmp/d2a_pipeline/test_char.h', 'include/')
                shutil.copy('/tmp/d2a_pipeline/chartables.c', 'include/')
        elif args.project == 'nginx':
            # Generate header files (in NGINX's case its pretty fast)
            completed_process = subprocess.run(['./auto/configure'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if completed_process.returncode != 0:
                print(completed_process.stdout.decode('utf-8'))
                print(completed_process.stderr.decode('utf-8'))
                print(f"{WARNING}WARNING{ENDC}: construction_phase_d2a.py: command './auto/configure' failed!", file=sys.stderr)
        elif args.project == 'libtiff':
            # Generate header files (in LIBTIFF's case its pretty fast)
            completed_process = subprocess.run(['./autogen.sh'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if completed_process.returncode != 0:
                print(completed_process.stdout.decode('utf-8'))
                print(completed_process.stderr.decode('utf-8'))
                print(f"{WARNING}WARNING{ENDC}: construction_phase_d2a.py: command './autogen.sh' failed!", file=sys.stderr)
            completed_process = subprocess.run(['./configure'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if completed_process.returncode != 0:
                print(completed_process.stdout.decode('utf-8'))
                print(completed_process.stderr.decode('utf-8'))
                print(f"{WARNING}WARNING{ENDC}: construction_phase_d2a.py: command './configure' failed!", file=sys.stderr)
        elif args.project == 'ffmpeg':
            # Generate config.h (in FFMPEG's case its pretty fast)
            completed_process = subprocess.run(['./configure'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if completed_process.returncode != 0:
                print(completed_process.stdout.decode('utf-8'))
                print(completed_process.stderr.decode('utf-8'))
                print(f"{WARNING}WARNING{ENDC}: construction_phase_d2a.py: command './configure' failed!", file=sys.stderr)
            # Generate code_names.h and version.h (we dont care if these commands fail, since the files are missing in some commits)
            command_code_names = 'gcc -I. -E libavcodec/avcodec.h | libavcodec/codec_names.sh config.h libavcodec/codec_names.h'
            subprocess.run(command_code_names, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.run(['make', 'version.h'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        elif args.project == 'libav':
            # Generate config.h (this might need to be optimized)
            completed_process = subprocess.run(['./configure'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if completed_process.returncode != 0:
                print(completed_process.stdout.decode('utf-8'))
                print(completed_process.stderr.decode('utf-8'))
                print(f"{WARNING}WARNING{ENDC}: construction_phase_d2a.py: command './configure' failed!", file=sys.stderr)
            subprocess.run(['make', 'version.h'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        elif args.project == 'openssl':
            # Generate include/openssl/configuration.h (in openssl's case its surprisingly fast)
            # Configure script in older version needs as a 1st argument name of used compiler.
            # Fortunately later version are backward compatible and take the same arg as well,
            # but they don't need it.
            completed_process = subprocess.run('./Configure gcc', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if completed_process.returncode != 0:
                print(completed_process.stdout.decode('utf-8'))
                print(completed_process.stderr.decode('utf-8'))
                print(f"{WARNING}WARNING{ENDC}: construction_phase_d2a.py: command './Configure gcc' failed!", file=sys.stderr)

            # Generate bn_conf.h
            subprocess.run('make crypto/include/internal/bn_conf.h', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # Generate progs.h
            subprocess.run('make apps/progs.h', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # We need to resolve typo in nb_lcl.h file in this commit, so that compilation will succeed
            if hash == 'fb81ac5e6be4041e59df9362cd63880b21e2cafc':
                subprocess.run(r"sed -i '206s/\]/\\/' crypto/bn/bn_lcl.h", shell=True)

        # Iterate over samples in each commit
        # (multiple samples could have been found in a single commit)
        for id, val in d2a_hashes[hash].items():
            # OPTIMIZE: if the files are the same as in the last sample, no need to run the compilation (it happens very often)

            if args.project == 'httpd':
                # We don't want to delete srclib/ (external libs) and generated .h files (they are up2date now)
                subprocess.run(['git', 'clean', '-dfx', '--exclude=srclib/', '--exclude=include/'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                # This needs to copied back to server/, because it was deleted by previous command
                shutil.copy('include/test_char.h', 'server/')
            elif args.project == 'nginx':
                # We don't want to delete objs/ (generated .h files)
                subprocess.run(['git', 'clean', '-dfx', '--exclude=objs/'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            elif args.project == 'libtiff':
                # We don't want to delete generated .h files
                subprocess.run(['git', 'clean', '-dfx', '--exclude=config/', '--exclude=port/', '--exclude=libtiff/'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            elif args.project == 'ffmpeg':
                # We don't want to delete generated config.h, version.h and some other files
                subprocess.run(['git', 'clean', '-dfx', '--exclude=config.h', '--exclude=version.h', '--exclude=libavutil/', '--exclude=libavformat/', '--exclude=libavcodec/'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            elif args.project == 'libav':
                # We don't want to delete generated config.h, version.h and some other files (this is not copy paste error
                # FFMPEG and LIBAV have same libraries thus we aren't removing similiar files)
                subprocess.run(['git', 'clean', '-dfx', '--exclude=config.h', '--exclude=version.h', '--exclude=libavutil/', '--exclude=libavformat/', '--exclude=libavcodec/'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            elif args.project == 'openssl':
                # We don't want to delete generated include/openssl/configuration.h and crypto/opensslconf.h
                subprocess.run(['git', 'clean', '-dfx', '--exclude=include/', '--exclude=crypto/', '--exclude=apps/'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Check for existing .bc files (we dont want those)
            old_bc_files = find_bitcode_files()

            # Create .bc files
            run_compile_commands(val['compiler_args'])

            new_bc_files = find_bitcode_files()
            added_bc_files = new_bc_files - old_bc_files

            # Get number of .c in current sample
            src_file_cnt = len([f for f in val['compiler_args'].keys() if f.endswith('.c')])

            # Check if we have .bc file for each compilation command
            if len(added_bc_files) == src_file_cnt:
                # Save all the created bitcode files into a single one
                save_bitcode(id, added_bc_files, args.output_dir)
                print(f"{OK}SUCCESS{ENDC}: construction_phase_d2a.py: source files of bug '{id}' were successfully compiled!", file=sys.stderr)
                success_samples += 1
            else:
                # Some .bc files may have been generated, but since some are missing we CAN'T use this sample
                print(f"{ERROR}ERROR{ENDC}: construction_phase_d2a.py: compilation of bug '{id}' failed to generate all .bc files!", file=sys.stderr)

        prev_hash = hash

    # Print final statistics
    if failed_samples:
        print(f"{WARNING}WARNING{ENDC}: construction_phase_d2a.py: {failed_samples} samples out of {failed_samples + success_samples} failed to compile!", file=sys.stderr)
    else:
        print(f"{OK}SUCCESS{ENDC}: construction_phase_d2a.py: all {success_samples} samples successfully compiled!", file=sys.stderr)

    # git checkout branchname       - to lastest commit in given branch
    # git checkout hash             - to given commit
    # git clean -df                 - to remove untracked files (.bc files)
    # git reset --hard HEAD         - hard reset of repo - restore tracked files and remove untracked
    # git rev-parse 6231d0983b      - to get whole hash from its abbreviated version
