# The experiment demonstrates the use of the Joern tool in "batch" mode, which
# eliminates the need to restart/shutdown the tool for each sample. At the same
# time, there is no need to reset the workspace. The 'save' command in
# 'joern_script_batch' must be present at the end to generate an identical graph
# as when processing a single CPG. If 'save' is omitted, the output graph is
# incomplete, even though the documentation says that everything is saved when
# the script exits (not true).

# The experiment is successful if the command ends without error.
# To run the experiment: 'make' or 'make experiment'.

# Slicing info for the bitcode:
#				  rc, bug_id                                            , entry_function   , file                   , fun, line, variable
# 	(1)		0 , openssl_4fe3bdc0e03efa4b891edc0920f73329305d034f_1, CRYPTO_128_unwrap, crypto/modes/wrap128.c ,    , 118 ,
# 	(2)		0 , openssl_d5bfd1d5930306bc81e4278b5a3529d12c63518a_1, des_quad_cksum   , crypto/des/qud_cksm.c  ,    , 127 ,
# 	(3)		0 , openssl_fc488cb514923abadce40a685eeb7f2258e6b55c_1, BUF_strnlen      , crypto/buffer/buf_str.c,    , 68  ,

.PHONY: all experiment clean 1 2 3

all: clean experiment

clean:
	rm -rf workspace workspaces *.bin */*.bin */*.zip */*_sliced.bc 1_cpg* 2_cpg* 3_cpg*

experiment: batch

1:
	llvm-slicer --entry=CRYPTO_128_unwrap --sc="crypto/modes/wrap128.c##118#" -o=1/1_sliced.bc 1/openssl_4fe3bdc0e03efa4b891edc0920f73329305d034f_1.bc
	llvm2cpg 1/1_sliced.bc --output=1/1_sliced.cpg.bin.zip
	joern --script 1_joern_script
	mv workspace/1_sliced.cpg.bin.zip/cpg.bin 1_cpg.bin
	rm -rf workspace
	joern-export --repr all --format neo4jcsv -o 1_cpg 1_cpg.bin

2:
	llvm-slicer --entry=des_quad_cksum --sc="crypto/des/qud_cksm.c##127#" -o=2/2_sliced.bc 2/openssl_d5bfd1d5930306bc81e4278b5a3529d12c63518a_1.bc
	llvm2cpg 2/2_sliced.bc --output=2/2_sliced.cpg.bin.zip
	joern --script 2_joern_script
	mv workspace/2_sliced.cpg.bin.zip/cpg.bin 2_cpg.bin
	rm -rf workspace
	joern-export --repr all --format neo4jcsv -o 2_cpg 2_cpg.bin

3:
	llvm-slicer --entry=BUF_strnlen --sc="crypto/buffer/buf_str.c##68#" -o=3/3_sliced.bc 3/openssl_fc488cb514923abadce40a685eeb7f2258e6b55c_1.bc
	llvm2cpg 3/3_sliced.bc --output=3/3_sliced.cpg.bin.zip
	joern --script 3_joern_script
	mv workspace/3_sliced.cpg.bin.zip/cpg.bin 3_cpg.bin
	rm -rf workspace
	joern-export --repr all --format neo4jcsv -o 3_cpg 3_cpg.bin

batch: 1 2 3
	joern --script joern_script_batch
	mv workspace/1_sliced.cpg.bin.zip/cpg.bin 1_batch_cpg.bin
	mv workspace/2_sliced.cpg.bin.zip/cpg.bin 2_batch_cpg.bin
	mv workspace/3_sliced.cpg.bin.zip/cpg.bin 3_batch_cpg.bin
	joern-export --repr all --format neo4jcsv -o 1_cpg_batch 1_batch_cpg.bin
	joern-export --repr all --format neo4jcsv -o 2_cpg_batch 2_batch_cpg.bin
	joern-export --repr all --format neo4jcsv -o 3_cpg_batch 3_batch_cpg.bin
	diff -r 1_cpg 1_cpg_batch/
	diff -r 2_cpg 2_cpg_batch/
	diff -r 3_cpg 3_cpg_batch/
