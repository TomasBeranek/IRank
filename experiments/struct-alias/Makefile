# ******************************************************************************
#  File:            Makefile
#  Master's Thesis: Evaluating Reliability of Static Analysis Results
#                   Using Machine Learning
#  Author:          Beranek Tomas (xberan46)
#  Date:            14.5.2024
#  Up2date sources: https://github.com/TomasBeranek/but-masters-thesis
#  Description:     Part of the struct-alias experiment.
# ******************************************************************************

# This experiment demonstrates that aliases are discarded and TYPE_DECL nodes
# for structures are correctly removed. The demonstration is run with 'make',
# and in the graph, look for TYPE node 7 and MEMBER nodes 10 and 12. These
# nodes together form the Point structurefrom the point.h file.

all: run

clean:
	rm -rf main.bc point.bc combined.bc combined.cpg.bin.zip graph workspace out combined.ll

run: clean
	clang -c -emit-llvm -g -grecord-command-line -fno-inline-functions -fno-builtin -O0 main.c -o main.bc
	clang -c -emit-llvm -g -grecord-command-line -fno-inline-functions -fno-builtin -O0 point.c -o point.bc
	llvm-link main.bc point.bc -o combined.bc
	llvm2cpg combined.bc --output=combined.cpg.bin.zip
	joern --script joern_script
	llvm-dis combined.bc
	joern-export --repr all --format neo4jcsv -o graph workspace/combined.cpg.bin.zip/cpg.bin
	python3.8 ../../model/schemas/feature_engineering.py graph
