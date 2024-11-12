# ******************************************************************************
#  File:            Makefile
#  Master's Thesis: Evaluating Reliability of Static Analysis Results
#                   Using Machine Learning
#  Author:          Beranek Tomas (xberan46)
#  Date:            14.5.2024
#  Up2date sources: https://github.com/TomasBeranek/but-masters-thesis
#  Description:     Part of the arg-passing experiment.
# ******************************************************************************

# This experiment serves as a demonstration of argument passing. The graph can
# be visualized using 'make'. In the graph, look for arguments as LITERAL nodes
# 18 and 19.


all: run

clean:
	rm -rf main.bc add.bc combined.bc combined.cpg.bin.zip graph workspace out combined.ll

run: clean
	clang -c -emit-llvm -g -grecord-command-line -fno-inline-functions -fno-builtin -O0 main.c -o main.bc
	clang -c -emit-llvm -g -grecord-command-line -fno-inline-functions -fno-builtin -O0 add.c -o add.bc
	llvm-link main.bc add.bc -o combined.bc
	llvm2cpg combined.bc --output=combined.cpg.bin.zip
	joern --script joern_script
	llvm-dis combined.bc
	joern-export --repr all --format neo4jcsv -o graph workspace/combined.cpg.bin.zip/cpg.bin
	python3.8 ../../model/schemas/feature_engineering.py graph
