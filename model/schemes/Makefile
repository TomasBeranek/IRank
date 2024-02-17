HEADERS_DIR = header_cache/
EXAMPLES_DIR = examples
EXAMPLE_STRUCT_ALIAS_DIR = $(EXAMPLES_DIR)/struct-alias
EXAMPLE_ARG_PASSING_DIR = $(EXAMPLES_DIR)/arg-passing


.PHONY: experiment all httpd-sample

all: experiment

# This experiment contains all types of invalid nodes for demonstration
experiment:
	python3 visualize_graph.py experiment/sample_graph

experiment2:
	python3 visualize_graph.py ../../D2A-CPG/1

httpd-sample:
	python3 visualize_graph.py httpd_sample

process-d2a:
	 find ../../D2A-CPG -mindepth 2 -type d | shuf -n 5000 | python3 visualize_graph.py

# Copies a single instance of each type of header file to $(HEADERS_DIR)
header-cache:
	rm -rf $(HEADERS_DIR)
	mkdir $(HEADERS_DIR)
	find ../../D2A-CPG/ -type f -name "*_header.csv" | awk -F/ '!seen[$$NF]++ {print}' | while read -r file; do cp "$$file" "$(HEADERS_DIR)"; done

# Use this as: 'time make measure-time'
measure-time:
	find ../../D2A-CPG -mindepth 2 -type d | head -n 50 | python3 visualize_graph.py

clean-example-struct-alias:
	rm -rf $(EXAMPLE_STRUCT_ALIAS_DIR)/main.bc $(EXAMPLE_STRUCT_ALIAS_DIR)/point.bc $(EXAMPLE_STRUCT_ALIAS_DIR)/combined.bc $(EXAMPLE_STRUCT_ALIAS_DIR)/combined.cpg.bin.zip $(EXAMPLE_STRUCT_ALIAS_DIR)/graph $(EXAMPLE_STRUCT_ALIAS_DIR)/workspace $(EXAMPLE_STRUCT_ALIAS_DIR)/out

# This target visualizes graph of custom code in 'model/schemes/examples/struct-alias'
example1: example-struct-alias
example-struct-alias: clean-example-struct-alias
	clang -c -emit-llvm -g -grecord-command-line -fno-inline-functions -fno-builtin -O0 $(EXAMPLE_STRUCT_ALIAS_DIR)/main.c -o $(EXAMPLE_STRUCT_ALIAS_DIR)/main.bc
	clang -c -emit-llvm -O0 $(EXAMPLE_STRUCT_ALIAS_DIR)/point.c -o $(EXAMPLE_STRUCT_ALIAS_DIR)/point.bc
	llvm-link $(EXAMPLE_STRUCT_ALIAS_DIR)/main.bc $(EXAMPLE_STRUCT_ALIAS_DIR)/point.bc -o $(EXAMPLE_STRUCT_ALIAS_DIR)/combined.bc
	llvm2cpg $(EXAMPLE_STRUCT_ALIAS_DIR)/combined.bc --output=$(EXAMPLE_STRUCT_ALIAS_DIR)/combined.cpg.bin.zip
	cd $(EXAMPLE_STRUCT_ALIAS_DIR) && joern --script joern_script
	joern-export --repr all --format neo4jcsv -o $(EXAMPLE_STRUCT_ALIAS_DIR)/graph $(EXAMPLE_STRUCT_ALIAS_DIR)/workspace/combined.cpg.bin.zip/cpg.bin
	python3 visualize_graph.py $(EXAMPLE_STRUCT_ALIAS_DIR)/graph

clean-example-arg-passing:
	rm -rf $(EXAMPLE_ARG_PASSING_DIR)/main.bc $(EXAMPLE_ARG_PASSING_DIR)/add.bc $(EXAMPLE_ARG_PASSING_DIR)/combined.bc $(EXAMPLE_ARG_PASSING_DIR)/combined.cpg.bin.zip $(EXAMPLE_ARG_PASSING_DIR)/graph $(EXAMPLE_ARG_PASSING_DIR)/workspace $(EXAMPLE_ARG_PASSING_DIR)/out

# This target visualizes graph of custom code in 'model/schemes/examples/arg-passing'
example2: example-arg-passing
example-arg-passing: clean-example-arg-passing
	clang -c -emit-llvm -g -grecord-command-line -fno-inline-functions -fno-builtin -O0 $(EXAMPLE_ARG_PASSING_DIR)/main.c -o $(EXAMPLE_ARG_PASSING_DIR)/main.bc
	clang -c -emit-llvm -O0 $(EXAMPLE_ARG_PASSING_DIR)/add.c -o $(EXAMPLE_ARG_PASSING_DIR)/add.bc
	llvm-link $(EXAMPLE_ARG_PASSING_DIR)/main.bc $(EXAMPLE_ARG_PASSING_DIR)/add.bc -o $(EXAMPLE_ARG_PASSING_DIR)/combined.bc
	llvm2cpg $(EXAMPLE_ARG_PASSING_DIR)/combined.bc --output=$(EXAMPLE_ARG_PASSING_DIR)/combined.cpg.bin.zip
	cd $(EXAMPLE_ARG_PASSING_DIR) && joern --script joern_script
	joern-export --repr all --format neo4jcsv -o $(EXAMPLE_ARG_PASSING_DIR)/graph $(EXAMPLE_ARG_PASSING_DIR)/workspace/combined.cpg.bin.zip/cpg.bin
	python3 visualize_graph.py $(EXAMPLE_ARG_PASSING_DIR)/graph
