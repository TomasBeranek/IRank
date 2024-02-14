HEADERS_DIR = header_cache/
TEST_DIR = test


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

clean-test:
	rm -rf $(TEST_DIR)/main.bc $(TEST_DIR)/point.bc $(TEST_DIR)/combined.bc $(TEST_DIR)/combined.cpg.bin.zip $(TEST_DIR)/graph $(TEST_DIR)/workspace $(TEST_DIR)/out

# This target visualizes graph of custom code in 'model/schemes/test/'
test: clean-test
	clang -c -emit-llvm -g -grecord-command-line -fno-inline-functions -fno-builtin -O0 $(TEST_DIR)/main.c -o $(TEST_DIR)/main.bc
	clang -c -emit-llvm -O0 $(TEST_DIR)/point.c -o $(TEST_DIR)/point.bc
	llvm-link $(TEST_DIR)/main.bc $(TEST_DIR)/point.bc -o $(TEST_DIR)/combined.bc
	llvm2cpg $(TEST_DIR)/combined.bc --output=$(TEST_DIR)/combined.cpg.bin.zip
	cd $(TEST_DIR) && joern --script joern_script
	joern-export --repr all --format neo4jcsv -o $(TEST_DIR)/graph $(TEST_DIR)/workspace/combined.cpg.bin.zip/cpg.bin
	python3 visualize_graph.py $(TEST_DIR)/graph
