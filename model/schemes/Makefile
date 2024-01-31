HEADERS_DIR = header_cache/

.PHONY: experiment all httpd-sample

all: experiment

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
