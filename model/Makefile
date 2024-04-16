all: train-model

ID ?= --latest

train-model:
	python3 src/mixed_nodes_model.py schemas/mixed_nodes/extended_cpg.pbtxt ../D2A-CPG

evaluate-model:
	python3 src/evaluate_model.py schemas/mixed_nodes/extended_cpg.pbtxt ../D2A-CPG $(ID)
