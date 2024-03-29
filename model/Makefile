all: train-model

train-model:
	python3 src/mixed_nodes_model.py schemas/mixed_nodes/extended_cpg.pbtxt ../D2A-CPG/libtiff

evaluate-model:
	python3 src/evaluate_model.py schemas/mixed_nodes/extended_cpg.pbtxt ../D2A-CPG/libtiff
