SCHEMA_PATH=schemas/mixed_nodes/extended_cpg.pbtxt
MODELS_DIR=saved_models
DATASET_DIR=../D2A-CPG
ID ?= 1

all: train-model

train-model:
	python3 src/mixed_nodes_model.py $(SCHEMA_PATH) $(DATASET_DIR) $(MODELS_DIR)

train-combined-model:
	python3 src/mixed_nodes_model.py $(SCHEMA_PATH) $(DATASET_DIR) $(MODELS_DIR) combined

evaluate-model-val:
	python3 src/evaluate_model.py $(SCHEMA_PATH) $(DATASET_DIR) $(MODELS_DIR) $(ID) val

evaluate-model-test:
	python3 src/evaluate_model.py $(SCHEMA_PATH) $(DATASET_DIR) $(MODELS_DIR) $(ID) test

scenario-combined:
	python3 src/evaluate_model.py $(SCHEMA_PATH) $(DATASET_DIR) $(MODELS_DIR) combined test

scenario-httpd:
	python3 src/evaluate_model.py $(SCHEMA_PATH) $(DATASET_DIR) $(MODELS_DIR) httpd test

scenario-libtiff:
	python3 src/evaluate_model.py $(SCHEMA_PATH) $(DATASET_DIR) $(MODELS_DIR) libtiff test

scenario-nginx:
	python3 src/evaluate_model.py $(SCHEMA_PATH) $(DATASET_DIR) $(MODELS_DIR) nginx test

scenario-cross-analysis:
	python3 src/evaluate_model.py $(SCHEMA_PATH) $(DATASET_DIR) $(MODELS_DIR) libav test

scenario-chatgpt:
	python3 src/evaluate_model.py $(SCHEMA_PATH) experiments/comparison_with_chatgpt/ $(MODELS_DIR) chatgpt test
