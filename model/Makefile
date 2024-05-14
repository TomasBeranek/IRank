# ******************************************************************************
#  File:            Makefile
#  Master's Thesis: Evaluating Reliability of Static Analysis Results
#                   Using Machine Learning
#  Author:          Beranek Tomas (xberan46)
#  Date:            14.5.2024
#  Up2date sources: https://github.com/TomasBeranek/but-masters-thesis
#  Description:     Makefile for model training and testing.
# ******************************************************************************

SCHEMA_PATH=schemas/extended_cpg.pbtxt
MODELS_DIR=saved_models
DATASET_DIR=tfrecords
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
	python3 src/evaluate_model.py $(SCHEMA_PATH) ../experiments/comparison-with-chatgpt/ $(MODELS_DIR) chatgpt test
