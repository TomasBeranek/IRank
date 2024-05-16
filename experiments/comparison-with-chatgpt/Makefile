# ******************************************************************************
#  File:            Makefile
#  Master's Thesis: Evaluating Reliability of Static Analysis Results
#                   Using Machine Learning
#  Author:          Beranek Tomas (xberan46)
#  Date:            14.5.2024
#  Up2date sources: https://github.com/TomasBeranek/but-masters-thesis
#  Description:     Part of the comparison-with-chatgpt experiment.
# ******************************************************************************

all: scenario-chatgpt

clean:
	rm ROC_curves.svg

scenario-chatgpt: clean
	python3 ../../model/src/evaluate_model.py ../../model/schemas/extended_cpg.pbtxt . ../../model/saved_models/ chatgpt test
