all: scenario-chatgpt

clean:
	rm ROC_curves.svg

scenario-chatgpt: clean
	python3 ../../model/src/evaluate_model.py ../../model/schemas/extended_cpg.pbtxt . ../../model/saved_models/ chatgpt test
