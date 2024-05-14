EXAMPLES_DIR = examples
EXAMPLE_STRUCT_ALIAS_DIR = $(EXAMPLES_DIR)/struct-alias
EXAMPLE_ARG_PASSING_DIR = $(EXAMPLES_DIR)/arg-passing
EXAMPLE_GLOBAL_VARS_DIR = $(EXAMPLES_DIR)/global-vars
GRAPH_DIR = ../../dataset/graph-d2a
TFRECORDS_DIR = ../tfrecords

.PHONY: all

all: transform-libtiff-1

# Use this as: 'time make measure-time'
measure-time:
	find $(GRAPH_DIR) -mindepth 2 -type d | head -n 50 | python3 feature_engineering.py mixed_nodes/extended_cpg.pbtxt ffmpeg_1.tfrecords ffmpeg 1

transform-libtiff-1:
	find $(GRAPH_DIR)/libtiff_1 -mindepth 1 -type d | python3 feature_engineering.py mixed_nodes/extended_cpg.pbtxt $(TFRECORDS_DIR)/libtiff_1.tfrecords libtiff 1 ../../dataset/d2a/splits.csv ../../dataset/d2a/libtiff_labeler_1.pickle.gz ../../dataset/slicing-info/libtiff_labeler_1.csv

transform-libtiff-0:
	find $(GRAPH_DIR)/libtiff_0 -mindepth 1 -type d | python3 feature_engineering.py mixed_nodes/extended_cpg.pbtxt $(TFRECORDS_DIR)/libtiff_0.tfrecords libtiff 0 ../../dataset/d2a/splits.csv ../../dataset/d2a/libtiff_labeler_0.pickle.gz ../../dataset/slicing-info/libtiff_labeler_0.csv

transform-httpd-1:
	find $(GRAPH_DIR)/httpd_1 -mindepth 1 -type d | python3 feature_engineering.py mixed_nodes/extended_cpg.pbtxt $(TFRECORDS_DIR)/httpd_1.tfrecords httpd 1 ../../dataset/d2a/splits.csv ../../dataset/d2a/httpd_labeler_1.pickle.gz ../../dataset/slicing-info/httpd_labeler_1.csv

transform-httpd-0:
	find $(GRAPH_DIR)/httpd_0 -mindepth 1 -type d | python3 feature_engineering.py mixed_nodes/extended_cpg.pbtxt $(TFRECORDS_DIR)/httpd_0.tfrecords httpd 0 ../../dataset/d2a/splits.csv ../../dataset/d2a/httpd_labeler_0.pickle.gz ../../dataset/slicing-info/httpd_labeler_0.csv

transform-nginx-1:
	find $(GRAPH_DIR)/nginx_1 -mindepth 1 -type d | python3 feature_engineering.py mixed_nodes/extended_cpg.pbtxt $(TFRECORDS_DIR)/nginx_1.tfrecords nginx 1 ../../dataset/d2a/splits.csv ../../dataset/d2a/nginx_labeler_1.pickle.gz ../../dataset/slicing-info/nginx_labeler_1.csv

transform-nginx-0:
	find $(GRAPH_DIR)/nginx_0 -mindepth 1 -type d | python3 feature_engineering.py mixed_nodes/extended_cpg.pbtxt $(TFRECORDS_DIR)/nginx_0.tfrecords nginx 0 ../../dataset/d2a/splits.csv ../../dataset/d2a/nginx_labeler_0.pickle.gz ../../dataset/slicing-info/nginx_labeler_0.csv

transform-libav-1:
	find $(GRAPH_DIR)/libav_1 -mindepth 1 -type d | python3 feature_engineering.py mixed_nodes/extended_cpg.pbtxt $(TFRECORDS_DIR)/libav_1.tfrecords libav 1 ../../dataset/d2a/splits.csv ../../dataset/d2a/libav_labeler_1.pickle.gz ../../dataset/slicing-info/libav_labeler_1.csv

transform-libav-0:
	find $(GRAPH_DIR)/libav_0 -mindepth 1 -type d | python3 feature_engineering.py mixed_nodes/extended_cpg.pbtxt $(TFRECORDS_DIR)/libav_0.tfrecords libav 0 ../../dataset/d2a/splits.csv ../../dataset/d2a/libav_labeler_0.pickle.gz ../../dataset/slicing-info/libav_labeler_0.csv

clean-example-global-vars:
	rm -rf $(EXAMPLE_GLOBAL_VARS_DIR)/main.bc $(EXAMPLE_GLOBAL_VARS_DIR)/add.bc $(EXAMPLE_GLOBAL_VARS_DIR)/combined.bc $(EXAMPLE_GLOBAL_VARS_DIR)/combined.cpg.bin.zip $(EXAMPLE_GLOBAL_VARS_DIR)/graph $(EXAMPLE_GLOBAL_VARS_DIR)/workspace $(EXAMPLE_GLOBAL_VARS_DIR)/out $(EXAMPLE_GLOBAL_VARS_DIR)/combined.ll

# This target visualizes graph of custom code in 'model/schemas/examples/arg-passing'
example3: example-global-vars
example-global-vars: clean-example-global-vars
	clang -c -emit-llvm -g -grecord-command-line -fno-inline-functions -fno-builtin -O0 $(EXAMPLE_GLOBAL_VARS_DIR)/main.c -o $(EXAMPLE_GLOBAL_VARS_DIR)/main.bc
	clang -c -emit-llvm -g -grecord-command-line -fno-inline-functions -fno-builtin -O0 $(EXAMPLE_GLOBAL_VARS_DIR)/add.c -o $(EXAMPLE_GLOBAL_VARS_DIR)/add.bc
	llvm-link $(EXAMPLE_GLOBAL_VARS_DIR)/main.bc $(EXAMPLE_GLOBAL_VARS_DIR)/add.bc -o $(EXAMPLE_GLOBAL_VARS_DIR)/combined.bc
	llvm2cpg $(EXAMPLE_GLOBAL_VARS_DIR)/combined.bc --output=$(EXAMPLE_GLOBAL_VARS_DIR)/combined.cpg.bin.zip
	cd $(EXAMPLE_GLOBAL_VARS_DIR) && joern --script joern_script && llvm-dis combined.bc
	joern-export --repr all --format neo4jcsv -o $(EXAMPLE_GLOBAL_VARS_DIR)/graph $(EXAMPLE_GLOBAL_VARS_DIR)/workspace/combined.cpg.bin.zip/cpg.bin
	python3 feature_engineering.py $(EXAMPLE_GLOBAL_VARS_DIR)/graph

extract-norm-coeffs-libtiff:
	python3 find_normalization_coefficients.py $(GRAPH_DIR)/libtiff_0 $(GRAPH_DIR)/libtiff_1 libtiff ../../dataset/d2a/splits.csv ../../dataset/slicing-info/libtiff_labeler_0.csv ../../dataset/slicing-info/libtiff_labeler_1.csv

extract-norm-coeffs-httpd:
	python3 find_normalization_coefficients.py $(GRAPH_DIR)/httpd_0 $(GRAPH_DIR)/httpd_1 httpd ../../dataset/d2a/splits.csv ../../dataset/slicing-info/httpd_labeler_0.csv ../../dataset/slicing-info/httpd_labeler_1.csv

extract-norm-coeffs-nginx:
	python3 find_normalization_coefficients.py $(GRAPH_DIR)/nginx_0 $(GRAPH_DIR)/nginx_1 nginx ../../dataset/d2a/splits.csv ../../dataset/slicing-info/nginx_labeler_0.csv ../../dataset/slicing-info/nginx_labeler_1.csv
