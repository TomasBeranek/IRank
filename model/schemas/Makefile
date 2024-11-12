# ******************************************************************************
#  File:            Makefile
#  Master's Thesis: Evaluating Reliability of Static Analysis Results
#                   Using Machine Learning
#  Author:          Beranek Tomas (xberan46)
#  Date:            14.5.2024
#  Up2date sources: https://github.com/TomasBeranek/but-masters-thesis
#  Description:     Makefile for normalization coefficients extraction and
#                   generation of TFRecords files..
# ******************************************************************************

SLICING_INFO_DIR = ../../dataset/slicing-info
D2A_DIR = ../../dataset/d2a
GRAPH_DIR = ../../dataset/graph-d2a
TFRECORDS_DIR = ../tfrecords

.PHONY: all

all: transform-libtiff-1

# Use this as: 'time make measure-time'
measure-time:
	find $(GRAPH_DIR) -mindepth 2 -type d | head -n 50 | python3.8 feature_engineering.py extended_cpg.pbtxt ffmpeg_1.tfrecords ffmpeg 1

transform-libtiff-1:
	find $(GRAPH_DIR)/libtiff_1 -mindepth 1 -type d | python3.8 feature_engineering.py extended_cpg.pbtxt $(TFRECORDS_DIR)/libtiff_1.tfrecords libtiff 1 $(D2A_DIR)/splits.csv $(D2A_DIR)/libtiff_labeler_1.pickle.gz $(SLICING_INFO_DIR)/libtiff_labeler_1.csv

transform-libtiff-0:
	find $(GRAPH_DIR)/libtiff_0 -mindepth 1 -type d | python3.8 feature_engineering.py extended_cpg.pbtxt $(TFRECORDS_DIR)/libtiff_0.tfrecords libtiff 0 $(D2A_DIR)/splits.csv $(D2A_DIR)/libtiff_labeler_0.pickle.gz $(SLICING_INFO_DIR)/libtiff_labeler_0.csv

transform-httpd-1:
	find $(GRAPH_DIR)/httpd_1 -mindepth 1 -type d | python3.8 feature_engineering.py extended_cpg.pbtxt $(TFRECORDS_DIR)/httpd_1.tfrecords httpd 1 $(D2A_DIR)/splits.csv $(D2A_DIR)/httpd_labeler_1.pickle.gz $(SLICING_INFO_DIR)/httpd_labeler_1.csv

transform-httpd-0:
	find $(GRAPH_DIR)/httpd_0 -mindepth 1 -type d | python3.8 feature_engineering.py extended_cpg.pbtxt $(TFRECORDS_DIR)/httpd_0.tfrecords httpd 0 $(D2A_DIR)/splits.csv $(D2A_DIR)/httpd_labeler_0.pickle.gz $(SLICING_INFO_DIR)/httpd_labeler_0.csv

transform-nginx-1:
	find $(GRAPH_DIR)/nginx_1 -mindepth 1 -type d | python3.8 feature_engineering.py extended_cpg.pbtxt $(TFRECORDS_DIR)/nginx_1.tfrecords nginx 1 $(D2A_DIR)/splits.csv $(D2A_DIR)/nginx_labeler_1.pickle.gz $(SLICING_INFO_DIR)/nginx_labeler_1.csv

transform-nginx-0:
	find $(GRAPH_DIR)/nginx_0 -mindepth 1 -type d | python3.8 feature_engineering.py extended_cpg.pbtxt $(TFRECORDS_DIR)/nginx_0.tfrecords nginx 0 $(D2A_DIR)/splits.csv $(D2A_DIR)/nginx_labeler_0.pickle.gz $(SLICING_INFO_DIR)/nginx_labeler_0.csv

transform-libav-1:
	find $(GRAPH_DIR)/libav_1 -mindepth 1 -type d | python3.8 feature_engineering.py extended_cpg.pbtxt $(TFRECORDS_DIR)/libav_1.tfrecords libav 1 $(D2A_DIR)/splits.csv $(D2A_DIR)/libav_labeler_1.pickle.gz $(SLICING_INFO_DIR)/libav_labeler_1.csv

transform-libav-0:
	find $(GRAPH_DIR)/libav_0 -mindepth 1 -type d | python3.8 feature_engineering.py extended_cpg.pbtxt $(TFRECORDS_DIR)/libav_0.tfrecords libav 0 $(D2A_DIR)/splits.csv $(D2A_DIR)/libav_labeler_0.pickle.gz $(SLICING_INFO_DIR)/libav_labeler_0.csv

extract-norm-coeffs-libtiff:
	python3.8 find_normalization_coefficients.py $(GRAPH_DIR)/libtiff_0 $(GRAPH_DIR)/libtiff_1 libtiff $(D2A_DIR)/splits.csv $(SLICING_INFO_DIR)/libtiff_labeler_0.csv $(SLICING_INFO_DIR)/libtiff_labeler_1.csv

extract-norm-coeffs-httpd:
	python3.8 find_normalization_coefficients.py $(GRAPH_DIR)/httpd_0 $(GRAPH_DIR)/httpd_1 httpd $(D2A_DIR)/splits.csv $(SLICING_INFO_DIR)/httpd_labeler_0.csv $(SLICING_INFO_DIR)/httpd_labeler_1.csv

extract-norm-coeffs-nginx:
	python3.8 find_normalization_coefficients.py $(GRAPH_DIR)/nginx_0 $(GRAPH_DIR)/nginx_1 nginx $(D2A_DIR)/splits.csv $(SLICING_INFO_DIR)/nginx_labeler_0.csv $(SLICING_INFO_DIR)/nginx_labeler_1.csv
