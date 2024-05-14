# ******************************************************************************
#  File:            Makefile
#  Master's Thesis: Evaluating Reliability of Static Analysis Results
#                   Using Machine Learning
#  Author:          Beranek Tomas (xberan46)
#  Date:            14.5.2024
#  Up2date sources: https://github.com/TomasBeranek/but-masters-thesis
#  Description:     Makefile for generating LLVM bitcode, Graph D2A and slicing
#                   criteria.
# ******************************************************************************

GRAPH_DIR = graph-d2a
SLICING_INFO_DIR = slicing-info
BITCODE_DIR = d2a-bitcode
D2A_FILTERED_DIR = d2a-filtered
D2A_DIR = d2a
PROJECTS_DIR = projects

all: slicing-info

clean-slicing-info:
	rm -rf $(SLICING_INFO_DIR)

slicing-info: clean-slicing-info
	mkdir -p $(SLICING_INFO_DIR)
	python3 ../inference-pipeline/slicing_criteria_extraction.py --d2a $(D2A_FILTERED_DIR)/ffmpeg_labeler_0.pickle.gz > $(SLICING_INFO_DIR)/ffmpeg_labeler_0.csv
	python3 ../inference-pipeline/slicing_criteria_extraction.py --d2a $(D2A_FILTERED_DIR)/ffmpeg_labeler_1.pickle.gz > $(SLICING_INFO_DIR)/ffmpeg_labeler_1.csv
	python3 ../inference-pipeline/slicing_criteria_extraction.py --d2a $(D2A_FILTERED_DIR)/httpd_labeler_0.pickle.gz > $(SLICING_INFO_DIR)/httpd_labeler_0.csv
	python3 ../inference-pipeline/slicing_criteria_extraction.py --d2a $(D2A_FILTERED_DIR)/httpd_labeler_1.pickle.gz > $(SLICING_INFO_DIR)/httpd_labeler_1.csv
	python3 ../inference-pipeline/slicing_criteria_extraction.py --d2a $(D2A_FILTERED_DIR)/libav_labeler_0.pickle.gz > $(SLICING_INFO_DIR)/libav_labeler_0.csv
	python3 ../inference-pipeline/slicing_criteria_extraction.py --d2a $(D2A_FILTERED_DIR)/libav_labeler_1.pickle.gz > $(SLICING_INFO_DIR)/libav_labeler_1.csv
	python3 ../inference-pipeline/slicing_criteria_extraction.py --d2a $(D2A_FILTERED_DIR)/libtiff_labeler_0.pickle.gz > $(SLICING_INFO_DIR)/libtiff_labeler_0.csv
	python3 ../inference-pipeline/slicing_criteria_extraction.py --d2a $(D2A_FILTERED_DIR)/libtiff_labeler_1.pickle.gz > $(SLICING_INFO_DIR)/libtiff_labeler_1.csv
	python3 ../inference-pipeline/slicing_criteria_extraction.py --d2a $(D2A_FILTERED_DIR)/nginx_labeler_0.pickle.gz > $(SLICING_INFO_DIR)/nginx_labeler_0.csv
	python3 ../inference-pipeline/slicing_criteria_extraction.py --d2a $(D2A_FILTERED_DIR)/nginx_labeler_1.pickle.gz > $(SLICING_INFO_DIR)/nginx_labeler_1.csv
	python3 ../inference-pipeline/slicing_criteria_extraction.py --d2a $(D2A_FILTERED_DIR)/openssl_labeler_0.pickle.gz > $(SLICING_INFO_DIR)/openssl_labeler_0.csv
	python3 ../inference-pipeline/slicing_criteria_extraction.py --d2a $(D2A_FILTERED_DIR)/openssl_labeler_1.pickle.gz > $(SLICING_INFO_DIR)/openssl_labeler_1.csv

httpd-1:
	python3 generate_bitcode.py -f $(D2A_FILTERED_DIR)/httpd_labeler_1.pickle.gz  -r $(PROJECTS_DIR)/httpd/ --project httpd -o $(BITCODE_DIR)/httpd_1

httpd-0:
	python3 generate_bitcode.py -f $(D2A_FILTERED_DIR)/httpd_labeler_0.pickle.gz  -r $(PROJECTS_DIR)/httpd/ --project httpd -o $(BITCODE_DIR)/httpd_0

nginx-1:
	python3 generate_bitcode.py -f $(D2A_FILTERED_DIR)/nginx_labeler_1.pickle.gz  -r $(PROJECTS_DIR)/nginx/ --project nginx -o $(BITCODE_DIR)/nginx_1

nginx-0:
	python3 generate_bitcode.py -f $(D2A_FILTERED_DIR)/nginx_labeler_0.pickle.gz  -r $(PROJECTS_DIR)/nginx/ --project nginx -o $(BITCODE_DIR)/nginx_0

libtiff-1:
	python3 generate_bitcode.py -f $(D2A_FILTERED_DIR)/libtiff_labeler_1.pickle.gz  -r $(PROJECTS_DIR)/libtiff/ --project libtiff -o $(BITCODE_DIR)/libtiff_1

libtiff-0:
	python3 generate_bitcode.py -f $(D2A_FILTERED_DIR)/libtiff_labeler_0.pickle.gz  -r $(PROJECTS_DIR)/libtiff/ --project libtiff -o $(BITCODE_DIR)/libtiff_0

libav-1:
	python3 generate_bitcode.py -f $(D2A_FILTERED_DIR)/libav_labeler_1.pickle.gz  -r $(PROJECTS_DIR)/libav/ --project libav -o $(BITCODE_DIR)/libav_1

libav-0:
	python3 generate_bitcode.py -f $(D2A_FILTERED_DIR)/libav_labeler_0.pickle.gz  -r $(PROJECTS_DIR)/libav/ --project libav -o $(BITCODE_DIR)/libav_0

openssl-1:
	python3 generate_bitcode.py -f $(D2A_FILTERED_DIR)/openssl_labeler_1.pickle.gz  -r $(PROJECTS_DIR)/openssl/ --project openssl -o $(BITCODE_DIR)/openssl_1

openssl-0:
	python3 generate_bitcode.py -f $(D2A_FILTERED_DIR)/openssl_labeler_0.pickle.gz  -r $(PROJECTS_DIR)/openssl/ --project openssl -o $(BITCODE_DIR)/openssl_0

ffmpeg-1:
	python3 generate_bitcode.py -f $(D2A_FILTERED_DIR)/ffmpeg_labeler_1.pickle.gz  -r $(PROJECTS_DIR)/FFmpeg/ --project ffmpeg -o $(BITCODE_DIR)/ffmpeg_1

ffmpeg-0:
	python3 generate_bitcode.py -f $(D2A_FILTERED_DIR)/ffmpeg_labeler_0.pickle.gz  -r $(PROJECTS_DIR)/FFmpeg/ --project ffmpeg -o $(BITCODE_DIR)/ffmpeg_0

graph-httpd-1:
	./construction_phase_d2a $(GRAPH_DIR)/httpd_1 $(SLICING_INFO_DIR)/httpd_labeler_1.csv $(BITCODE_DIR)/httpd_1

graph-httpd-0:
	./construction_phase_d2a $(GRAPH_DIR)/httpd_0 $(SLICING_INFO_DIR)/httpd_labeler_0.csv $(BITCODE_DIR)/httpd_0

graph-nginx-1:
	./construction_phase_d2a $(GRAPH_DIR)/nginx_1 $(SLICING_INFO_DIR)/nginx_labeler_1.csv $(BITCODE_DIR)/nginx_1

graph-nginx-0:
	./construction_phase_d2a $(GRAPH_DIR)/nginx_0 $(SLICING_INFO_DIR)/nginx_labeler_0.csv $(BITCODE_DIR)/nginx_0

graph-libtiff-1:
	./construction_phase_d2a $(GRAPH_DIR)/libtiff_1 $(SLICING_INFO_DIR)/libtiff_labeler_1.csv $(BITCODE_DIR)/libtiff_1

graph-libtiff-0:
	./construction_phase_d2a $(GRAPH_DIR)/libtiff_0 $(SLICING_INFO_DIR)/libtiff_labeler_0.csv $(BITCODE_DIR)/libtiff_0

graph-libav-1:
	./construction_phase_d2a $(GRAPH_DIR)/libav_1 $(SLICING_INFO_DIR)/libav_labeler_1.csv $(BITCODE_DIR)/libav_1

graph-libav-0:
	./construction_phase_d2a $(GRAPH_DIR)/libav_0 $(SLICING_INFO_DIR)/libav_labeler_0.csv $(BITCODE_DIR)/libav_0

graph-openssl-1:
	./construction_phase_d2a $(GRAPH_DIR)/openssl_1 $(SLICING_INFO_DIR)/openssl_labeler_1.csv $(BITCODE_DIR)/openssl_1

graph-openssl-0:
	./construction_phase_d2a $(GRAPH_DIR)/openssl_0 $(SLICING_INFO_DIR)/openssl_labeler_0.csv $(BITCODE_DIR)/openssl_0

graph-ffmpeg-1:
	./construction_phase_d2a $(GRAPH_DIR)/ffmpeg_1 $(SLICING_INFO_DIR)/ffmpeg_labeler_1.csv $(BITCODE_DIR)/ffmpeg_1

graph-ffmpeg-0:
	./construction_phase_d2a $(GRAPH_DIR)/ffmpeg_0 $(SLICING_INFO_DIR)/ffmpeg_labeler_0.csv $(BITCODE_DIR)/ffmpeg_0

remove-duplicates:
	python3 ../dev-utils/remove_duplicates.py --d2a-dir $(D2A_DIR) --bitcode-dir $(BITCODE_DIR)

remove-invalid-symlinks:
	python3 ../dev-utils/remove_invalid_symlinks.py $(BITCODE_DIR)

stats:
	python3 ../dev-utils/stats.py -d $(D2A_DIR)
