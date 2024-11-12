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

all: filter-d2a

filter-d2a:
	python3.8 filter.py -d $(D2A_DIR) -o $(D2A_FILTERED_DIR)

clean-slicing-info:
	rm -rf $(SLICING_INFO_DIR)

slicing-info: clean-slicing-info
	mkdir -p $(SLICING_INFO_DIR)
	python3.8 ../inference-pipeline/slicing_criteria_extraction.py --d2a $(D2A_FILTERED_DIR)/ffmpeg_labeler_0.pickle.gz > $(SLICING_INFO_DIR)/ffmpeg_labeler_0.csv
	python3.8 ../inference-pipeline/slicing_criteria_extraction.py --d2a $(D2A_FILTERED_DIR)/ffmpeg_labeler_1.pickle.gz > $(SLICING_INFO_DIR)/ffmpeg_labeler_1.csv
	python3.8 ../inference-pipeline/slicing_criteria_extraction.py --d2a $(D2A_FILTERED_DIR)/httpd_labeler_0.pickle.gz > $(SLICING_INFO_DIR)/httpd_labeler_0.csv
	python3.8 ../inference-pipeline/slicing_criteria_extraction.py --d2a $(D2A_FILTERED_DIR)/httpd_labeler_1.pickle.gz > $(SLICING_INFO_DIR)/httpd_labeler_1.csv
	python3.8 ../inference-pipeline/slicing_criteria_extraction.py --d2a $(D2A_FILTERED_DIR)/libav_labeler_0.pickle.gz > $(SLICING_INFO_DIR)/libav_labeler_0.csv
	python3.8 ../inference-pipeline/slicing_criteria_extraction.py --d2a $(D2A_FILTERED_DIR)/libav_labeler_1.pickle.gz > $(SLICING_INFO_DIR)/libav_labeler_1.csv
	python3.8 ../inference-pipeline/slicing_criteria_extraction.py --d2a $(D2A_FILTERED_DIR)/libtiff_labeler_0.pickle.gz > $(SLICING_INFO_DIR)/libtiff_labeler_0.csv
	python3.8 ../inference-pipeline/slicing_criteria_extraction.py --d2a $(D2A_FILTERED_DIR)/libtiff_labeler_1.pickle.gz > $(SLICING_INFO_DIR)/libtiff_labeler_1.csv
	python3.8 ../inference-pipeline/slicing_criteria_extraction.py --d2a $(D2A_FILTERED_DIR)/nginx_labeler_0.pickle.gz > $(SLICING_INFO_DIR)/nginx_labeler_0.csv
	python3.8 ../inference-pipeline/slicing_criteria_extraction.py --d2a $(D2A_FILTERED_DIR)/nginx_labeler_1.pickle.gz > $(SLICING_INFO_DIR)/nginx_labeler_1.csv
	python3.8 ../inference-pipeline/slicing_criteria_extraction.py --d2a $(D2A_FILTERED_DIR)/openssl_labeler_0.pickle.gz > $(SLICING_INFO_DIR)/openssl_labeler_0.csv
	python3.8 ../inference-pipeline/slicing_criteria_extraction.py --d2a $(D2A_FILTERED_DIR)/openssl_labeler_1.pickle.gz > $(SLICING_INFO_DIR)/openssl_labeler_1.csv

download-repos:
	cd projects && git clone https://github.com/openssl/openssl
	cd projects && git clone https://github.com/libav/libav
	cd projects && git clone https://github.com/nginx/nginx
	cd projects && git clone https://gitlab.com/libtiff/libtiff
	cd projects && git clone https://github.com/apache/httpd
	cd projects && git clone https://github.com/FFmpeg/FFmpeg

bitcode-httpd-1:
	python3.8 generate_bitcode.py -f $(D2A_FILTERED_DIR)/httpd_labeler_1.pickle.gz  -r $(PROJECTS_DIR)/httpd/ --project httpd -o $(BITCODE_DIR)/httpd_1

bitcode-httpd-0:
	python3.8 generate_bitcode.py -f $(D2A_FILTERED_DIR)/httpd_labeler_0.pickle.gz  -r $(PROJECTS_DIR)/httpd/ --project httpd -o $(BITCODE_DIR)/httpd_0

bitcode-nginx-1:
	python3.8 generate_bitcode.py -f $(D2A_FILTERED_DIR)/nginx_labeler_1.pickle.gz  -r $(PROJECTS_DIR)/nginx/ --project nginx -o $(BITCODE_DIR)/nginx_1

bitcode-nginx-0:
	python3.8 generate_bitcode.py -f $(D2A_FILTERED_DIR)/nginx_labeler_0.pickle.gz  -r $(PROJECTS_DIR)/nginx/ --project nginx -o $(BITCODE_DIR)/nginx_0

bitcode-libtiff-1:
	python3.8 generate_bitcode.py -f $(D2A_FILTERED_DIR)/libtiff_labeler_1.pickle.gz  -r $(PROJECTS_DIR)/libtiff/ --project libtiff -o $(BITCODE_DIR)/libtiff_1

bitcode-libtiff-0:
	python3.8 generate_bitcode.py -f $(D2A_FILTERED_DIR)/libtiff_labeler_0.pickle.gz  -r $(PROJECTS_DIR)/libtiff/ --project libtiff -o $(BITCODE_DIR)/libtiff_0

bitcode-libav-1:
	python3.8 generate_bitcode.py -f $(D2A_FILTERED_DIR)/libav_labeler_1.pickle.gz  -r $(PROJECTS_DIR)/libav/ --project libav -o $(BITCODE_DIR)/libav_1

bitcode-libav-0:
	python3.8 generate_bitcode.py -f $(D2A_FILTERED_DIR)/libav_labeler_0.pickle.gz  -r $(PROJECTS_DIR)/libav/ --project libav -o $(BITCODE_DIR)/libav_0

bitcode-openssl-1:
	python3.8 generate_bitcode.py -f $(D2A_FILTERED_DIR)/openssl_labeler_1.pickle.gz  -r $(PROJECTS_DIR)/openssl/ --project openssl -o $(BITCODE_DIR)/openssl_1

bitcode-openssl-0:
	python3.8 generate_bitcode.py -f $(D2A_FILTERED_DIR)/openssl_labeler_0.pickle.gz  -r $(PROJECTS_DIR)/openssl/ --project openssl -o $(BITCODE_DIR)/openssl_0

bitcode-ffmpeg-1:
	python3.8 generate_bitcode.py -f $(D2A_FILTERED_DIR)/ffmpeg_labeler_1.pickle.gz  -r $(PROJECTS_DIR)/FFmpeg/ --project ffmpeg -o $(BITCODE_DIR)/ffmpeg_1

bitcode-ffmpeg-0:
	python3.8 generate_bitcode.py -f $(D2A_FILTERED_DIR)/ffmpeg_labeler_0.pickle.gz  -r $(PROJECTS_DIR)/FFmpeg/ --project ffmpeg -o $(BITCODE_DIR)/ffmpeg_0

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
	python3.8 ../dev-utils/remove_duplicates.py --d2a-dir $(D2A_DIR) --bitcode-dir $(BITCODE_DIR)

remove-invalid-symlinks:
	python3.8 ../dev-utils/remove_invalid_symlinks.py $(BITCODE_DIR)

stats:
	python3.8 ../dev-utils/stats.py -d $(D2A_DIR)
