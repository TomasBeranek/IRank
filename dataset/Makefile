GRAPH_DIR = ../D2A-CPG
SLICING_INFO_DIR = slicing-info
BITCODE_DIR = d2a-bitcode

all:

clean-slicing-info:
	rm -rf slicing-info

slicing-info: clean-slicing-info
	mkdir -p slicing-info
	python3 ../code-extraction/slicing_criteria_extraction.py --d2a d2a-filtered/ffmpeg_labeler_0.pickle.gz > slicing-info/ffmpeg_labeler_0.csv
	python3 ../code-extraction/slicing_criteria_extraction.py --d2a d2a-filtered/ffmpeg_labeler_1.pickle.gz > slicing-info/ffmpeg_labeler_1.csv
	python3 ../code-extraction/slicing_criteria_extraction.py --d2a d2a-filtered/httpd_labeler_0.pickle.gz > slicing-info/httpd_labeler_0.csv
	python3 ../code-extraction/slicing_criteria_extraction.py --d2a d2a-filtered/httpd_labeler_1.pickle.gz > slicing-info/httpd_labeler_1.csv
	python3 ../code-extraction/slicing_criteria_extraction.py --d2a d2a-filtered/libav_labeler_0.pickle.gz > slicing-info/libav_labeler_0.csv
	python3 ../code-extraction/slicing_criteria_extraction.py --d2a d2a-filtered/libav_labeler_1.pickle.gz > slicing-info/libav_labeler_1.csv
	python3 ../code-extraction/slicing_criteria_extraction.py --d2a d2a-filtered/libtiff_labeler_0.pickle.gz > slicing-info/libtiff_labeler_0.csv
	python3 ../code-extraction/slicing_criteria_extraction.py --d2a d2a-filtered/libtiff_labeler_1.pickle.gz > slicing-info/libtiff_labeler_1.csv
	python3 ../code-extraction/slicing_criteria_extraction.py --d2a d2a-filtered/nginx_labeler_0.pickle.gz > slicing-info/nginx_labeler_0.csv
	python3 ../code-extraction/slicing_criteria_extraction.py --d2a d2a-filtered/nginx_labeler_1.pickle.gz > slicing-info/nginx_labeler_1.csv
	python3 ../code-extraction/slicing_criteria_extraction.py --d2a d2a-filtered/openssl_labeler_0.pickle.gz > slicing-info/openssl_labeler_0.csv
	python3 ../code-extraction/slicing_criteria_extraction.py --d2a d2a-filtered/openssl_labeler_1.pickle.gz > slicing-info/openssl_labeler_1.csv

nginx-1:
	python3 generate_bitcode.py -f d2a-filtered/nginx_labeler_1.pickle.gz  -r projects/nginx/ nginx_labeler_1.csv

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
	python3 remove_duplicates.py --d2a-dir d2a --bitcode-dir d2a-bitcode

# remove symlinks with non-existent targets (removed 877 invalid links from original bitcode)
remove-invalid-symlinks:
	python3 remove_invalid_symlinks.py d2a-bitcode

# Removes specified already generated samples
remove-samples:
	awk -F, 'NR>=10 && NR<=50 {print "rm -rf ../D2A-CPG/httpd_1/"$$2}' $(SLICING_INFO_DIR)/httpd_labeler_1.csv | sh
