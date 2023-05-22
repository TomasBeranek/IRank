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
	python3 construct_phase_d2a.py -f d2a-filtered/nginx_labeler_1.pickle.gz  -r projects/nginx/ -s slicing-info/nginx_labeler_1.csv
