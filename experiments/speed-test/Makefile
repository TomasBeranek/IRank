# This experiment shows a comparison of the speed of recursively searching a project
# versus just searching the current folder. This search is necessary to find all
# the bitcode files created for each sample. On the FFmpeg project, the speedup
# is ~400x, which for ~1,300,000M samples will make a difference of several hours
# (in single-threaded mode).

# To start the experiment, run "make" (the default project is FFmpeg)

run: FFmpeg

FFmpeg:
	python3 speed_test.py ../../dataset/projects/FFmpeg

httpd:
	python3 speed_test.py ../../dataset/projects/httpd

libav:
	python3 speed_test.py ../../dataset/projects/libav

libtiff:
	python3 speed_test.py ../../dataset/projects/libtiff

nginx:
	python3 speed_test.py ../../dataset/projects/nginx

openssl:
	python3 speed_test.py ../../dataset/projects/openssl
