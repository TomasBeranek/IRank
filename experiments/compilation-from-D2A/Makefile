# ******************************************************************************
#  File:            Makefile
#  Master's Thesis: Evaluating Reliability of Static Analysis Results
#                   Using Machine Learning
#  Author:          Beranek Tomas (xberan46)
#  Date:            14.5.2024
#  Up2date sources: https://github.com/TomasBeranek/but-masters-thesis
#  Description:     Part of the compilation-from-D2A experiment.
# ******************************************************************************

# This expriment shows that it is not possible to simply extract code from
# 'functions' records in D2A and use it as compilation input. In order to
# successfully compile it (to generate LLVM bitcode) it is necessary to include
# all the #define and other symbols defined in the original .c file that contain
# the function. However, attempting to include this file will cause its contents
# to be copied (this is the #include property in C) into an artificially created
# file containing the function. This copying, however, creates duplicate symbols
# that prevent compilation.

# To start the experiment, run 'make'

all: compile

clean:
	rm -f a.out

compile: clean
	clang extracted_function.c
