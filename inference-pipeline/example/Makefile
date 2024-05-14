# ******************************************************************************
#  File: 						Makefile
#  Master's Thesis: Evaluating Reliability of Static Analysis Results
#                   Using Machine Learning
#  Author: 					Beranek Tomas (xberan46)
#  Date: 						14.5.2024
#  Up2date sources: https://github.com/TomasBeranek/but-masters-thesis
#  Description:			Makefile for compiling a project to demonstrate the
#										inference pipeline.
# ******************************************************************************

CC = gcc
CFLAGS = -I.
DEPS = add_one.h
OBJ = main.o add_one.o
BIN = a.out

.PHONY: all install clean

all: install

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

install: $(OBJ)
	$(CC) -o $(BIN) $^ $(CFLAGS)

clean:
	rm -rf *.ll *.o a.out bitcode.* *.bc infer-out/ out/ workspace/
