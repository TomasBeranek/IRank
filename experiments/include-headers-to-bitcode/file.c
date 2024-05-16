// *****************************************************************************
//  File:            file.c
//  Master's Thesis: Evaluating Reliability of Static Analysis Results
//                   Using Machine Learning
//  Author:          Beranek Tomas (xberan46)
//  Date:            14.5.2024
//  Up2date sources: https://github.com/TomasBeranek/but-masters-thesis
//  Description:     Part of the include-headers-to-bitcode experiment.
// *****************************************************************************

#include "header.h"


int fun1(int idx) {
  int arr[10];
  arr[idx] = 42*idx; /* Bufferoverrun happens here */
  return arr[idx];
}


/* 'main' is here only to test if the includes work despite C99 warning */
int main() {
  int y = fun2();
  return y;
}
