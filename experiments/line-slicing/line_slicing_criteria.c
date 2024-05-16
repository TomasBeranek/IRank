// *****************************************************************************
//  File:            line_slicing_criteria.c
//  Master's Thesis: Evaluating Reliability of Static Analysis Results
//                   Using Machine Learning
//  Author:          Beranek Tomas (xberan46)
//  Date:            14.5.2024
//  Up2date sources: https://github.com/TomasBeranek/but-masters-thesis
//  Description:     Part of the line-slicing experiment.
// *****************************************************************************

#include <stdio.h>

int fun(int x, int *y){
  int z = x + (*y);
  return z;
}

int main(int argc, char const *argv[]) {
  // Part 1
  int a = argc + 10;
  int *p = &a;
  int b = 123;
  int c = fun(a, p);

  // Part 2
  int e = argc * 12;
  //int *q = &e;
  int f = 567;
  int g = 0; //fun(e, q);

  // Depends on part 2
  int h = g * f;
  // Depends on part 1
  int d = c - b;

  // Depends on both
  if ((d > 0) && (h > 0))
    return 0;
  else
    return 1;
}

// More information about the experiment is in the Makefile
