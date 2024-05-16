// *****************************************************************************
//  File:            add.c
//  Master's Thesis: Evaluating Reliability of Static Analysis Results
//                   Using Machine Learning
//  Author:          Beranek Tomas (xberan46)
//  Date:            14.5.2024
//  Up2date sources: https://github.com/TomasBeranek/but-masters-thesis
//  Description:     Part of the global-vars experiment.
// *****************************************************************************

#include "add.h"
#include <stdio.h>

int x;
int y;
int add(int a, int b) {
  x = getchar();
  y = getchar();
  return a + b;
}
