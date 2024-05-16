// *****************************************************************************
//  File:            original.c
//  Master's Thesis: Evaluating Reliability of Static Analysis Results
//                   Using Machine Learning
//  Author:          Beranek Tomas (xberan46)
//  Date:            14.5.2024
//  Up2date sources: https://github.com/TomasBeranek/but-masters-thesis
//  Description:     Part of the compilation-from-D2A experiment.
// *****************************************************************************

#include <stdio.h>

#define EXPERIMENT 123

int main(int argc, char const *argv[]) {
  double x = 1.0;
  int b = 100;

#ifdef EXPERIMENT
if (argc > b){
  printf("DEFINED\nTrue\n");
  return EXPERIMENT;
} else {
  printf("DEFINED\nFalse\n");
  return EXPERIMENT;
}
#else
  if (argc > b){
    printf("NOT DEFINED\nTrue\n");
    return argc % b;
  } else {
    printf("NOT DEFINED\nFalse\n");
    return b % argc;
  }
#endif
}

int not_include(int x){
  return x + 10;
}
