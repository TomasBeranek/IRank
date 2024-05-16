// *****************************************************************************
//  File:            scenario2.c
//  Master's Thesis: Evaluating Reliability of Static Analysis Results
//                   Using Machine Learning
//  Author:          Beranek Tomas (xberan46)
//  Date:            14.5.2024
//  Up2date sources: https://github.com/TomasBeranek/but-masters-thesis
//  Description:     Part of the entry-function experiment.
// *****************************************************************************

#include <stdio.h>
#include <stdlib.h>

int* f(){
  /********************** Start of the bug **********************/
  int* p;
  int x;

  // Nondeterministic if construct
  if (getchar() > 1) {
    p = NULL;
  } else {
    p = &x;
  }
  /**************************************************************/

  return p;
}

int main(int argc, char const *argv[]) {
  int* a;

  a = f();

  /****************** Manifestation of the bug ******************/
  int y = 2;
  // NULL_DEREFERENCE happens when some args were passed to the program
  *a = y;
  /**************************************************************/

  return 0;
}
