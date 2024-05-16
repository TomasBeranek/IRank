// *****************************************************************************
//  File:            main.c
//  Master's Thesis: Evaluating Reliability of Static Analysis Results
//                   Using Machine Learning
//  Author:          Beranek Tomas (xberan46)
//  Date:            14.5.2024
//  Up2date sources: https://github.com/TomasBeranek/but-masters-thesis
//  Description:     Part of the arg-passing experiment.
// *****************************************************************************

#include <stdio.h>
#include "add.h"


int main() {
    int z = add(10, 20);
    if (z)
      return 1;
    else
      return 2;
}
