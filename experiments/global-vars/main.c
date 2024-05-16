// *****************************************************************************
//  File:            main.c
//  Master's Thesis: Evaluating Reliability of Static Analysis Results
//                   Using Machine Learning
//  Author:          Beranek Tomas (xberan46)
//  Date:            14.5.2024
//  Up2date sources: https://github.com/TomasBeranek/but-masters-thesis
//  Description:     Part of the global-vars experiment.
// *****************************************************************************

#include <stdio.h>
#include "add.h"

extern int x;
extern int y;
int main() {

    int z = add(10, 20);
    if (z)
      return y + 1;
    else
      return x + z;
}
