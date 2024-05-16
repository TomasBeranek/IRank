// *****************************************************************************
//  File:            header.h
//  Master's Thesis: Evaluating Reliability of Static Analysis Results
//                   Using Machine Learning
//  Author:          Beranek Tomas (xberan46)
//  Date:            14.5.2024
//  Up2date sources: https://github.com/TomasBeranek/but-masters-thesis
//  Description:     Part of the include-headers-to-bitcode experiment.
// *****************************************************************************

int fun3() {
  int i = 13221548; /* This causes the bufferoverrun */
  int x = fun1(i); /* Infer reports the bufferoverrun here */
  return x;
}

int fun2() {
  int i = 12345; /* This causes the bufferoverrun */
  int x = fun1(i); /* Infer reports the bufferoverrun here */
  return x;
}
