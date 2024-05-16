// *****************************************************************************
//  File:            point.c
//  Master's Thesis: Evaluating Reliability of Static Analysis Results
//                   Using Machine Learning
//  Author:          Beranek Tomas (xberan46)
//  Date:            14.5.2024
//  Up2date sources: https://github.com/TomasBeranek/but-masters-thesis
//  Description:     Part of the struct-alias experiment.
// *****************************************************************************

#include "point.h"

MyPointStruct addPoints(MyPointStruct pointA, MyPointStruct pointB) {
  MyPointStruct resultPoint;
  resultPoint.x = pointA.x + pointB.x;
  resultPoint.y = pointA.y + pointB.y;
  return resultPoint;
}
