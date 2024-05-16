// *****************************************************************************
//  File:            point.h
//  Master's Thesis: Evaluating Reliability of Static Analysis Results
//                   Using Machine Learning
//  Author:          Beranek Tomas (xberan46)
//  Date:            14.5.2024
//  Up2date sources: https://github.com/TomasBeranek/but-masters-thesis
//  Description:     Part of the struct-alias experiment.
// *****************************************************************************

#ifndef POINT_H
#define POINT_H

// Define a struct for a point in 2D space
struct Point {
    int x;
    int y;
};

// Create an alias for the struct Point type
typedef struct Point MyPointStruct;

MyPointStruct addPoints(MyPointStruct pointA,  MyPointStruct pointB);

#endif
