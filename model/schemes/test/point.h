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
