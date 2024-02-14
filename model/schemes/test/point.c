#include "point.h"

MyPointStruct addPoints(MyPointStruct pointA, MyPointStruct pointB) {
  MyPointStruct resultPoint;
  resultPoint.x = pointA.x + pointB.x;
  resultPoint.y = pointA.y + pointB.y;
  return resultPoint;
}
