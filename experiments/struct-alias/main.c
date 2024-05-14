#include <stdio.h>
#include "point.h"


int main() {
    // Create instances of Point using the alias
    MyPointStruct p1, p2, newPoint;

    // Assign values
    p1.x = 10;
    p1.y = 20;
    p2.x = 20;
    p2.y = 10;

    // Call function which sums coordinates of both points
    newPoint = addPoints(p1, p2);

    // Print the values
    printf("Point coordinates: (%d, %d)\n", newPoint.x, newPoint.y);

    return 0;
}
