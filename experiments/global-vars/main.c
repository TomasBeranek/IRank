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
