#include <stdio.h>
#include "add.h"


int main() {
    int z = add(10, 20);
    if (z)
      return 1;
    else
      return 2;
}
