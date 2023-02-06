#include <stdlib.h>
#include <stdio.h>
#include "add_one.h"

int f(int a) {
  return add_one(a);
}

int g(int b) {
  return b;
}

/* An unnecessarily complicated program which returns 0, if no args were passed
   otherwise it returns the number of args passed (program name included). But
   when some args are passed NULL_DEREFERENCE happens in add_one function. */
int main(int argc, char const *argv[]) {
  int x = argc - 1;
  int y;

  printf("%d\n", x);

  if (x > 0) {
    y = f(x);
  } else {
    y = g(x);
  }

  return y;
}
