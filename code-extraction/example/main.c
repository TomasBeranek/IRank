#include <stdlib.h>
#include <stdio.h>

int f(int a) {
  return a + 1;
}

int g(int b) {
  return b;
}

/* An unnecessarily complicated program which returns 0, if no args were passed
   otherwise it returns the number of args passed (program name included). */
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
