#include <stdio.h>
#include <stdlib.h>

int* f(int argc){
  /********************** Start of the bug **********************/
  int* p;
  int x;

  // Nondeterministic if construct
  if (argc > 1) {
    p = NULL;
  } else {
    p = &x;
  }
  /**************************************************************/

  return p;
}

int main(int argc, char const *argv[]) {
  int* a;

  a = f(argc);

  /****************** Manifestation of the bug ******************/
  int y = 2;
  // NULL_DEREFERENCE happens when some args were passed to the program
  *a = y;
  /**************************************************************/

  return 0;
}
