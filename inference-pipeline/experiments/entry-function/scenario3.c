#include <stdio.h>
#include <stdlib.h>


int f(int argc){
  /***************** Has some effect on the bug *****************/
  // Nondeterministic if construct
  if (argc > 1) {
    return 0;
  } else {
    return 1;
  }
  /**************************************************************/
}

int main(int argc, char const *argv[]) {
  /********************** Start of the bug **********************/
  int y = 2;
  int* p = NULL;
  /**************************************************************/

  int condition;
  condition = f(argc);

  /****************** Manifestation of the bug ******************/
  if (condition) {
    y += 1;
  } else {
    // NULL_DEREFERENCE happens when some args were passed to the program
    *p = y;
  }
  /**************************************************************/

  return y;
}
