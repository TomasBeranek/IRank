#include "header.h"


int fun1(int idx) {
  int arr[10];
  arr[idx] = 42*idx; /* Bufferoverrun happens here */
  return arr[idx];
}


/* 'main' is here only to test if the includes work despite C99 warning */
int main() {
  int y = fun2();
  return y;
}
