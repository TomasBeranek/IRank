#include <original.c>

int main(int argc, char const *argv[]) {
  double x = 1.0;
  int b = 100;

#ifdef EXPERIMENT
if (argc > b){
  printf("DEFINED\nTrue\n");
  return EXPERIMENT;
} else {
  printf("DEFINED\nFalse\n");
  return EXPERIMENT;
}
#else
  if (argc > b){
    printf("NOT DEFINED\nTrue\n");
    return argc % b;
  } else {
    printf("NOT DEFINED\nFalse\n");
    return b % argc;
  }
#endif
}
