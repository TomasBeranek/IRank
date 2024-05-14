int fun3() {
  int i = 13221548; /* This causes the bufferoverrun */
  int x = fun1(i); /* Infer reports the bufferoverrun here */
  return x;
}

int fun2() {
  int i = 12345; /* This causes the bufferoverrun */
  int x = fun1(i); /* Infer reports the bufferoverrun here */
  return x;
}
