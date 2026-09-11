
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define N 1024

int main(void) {
  static double a[N];
#pragma omp target enter data map(tofrom : a[0 : N])
  return 0;
}
