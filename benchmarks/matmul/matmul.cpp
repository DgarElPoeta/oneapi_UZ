#include <omp.h>

Matmul* matmul = reinterpret_cast<Matmul*>(opts->p_problem);
int mmsize = matmul->size;

//#define THREAD_NUM 4
//omp_set_thread_num(THREAD_NUM);

//#pragma omp parallel for
//for (size_t i = 0; i < size; ++i) {
for (size_t i = offset; i < offset+size; ++i) {
  for (size_t j = 0; j < mmsize; ++j) {
    int sum = 0;
    for (size_t k = 0; k < mmsize; ++k) {
      sum += matmul->a[i * mmsize + k] * matmul->b[k * mmsize + j];
    }
    matmul->c[i * mmsize + j] = sum;
  }
}
