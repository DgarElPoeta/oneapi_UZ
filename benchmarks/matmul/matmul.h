#ifndef MATMUL_H
#define MATMUL_H

#include <vector>
#include <cstdint>

typedef float ptype;

struct Matmul {
  std::vector<ptype> a;
  std::vector<ptype> b;
  std::vector<ptype> c;
  uint64_t size;
};

#endif //MATMUL_H

