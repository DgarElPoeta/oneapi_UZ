#ifndef MATADD_H
#define MATADD_H

#include <vector>
#include <cstdint>
#include <sycl/sycl.hpp>

typedef float ptype;

struct Matadd {
  std::vector<ptype> a;
  std::vector<ptype> b;
  std::vector<ptype> c;
  uint64_t size;
};

#endif //MATADD_H

