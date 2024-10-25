#ifndef RAP_H
#define RAP_H

#include <vector>
#include <cstdint>

typedef int32_t ptype;

struct Rap {
  std::vector<ptype> a;
  std::vector<ptype> func;
  std::vector<ptype> b;
  uint64_t size;
  uint64_t M;
};

#endif //RAP_H

