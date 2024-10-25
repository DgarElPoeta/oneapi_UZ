#ifndef GAUSSIAN_H
#define GAUSSIAN_H

#include <sycl/sycl.hpp>
#include <vector>
#include <cstdint>


constexpr size_t filterDim = 5;

typedef sycl::uchar3 ptype;
typedef float ftype;

struct Gaussian {
  std::vector<ptype> input;
  std::vector<ftype> filter;
  std::vector<ptype> blurred;
  uint64_t size;
};

#endif //GAUSSIAN_H

