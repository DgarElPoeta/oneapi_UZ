#ifndef GAUSSIAN_H
#define GAUSSIAN_H

#include <sycl/sycl.hpp>


constexpr size_t filterDim = 5;
typedef sycl::uchar3 ptype;

struct Gaussian {
  std::vector<ptype> input;
  std::vector<float> filter;
  std::vector<ptype> blurred;
  uint64_t size;
};

#endif //GAUSSIAN_H

