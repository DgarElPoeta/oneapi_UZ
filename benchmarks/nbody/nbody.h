#ifndef NBODY_H
#define NBODY_H

#include <sycl/sycl.hpp>

constexpr float DT = 0.005f;
constexpr float SofteningSquared = 1e-3f;
constexpr float G = 6.67259e-11f;

typedef sycl::float3 ptype;

struct Nbody {
  std::vector<ptype> pos_in;
  std::vector<ptype> vel_in;
  std::vector<ptype> pos_out;
  std::vector<ptype> vel_out;
  std::vector<float> body_mass;
  uint64_t size;
};

#endif //NBODY_H

