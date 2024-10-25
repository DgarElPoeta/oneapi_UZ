#ifndef NBODY_H
#define NBODY_H

#include <sycl/sycl.hpp>
#include <vector>
#include <cstdint>

constexpr float DT = 0.005;
constexpr float SofteningSquared = 1e-3;
constexpr float G = 6.67259e-11;

typedef sycl::float3 ptype;
typedef float mtype;

struct Nbody {
  std::vector<ptype> pos_in;
  std::vector<ptype> vel_in;
  std::vector<ptype> pos_out;
  std::vector<ptype> vel_out;
  std::vector<mtype> body_mass;
  uint64_t size;
};

#endif //NBODY_H

