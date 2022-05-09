#ifndef VECTOR_ADD_RAY_H
#define VECTOR_ADD_RAY_H

#include <CL/sycl.hpp>
//#include "defs.h"

#include <iostream>
#include <iomanip>
#include <ctime>
#include <sys/time.h>
//#include <cstdlib>
//#include "sycl_exceptions.hpp"

#include <chrono>
#include <array>
#include <mutex>
#include <condition_variable>
#include <string>
#include <thread>

// Gaussian
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include <cmath>
#include <cstdlib>
#include <omp.h>

#include <stdexcept>

// #include "../dpc_common.hpp"
#include "io.hpp"

#ifdef __SYCL_DEVICE_ONLY__
#define CONSTANT __attribute__((opencl_constant))
#else
#define CONSTANT
#endif

#define BENCHMARK_RAY 1


// using sycl::float;
using sycl::float4;
using sycl::uchar4;
using sycl::uchar;

using cl::sycl::sqrt;
using cl::sycl::normalize;
using cl::sycl::clamp;

using std::ostream;
using std::vector;
using std::runtime_error;
using std::cout;

// using sycl::cl_float4;
// using sycl::cl_uchar4;
using sycl::device_selector;
using sycl::cpu_selector;
using sycl::gpu_selector;
using sycl::queue;
// To do Profiling
// using dpc::queue;
using sycl::handler;
using sycl::id;
using sycl::item;
using sycl::nd_item;

// in EngineCL it can be decied directly in the kernel
#define USE_LOCAL_MEM 0
// DO NOT USE THIS:
#define USM_USE_LOCAL_MEM_NOASYNC 0

#define THRESHOLD 0.51
//#define THRESHOLD 0.10

// I was thinking about extrapolating borders to do a proper gaussian filter for an image,
// but the algo will be changed from the originally defined by AMD, so I skip it
//#define EXTRAYOLATE_BORDERS 1

// To mimic the EngineCL behavior, we send the whole image, if not we will have new borders per every package sent
// but we will need to calculate the proper offset inside the kernel (not as in the range)
#define INPUT_BUFFER_SENT_ALL 1

typedef cl_float4 ptype;

typedef float4 Color;

//using namespace cl::sycl;

// enum class prim_type
// {
//   PLANE = 0,
//   SPHERE = 1
// };
typedef enum prim_type
{
  PLANE = 0,
  SPHERE = 1
} prim_type;

// typedef struct Primitive
// {
//   Color m_color;                        // material_color
//   cl_float m_refl;                      // material_reflection(_factor)
//   cl_float m_diff;                      // material_diffuse(_lighting_factor)
//   cl_float m_refr;                      // material_refraction(_factor)
//   cl_float m_refr_index;                // material_refraction_index
//   cl_float m_spec;                      // material_specular(_lighting_factor)
//   cl_float dummy_3;                     // dummy value
//   prim_type type;                       // primitive type
//   cl_bool is_light;                     // light?
//   cl_float4 normal;                     // normal used to define a plane
//   cl_float4 center;                     // center used to define a sphere
//   cl_float depth;                       // depth used to define a plane
//   cl_float radius, sq_radius, r_radius; // radius used to define a sphere
// } Primitive;
typedef struct Primitive
{
  Color m_color;                        // material_color
  float m_refl;                      // material_reflection(_factor)
  float m_diff;                      // material_diffuse(_lighting_factor)
  float m_refr;                      // material_refraction(_factor)
  float m_refr_index;                // material_refraction_index
  float m_spec;                      // material_specular(_lighting_factor)
  float dummy_3;                     // dummy value
  prim_type type;                       // primitive type
  bool is_light;                     // light?
  float4 normal;                     // normal used to define a plane
  float4 center;                     // center used to define a sphere
  float depth;                       // depth used to define a plane
  float radius, sq_radius, r_radius; // radius used to define a sphere
} Primitive;

typedef struct data_t
{
  int width;
  int height;
  Primitive* A;   // primitives (in)
  Pixel* C;       // pixels (out)
  size_t total_size; // width * height
  int depth;
  int fast_norm;
  int buil_norm;
  int nati_sqrt;
  int buil_dot;
  int buil_len;
  float viewp_w;
  float viewp_h;
  float camera_x;
  float camera_y;
  float camera_z;
  const char* scene;
  const char* out_file;
  const char* progname;
  int n_primitives;
  int retval;
} data_t;

struct Ray {
  Primitive* prim_ptr;
  Pixel* pixels_ptr;
  size_t size;
  size_t n_primitives;
  data_t* data;
};

void
data_t_init(data_t* data);

int
ray_begin(data_t* data);

int
ray_end(data_t* data);

// Inside ray.cl kernel

#ifdef D_FAST_NORMALIZE
#define NORMALIZE(A) fast_normalize(A)
#elif defined D_BUILTIN_NORMALIZE
#define NORMALIZE(A) normalize(A)
#else
#define NORMALIZE(A) soft_normalize(A)
#endif

#ifdef D_NATIVE_SQRT
#define SQRT(A) native_sqrt(A)
#else
#define SQRT(A) sqrt(A)
#endif

#ifdef D_BUILTIN_DOT
#define DOT(A, B) dot(A, B)
#else
#define DOT(A, B) soft_dot(A, B)
#endif

#ifdef D_BUILTIN_LEN
#define LENGTH(A) length(A)
#else
#define LENGTH(A) soft_length(A)
#endif

// cant have dynamic allocation in the kernel
#ifdef D_TRACEDEPTH_0
#define TRACEDEPTH 0
#define MAX_RAY_COUNT 1
#elif defined D_TRACEDEPTH_1
#define TRACEDEPTH 1
#define MAX_RAY_COUNT 2
#elif defined D_TRACEDEPTH_2
#define TRACEDEPTH 2
#define MAX_RAY_COUNT 4
#elif defined D_TRACEDEPTH_3
#define TRACEDEPTH 3
#define MAX_RAY_COUNT 8
#elif defined D_TRACEDEPTH_4
#define TRACEDEPTH 4
#define MAX_RAY_COUNT 32
#elif defined D_TRACEDEPTH_5
#define TRACEDEPTH 5
#define MAX_RAY_COUNT 64
#else
#define TRACEDEPTH 0
#define MAX_RAY_COUNT 1
#endif

// Intersection method return values
#define HIT 1     // Ray hit primitive
#define MISS 0    // Ray missed primitive
#define INPRIM -1 // Ray started inside primitive

#define EPSILON 0.001f

// ray queue to simulate recursion

#define PUSH_RAY(q, r, c, n)                                                                       \
  if (c >= MAX_RAY_COUNT)                                                                          \
    c = 0;                                                                                         \
  q[c++] = r;                                                                                      \
  n++;

#define POP_RAY(q, r, c, n)                                                                        \
  if (c >= MAX_RAY_COUNT)                                                                          \
    c = 0;                                                                                         \
  r = q[c++];                                                                                      \
  n--;

  // -----
  // Inside the ray.cl kernel

  float4
  soft_normalize(float4 vec)
  {
    float l = 1 / SQRT(vec.x() * vec.x() + vec.y() * vec.y() + vec.z() * vec.z());
    return float4{vec.x() *= l, vec.y() *= l, vec.z() *= l, 0}; // TODO: review conversion
  }

  float
  soft_dot(float4 vec_a, float4 vec_b)
  {
    return vec_a.x() * vec_b.x() + vec_a.y() * vec_b.y() + vec_a.z() * vec_b.z();
  }

  float
  soft_length(float4 vec)
  {
    return SQRT(vec.x() * vec.x() + vec.y() * vec.y() + vec.z() * vec.z());
  }


  // typedefs - repeated from common.h, but for the kernel
  // typedef uchar4 Pixel;
  // typedef float4 Color;

  // Note - these enums were anonymous previously, and compiled fine under AMD's OpenCL drivers.
  // Using Intels, the compiler throws an error with anonymous enums.
  typedef enum raytype
  {
    ORIGIN = 0,
    REFLECTED = 1,
    REFRACTED = 2
  } ray_type;

  // typedef enum prim_type // modified
  // {
  //   PLANE = 0,
  //   SPHERE = 1
  // } prim_type;

  // The dummy_3 value is used to align the struct properly.
  // OpenCL requires 16-byte alignment (ie 4 floats) for its vector data types. Without
  // the dummy_3 value, the Primitive struct will not copy from the host correctly.
  // See common.h for structure value information
  // typedef struct
  // {
  //   Color m_color;
  //   float m_refl;
  //   float m_diff;
  //   float m_refr;
  //   float m_refr_index;
  //   float m_spec;
  //   float dummy_3;
  //   prim_type type; // mod
  //   bool is_light;
  //   float4 normal;
  //   float4 center;
  //   float depth;
  //   float radius, sq_radius, r_radius;
  // } Primitive;

  typedef struct
  {
    float4 origin;
    float4 direction;
    float weight;
    float depth;
    int origin_primitive;
    ray_type type;
    float r_index;
    Color transparency;
  } RayK;

  // functions
  // int
  // plane_intersect(local Primitive* p, Ray* ray, float* cumu_dist)
  // {
  int
  plane_intersect(Primitive* p, RayK* ray, float* cumu_dist)
  {
    float d = DOT(p->normal, ray->direction);
    if (d != 0) {
      float dist = -(DOT(p->normal, ray->origin) + p->depth) / d;
      if (dist > 0 && dist < *cumu_dist) {
        *cumu_dist = dist;
        return HIT;
      }
  }
  return MISS;
  }

  // int
  // sphere_intersect(local Primitive* p, Ray* ray, float* cumu_dist)
  int
  sphere_intersect(Primitive* p, RayK* ray, float* cumu_dist)
  {
    float4 v = ray->origin - p->center;
    float b = -DOT(v, ray->direction);
    float det = (b * b) - DOT(v, v) + p->sq_radius;
    int retval = MISS;
    if (det > 0) {
      det = SQRT(det);
      float i1 = b - det;
      float i2 = b + det;
      if (i2 > 0) {
        if (i1 < 0) {
          if (i2 < *cumu_dist) {
            *cumu_dist = i2;
            retval = INPRIM;
          }
        } else {
          if (i1 < *cumu_dist) {
            *cumu_dist = i1;
            retval = HIT;
          }
        }
      }
    }
    return retval;
  }

  // int
  // intersect(local Primitive* p, Ray* ray, float* cumu_dist)
  int
  intersect(Primitive* p, RayK* ray, float* cumu_dist)
  {
    switch (p->type) {
      case PLANE:
        return plane_intersect(p, ray, cumu_dist);
      case SPHERE:
        return sphere_intersect(p, ray, cumu_dist);
    }
    return MISS;
  }

  // float4
  // get_normal(local Primitive* p, float4 point)
  float4
  get_normal(Primitive* p, float4 point)
  {
    switch (p->type) {
      case PLANE:
        return (p->normal);
      case SPHERE:
        return (point - p->center) * p->r_radius;
    }
    // return (float4)(0, 0, 0, 0);
    return float4{0, 0, 0, 0};
  }

  // int
  // raytrace(Ray* a_ray,
  //          Color* a_acc,
  //          float* a_dist,
  //          float4* point_intersect,
  //          int* result,
  //          local Primitive* primitives,
  // int n_primitives)
  // int
  // raytrace_x(RayK* a_ray,
  //          Color* a_acc,
  //          float* a_dist,
  //          float4* point_intersect,
  //          int* result,
  //          Primitive* primitives,
  //          int n_primitives)
  // {
  //   *a_dist = MAXFLOAT;
  //   int prim_index = -1;
  //
  // // find nearest intersection
  //   for (int s = 0; s < n_primitives; s++) {
  //     int res;
  //     if (res = intersect(&primitives[s], a_ray, a_dist)) {
  //       prim_index = s;
  //       *result = res;
  //     }
  //   }
  // // no hit
  //   if (prim_index == -1)
  //     return -1;
  // // handle hit
  //   if (primitives[prim_index].is_light) {
  //     *a_acc = primitives[prim_index].m_color;
  //   } else {
  //     *point_intersect = a_ray->origin + (a_ray->direction * (*a_dist));
  // // trace lights
  //     for (int l = 0; l < n_primitives; l++) {
  //       if (primitives[l].is_light) {
  // // point light source shadows
  //         float shade = 1.0f;
  //         float L_LEN = LENGTH(primitives[l].center - *point_intersect);
  //         float4 L = NORMALIZE(primitives[l].center - *point_intersect);
  //         if (primitives[l].type == SPHERE) {
  //           RayK r;
  //           r.origin = *point_intersect + L * EPSILON;
  //           r.direction = L;
  //           int s = 0;
  //           while (s < n_primitives) {
  //             if (&primitives[s] != &primitives[l] && !primitives[s].is_light &&
  //                 intersect(&primitives[s], &r, &L_LEN)) {
  //               shade = 0;
  //             }
  //             s++;
  //           }
  //         }
  // // Calculate diffuse shading
  //         float4 N = get_normal(&primitives[prim_index], *point_intersect);
  //         if (primitives[prim_index].m_diff > 0) {
  //           float dot_prod = DOT(N, L);
  //           if (dot_prod > 0) {
  //             float diff = dot_prod * primitives[prim_index].m_diff * shade;
  //             *a_acc += diff * primitives[prim_index].m_color * primitives[l].m_color;
  //           }
  //         }
  // // Calculate specular shading
  //         if (primitives[prim_index].m_spec > 0) {
  //           float4 V = a_ray->direction;
  //           float4 R = L - 1.5f * DOT(L, N) * N;
  //           float dot_prod = DOT(V, R);
  //           if (dot_prod > 0) {
  //             // TODO: review, originally native_powr(dot_prod, 20)
  //             float spec = cl::sycl::powr(dot_prod, 20.0f) * primitives[prim_index].m_spec * shade;
  //             *a_acc += spec * primitives[l].m_color;
  //           }
  //         }
  //       }
  //     }
  //   }
  //
  //   return prim_index;
  // }

  // Outside the ray.cl kernel
  // -----
// Outside ray.cl kernel

#endif //VECTOR_ADD_RAY_H
