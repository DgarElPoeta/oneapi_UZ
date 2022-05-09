#ifndef VECTOR_ADD_MANDELBROT_H
#define VECTOR_ADD_MANDELBROT_H

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

#define BENCHMARK_MANDELBROT 1

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

#define THRESHOLD 0.51
//#define THRESHOLD 0.10

// I was thinking about extrapolating borders to do a proper gaussian filter for an image,
// but the algo will be changed from the originally defined by AMD, so I skip it
//#define EXTMANDELBROTOLATE_BORDERS 1

// To mimic the EngineCL behavior, we send the whole image, if not we will have new borders per every package sent
// but we will need to calculate the proper offset inside the kernel (not as in the range)
#define INPUT_BUFFER_SENT_ALL 1

using sycl::cos;
using sycl::log;
using sycl::int4;
using sycl::float4;
using sycl::uchar4;
#include <cmath>
/* #define MUL_ADD fma */
#define MUL_ADD sycl::mad
typedef uchar4 ptype;

struct Mandelbrot {
  ptype* out;
  size_t size;
  size_t width;

  float leftxF;
  float topyF;
  float xstepF;
  float ystepF;
  uint max_iterations;
  uint numDevices;
  int bench;

  void setup(int _size, int iterations){
    width = _size;
    max_iterations = iterations;

    // Make sure width is a multiple of 4
    // width = (width + 3) & ~(4 - 1);
    width = (width / 4) ? (width / 4) * 4 : 4; // convierte a multiplo de 4

    auto height = width;
    auto xstep = (4.0 / (double)width);
    auto ystep = (-4.0 / (double)height);
    auto xpos = 0.0;
    auto ypos = 0.0;

    int size_matrix = width * height;
    size = size_matrix;

    // size_t lws = 256;
    // size_t gws = size_matrix >> 2;

    numDevices = 1;
    bench = 0;
    auto xsize = 4.0;

    auto larger = true; // the set is larger than the default
    if (larger) {
      xsize = 4 * xsize / 7;
      xpos = -0.65;
      ypos = 0.3;
    }

    double aspect = (double)width / (double)height;
    xstep = (xsize / (double)width);

    // Adjust for aspect ratio
    double ysize = xsize / aspect;
    ystep = (-(xsize / aspect) / height);
    auto leftx = (xpos - xsize / 2.0);

    auto i = 0;
    auto topy = (ypos + ysize / 2.0 - ((double)i * ysize) / (double)numDevices);

    leftxF = (float)leftx;
    topyF = (float)topy;
    xstepF = (float)xstep;
    ystepF = (float)ystep;
  }
};
//using namespace cl::sycl;




inline ostream&
operator<<(ostream& os, cl_uchar4& t)
{
  os << "(" << (int)t.s[0] << "," << (int)t.s[1] << "," << (int)t.s[2] << "," << (int)t.s[3] << ")";
  return os;
}
// inline ostream&
// operator<<(ostream& os, uchar4& t)
// {
//   os << "(" << (int)t.s0() << "," << (int)t.s1() << "," << (int)t.s2() << "," << (int)t.s3() << ")";
//   return os;
// }

union u_uchar4
{
  struct __uchar_four
  {
    unsigned char s0;
    unsigned char s1;
    unsigned char s2;
    unsigned char s3;
  } ch;
  cl_uint num;
};

// struct int4;
// struct float4
// {
//   float s0;
//   float s1;
//   float s2;
//   float s3;
//
//   float4 operator*(float4& fl)
//   {
//     float4 temp;
//     temp.s0() = (this->s0) * fl.s0;
//     temp.s1() = (this->s1) * fl.s1;
//     temp.s2() = (this->s2) * fl.s2;
//     temp.s3() = (this->s3) * fl.s3;
//     return temp;
//   }
//
//   float4 operator*(float scalar)
//   {
//     float4 temp;
//     temp.s0() = (this->s0) * scalar;
//     temp.s1() = (this->s1) * scalar;
//     temp.s2() = (this->s2) * scalar;
//     temp.s3() = (this->s3) * scalar;
//     return temp;
//   }
//
//   float4 operator+(float4& fl)
//   {
//     float4 temp;
//     temp.s0() = (this->s0) + fl.s0;
//     temp.s1() = (this->s1) + fl.s1;
//     temp.s2() = (this->s2) + fl.s2;
//     temp.s3() = (this->s3) + fl.s3;
//     return temp;
//   }
//
//   float4 operator-(float4 fl)
//   {
//     float4 temp;
//     temp.s0() = (this->s0) - fl.s0;
//     temp.s1() = (this->s1) - fl.s1;
//     temp.s2() = (this->s2) - fl.s2;
//     temp.s3() = (this->s3) - fl.s3;
//     return temp;
//   }
//
//   friend float4 operator*(float scalar, float4& fl);
//   friend float4 convert_float4(int4 i);
// };
//
// float4 operator*(float scalar, float4& fl)
// {
//   float4 temp;
//   temp.s0() = fl.s0() * scalar;
//   temp.s1() = fl.s1() * scalar;
//   temp.s2() = fl.s2() * scalar;
//   temp.s3() = fl.s3() * scalar;
//   return temp;
// }
//
// struct double4
// {
//   double s0;
//   double s1;
//   double s2;
//   double s3;
//
//   double4 operator*(double4& fl)
//   {
//     double4 temp;
//     temp.s0() = (this->s0) * fl.s0;
//     temp.s1() = (this->s1) * fl.s1;
//     temp.s2() = (this->s2) * fl.s2;
//     temp.s3() = (this->s3) * fl.s3;
//     return temp;
//   }
//
//   double4 operator*(double scalar)
//   {
//     double4 temp;
//     temp.s0() = (this->s0) * scalar;
//     temp.s1() = (this->s1) * scalar;
//     temp.s2() = (this->s2) * scalar;
//     temp.s3() = (this->s3) * scalar;
//     return temp;
//   }
//
//   double4 operator+(double4& fl)
//   {
//     double4 temp;
//     temp.s0() = (this->s0) + fl.s0;
//     temp.s1() = (this->s1) + fl.s1;
//     temp.s2() = (this->s2) + fl.s2;
//     temp.s3() = (this->s3) + fl.s3;
//     return temp;
//   }
//
//   double4 operator-(double4 fl)
//   {
//     double4 temp;
//     temp.s0() = (this->s0) - fl.s0;
//     temp.s1() = (this->s1) - fl.s1;
//     temp.s2() = (this->s2) - fl.s2;
//     temp.s3() = (this->s3) - fl.s3;
//     return temp;
//   }
//
//   friend double4 operator*(double scalar, double4& fl);
//   friend double4 convert_double4(int4 i);
// };
//
// double4 operator*(double scalar, double4& fl)
// {
//   double4 temp;
//   temp.s0() = fl.s0() * scalar;
//   temp.s1() = fl.s1() * scalar;
//   temp.s2() = fl.s2() * scalar;
//   temp.s3() = fl.s3() * scalar;
//   return temp;
// }
//
// struct int4
// {
//   int s0;
//   int s1;
//   int s2;
//   int s3;
//
//   int4 operator*(int4& fl)
//   {
//     int4 temp;
//     temp.s0() = (this->s0) * fl.s0;
//     temp.s1() = (this->s1) * fl.s1;
//     temp.s2() = (this->s2) * fl.s2;
//     temp.s3() = (this->s3) * fl.s3;
//     return temp;
//   }
//
//   int4 operator*(int scalar)
//   {
//     int4 temp;
//     temp.s0() = (this->s0) * scalar;
//     temp.s1() = (this->s1) * scalar;
//     temp.s2() = (this->s2) * scalar;
//     temp.s3() = (this->s3) * scalar;
//     return temp;
//   }
//
//   int4 operator+(int4& fl)
//   {
//     int4 temp;
//     temp.s0() = (this->s0) + fl.s0;
//     temp.s1() = (this->s1) + fl.s1;
//     temp.s2() = (this->s2) + fl.s2;
//     temp.s3() = (this->s3) + fl.s3;
//     return temp;
//   }
//
//   int4 operator-(int4 fl)
//   {
//     int4 temp;
//     temp.s0() = (this->s0) - fl.s0;
//     temp.s1() = (this->s1) - fl.s1;
//     temp.s2() = (this->s2) - fl.s2;
//     temp.s3() = (this->s3) - fl.s3;
//     return temp;
//   }
//
//   int4 operator+=(int4 fl)
//   {
//     s0 += fl.s0;
//     s1 += fl.s1;
//     s2 += fl.s2;
//     s3 += fl.s3;
//     return (*this);
//   }
//
//   friend float4 convert_float4(int4 i);
//   friend double4 convert_double4(int4 i);
// };
//
static inline float4
convert_float4(int4 i)
{
  float4 temp;
  temp.s0() = (float)i.s0();
  temp.s1() = (float)i.s1();
  temp.s2() = (float)i.s2();
  temp.s3() = (float)i.s3();
  return temp;
}

// double4
// convert_double4(int4 i)
// {
//   double4 temp;
//   temp.s0() = (double)i.s0();
//   temp.s1() = (double)i.s1();
//   temp.s2() = (double)i.s2();
//   temp.s3() = (double)i.s3();
//   return temp;
// }

static inline float
native_log2(float in)
{
  return log(in) / log(2.0f);
}

static inline float
native_cos(float in)
{
  return cos(in);
}

static inline double
native_log2(double in)
{
  return log(in) / log(2.0f);
}

inline double
native_cos(double in)
{
  return cos(in);
}

#ifndef min
int
min(int a1, int a2)
{
  return ((a1 < a2) ? a1 : a2);
}
#endif




#endif //VECTOR_ADD_MANDELBROT_H

