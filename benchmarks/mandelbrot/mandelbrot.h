#ifndef MANDELBROT_H
#define MANDELBROT_H

#include <complex>
#include <vector>
#include <cstdint>

typedef uint64_t ptype;
typedef std::complex<float> ctype;

constexpr float MINX = -2.0f;
constexpr float MAXX = 1.0f;
constexpr float MINY = -1.0f;
constexpr float MAXY = 1.0f;
constexpr uint64_t MAXITERATIONS = 100;

struct Mandelbrot {
  std::vector<ptype> image;
  uint64_t size;
};

#endif //MANDELBROT_H

