#ifndef ENGINECL_EXAMPLES_COMMON_IO_HPP
#define ENGINECL_EXAMPLES_COMMON_IO_HPP 1

#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include <CL/sycl.hpp>
// #include <CL/cl.h>

using std::ifstream;
using std::ios;
using std::move;
using std::string;
using std::stringstream;
using std::vector;

using sycl::uchar4;
typedef uchar4 Pixel;
// Previously was CL/cl.h and here only cl_uchar4

#define PIXEL_BIT_DEPTH 24
#define BITMAP_HEADER_SIZE 14
#define BITMAP_INFO_HEADER_SIZE 40

typedef struct bmp_magic
{
  unsigned char magic[2];
} bmp_magic_t;

typedef struct
{
  uint32_t filesz;
  uint16_t creator1;
  uint16_t creator2;
  uint32_t bmp_offset;
} BMP_HEADER;

typedef struct
{
  uint32_t header_sz;
  int32_t width;
  int32_t height;
  uint16_t nplanes;
  uint16_t bitspp;
  uint32_t compress_type;
  uint32_t bmp_bytesz;
  int32_t hres;
  int32_t vres;
  uint32_t ncolors;
  uint32_t nimpcolors;
} BMP_INFO_HEADER;

int
write_bmp_file(Pixel* pixels, int width, int height, const char* filename);

string
file_read(const string& path);

std::vector<char>
file_read_binary(const string& path);

#endif // ENGINECL_EXAMPLES_COMMON_IO_HPP
