#define PACKED 0

Gaussian* gaussian = reinterpret_cast<Gaussian*>(opts->p_problem);
auto input = gaussian->_a;
auto filterWeight = gaussian->_b;
auto blurred = gaussian->_c;
auto cols = gaussian->_width;
auto rows = gaussian->_width;
auto filterWidth = gaussian->_filter_width;

//#pragma omp parallel for
for(size_t tid=offset; tid< size+offset; tid++){
//for(size_t tid=0; tid< size; tid++){
  int r = tid / cols; // current row
  int c = tid % cols; // current column

  int middle = filterWidth / 2;
#if PACKED
float4 blur{0.f};
#else
  float blurX = 0.f; // will contained blurred value
  float blurY = 0.f; // will contained blurred value
  float blurZ = 0.f; // will contained blurred value
#endif
  int width = cols - 1;
  int height = rows - 1;

  for (int i = -middle; i <= middle; ++i) // rows
  {
    for (int j = -middle; j <= middle; ++j) // columns
    {

      int h = r + i;
      int w = c + j;
      if (h > height || h < 0 || w > width || w < 0) {
        continue;
      }

      int idx = w + cols * h; // current pixel index

#if PACKED
float4 pixel = input[idx].convert<float>();
#else
      float pixelX = (input[idx].x()); //s[0]);
      float pixelY = (input[idx].y()); //s[1]);
      float pixelZ = (input[idx].z()); //s[2]);
#endif

      idx = (i + middle) * filterWidth + j + middle;
      float weight = filterWeight[idx];

#if PACKED
blur += pixel * weight;
#else
      blurX += pixelX * weight;
      blurY += pixelY * weight;
      blurZ += pixelZ * weight;
#endif
    }
  }

#if PACKED
    blurred[tid] = (cl::sycl::round(blur)).convert<uchar>();
#else
    blurred[tid].x() = (unsigned char) cl::sycl::round(blurX);
    blurred[tid].y() = (unsigned char) cl::sycl::round(blurY);
    blurred[tid].z() = (unsigned char) cl::sycl::round(blurZ);
#endif
}
