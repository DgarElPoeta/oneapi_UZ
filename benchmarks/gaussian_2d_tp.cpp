/* It is not 2d anymore. Reverted to 1d */
#define THREADS 6

#include <CL/sycl.hpp>
#include <iostream>
#include <iomanip>
#include <ctime>
#include <sys/time.h>
#include <omp.h>
#include <cstdlib>

#include "../tpool.h"
#include "../dpc_common.hpp"
#include "../benchmarks/io.hpp"
// using namespace std;
using sycl::uchar;
using sycl::uchar4;
using sycl::float4;
// using sycl::queue;
using sycl::buffer;
using sycl::accessor;
using sycl::range;

using std::ios_base;
using std::cout;
using std::setprecision;
using std::fixed;
using std::ostream;

#if !defined(USM)
#error define USM to 0 or 1
#endif

#ifdef __SYCL_DEVICE_ONLY__
#define CONSTANT __attribute__((opencl_constant))
#else
#define CONSTANT
#endif

typedef float op_type;

// constexpr int size = 1024;
// constexpr int filter_size = 61;

const int its = 3;

inline ostream &
operator<<(ostream &os, uchar4 &t) {
  os << "(" << (int) t.x() << "," << (int) t.y() << "," << (int) t.z() << "," << (int) t.w() << ")";
  return os;
}

void fill_filter(float* filter, int filter_width) {
  const float sigma = 2.f;

  const int half = filter_width / 2;
  float sum = 0.f;

  // vector<cl_float>* res = new vector<cl_float>(width * width);
  // vector<cl_float> res(width * width);
  // float* res = (float*)malloc(width * width * sizeof(float));

  int r;
  for (r = -half; r <= half; ++r) {
    int c;
    for (c = -half; c <= half; ++c) {
      float weight = expf(-(float) (c * c + r * r) / (2.0f * sigma * sigma));
      // float weight = expf(-(float)(c * c + r * r) / (2.f * sigma * sigma));
      int idx = (r + half) * filter_width + c + half;

      filter[idx] = weight;
      // res.at(idx) = weight;
      sum += weight;
    }
  }

  float normal = 1.0f / sum;

  for (r = -half; r <= half; ++r) {
    int c;
    for (c = -half; c <= half; ++c) {
      int idx = (r + half) * filter_width + c + half;

      filter[idx] *= normal;
      // res[idx] *= normal;
    }
  }
}

double time_computing_cpu = 0.0;
double time_computing_gpu = 0.0;

tpool::tpooler pool(THREADS);
int cpu_queue_detail(int thread_, int thread, int thread_chunk_offset, int thread_chunk_limit, float* filterWeight, uchar4* input, uchar4* blurred, size_t filterWidth, size_t size) {
  auto cols = size;
  auto rows = size;

  std::cout << "thread " << thread << " from " << thread_chunk_offset << " to " << thread_chunk_limit << " (size: "
            << (thread_chunk_limit - thread_chunk_offset) << ")\n";

      for (size_t tid = thread_chunk_offset; tid < thread_chunk_limit; ++tid) {
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

#if packed
            float4 pixel = input[idx].convert<float>();
#else
            float pixelx = (input[idx].x()); //s[0]);
            float pixely = (input[idx].y()); //s[1]);
            float pixelz = (input[idx].z()); //s[2]);
#endif

            idx = (i + middle) * filterWidth + j + middle;
            float weight = filterWeight[idx];

#if packed
            blur += pixel * weight;
#else
            blurX += pixelx * weight;
            blurY += pixely * weight;
            blurZ += pixelz * weight;
#endif
          }
        }

#if packed
        blurred[tid] = (cl::sycl::round(blur)).convert<uchar>();
#else
        blurred[tid].x() = (unsigned char) cl::sycl::round(blurX);
        blurred[tid].y() = (unsigned char) cl::sycl::round(blurY);
        blurred[tid].z() = (unsigned char) cl::sycl::round(blurZ);
#endif
      }

      std::cout << "thread " << thread << " finished\n";
      return thread;
}
// #define PACKED
void cpu_queue(float* vfilter_ptr, uchar4* vinput_ptr, uchar4* vblurred_ptr, size_t filter_size, size_t chunk_size, size_t chunk_offset, size_t size, size_t N) {
  struct timeval compute_start, compute_end;
  auto filterWidth = filter_size;
  float* filterWeight = vfilter_ptr;
  uchar4* input = vinput_ptr;
  uchar4* blurred = vblurred_ptr;
  auto cols = size;
  auto rows = size;
  gettimeofday(&compute_start, NULL);

  std::future<int> fthreads[THREADS];
  for (int thread=0; thread<THREADS; ++thread) {
    auto thread_chunk_size = chunk_size / THREADS;
    auto thread_chunk_offset = chunk_offset + (thread * (thread_chunk_size));
    auto thread_chunk_limit = (thread_chunk_size + thread_chunk_offset);
    fthreads[thread] = pool.push(cpu_queue_detail, thread, thread_chunk_offset, thread_chunk_limit, filterWeight, input, blurred, filterWidth, size);
//     fthreads[thread] = pool.push([&,thread,chunk_offset](int id) {
//       auto thread_chunk_size = chunk_size / THREADS;
//       auto thread_chunk_offset = chunk_offset + (thread * (thread_chunk_size));
//       auto thread_chunk_limit = (thread_chunk_size + thread_chunk_offset);
//       std::cout << "thread " << id << " from " << thread_chunk_offset << " to " << thread_chunk_limit << " (size: "
//                 << thread_chunk_size << ")\n";
//
//       for (size_t tid = thread_chunk_offset; tid < thread_chunk_limit; ++tid) {
//         int r = tid / cols; // current row
//         int c = tid % cols; // current column
//
//         int middle = filterWidth / 2;
// #if PACKED
//         float4 blur{0.f};
// #else
//         float blurX = 0.f; // will contained blurred value
//         float blurY = 0.f; // will contained blurred value
//         float blurZ = 0.f; // will contained blurred value
// #endif
//         int width = cols - 1;
//         int height = rows - 1;
//
//         for (int i = -middle; i <= middle; ++i) // rows
//         {
//           for (int j = -middle; j <= middle; ++j) // columns
//           {
//
//             int h = r + i;
//             int w = c + j;
//             if (h > height || h < 0 || w > width || w < 0) {
//               continue;
//             }
//
//             int idx = w + cols * h; // current pixel index
//
// #if packed
//             float4 pixel = input[idx].convert<float>();
// #else
//             float pixelx = (input[idx].x()); //s[0]);
//             float pixely = (input[idx].y()); //s[1]);
//             float pixelz = (input[idx].z()); //s[2]);
// #endif
//
//             idx = (i + middle) * filterWidth + j + middle;
//             float weight = filterWeight[idx];
//
// #if packed
//             blur += pixel * weight;
// #else
//             blurX += pixelx * weight;
//             blurY += pixely * weight;
//             blurZ += pixelz * weight;
// #endif
//           }
//         }
//
// #if packed
//         blurred[tid] = (cl::sycl::round(blur)).convert<uchar>();
// #else
//         blurred[tid].x() = (unsigned char) cl::sycl::round(blurX);
//         blurred[tid].y() = (unsigned char) cl::sycl::round(blurY);
//         blurred[tid].z() = (unsigned char) cl::sycl::round(blurZ);
// #endif
//       }
//
//       return id;
//     }); // lambda
  }
  for (int thread=0; thread<THREADS; ++thread){
    std::cout << "future[" << thread << "]: " << fthreads[thread].get() << "\n";
  }
  gettimeofday(&compute_end, NULL);
  double time_taken;
  time_taken = (compute_end.tv_sec - compute_start.tv_sec) * 1e6;
  time_taken = (time_taken + (compute_end.tv_usec - compute_start.tv_usec)) * 1e-6;
  time_computing_cpu += time_taken;
}


#include <CL/sycl/INTEL/fpga_extensions.hpp>
bool first_dev = true;
void gpu_queue(float* vfilter_ptr, uchar4* vinput_ptr, uchar4* vblurred_ptr, size_t filter_size, size_t chunk_size, size_t chunk_offset, size_t size, size_t N){
  sycl::event ew, ek;
  struct timeval compute_start, compute_end;
  try {
    cl::sycl::property_list prop_list =
        cl::sycl::property_list{cl::sycl::property::queue::enable_profiling()};
    // sycl::cpu_selector d;
    //cl::sycl::queue q(sycl::gpu_selector{}, dpc::exception_handler, prop_list);
    //cl::sycl::queue q(sycl::INTEL::fpga_emulator_selector{}, dpc::exception_handler, prop_list);
    cl::sycl::queue q(sycl::INTEL::fpga_selector{}, dpc::exception_handler, prop_list);
    // sycl::queue q(sycl::cpu_selector{});

    // dpc::queue q(&sycl::cpu_selector{});
    // dpc::queue q(&d);

    if (first_dev){
      cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";
      first_dev = false;
    }


      // buffer<uchar4, 1> input_buf(vinput.data(), range(N));
      buffer<float, 1> filter_buf(vfilter_ptr, range<1>(filter_size));
      buffer<uchar4, 1> blurred_buf(vblurred_ptr, range<1>(chunk_size));

      range<1> Rinput(N);
      // buffer<uchar4, 1> input_buf(Rinput);
      buffer<uchar4, 1> input_buf(vinput_ptr, Rinput);

      // buffer<uchar4, 1> input_buf;
      // buffer<float, 1> filter_buf(vfilter.data(), range(Nfilter));
      // buffer<uchar4, 1> blurred_buf(vblurred.data(), range(N));

      range<1> workitems(chunk_size);

      gettimeofday(&compute_start, NULL);
      ios_base::sync_with_stdio(false);
      const int cols = size;
      const int rows = size;
      // maybe with usm?
      // ew = q.memcpy(input_buf, vinput, N * sizeof(uchar4));
      //ew = q.submit([&](auto &h) {
      //  auto input = input_buf.get_access<sycl::access::mode::write>(h);
      //  h.copy(vinput.data(), input);
      //});

      ek = q.submit([&](auto &h) {
        // accessor input(input_buf, h, sycl::access::mode::read);
        // accessor filter(filter_buf, h, sycl::access::mode::read);
        // accessor blurred(blurred_buf, h, sycl::access::mode::discard_write);

        auto input = input_buf.get_access<sycl::access::mode::read>(h);
        auto filter = filter_buf.get_access<sycl::access::mode::read>(h);
        auto blurred = blurred_buf.get_access<sycl::access::mode::discard_write>(h);

        h.parallel_for(workitems, [=](auto index) {
          // int r = tid / cols;
          // int c = tid % cols;
          // {
          //   static const CONSTANT char FMT[] = "[%d][%d]\n";
          //   sycl::ONEAPI::experimental::printf(FMT, r, c);
          // }
          auto tid = index.get_linear_id();

          int r = tid / cols;
          int c = tid % cols;

          int middle = filter_size / 2;
          float4 blur{0.f};

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

              // float4 pixel = input[idx].convert<float>();
              float4 pixel = input[idx].template convert<float>();

              idx = (i + middle) * filter_size + j + middle;
              float weight = filter[idx];

              blur += pixel * weight;
            }
          }

          // tid -= offset;
          // const size_t tid = index.get_linear_id();
          // {
          //   static const CONSTANT char FMT[] = "[%d][]\n";
          //   sycl::ONEAPI::experimental::printf(FMT, tid);
          // }
          // blurred[tid] = input[tid];
          // blurred[index] = input[index];
          blurred[index] = (cl::sycl::round(blur)).convert<uchar>();
        });
      });

    q.wait();
    gettimeofday(&compute_end, NULL);
    double time_taken;
    time_taken = (compute_end.tv_sec - compute_start.tv_sec) * 1e6;
    time_taken = (time_taken + (compute_end.tv_usec - compute_start.tv_usec)) * 1e-6;
    time_computing_gpu += time_taken;

  } catch (sycl::exception const &e) {
    cout << "sycl exception: " << e.what() << "\n";
    std::terminate();
  }

  auto profilingQueueS = dpc::ReportTime(ek) / 1000.0;
  cout << "profiling queue (ek kernel): " << profilingQueueS << "\n";
  // profilingQueueS = dpc::ReportTime(ew) / 1000.0;
  // cout << "profiling queue ew" << profilingQueueS << "\n";
}

int main(int argc, char *argv[]) {
  if (argc != 4) {
    std::cout << "usage: <size> <filter size> <chunks>\n";
    return 1;
  }
  std::chrono::high_resolution_clock::time_point tStart = std::chrono::high_resolution_clock::now();
  // int size = argc > 1 ? atoi(argv[1]) : (1024 * 1);
  // int filter_size = argc > 2 ? atoi(argv[2]) : 61;
  size_t size = std::stoi(argv[1]);
  size_t filter_size = std::stoi(argv[2]);
  size_t chunks = std::stoi(argv[3]);

  int N = size * size;
  int Nfilter = filter_size * filter_size;
  std::vector<uchar4> vinput(N);
  std::vector<float> vfilter(Nfilter);
  std::vector<uchar4> vblurred(N);

//#pragma omp parallel for num_threads(4)
#pragma omp parallel num_threads(THREADS)
  {
    printf("omp thread: %d\n", omp_get_thread_num());
  }

//   int channels = 4;
//   auto total = _total_size * channels;
// #pragma omp parallel for num_threads(omp_get_max_threads())
//   for (auto i = 0; i < total; i++) {
//     int mod = i % channels;
//     switch (mod) {
//       case 0:blurred[i / channels][0] = 0;
//         break;
//       case 1:blurred[i / channels][1] = 0;
//         break;
//       case 2:blurred[i / channels][2] = 0;
//         break;
//       case 3:blurred[i / channels][3] = 0;
//         break;
//     }
//   }
  srand(0);
#pragma omp for
  for (auto i = 0; i<N; ++i) {
    // vinput[i] = uchar4{55};
    vinput[i] = uchar4{rand() % 256, rand() % 256, rand() % 256, 0};
    vblurred[i] = uchar4{0};
  }

  cout << "init containers\n";

  fill_filter(vfilter.data(), filter_size);

  cout << "init filter\n";

  // auto R = sycl::range<1>(N);

  auto first = true;
  auto chunk_size = N / chunks;
  cout << "chunk: " << chunk_size << "\n";
  struct timeval start, end;
  gettimeofday(&start, NULL);

  // chunks = 1;
  // auto subchunks = 1;
  // auto subchunk_size = chunk_size / subchunks;

  auto to_cpu = true;


  // exit(0);
  for (auto chunk=0; chunk<chunks; ++chunk) {
    auto chunk_offset = chunk * chunk_size;

    if (to_cpu) {
      to_cpu = false;
      cpu_queue(vfilter.data(), vinput.data(), vblurred.data(), filter_size, chunk_size, chunk_offset, size, N);
    } else {
      to_cpu = true;
      gpu_queue(vfilter.data(), vinput.data(), vblurred.data(), filter_size, chunk_size, chunk_offset, size, N);
    }

  }

  gettimeofday(&end, NULL);

  auto tTemp = std::chrono::high_resolution_clock::now();
  auto diffTemp = (tTemp - tStart).count();
  auto diffTempS = diffTemp / 1e9;
  printf("Time taken by data init + queue is: %6.4f\n", diffTempS);
  double time_taken;
  time_taken = (end.tv_sec - start.tv_sec) * 1e6;
  time_taken = (time_taken + (end.tv_usec - start.tv_usec)) * 1e-6;
  cout << "Time taken by queue is: " << fixed << time_taken << setprecision(6) << " sec " << "\n";
  cout << "Time taken in computing regions (kernels) for CPU is: " << fixed << time_computing_cpu << setprecision(6) << " sec " << "\n";
  cout << "Time taken in computing regions (kernels) for GPU is: " << fixed << time_computing_gpu << setprecision(6) << " sec " << "\n";

  cout << "res: " << vblurred.data() << " => [0]" << vblurred[0] << ", [" << N-1 << "]" << vblurred[N - 1] << "\n";
  // printf("res: %p => (%d,%d,%d,%d), (%d,%d,%d,%d)\n", vblurred.data(), vblurred[0], vblurred[N-1]);

  char *image_str = getenv("IMAGE");
  bool image = (image_str != NULL && std::string(image_str) == "y");
  if (image){
    write_bmp_file((uchar4*)vinput.data(), size, size, "in.bmp");
    write_bmp_file((uchar4*)vblurred.data(), size, size, "out.bmp");
    cout << "images written\n";
  }

  exit(0);

// #if USM
//   printf("USM\n");
//   sycl::queue gpuQ(sycl::gpu_selector{});
//   auto v1_ptr = (op_type*)sycl::malloc_shared(sizeof(op_type) * N, gpuQ);
// #else
//   printf("not USM\n");
//   std::vector<op_type> v1(N);
//   auto v1_ptr = v1.data();
// #endif
// //  std::vector<op_type> v2(N);
//   for (int i = 0; i < N; i++) {
//     v1_ptr[i] = i;
// //    v2[i] = i;
//   }
// //struct timeval start, end, start1, end1;
//   struct timeval start, compute, end;
//   gettimeofday(&start, NULL);
// #if USM
// #else
//   sycl::queue gpuQ(sycl::gpu_selector{});
// #endif
//   {
// //Splitting input data into 2 halves for gpu and iGPU
// #if USM
//     auto agpuv1 = v1_ptr;
// #else
//     sycl::buffer<op_type, 1> bufgpuv1((v1.data()), R);
// #endif
//     std::cout << "Running on: " << gpuQ.get_device().get_info<sycl::info::device::name>() << " (size: " << N << ")\n";
//     gettimeofday(&compute, NULL);
//     ios_base::sync_with_stdio(false);
//     auto ev = gpuQ.submit([&](sycl::handler &h) {
// #if USM
// #else
//       auto agpuv1 = bufgpuv1.get_access<sycl::access::mode::read_write>(h);
// #endif
//       //auto agpuv2 = bufgpuv2.get_access<sycl::access::mode::read_write>(h);
//       h.parallel_for(R, [=](sycl::id<1> i) {
//         //agpuv1[i]+=agpuv2[i];
//         for (int ii = 1; ii < its; ii++) {
//           op_type tanval = (sycl::sin(agpuv1[i]) * ii) / (sycl::cos(agpuv1[i]) * ii);
//           op_type secval = 1.0 / sycl::cos(agpuv1[i]);
//           agpuv1[i] = (secval * secval) - (tanval * tanval);
//         }
//       });
//     });
//     // gpuQ.wait(); // IMPORTANT!!!
//     ev.wait(); // IMPORTANT!!!
//   }
//   gettimeofday(&end, NULL);
//   auto tTemp = std::chrono::high_resolution_clock::now();
//   auto diffTemp = (tTemp - tStart).count();
//   auto diffTempS = diffTemp / 1e9;
//   printf("Time taken by data init + queue is: %6.4f\n", diffTempS);
//   double time_taken;
//   time_taken = (end.tv_sec - start.tv_sec) * 1e6;
//   time_taken = (time_taken + (end.tv_usec - start.tv_usec)) * 1e-6;
//   cout << "Time taken by queue is : " << fixed << time_taken << setprecision(6) << " sec " << "\n";
//   time_taken = (end.tv_sec - compute.tv_sec) * 1e6;
//   time_taken = (time_taken + (end.tv_usec - compute.tv_usec)) * 1e-6;
//   cout << "Time taken by kernel is : " << fixed << time_taken << setprecision(6) << " sec " << "\n";
// //  std::cout << "Sample values on GPU:\n";
// //  int step = N / 10;
// //  for (int i = 0; i < 10; i++) {
// //    std::cout << "[" << (i * step) << "] " << v1[(i * step)] << "\n";
// //  }
//
// //  for (int i=0; i<10; ++i){
// //    if (v1[i] == i){
// //      std::cout << "Failure at [" << i << "] " << v1[i] << "\n";
// //      break;
// //    }
// //  }
// //std::vector<int> validate_vector(N,N);
// //validate_vector==v1?std::cout<<"Vector addition: Success\n":std::cout<<"Vector addition: Failure\n";
//   printf("res: %p => %f, %f\n", v1_ptr, v1_ptr[0], v1_ptr[N - 1]);
  // may_verify(v1_ptr, its, N);
  return 0;
}
