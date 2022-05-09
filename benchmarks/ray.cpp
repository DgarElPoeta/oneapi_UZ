//
// Created by radon on 17/09/20.
//

#include "ray.h"
#include "ray/queue.cpp"

using namespace std::chrono;

using std::ostream;

inline ostream&
operator<<(ostream& os, uchar4& t)
{
  os << "(" << (int)t[0] << "," << (int)t[1] << "," << (int)t[2] << "," << (int)t[3] << ")";
  return os;
}


#if !defined(USM)
#error USM should be 0 or 1
#endif
//typedef sycl::cl_uchar4 cl_uchar4;

//---

inline ostream &
operator<<(ostream &os, cl_uchar4 &t) {
  os << "(" << (int) t.s[0] << "," << (int) t.s[1] << "," << (int) t.s[2] << "," << (int) t.s[3] << ")";
  return os;
}

#include "schedulers/st.cpp"
#include "schedulers/dyn.cpp"
#include "schedulers/hg.cpp"

#define IF_LOGGING(x)

int
parse_scene_primitive(char* buf, Primitive* prim_list, int i)
{
  int type = 0, light = 0;
  cl_float4 center_normal;
  center_normal.s[0] = 0;
  center_normal.s[1] = 0;
  center_normal.s[2] = 0;
  center_normal.s[3] = 0;
  cl_float radius_depth = 1;
  if (sscanf(buf,
             "%d,%f,%f,%f,%f,%f,%f,%f,%f,%d,%f,%f,%f,%f",
             &type,
             &prim_list[i].m_color[0],
             &prim_list[i].m_color[1],
             &prim_list[i].m_color[2],
             &prim_list[i].m_refl,
             &prim_list[i].m_refr,
             &prim_list[i].m_refr_index,
             &prim_list[i].m_diff,
             &prim_list[i].m_spec,
             &light,
             &center_normal.s[0],
             &center_normal.s[1],
             &center_normal.s[2],
             &radius_depth) != 14) {
    // buf[0] != '#' && buf[0] != '\n') {
    /* allows comments if start with # */
    fprintf(stderr, "Scene file format invalid. Primitive %d\n", i + 1);
    return -1;
  }

  prim_list[i].is_light = (light == 0) ? 0 : 1;

  switch (type) {
    case 0:
      prim_list[i].type = prim_type::PLANE;
      prim_list[i].normal[0] = center_normal.s[0];
      prim_list[i].normal[1] = center_normal.s[1];
      prim_list[i].normal[2] = center_normal.s[2];
      prim_list[i].depth = radius_depth;
      break;
    case 1:
      prim_list[i].type = prim_type::SPHERE;
      prim_list[i].center[0] = center_normal.s[0];
      prim_list[i].center[1] = center_normal.s[1];
      prim_list[i].center[2] = center_normal.s[2];
      prim_list[i].radius = radius_depth;
      prim_list[i].sq_radius = radius_depth * radius_depth;
      prim_list[i].r_radius = 1.0f / radius_depth;
      break;
    default:
      fprintf(stderr, "Scene file format invalid. Primitive %d\n", i + 1);
      return -1;
  }
  IF_LOGGING(printf("--->%f\n", prim_list[i].m_refl));
  return 0;
}

Primitive*
load_scene(data_t* data, Options* opts)
{
  const char* scene = data->scene;

  int num_primitives = 0;

  FILE* scene_file;
  scene_file = fopen(scene, "r");

  if (scene_file == NULL) {
    fprintf(stderr, "Scene file not found\n");
    return NULL;
  }

  Primitive* prim_list;

  size_t bsize = 256;
  char buf[bsize];
  // scenefile format
  // line 1: n_primitives
  // line 2+: primitive
  // type,r,g,b,refl,refr,refr_index,diff,spec,is_light,center/normal_x,center/normal_y,center/normal_z,radius/depth
  // type 0 = PLANE, type 1 = SPHERE; is_light 0 = FALSE, 1 = TRUE

  // bool ok = true;
  bool first = true;
  int dummy;
  enum Phase
  {
    COUNTING = 2,
    PARSING = 1,
  };
  int phase = COUNTING; // with 0 will finish
  int parsed_primitive = 0;
  // 2 = counting, 1 = parsing
  while (phase) {
    first = true;
    while (fgets(buf, bsize, scene_file) != NULL) {
      int c = buf[0];
      if (c != '#' && c != '\n' && strncmp(buf, "type,", 5) != 0) {
        if (index(buf, ',') != NULL) {
          if (phase == COUNTING) {
            num_primitives++;
          } else { // parsing
            if (parse_scene_primitive(buf, prim_list, parsed_primitive) == -1) {
              return NULL;
            } else {
              parsed_primitive++;
            }
          }
        }
      }
    } // read

    if (phase == COUNTING) {
      fseek(scene_file, 0L, SEEK_SET);
      IF_LOGGING(cout << "num primitives: " << num_primitives << "\n");
      // prim_list = (Primitive*)malloc(sizeof(Primitive) * num_primitives);
#if USM
      prim_list = (Primitive*)sycl::malloc_shared(num_primitives * sizeof(Primitive), opts->gpuQ);
#else
      if (opts->usm) {
        // prim_ptr = (Primitive*)sycl::malloc_shared(data.n_primitives * sizeof(Primitive), opts->gpuQ);
        prim_list = (Primitive*)sycl::malloc_shared(num_primitives * sizeof(Primitive), opts->gpuQ);
      } else {
        // prim_ptr = (Primitive*)malloc(data.n_primitives * sizeof(Primitive));
        prim_list = (Primitive*)malloc(num_primitives * sizeof(Primitive));
      }
#endif

      if (prim_list == NULL) {
        fprintf(stderr, "Error: Failed to allocate primitive list memory on host.\n");
        return NULL;
      }
      memset(prim_list, 0, sizeof(Primitive) * (num_primitives));
    }

    phase--;
  }

  // rnoz
  data->n_primitives = num_primitives;

  return prim_list;
}


int
ray_begin(data_t* data, Options* opts)
{
  const char* scene = data->scene;
  int total_size = data->total_size;

  // Load scene
  IF_LOGGING(printf("Loading scene..\n"));
  Primitive* primitive_list;
  primitive_list = load_scene(data, opts);

  IF_LOGGING(cout << "primitives: " << data->n_primitives << "\n");

  IF_LOGGING(printf("- primitive: %f\n", primitive_list[2].m_refl));
  if (primitive_list == NULL) {
    fprintf(stderr, "Failed to load scene from file: %s\n", scene);
    return 1;
  }

  // Allocate pixels for result
  Pixel* out_pixels;

  cl_uint pixel_size_bytes = sizeof(Pixel) * total_size;
  // out_pixels = (Pixel*)malloc(pixel_size_bytes);

#if USM
  out_pixels = (Pixel*)sycl::malloc_shared(total_size * sizeof(Pixel), opts->gpuQ);
#else
  if (opts->usm) {
    // prim_ptr = (Primitive*)sycl::malloc_shared(data.n_primitives * sizeof(Primitive), opts->gpuQ);
    out_pixels = (Pixel*)sycl::malloc_shared(total_size * sizeof(Pixel), opts->gpuQ);
  } else {
    // prim_ptr = (Primitive*)malloc(data.n_primitives * sizeof(Primitive));
    out_pixels = (Pixel*)malloc(total_size * sizeof(Pixel));
  }
#endif

  if (out_pixels == NULL) {
    fprintf(stderr, "Error: Failed to allocate output pixel memory on host.\n");
    return 1;
  }

  // null all colors to 0
  memset(out_pixels, 0, pixel_size_bytes);

  data->A = primitive_list;
  data->C = out_pixels;
  return 0;
}

int
ray_end(data_t* data)
{
  data->retval = write_bmp_file(data->C, data->width, data->height, data->out_file);
  /* int r = 0; */
  /* for (r = 0; r < width * height; r++) { */
  /*   //	fprintf(stdout, "%d\n", out_pixels[r].x); */
  /* } */
  /* if (retval == 1) { */
  /*   fprintf(stderr, "Image write failed.\n"); */
  /*   return 1; */
  /* } */
  return data->retval;
}

void
data_t_init(data_t* data)
{
  data->depth = 5;
  data->fast_norm = 0;
  data->buil_norm = 0;
  data->nati_sqrt = 0;
  data->buil_dot = 0;
  data->buil_len = 0;
  data->viewp_w = 6.0;
  data->viewp_h = 4.5;
  data->camera_x = 0.0;
  data->camera_y = 0.25;
  data->camera_z = -7.0;
  data->scene = "def.scn";
  data->out_file = "/tmp/ray_out.bmp";
  data->progname = "raytracer";
  data->n_primitives = 0;
  data->retval = 0;
}

void process(bool cpu, Options* opts, int thr_id) {
  if (opts->algo == Algo::Dynamic) {
    process_dynamic(cpu, opts);
  } else if (opts->algo == Algo::HGuided) {
    process_hguided(cpu, opts);
  } else {
    process_static(cpu, opts, thr_id);
  }
}

int usage() {
  std::cout
      << "usage: <cpu|gpu|cpugpu> <static|dynamic> <num pkgs (dyn)|cpu proportion (st|hg)> <image width> <scene> \n"
      << "DEBUG=y\n"
      << "CHECK=y\n"
      << "MIN_CHUNK_MULTIPLIER=1,1 (cpu,gpu)\n";
  return 1;
}

bool verify(Ray* ray) {
  // std::vector<ptype> b2(size, 0);
  bool verification_passed = true;

  /*
  bool debug = false;
  int show_errors = 5;
  for (size_t i = 0; i<ray->size; ++i){
    const ptype kernel_value = b[i];
    const ptype host_value = b2[i];
    if (kernel_value != host_value) {
      fprintf(stderr, "VERIFICATION FAILED for element %ld: %d != %d (kernel != host value)\n", i, kernel_value, host_value);
      show_errors--;
      verification_passed = false;
      if (show_errors == 0){
        break;
      }
    }
  }
  */
  return verification_passed;
}

int main(int argc, char *argv[]) {
  argc--;
  if (argc < 5) {
    return usage();
  }

  std::chrono::high_resolution_clock::time_point tStart = std::chrono::high_resolution_clock::now();
  std::string type;
  std::string n;
  Mode mode = Mode::CPU;
  bool check = false;
  float cpu_prop = 0.0;
  //float gpu_prop = 0.0;
  int pkg_size = 0; // old dyn
  int num_pkgs = 0; // dyn
  bool usm = false;

  // num cpp threads
  int num_cpp_threads = 1;
  if(argc >= 6) num_cpp_threads = atoi(argv[6]);

  // scene (path)
  std::string scene = argv[5]; // 1000

  // size arg
  size_t image_size = atoi(argv[4]); // 1000

  // gpu proportion arg
  cpu_prop = 1 - atof(argv[3]);

  // scheduler arg
  Algo algo;
  std::string algo_str = argv[2];
  if (algo_str == "static") {
    algo = Algo::Static;
  }
  else if (algo_str == "dynamic") {
    algo = Algo::Dynamic;
    num_pkgs = atof(argv[3]);
  }
  else if (algo_str == "hguided") {
    algo = Algo::HGuided;
  }
  else {
    return usage();
  }

  // devices arg
  std::string mode_str = argv[1];
  if (mode_str == "cpugpu") {
    mode = Mode::CPUGPU;
  }
  else if (mode_str == "cpu") {
    mode = Mode::CPU;
    if (algo == Algo::Static) {
      cpu_prop = 1.0;
    }
  }
  else if (mode_str == "gpu") {
    mode = Mode::GPU;
    if (algo == Algo::Static) {
      cpu_prop = 0.0;
    }
  }
  else {
    return usage();
  }

  // env variables
  char *debug_str = getenv("DEBUG");
  bool debug = (debug_str != NULL && std::string(debug_str) == "y");

  char *check_str = getenv("CHECK");
  check = (check_str != NULL && std::string(check_str) == "y");

  char *K_str = getenv("HGUIDED_K");
  float K = 2.0;
  if (K_str != nullptr) {
    float K_ = std::stof(K_str);
    //if (K_ > 1.0 && K_ < 10) {
      K = K_;
    //}
  }

  data_t data;
  data_t_init(&data);

  data.width = image_size;
  data.height = image_size;
  size_t size = image_size * image_size;
  data.total_size = size;
  data.scene = scene.c_str();

  int depth = data.depth;
  int fast_norm = data.fast_norm;
  int buil_norm = data.buil_norm;
  int nati_sqrt = data.nati_sqrt;
  int buil_dot = data.buil_dot;
  int buil_len = data.buil_len;
  int width = data.width;
  int height = data.height;
  float viewp_w = data.viewp_w;
  float viewp_h = data.viewp_h;
  float camera_x = data.camera_x;
  float camera_y = data.camera_y;
  float camera_z = data.camera_z;


  Ray ray;
  Options opts;
  // ray.size = x * y;
  // ray.M = M;

  // std::vector<ptype> a;
  // std::vector<ptype> b;
  // std::vector<ptype> func;
  // Primitive* prim_ptr;
  // Pixel* pixels_ptr;
  // ptype* func_ptr;

  ray_begin(&data, &opts);
  ray.prim_ptr = data.A;
  ray.pixels_ptr = data.C;
  ray.n_primitives = data.n_primitives;
  ray.size = data.total_size;
  ray.data = &data;

  std::mutex m;
  // int size = ray.size;
  size_t offset = 0;
  //  int pkg_size = 1024 * 100;
  int pkgid = 1;

  opts.K = K;
  opts.usm = usm;
  opts.debug = debug;
  opts.p_total_size = size;
  opts.p_rest_size = &size;
  //  opts.pkg_size = pkg_size;
  opts.num_pkgs = num_pkgs;
  opts.cpu_prop = cpu_prop;
  opts.p_offset = &offset;
  opts.p_pkgid = &pkgid;
  opts.m = &m;
  opts.ptr1 = nullptr;
  opts.tStart = tStart;
  opts.algo = algo;
  opts.mode = mode;
  opts.num_cpp_threads = num_cpp_threads;

  int min_multiplier[2] = {1, 1};
  std::string multiplier("");
  char *val = getenv("MIN_CHUNK_MULTIPLIER");
  if (val != nullptr) {
    multiplier = std::string(val);
    std::stringstream ss(multiplier);
    std::string item;
    auto i = 0;
    while (std::getline(ss, item, ',')) {
      min_multiplier[i] = std::stoi(item);
      i++;
    }
  }
  opts.min_multiplier_gpu = min_multiplier[0];
  opts.min_multiplier_cpu = min_multiplier[1];

  // std::mutex m_profiling_acc;
  // opts.m_profiling_acc = &m_profiling_acc;
  double cpu_end = 0.0, gpu_end = 0.0, compute_cpu = 0.0, compute_gpu = 0.0, profiling_q_cpu = 0.0,
      profiling_q_gpu = 0.0;
  opts.cpu_end = &cpu_end;
  opts.gpu_end = &gpu_end;
  opts.compute_cpu = &compute_cpu;
  opts.compute_gpu = &compute_gpu;
  opts.profiling_q_cpu = &profiling_q_cpu;
  opts.profiling_q_gpu = &profiling_q_gpu;

  if (debug) {
    std::cout << "N: " << ray.size << " cpu prop: " << opts.cpu_prop << " min pkg size: " << opts.pkg_size <<
              " min_multiplier(hg: cpu,gpu) " << opts.min_multiplier_cpu << ","
              << opts.min_multiplier_gpu << "\n";
  }

  // Matmul
//  char* filter_size_str = getenv("FILTER_SIZE");
//  if (filter_size_str == NULL){
//    std::cout << "needs FILTER_SIZE\n";
//    return usage();
//  }
//  int filterSize = atoi(filter_size_str);
  int imageWidth, imageHeight;
  imageWidth = imageHeight = size;

  int problem_size = ray.size;
  *opts.p_rest_size = problem_size;
  size_t lws = 128;
  size_t gws = problem_size;

  // struct timeval start, end;
  // gettimeofday(&start, NULL);
  auto tLaunchStart = std::chrono::high_resolution_clock::now();
  opts.tLaunchStart = tLaunchStart;
  opts.p_problem = &ray;
  opts.lws = lws;
  opts.pkg_size_multiple = lws;
  opts.setup();

  if (mode == Mode::CPU) {
    std::vector<std::thread> vecOfThreads;
    for(int i=1; i<num_cpp_threads; i++){
      vecOfThreads.push_back(std::thread(process, true, &opts, i));
    }
    tLaunchStart = std::chrono::high_resolution_clock::now();
    opts.tLaunchStart = tLaunchStart;
    process(true, &opts, 0);
    for (std::thread & th : vecOfThreads){
      if (th.joinable()) th.join();
    }
  } else if (mode == Mode::GPU) {
    process(false, &opts, 0);
  } else {
    std::vector<std::thread> vecOfThreads;
    for(int i=0; i<num_cpp_threads; i++){
      vecOfThreads.push_back(std::thread(process, true, &opts, i));
    }
    tLaunchStart = std::chrono::high_resolution_clock::now();
    opts.tLaunchStart = tLaunchStart;
    process(false, &opts, 0);
    for (std::thread & th : vecOfThreads){
      if (th.joinable()) th.join();
    }
  }

  auto tLaunchEnd = std::chrono::high_resolution_clock::now();
  auto diffLaunchEnd = (tLaunchEnd - tLaunchStart).count();
  auto diffLaunchEndMs = diffLaunchEnd / 1e6;

  // gettimeofday(&end, NULL);
  // double time_taken;
  // time_taken = (end.tv_sec - start.tv_sec) * 1e6;
  // time_taken = (time_taken + (end.tv_usec - start.tv_usec)) * 1e-6;
  // cout << "Time taken by queue is : " << std::fixed << time_taken << std::setprecision(6) << " sec " << "\n";
  // std::cout << "Sample values on GPU and CPU\n";
  printf("More Options:\n");
  printf("  size: %lu\n", image_size);
  printf("  scene path: %s\n", scene.c_str());
  // printf("Runtime init timestamp: %.10s\n", std::to_string(opts.tStart.time_since_epoch().count()).c_str());
  printf("Program init timestamp: %.3f\n", (std::chrono::duration_cast<std::chrono::milliseconds>(tStart.time_since_epoch()).count() / 1000.0));
  printf("Runtime init timestamp: %.3f\n", (std::chrono::duration_cast<std::chrono::milliseconds>(tLaunchStart.time_since_epoch()).count() / 1000.0));
  printf("Kernel: raytracer_kernel\n");
  printf("OpenCL gws: (%lu) lws: (%lu)\n", ray.size, lws);
#if USE_LOCAL_MEM == 1
  printf("Kernel mode: local mem\n");
#else
  printf("Kernel mode: global mem\n");
#endif
  printf("Memory mode: %s\n", false ? "usm" : "normal");

  if (mode == Mode::GPU || mode == Mode::CPUGPU) {
    printf("Device id: 0\n");
    {
      sycl::queue q(sycl::accelerator_selector{});
      printf("Selected device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    }
    cout << "works: " << opts.mChunksGPU.size() << " works_size: " << opts.worksizeGPU << "\n";
    printf("duration increments:\n");
    printf(" completeKernel: %d ms. (and Read)\n", (int)std::round(*opts.profiling_q_gpu));
    printf(" total: %d ms.\n", (int)std::round(*opts.compute_gpu));
    printf("duration offsets from init:\n");
    printf(" deviceEnd: %d ms.\n", (int)std::round(*opts.gpu_end));

    cout << "chunks (mOffset+mSize:ts_ms+duration_ms+duration_read_ms)";
    cout << "type-chunks,";
    for (auto chunk : opts.mChunksGPU) {
      cout << chunk.offset << "+" << chunk.size << ":" << chunk.ts_ms << "+" << chunk.duration_ms
           << "+" << chunk.duration_read_ms << ",";
    }
    cout << "\n";
  }
  if (mode == Mode::CPU || mode == Mode::CPUGPU) {
    printf("Device id: 1\n");
    {
      sycl::queue q(cpu_selector{});
      printf("Selected device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    }
    cout << "works: " << opts.mChunksCPU.size() << " works_size: " << opts.worksizeCPU << "\n";
    printf("duration increments:\n");
    printf(" completeKernel: %d ms. (and Read)\n", (int)std::round(*opts.profiling_q_cpu));
    printf(" total: %d ms.\n", (int)std::round(*opts.compute_cpu));
    printf("duration offsets from init:\n");
    printf(" deviceEnd: %d ms.\n", (int)std::round(*opts.cpu_end));

    cout << "chunks (mOffset+mSize:ts_ms+duration_ms+duration_read_ms)";
    cout << "type-chunks,";
    for (auto chunk : opts.mChunksCPU) {
      cout << chunk.offset << "+" << chunk.size << ":" << chunk.ts_ms << "+" << chunk.duration_ms
           << "+" << chunk.duration_read_ms << ",";
    }
    cout << "\n";
  }

  printf("scheduler: ");
  if (algo == Algo::Static){
    printf("Static\n");
  } else if (algo == Algo::Dynamic) {
    printf("Dynamic\n");
  } else if (algo == Algo::HGuided) {
    printf("HGuided\n");
    printf("scheduler parameters:\n");
    printf(" K: %.1f\n", opts.K);
    printf(" minChunkMultiplier (gpu,cpu): %d,%d\n", opts.min_multiplier_gpu, opts.min_multiplier_cpu);
  }
  // printf("chunks: %d\n", *opts.p_pkgid - 1);
  printf("chunks: %lu\n", opts.mChunksCPU.size() + opts.mChunksGPU.size());
  printf("duration offsets from init:\n");
  printf(" schedulerEnd: %d ms.\n", (int)std::round(diffLaunchEndMs));

  if (check) {
    if (verify(&ray)) {
      std::cout << "Success\n";
    } else {
      std::cout << "Failure\n";
    }
  }

  char *image_str = getenv("IMAGE");
  bool image = (image_str != NULL && std::string(image_str) == "y");
  if (image){
    data.out_file = "ray.bmp";
    ray_end(&data);
    // transform_image((cl_uchar4*)mandelbrot.out, mandelbrot.width, mandelbrot.width);
    // write_bmp_file((Pixel*)ray.pixels_ptr, ray.data->width, ray.data->height, "ray.bmp");
  }

  // printf("forcing the computation and read: %d...%d\n", ray.pixels_ptr[0][0], ray.pixels_ptr[ray.size - 1][0]);
  cout << "Output values: " << ray.pixels_ptr[0] << "..." << ray.pixels_ptr[size - 1] << "\n";
  if (usm){
    sycl::free(ray.prim_ptr, opts.gpuQ);
    sycl::free(ray.pixels_ptr, opts.gpuQ);
  } else {
    free(ray.prim_ptr);
    free(ray.pixels_ptr);
  }

  return 0;
}
