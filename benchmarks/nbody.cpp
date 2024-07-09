//
// Created by radon on 17/09/20.
//
#include "nbody.h"
#include "nbody/queue.cpp"

using namespace std::chrono;

using std::ostream;

inline ostream&
operator<<(ostream& os, float4& t)
{
  os << "(" << (int)t[0] << "," << (int)t[1] << "," << (int)t[2] << "," << (int)t[3] << ")";
  return os;
}

//typedef sycl::cl_uchar4 cl_uchar4;

#include "schedulers/st.cpp"
#include "schedulers/dyn.cpp"
#include "schedulers/hg.cpp"

void process(bool cpu, Options* opts, int thr_id) {
  if (opts->algo == Algo::Dynamic) {
    process_dynamic(cpu, opts);
  } else if (opts->algo == Algo::HGuided) {
    process_hguided(cpu, opts);
  } else {
    process_static(cpu, opts, thr_id);
  }
}

float
random(float rand_max, float rand_min)
{
    float result;
    result = (float)rand() / (float)RAND_MAX;

    return ((1.0f - result) * rand_min + result * rand_max);
}

int usage() {
  std::cout
      << "usage: <cpu|gpu|cpugpu> <static|dynamic> <num pkgs (dyn)|gpu proportion (st|hg)> <size> \n"
      << "DEBUG=y\n"
      << "CHECK=y\n"
      << "MIN_CHUNK_MULTIPLIER=1,1 (cpu,gpu)\n";
  return 1;
}

void
nBodyCPUReference(uint numBodies,
                  float delT,
                  float espSqr,
                  float* currentPos,
                  float* currentVel,
                  float* newPos,
                  float* newVel)
{
    // Iterate for all samples
    for (cl_uint i = 0; i < numBodies; ++i) {
        int myIndex = 4 * i;
        float acc[3] = { 0.0f, 0.0f, 0.0f };
        for (cl_uint j = 0; j < numBodies; ++j) {
            float r[3];
            int index = 4 * j;
            float distSqr = 0.0f;
            for (int k = 0; k < 3; ++k) {
                r[k] = currentPos[index + k] - currentPos[myIndex + k];
                distSqr += r[k] * r[k];
            }
            float invDist = 1.0f / sqrt(distSqr + espSqr);
            float invDistCube = invDist * invDist * invDist;
            float s = currentPos[index + 3] * invDistCube;
            for (int k = 0; k < 3; ++k) {
                acc[k] += s * r[k];
            }
        }
        for (int k = 0; k < 3; ++k) {
            newPos[myIndex + k] =
                    currentPos[myIndex + k] + currentVel[myIndex + k] * delT + 0.5f * acc[k] * delT * delT;
            newVel[myIndex + k] = currentVel[myIndex + k] + acc[k] * delT;
        }
        newPos[myIndex + 3] = currentPos[myIndex + 3];
    }
}


bool verify(Nbody* nbody) {
  auto num_bodies = nbody->size;
  bool verification_passed = true;

  float4* pos_out = (float4*)malloc(num_bodies * sizeof(float4));
  float4* vel_out = (float4*)malloc(num_bodies * sizeof(float4));
  nBodyCPUReference(num_bodies, nbody->delT, nbody->espSqr, (float*)nbody->pos_in, (float*)nbody->vel_in, (float*)pos_out, (float*)vel_out);

  bool debug = false;

    int show_errors = 5;
    uint buffer_size = num_bodies;
    for (size_t i = 0; i < buffer_size; ++i) {
        float4 kernel_value = nbody->pos_out[i];
        float4 host_value = pos_out[i];
        auto diffx = abs(kernel_value[0] - host_value[0]);
        auto diffy = abs(kernel_value[1] - host_value[1]);
        auto diffz = abs(kernel_value[2] - host_value[2]);
        if (diffx >= THRESHOLD || diffy >= THRESHOLD || diffz >= THRESHOLD) {
          fprintf(stderr, "VERIFICATION FAILED for element %ld: (%f,%f,%f) != (%f,%f,%f) (kernel != host value)\n", i, kernel_value[0], kernel_value[1], kernel_value[2], host_value[0], host_value[0], host_value[1]);
            show_errors--;
            verification_passed = false;
            if (show_errors == 0){
                break;
            }
        }
    }
    free(pos_out);
    free(vel_out);
  return verification_passed;
}

int main(int argc, char *argv[]) {
  argc--;
  if (argc < 4) {
    return usage();
  }

  std::chrono::high_resolution_clock::time_point tStart = std::chrono::high_resolution_clock::now();
  std::string type;
  std::string n;
  Mode mode = Mode::CPU;
  bool check = false;
  float cpu_prop = 0.0;
  float gpu_prop = 0.0;
  int pkg_size = 0; // old dyn
  int num_pkgs = 0; // dyn

  // num cpp threads
  int num_cpp_threads = 1;
  if(argc >= 5) num_cpp_threads = atoi(argv[5]);

  // size arg
  size_t size = atoi(argv[4]);

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

  Nbody nbody;
  uint groupSize = 64;
  uint numBodies = size;
  cl_float delT = DEL_T;
  cl_float espSqr = ESP_SQR;
  nbody.delT = delT;
  nbody.espSqr = espSqr;
  numBodies = (uint)(((size_t)numBodies < groupSize) ? groupSize : numBodies);
  numBodies = (uint)((numBodies / groupSize) * groupSize);
  size = numBodies;
  nbody.size = size;

  size_t lws = groupSize;
  size_t gws = size;

  std::vector<ptype> pos_in;
  std::vector<ptype> vel_in;
  std::vector<ptype> pos_out;
  std::vector<ptype> vel_out;
  pos_in = std::vector<ptype>(nbody.size);
  vel_in = std::vector<ptype>(nbody.size); // set 0 in loop
  pos_out = std::vector<ptype>(nbody.size);
  vel_out = std::vector<ptype>(nbody.size); // set 0 in loop
  ptype* pos_in_ptr;
  ptype* vel_in_ptr;
  ptype* pos_out_ptr;
  ptype* vel_out_ptr;
  pos_in_ptr = pos_in.data();
  vel_in_ptr = vel_in.data();
  pos_out_ptr = pos_out.data();
  vel_out_ptr = vel_out.data();
  nbody.pos_in = pos_in_ptr;
  nbody.vel_in = vel_in_ptr;
  nbody.pos_out = pos_out_ptr;
  nbody.vel_out = vel_out_ptr;

  srand(0);
  for (uint i = 0; i < nbody.size; ++i) {
      pos_in_ptr[i] = float4{random(3, 50), random(3, 50), random(3, 50), random(1, 1000)};
      vel_in_ptr[i] = float4{0.0f};
      auto pos_in_fptr = pos_in_ptr[i];
      auto vel_in_fptr = vel_in_ptr[i];
  }

  std::mutex m;
  size = nbody.size;
  size_t offset = 0;
  int pkgid = 1;

  Options opts;
  opts.K = K;
  opts.usm = false;
  opts.debug = debug;
  opts.p_total_size = size;
  opts.p_rest_size = &size;
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

  double cpu_end = 0.0, gpu_end = 0.0, compute_cpu = 0.0, compute_gpu = 0.0, profiling_q_cpu = 0.0,
      profiling_q_gpu = 0.0;
  opts.cpu_end = &cpu_end;
  opts.gpu_end = &gpu_end;
  opts.compute_cpu = &compute_cpu;
  opts.compute_gpu = &compute_gpu;
  opts.profiling_q_cpu = &profiling_q_cpu;
  opts.profiling_q_gpu = &profiling_q_gpu;

  if (debug) {
    std::cout << "N: " << nbody.size << " cpu prop: " << opts.cpu_prop << " min pkg size: " << opts.pkg_size <<
              " min_multiplier(hg: cpu,gpu) " << opts.min_multiplier_cpu << ","
              << opts.min_multiplier_gpu << "\n";
  }

  int problem_size = nbody.size;
  *opts.p_rest_size = problem_size;
//  int lws = 64;
//  int gws = problem_size;

  auto tLaunchStart = std::chrono::high_resolution_clock::now();
  opts.tLaunchStart = tLaunchStart;
  opts.p_problem = &nbody;
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

  printf("More Options:\n");
  printf("  size: %ld\n", nbody.size);
  printf("Program init timestamp: %.3f\n", (std::chrono::duration_cast<std::chrono::milliseconds>(tStart.time_since_epoch()).count() / 1000.0));
  printf("Runtime init timestamp: %.3f\n", (std::chrono::duration_cast<std::chrono::milliseconds>(tLaunchStart.time_since_epoch()).count() / 1000.0));
  printf("Kernel: nbody\n");
  printf("OpenCL gws: (%lu) lws: ()\n", nbody.size);

  if (mode == Mode::GPU || mode == Mode::CPUGPU) {
    printf("Device id: 0\n");
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
  printf("chunks: %lu\n", opts.mChunksCPU.size() + opts.mChunksGPU.size());
  printf("duration offsets from init:\n");
  printf(" schedulerEnd: %d ms.\n", (int)std::round(diffLaunchEndMs));

  if (check) {
    if (verify(&nbody)){
      std::cout << "Success\n";
    } else {
      std::cout << "Failure\n";
    }
  }

  cout << "Output values: \n"
       << "  positions with mass " << pos_out_ptr[0] << "..." << pos_out_ptr[size - 1] << "\n"
       << "  velocity " << vel_out_ptr[0] << "..." << vel_out_ptr[size - 1] << "\n";
  return 0;
}
