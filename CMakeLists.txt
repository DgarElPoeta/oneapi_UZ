cmake_minimum_required(VERSION 3.8)
project (vector-add)

set(INTEL_DIR "/opt/intel/oneapi")
set(ONEAPI_V "2021.1.1")
set(CMAKE_VERBOSE_MAKEFILE On)

#set(CXXFLAGS "-O3 -g -std=c++17 -ferror-limit=1 -qopenmp")
set(CXXFLAGS "-O3 -g -std=c++17 -fintelfpga -ferror-limit=1 -qopenmp")
#set(CXXFLAGS "-O0 -g -std=c++17 -fintelfpga -ferror-limit=1")

cmake_host_system_information(RESULT HOSTNAME QUERY HOSTNAME)
message("HOSTNAME ${HOSTNAME}")
set(OMP_LIB "")
SET(CMAKE_CXX_COMPILER "dpcpp")
set(OpenMP_CXX_FLAGS " ")
set(OpenMP_CXX_LIB_NAMES "iomp5")
set(OMP_LIB "${OpenMP_CXX_LIB_NAMES}")
set(OpenMP_iomp5_LIBRARY "${INTEL_DIR}/compiler/${ONEAPI_V}/linux/compiler/lib/intel64_lin/libiomp5.so")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXXFLAGS}")

include_directories(SYSTEM "${INTEL_DIR}/compiler/${ONEAPI_V}/linux/include/sycl/")
include_directories(SYSTEM "${INTEL_DIR}/compiler/${ONEAPI_V}/linux/compiler/include/")
include_directories(SYSTEM /usr/include/)

IF(OPENMP_FOUND)
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  message("FOUND OPENMP")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
ENDIF()

set(PROGRAM_LIST
    matadd
    matmul
    rap
    nbody
    gaussian
    mandelbrot
    ray
)
foreach(PROGRAM ${PROGRAM_LIST})
  set(BINARY "${PROGRAM}_ndr")
  add_executable("${BINARY}" "benchmarks/${PROGRAM}.cpp" benchmarks/io.cpp)
  target_compile_definitions("${BINARY}" PUBLIC -DUSM=0 "-DQUEUE_NDRANGE=1")
#  target_link_libraries("${BINARY}" "-lpthread -l${OMP_LIB} -fopenmp")
#  target_link_libraries("${BINARY}" "-Xsprofile -lpthread -l${OMP_LIB} -fopenmp")
#  target_link_libraries("${BINARY}" "-Xshardware" "-lpthread -l${OMP_LIB}")
 target_link_libraries("${BINARY}" "-Xshardware -reuse-exe=/home/raulhs/oneapi_UC_UZ/builds/optimizacion/${BINARY}" "-lpthread -l${OMP_LIB}")
#  target_link_libraries("${BINARY}" "-v -Xsfp-model=fast -Xsfma -Xsfp-relaxed -Xsfpc -Xsffp-contract=fast -Xshardware -Xsprofile -reuse-exe=/home/raulhs/oneapi_UC_UZ/builds/c++_fpga/${BINARY}" "-lpthread -l${OMP_LIB}")
#  target_link_libraries("${BINARY}" "-Xshardware -reuse-exe=/home/raulhs/oneapi_UC_UZ/builds/extra_c++_fpga/${BINARY}" "-lpthread -l${OMP_LIB}")
#  target_link_libraries("${BINARY}" "-Xshardware -reuse-exe=/home/raulhs/oneapi_UC_UZ/builds/extra_c++_fpga/${BINARY}" "-lpthread -l${OMP_LIB}")
#  target_link_libraries("${BINARY}" "-Xshardware -reuse-exe=/home/raulhs/oneapi_UC_UZ/builds/c++_fpga_optimizada/${BINARY}" "-lpthread -l${OMP_LIB}")
#  target_link_libraries("${BINARY}" "-Xshardware -fsycl-link=early" "-lpthread -l${OMP_LIB}")
#  target_link_libraries("${BINARY}" "-v -Xsfp-model=fast -Xsfma -Xsfp-relaxed -Xsfpc -Xsffp-contract=fast -Xshardware -fsycl-link=early" "-lpthread -l${OMP_LIB}")
#  target_link_libraries("${BINARY}" "-Xshardware -Xsprofile -fsycl-link=early" "-lpthread -l${OMP_LIB}")
endforeach()

# binomial
set(BINARY "binomial_ndr")
add_executable("${BINARY}" "benchmarks/binomial.cpp" benchmarks/io.cpp)
target_compile_definitions("${BINARY}" PUBLIC -DUSM=0 "-DQUEUE_NDRANGE=0")
target_link_libraries("${BINARY}" "-lpthread -l${OMP_LIB}")
#target_link_libraries("${BINARY}" "-Xshardware -Xsprofile -reuse-exe=/home/raulhs/oneapi_UC_UZ/builds/extra_c++_fpga/${BINARY}" "-lpthread -l${OMP_LIB}")

message("Done")
