cmake_minimum_required(VERSION 3.8)
project (coexecutor)

set(CMAKE_VERBOSE_MAKEFILE On)

SET(CMAKE_CXX_COMPILER "dpcpp")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fintelfpga")
SET(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} -lpthread")

set(PROGRAM_LIST
    matadd
    matmul
    rap
    nbody
    gaussian
    mandelbrot
    ray
    binomial
)
foreach(PROGRAM ${PROGRAM_LIST})
  set(BINARY "${PROGRAM}_ndr")
  add_executable("${BINARY}" "benchmarks/${PROGRAM}.cpp")
  target_compile_definitions("${BINARY}" PUBLIC -DUSM=0 "-DQUEUE_NDRANGE=1")
  target_link_libraries("${BINARY}" "")
#  target_link_libraries("${BINARY}" "-Xshardware -fsycl-link=early" "-lpthread -l${OMP_LIB}")
#  target_link_libraries("${BINARY}" "-v -Xsfp-model=fast -Xsfma -Xsfp-relaxed -Xsfpc -Xsffp-contract=fast -Xshardware -fsycl-link=early" "-lpthread -l${OMP_LIB}")
endforeach()

message("Done")
