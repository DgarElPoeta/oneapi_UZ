cmake_minimum_required(VERSION 3.13)
project(oneapi_UZ)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

set(CMAKE_VERBOSE_MAKEFILE On)

# Set the C++ SYCL compiler
SET(CMAKE_CXX_COMPILER "icpx")

# Set the compile flags
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -sycl-std=2020 -Wall -fno-sycl-id-queries-fit-in-int ${USER_OPT_FLAGS}")

# Set the link flags
SET(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} -fsycl -lpthread")

# If variable BENCHMARK is not defined, show an error message
if(NOT DEFINED BENCHMARK)
  message(FATAL_ERROR "BENCHMARK variable is not defined. Use -DBENCHMARK=<benchmark_name>\n\t posible benchmark names: binomial, gaussian, mandelbrot, matadd, matmul, nbody, rap, ray")
endif()

# Define variables with the names of the versions
set(BASE_VERSION "base")
set(OPT_VERSION "opt")
set(PRUEBA_VERSION "prueba")

# If variable VERSION is not defined, show an error message 
if(NOT DEFINED KERNEL_VERSION)
  message(FATAL_ERROR "VERSION variable is not defined. Use -DVERSION=<version_name>\n\t posible version names: base, opt")
elseif(NOT (KERNEL_VERSION STREQUAL BASE_VERSION OR KERNEL_VERSION STREQUAL OPT_VERSION OR KERNEL_VERSION STREQUAL PRUEBA_VERSION))
  message(FATAL_ERROR "Unknown version: ${VERSION}\n\t posible version names: base, opt")
endif()

# Define variables with the names of the benchmarks
set(GAUSSIAN "gaussian")
set(MANDELBROT "mandelbrot")
set(MATADD "matadd")
set(MATMUL "matmul")
set(NBODY "nbody")
set(RAP "rap")


# Add include directories
set(INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/include
                 ${CMAKE_SOURCE_DIR}/benchmarks/${BENCHMARK})
include_directories(${INCLUDE_DIRS})

# If variable BENCHMARK is equal to variable GAUSSIAN, add subdirectory benchmarks/gaussian
if(BENCHMARK STREQUAL GAUSSIAN)
  add_subdirectory(benchmarks/gaussian)
# If variable BENCHMARK is equal to variable MANDELBROT, add subdirectory benchmarks/mandelbrot
elseif(BENCHMARK STREQUAL MANDELBROT)
  add_subdirectory(benchmarks/mandelbrot)
# If variable BENCHMARK is equal to variable MATADD, add subdirectory benchmarks/matadd
elseif(BENCHMARK STREQUAL MATADD)
  add_subdirectory(benchmarks/matadd)
# If variable BENCHMARK is equal to variable MATMUL, add subdirectory benchmarks/matmul
elseif(BENCHMARK STREQUAL MATMUL)
  add_subdirectory(benchmarks/matmul)
# If variable BENCHMARK is equal to variable NBODY, add subdirectory benchmarks/nbody
elseif(BENCHMARK STREQUAL NBODY)
  add_subdirectory(benchmarks/nbody)
# If variable BENCHMARK is equal to variable RAP, add subdirectory benchmarks/rap
elseif(BENCHMARK STREQUAL RAP)
  add_subdirectory(benchmarks/rap)
# If variable BENCHMARK is equal to variable RAY, add subdirectory benchmarks/ray
else()
  message(FATAL_ERROR "Unknown benchmark: ${BENCHMARK}\n\t posible benchmark names: binomial, gaussian, mandelbrot, matadd, matmul, nbody, rap, ray")
endif()

