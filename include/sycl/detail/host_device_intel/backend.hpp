//==-------------- backend.hpp --- Host device backend ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

// Support for different backends for host device
//
// A backend is expected to implement a specific set of
// functions for kernel execution and related things.
//
// The backend is choosen at compile-time by specifying
// a corresponding build flag.
//
// Supported backens:
// - Serial(DPCPP_HOST_DEVICE_SERIAL option)
//   A kernel is executed serially in a loop, barriers
//   are not supported.
// - OpenMP(DPCPP_HOST_DEVICE_OPENMP options)
//   A kernel is executed using OpenMP threading.
// - OpenMP + Work-group transformations
//   (DPCPP_HOST_DEVICE_PERF_NATIVE options)
//   A kernel is transformed by a set of
//   work-group passes(e.g. barrier handling)
//   It's executed using OpenMP threading
//
// By default serial implementation is used.

// Choose default backend
#if !defined(DPCPP_HOST_DEVICE_OPENMP) &&                                      \
    !defined(DPCPP_HOST_DEVICE_PERF_NATIVE)
#define DPCPP_HOST_DEVICE_SERIAL 1
#endif

#if DPCPP_HOST_DEVICE_OPENMP
#define DPCPP_HOST_DEVICE_HAS_BARRIER 1
#include <CL/sycl/detail/host_device_intel/openmp_backend.hpp>
#elif DPCPP_HOST_DEVICE_PERF_NATIVE
#include <CL/sycl/detail/host_device_intel/perf_native_backend.hpp>
#elif DPCPP_HOST_DEVICE_SERIAL
// No additional include required
#else
#error "Unknown backend"
#endif
