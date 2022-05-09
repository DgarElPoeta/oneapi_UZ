//==-------------- perf_native_backend.hpp.hpp --- PERF backend ------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

// OpenMP based host device backend with additional work-group transformations.

#include <CL/sycl/detail/helpers.hpp>
#include <CL/sycl/id.hpp>
#include <CL/sycl/item.hpp>
#include <iostream>
#include <thread>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

using namespace cl;

// Settings for work-group size for range-based parallel_for for each dimension
#ifndef DPCPP_PERF_HOST_WG_SIZE0
#define DPCPP_PERF_HOST_WG_SIZE0 16
#endif

#ifndef DPCPP_PERF_HOST_WG_SIZE1
#define DPCPP_PERF_HOST_WG_SIZE1 16
#endif

#ifndef DPCPP_PERF_HOST_WG_SIZE2
#define DPCPP_PERF_HOST_WG_SIZE2 16
#endif

// Make the name available during parsing, it's resolved by WGLoopCreator pass.
extern "C" {
size_t __builtin_get_local_id(size_t);
}

// The functions marked as annotate("sycl_kernel") are transformed by the compiler
// in a following way:
// - Implicit loop over the function body is created with LocalRange* parameters
// as its boundaries. So even the parameters are not used in the code, they will
// have usage after transformations. For unused dimensions, e.g. LocalRange2 for 1D
// case, value 1 is expected.
// - __builtin_get_local_id is resolved to the current loop iteration number.
template <class KernelFunc>
__attribute__((noinline))
__attribute__((annotate("sycl_kernel")))
void ParallelForNDRangeImplKernel1D(size_t GroupId0, size_t /*GroupId1*/,
                                    size_t /*GroupId2*/, size_t /*LocalRange0*/,
                                    size_t /*LocalRange1 = 1*/, size_t /*LocalRange2 = 1*/,
                                    KernelFunc Kernel) {
// The function is transformed to the following code:
// for (size_t LocalId2 = 0; LocalId2 < LocalRange2; ++LocalId1) {
//   for (size_t LocalId1 = 0; LocalId1 < LocalRange1; ++LocalId1) {
//     for (size_t LocalId0 = 0; LocalId0 < LocalRange0; ++LocalId0) {
//       Kernel({GroupId0}, {LocalId0});
//     }
//   }
// }
  Kernel({GroupId0}, {__builtin_get_local_id(0)});
// TODO: Throw an exception if the code is compiled by a regular compiler
}

// Implementation of parallel_for with nd_range(different overloads for 1/2/3
// dimensions)
//
// Loop over work-groups is executed in parallel via #pragma omp parallel for.
// Each iteration executes loop over work-items(the loop is generated implicitly
// by the compiler) inside of ParallelForNDRangeImplKernel* functions.
template <class KernelFunc>
void ParallelForNDRangeImpl(sycl::range<1> LocalRange,
                            sycl::range<1> GroupRange, KernelFunc Kernel) {
#pragma omp parallel for
  for (size_t GroupID0 = 0; GroupID0 < GroupRange.get(0); ++GroupID0) {
    ParallelForNDRangeImplKernel1D(GroupID0, 0, 0, LocalRange.get(0), 1, 1,
                                   Kernel);
  }
}

template <class KernelFunc>
__attribute__((noinline))
__attribute__((annotate("sycl_kernel")))
void ParallelForNDRangeImplKernel2D(size_t GroupId0, size_t GroupId1,
                                    size_t /*GroupId2*/, size_t /*LocalRange0*/,
                                    size_t /*LocalRange1*/, size_t /*LocalRange2=1*/,
                                    KernelFunc Kernel) {
// The function is transformed to the following code:
// for (size_t LocalId2 = 0; LocalId2 < LocalRange2; ++LocalId1) {
//   for (size_t LocalId1 = 0; LocalId1 < LocalRange1; ++LocalId1) {
//     for (size_t LocalId0 = 0; LocalId0 < LocalRange0; ++LocalId0) {
//       Kernel({GroupId0, GroupId1}, {LocalId1, LocalId0});
//     }
//   }
// }
  Kernel({GroupId0, GroupId1},
        // For the generated code(from OpenCL passes) LocalId0 is a
        // fasted-moving index which doesn't match SYCL which assumes
        // LocalId1 as the fastest one. In order to get optimal performance
        // and preserve correctness do the following:
        // LocalSize0 <-> LocalSize1 (done before kernel execution)
        // LocalId0 <-> LocalId1
         {__builtin_get_local_id(1), __builtin_get_local_id(0)});
}

template <class KernelFunc>
void ParallelForNDRangeImpl(sycl::range<2> LocalRange,
                            sycl::range<2> GroupsRange, KernelFunc Kernel) {
#pragma omp parallel for collapse(2)
  for (size_t GroupID0 = 0; GroupID0 < GroupsRange.get(0); ++GroupID0) {
    for (size_t GroupID1 = 0; GroupID1 < GroupsRange.get(1); ++GroupID1) {
      // Flip the ranges to make LocalId1 a fastest moving index(see comment
      // in ParallelForNDRangeImplKernel2D).
      ParallelForNDRangeImplKernel2D(GroupID0, GroupID1, 0, LocalRange.get(1),
                                     LocalRange.get(0), 1, Kernel);
    }
  }
}

template <class KernelFunc>
__attribute__((noinline)) __attribute__((annotate("sycl_kernel"))) void
ParallelForNDRangeImplKernel3D(size_t GroupId0, size_t GroupId1,
                               size_t GroupId2, size_t /*LocalRange0*/,
                               size_t /*LocalRange1*/, size_t /*LocalRange2*/,
                               KernelFunc Kernel) {
// The function is transformed to the following code:
// for (size_t LocalId2 = 0; LocalId2 < LocalRange2; ++LocalId1) {
//   for (size_t LocalId1 = 0; LocalId1 < LocalRange1; ++LocalId1) {
//     for (size_t LocalId0 = 0; LocalId0 < LocalRange0; ++LocalId0) {
//       Kernel({GroupId0, GroupId1, GroupId2}, {LocalId2, LocalId1, LocalId0});
//     }
//   }
// }
  Kernel({GroupId0, GroupId1, GroupId2},
     // For the generated code(from OpenCL passes) LocalId0 is a
     // fasted-moving index which doesn't match SYCL which assumes
     // LocalId2 as the fastest one. In order to get optimal performance
     // and preserve correctness do the following:
     // LocalSize0 <-> LocalSize2 (done before kernel execution)
     // LocalId0 <-> LocalId2
      {__builtin_get_local_id(2), __builtin_get_local_id(1),
       __builtin_get_local_id(0)});
}

template <class KernelFunc>
void ParallelForNDRangeImpl(sycl::range<3> LocalRange,
                            sycl::range<3> GroupsRange, KernelFunc Kernel) {
#pragma omp parallel for collapse(3)
  for (size_t GroupID0 = 0; GroupID0 < GroupsRange.get(0); ++GroupID0) {
    for (size_t GroupID1 = 0; GroupID1 < GroupsRange.get(1); ++GroupID1) {
      for (size_t GroupID2 = 0; GroupID2 < GroupsRange.get(2); ++GroupID2) {
        // Flip the ranges to make LocalId2 a fastest moving index(see comment
        // in ParallelForNDRangeImplKernel3D).
        ParallelForNDRangeImplKernel3D(GroupID0, GroupID1, GroupID2,
                                       LocalRange.get(2), LocalRange.get(1),
                                       LocalRange.get(0), Kernel);
      }
    }
  }
}

template <int Dim, class KernelFunc>
void ParallelForNDRange(sycl::range<Dim> LocalRange,
                        sycl::range<Dim> GroupsRange, KernelFunc Kernel) {
  ParallelForNDRangeImpl(LocalRange, GroupsRange, Kernel);
}

// Implementation of parallel_for without nd_range(different overloads for 1/2/3
// dimensions)
//
// Split the global range into work-groups of predefined size(
// specified by DPCPP_PERF_HOST_WG_SIZE[dimension] macro). Loop over work-groups
// is executed in parallel via #pragma omp parallel for. On each iteration a
// loop over work-items which is generated by the compiler is executed(same
// approach as parallel_for with nd_range).
template <typename KernelFunc>
void ParallelForImpl(const sycl::range<3> GlobalRange,
                     const KernelFunc Kernel) {
  // Create a range of predefined work-group sizes to simplify calculations
  const sycl::range<3> WGSize = {DPCPP_PERF_HOST_WG_SIZE0,
                                 DPCPP_PERF_HOST_WG_SIZE1,
                                 DPCPP_PERF_HOST_WG_SIZE2};

  auto GroupsRange = GlobalRange / WGSize;
  auto Reminder = GlobalRange % WGSize;
  GroupsRange += (Reminder > 0);

#pragma omp parallel for collapse(3)
  for (size_t GroupID0 = 0; GroupID0 < GroupsRange.get(0); ++GroupID0) {
    for (size_t GroupID1 = 0; GroupID1 < GroupsRange.get(1); ++GroupID1) {
      for (size_t GroupID2 = 0; GroupID2 < GroupsRange.get(2); ++GroupID2) {
        // Handle remainder part for each dimension: if the at the last group
        // iteration run only remaining number if iterations by a internal loop.
        auto WGSize0 =
            ((GroupID0 == GroupsRange.get(0) - 1) && Reminder.get(0) > 0)
                ? Reminder.get(0)
                : DPCPP_PERF_HOST_WG_SIZE0;
        auto WGSize1 =
            ((GroupID1 == GroupsRange.get(1) - 1) && Reminder.get(1) > 0)
                ? Reminder.get(1)
                : DPCPP_PERF_HOST_WG_SIZE1;
        auto WGSize2 =
            ((GroupID2 == GroupsRange.get(2) - 1) && Reminder.get(2) > 0)
                ? Reminder.get(2)
                : DPCPP_PERF_HOST_WG_SIZE2;

        ParallelForNDRangeImplKernel3D(
            GroupID0, GroupID1, GroupID2,
            // Flip the ranges to make LocalId2 a fastest moving index(see comment
            // in ParallelForNDRangeImplKernel3D).
            WGSize2, WGSize1, WGSize0,
            [&Kernel, WGSize](sycl::id<3> GroupId, sycl::id<3> LocalId) {
              Kernel(GroupId * sycl::id<3>(WGSize) + LocalId);
            });
      }
    }
  }
}

template <typename KernelFunc>
void ParallelForImpl(const sycl::range<2> GlobalRange,
                     const KernelFunc Kernel) {
  // Create a range of predefined work-group sizes to simplify calculations
  const sycl::range<2> WGSize = {
      DPCPP_PERF_HOST_WG_SIZE0,
      DPCPP_PERF_HOST_WG_SIZE1,
  };

  auto GroupsRange = GlobalRange / WGSize;
  auto Reminder = GlobalRange % WGSize;
  GroupsRange += (Reminder > 0);

#pragma omp parallel for collapse(2)
  for (size_t GroupID0 = 0; GroupID0 < GroupsRange.get(0); ++GroupID0) {
    for (size_t GroupID1 = 0; GroupID1 < GroupsRange.get(1); ++GroupID1) {
      // Handle remainder part for each dimension: if the at the last group
      // iteration run only remaining number if iterations by a internal loop.
      auto WGSize0 =
          ((GroupID0 == GroupsRange.get(0) - 1) && Reminder.get(0) > 0)
              ? Reminder.get(0)
              : DPCPP_PERF_HOST_WG_SIZE0;
      auto WGSize1 =
          ((GroupID1 == GroupsRange.get(1) - 1) && Reminder.get(1) > 0)
              ? Reminder.get(1)
              : DPCPP_PERF_HOST_WG_SIZE1;

      ParallelForNDRangeImplKernel2D(
          GroupID0, GroupID1, 0,
          // Flip the ranges to make LocalId1 a fastest moving index(see comment
          // in ParallelForNDRangeImplKernel2D).
          WGSize1, WGSize0, 1,
          [&Kernel, WGSize](sycl::id<2> GroupId, sycl::id<2> LocalId) {
            Kernel(GroupId * sycl::id<2>(WGSize) + LocalId);
          });
    }
  }
}

template <typename KernelFunc>
void ParallelForImpl(const sycl::range<1> GlobalRange,
                     const KernelFunc Kernel) {
  auto GroupsRange = GlobalRange / DPCPP_PERF_HOST_WG_SIZE0;
  auto Reminder = GlobalRange % DPCPP_PERF_HOST_WG_SIZE0;
  GroupsRange += (Reminder > 0);

#pragma omp parallel for
  for (size_t GroupID0 = 0; GroupID0 < GroupsRange.get(0); ++GroupID0) {
    auto WGSize0 = ((GroupID0 == GroupsRange.get(0) - 1) && Reminder.get(0) > 0)
                       ? Reminder.get(0)
                       : DPCPP_PERF_HOST_WG_SIZE0;

    ParallelForNDRangeImplKernel1D(
        GroupID0, 0, 0, WGSize0, 1, 1,
        [&Kernel](sycl::id<1> GroupId, sycl::id<1> LocalId) {
          Kernel(GroupId * DPCPP_PERF_HOST_WG_SIZE0 + LocalId);
        });
  }
}

template <int Dim, class KernelFunc>
void ParallelFor(sycl::range<Dim> GlobalRange, KernelFunc Kernel) {
  ParallelForImpl(GlobalRange, Kernel);
}

template <class KernelFunc>
void ParallelForWorkGroupImpl(sycl::range<1> GroupsRange, KernelFunc Kernel) {
#pragma omp parallel for
  for (size_t GroupID0 = 0; GroupID0 < GroupsRange.get(0); ++GroupID0) {
    Kernel({GroupID0});
  }
}

template <class KernelFunc>
void ParallelForWorkGroupImpl(sycl::range<2> GroupsRange, KernelFunc Kernel) {
#pragma omp parallel for collapse(2)
  for (size_t GroupID0 = 0; GroupID0 < GroupsRange.get(0); ++GroupID0) {
    for (size_t GroupID1 = 0; GroupID1 < GroupsRange.get(1); ++GroupID1) {
      Kernel({GroupID0, GroupID1});
    }
  }
}

template <class KernelFunc>
void ParallelForWorkGroupImpl(sycl::range<3> GroupsRange, KernelFunc Kernel) {
#pragma omp parallel for collapse(3)
  for (size_t GroupID0 = 0; GroupID0 < GroupsRange.get(0); ++GroupID0) {
    for (size_t GroupID1 = 0; GroupID1 < GroupsRange.get(1); ++GroupID1) {
      for (size_t GroupID2 = 0; GroupID2 < GroupsRange.get(2); ++GroupID2) {
        Kernel({GroupID0, GroupID1, GroupID2});
      }
    }
  }
}

template <int Dim, class KernelFunc>
void ParallelForWorkGroup(sycl::range<Dim> GroupRange, KernelFunc Kernel) {
  ParallelForWorkGroupImpl(GroupRange, Kernel);
}

template <class KernelFunc>
void ParallelForWorkItemImpl(sycl::range<1> LocalRange, KernelFunc Kernel) {
#pragma omp simd
  for (size_t LocalID0 = 0; LocalID0 < LocalRange.get(0); ++LocalID0) {
    Kernel({LocalID0});
  }
}

template <class KernelFunc>
void ParallelForWorkItemImpl(sycl::range<2> LocalRange, KernelFunc Kernel) {
#pragma omp simd collapse(2)
  for (size_t LocalID0 = 0; LocalID0 < LocalRange.get(0); ++LocalID0) {
    for (size_t LocalID1 = 0; LocalID1 < LocalRange.get(1); ++LocalID1) {
      Kernel({LocalID0, LocalID1});
    }
  }
}

template <class KernelFunc>
void ParallelForWorkItemImpl(sycl::range<3> LocalRange, KernelFunc Kernel) {
#pragma omp simd collapse(3)
  for (size_t LocalID0 = 0; LocalID0 < LocalRange.get(0); ++LocalID0) {
    for (size_t LocalID1 = 0; LocalID1 < LocalRange.get(1); ++LocalID1) {
      for (size_t LocalID2 = 0; LocalID2 < LocalRange.get(2); ++LocalID2) {
        Kernel({LocalID0, LocalID1, LocalID2});
      }
    }
  }
}

template <int Dim, class KernelFunc>
void ParallelForWorkItem(sycl::range<Dim> LocalRange, KernelFunc Kernel) {
  ParallelForWorkItemImpl(LocalRange, Kernel);
}

inline void NDRangeBarrier(sycl::access::fence_space) {
  std::cerr << "Barrier is not implemented\n";
  std::abort();
}

inline cl_uint getMaxComputeUnits() {
  return std::thread::hardware_concurrency();
}

inline size_t getWorkGroupSize() {
    // Use the number from OpenCL CPU RT
    return 8192;
}

inline sycl::id<3> getMaxWorkItemSizes() {
  auto maxWGSize = getWorkGroupSize();
  return {maxWGSize, maxWGSize, maxWGSize};
}

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
