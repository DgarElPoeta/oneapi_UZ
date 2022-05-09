//==-------------- openmp_backend.hpp --- OpenMP backend -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

// OpenMP based host device backend.
// A kernel is executed via OpenMP in a following way:
// - parallel_for without nd_range:
//   Execution is mapped to #pragma omp parallel for simd
// - parallel_for with nd_range/barrier:
//   Each work-item in a work-group is executed in a separate
//   thread via OpenMP, barrier is mapped to #pragma omp barrier.
//   Work-group are executed sequentially in respect to each other.
// - parallel_for_work_group:
//   Mapped to #pragma omp parallel for
// - parallel_for_work_item:
//   Mapped to #pragma omp simd

#include <CL/sycl/detail/helpers.hpp>
#include <CL/sycl/id.hpp>
#include <CL/sycl/item.hpp>
#include <thread>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

using namespace cl;

template <typename KernelFunc>
void ParallelForImpl(const sycl::range<3> GlobalRange,
                     const KernelFunc Kernel) {
#pragma omp parallel for collapse(2)
  for (size_t GlobalID0 = 0; GlobalID0 < GlobalRange.get(0); ++GlobalID0) {
    for (size_t GlobalID1 = 0; GlobalID1 < GlobalRange.get(1); ++GlobalID1) {
#pragma omp simd
      for (size_t GlobalID2 = 0; GlobalID2 < GlobalRange.get(2); ++GlobalID2) {
        Kernel({GlobalID0, GlobalID1, GlobalID2});
      }
    }
  }
}

template <typename KernelFunc>
void ParallelForImpl(const sycl::range<2> GlobalRange,
                     const KernelFunc Kernel) {
#pragma omp parallel for
  for (size_t GlobalID0 = 0; GlobalID0 < GlobalRange.get(0); ++GlobalID0) {
#pragma omp simd
    for (size_t GlobalID1 = 0; GlobalID1 < GlobalRange.get(1); ++GlobalID1) {
      Kernel({GlobalID0, GlobalID1});
    }
  }
}

template <typename KernelFunc>
void ParallelForImpl(const sycl::range<1> GlobalRange,
                     const KernelFunc Kernel) {
#pragma omp parallel for simd
  for (size_t GlobalID0 = 0; GlobalID0 < GlobalRange.get(0); ++GlobalID0) {
    Kernel({GlobalID0});
  }
}

template <int Dim, class KernelFunc>
void ParallelFor(sycl::range<Dim> GlobalRange, KernelFunc Kernel) {
  ParallelForImpl(GlobalRange, Kernel);
}

template <class KernelFunc>
void ParallelForNDRangeImpl(sycl::range<1> LocalRange,
                            sycl::range<1> GroupRange, KernelFunc Kernel) {
  const size_t LocalSize = LocalRange.get(0);

  for (size_t GroupID0 = 0; GroupID0 < GroupRange.get(0); ++GroupID0) {
  // Run each iteration(work-item) in a separate thread
#pragma omp parallel for schedule(static, 1) num_threads(LocalSize)
    for (size_t LocalID0 = 0; LocalID0 < LocalRange.get(0); ++LocalID0) {
      Kernel({GroupID0}, {LocalID0});
    }
  }
}

template <class KernelFunc>
void ParallelForNDRangeImpl(sycl::range<2> LocalRange,
                            sycl::range<2> GroupsRange, KernelFunc Kernel) {
  // Calculate the total size to determine the required number of threads
  const size_t LocalSize = LocalRange.get(0) * LocalRange.get(1);

  for (size_t GroupID0 = 0; GroupID0 < GroupsRange.get(0); ++GroupID0) {
    for (size_t GroupID1 = 0; GroupID1 < GroupsRange.get(1); ++GroupID1) {
    // Run each iteration(work-item) in a separate thread
#pragma omp parallel for schedule(static, 1) collapse(2) num_threads(LocalSize)
      for (size_t LocalID0 = 0; LocalID0 < LocalRange.get(0); ++LocalID0) {
        for (size_t LocalID1 = 0; LocalID1 < LocalRange.get(1); ++LocalID1) {
          Kernel({GroupID0, GroupID1}, {LocalID0, LocalID1});
        }
      }
    }
  }
}

template <class KernelFunc>
void ParallelForNDRangeImpl(sycl::range<3> LocalRange,
                            sycl::range<3> GroupsRange, KernelFunc Kernel) {
  // Calculate the total size to determine the required number of threads
  const size_t LocalSize =
      LocalRange.get(0) * LocalRange.get(1) * LocalRange.get(2);

  for (size_t GroupID0 = 0; GroupID0 < GroupsRange.get(0); ++GroupID0) {
    for (size_t GroupID1 = 0; GroupID1 < GroupsRange.get(1); ++GroupID1) {
      for (size_t GroupID2 = 0; GroupID2 < GroupsRange.get(2); ++GroupID2) {
      // Run each iteration(work-item) in a separate thread
#pragma omp parallel for schedule(static, 1) collapse(3) num_threads(LocalSize)
        for (size_t LocalID0 = 0; LocalID0 < LocalRange.get(0); ++LocalID0) {
          for (size_t LocalID1 = 0; LocalID1 < LocalRange.get(1); ++LocalID1) {
            for (size_t LocalID2 = 0; LocalID2 < LocalRange.get(2);
                 ++LocalID2) {
              Kernel({GroupID0, GroupID1, GroupID2},
                     {LocalID0, LocalID1, LocalID2});
            }
          }
        }
      }
    }
  }
}

template <int Dim, class KernelFunc>
void ParallelForNDRange(sycl::range<Dim> LocalRange,
                        sycl::range<Dim> GroupsRange, KernelFunc Kernel) {
  ParallelForNDRangeImpl(LocalRange, GroupsRange, Kernel);
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
#pragma omp barrier
}

inline cl_uint getMaxComputeUnits() {
  return std::thread::hardware_concurrency();
}

inline size_t getWorkGroupSize() { return 8 * getMaxComputeUnits(); }

inline sycl::id<3> getMaxWorkItemSizes() {
  auto maxWGSize = getWorkGroupSize();
  return {maxWGSize, maxWGSize, maxWGSize};
}

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
