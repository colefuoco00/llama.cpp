// disjoint memory-range tracker for graph-level concurrency analysis
//
// originally lifted from ggml-metal-common.{h,cpp}; kept backend-neutral so
// that other backends (e.g. CUDA) can use the same overlap algebra.

#pragma once

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_tensor;

enum ggml_mem_range_type {
    MEM_RANGE_TYPE_SRC = 0,
    MEM_RANGE_TYPE_DST = 1,
};

// a helper object that can be used for reordering operations to improve concurrency
//
// the fundamental idea is that a set of tasks (either ggml ops, or something else) can run concurrently if they
//   don't write to a memory that is being read by another task or written to by another task in the set
//
// with this structure, we can add tasks to the set, setting memory constraints. we can also check if a new task
//   can be added to the set without violating the constraints (i.e. if it can be executed concurrently with the
//   tasks already in the set)
//
typedef struct ggml_mem_ranges * ggml_mem_ranges_t;

ggml_mem_ranges_t ggml_mem_ranges_init(int debug);
void ggml_mem_ranges_free(ggml_mem_ranges_t mrs);

// remove all ranges from the set
void ggml_mem_ranges_reset(ggml_mem_ranges_t mrs);

// add a single tensor as a src or dst range
bool ggml_mem_ranges_add_src(ggml_mem_ranges_t mrs, const struct ggml_tensor * tensor);
bool ggml_mem_ranges_add_dst(ggml_mem_ranges_t mrs, const struct ggml_tensor * tensor);

// add all srcs of the tensor and the tensor itself as a dst range
bool ggml_mem_ranges_add(ggml_mem_ranges_t mrs, const struct ggml_tensor * tensor);

// return false if a single src or dst range conflicts with the existing set
bool ggml_mem_ranges_check_src(ggml_mem_ranges_t mrs, const struct ggml_tensor * tensor);
bool ggml_mem_ranges_check_dst(ggml_mem_ranges_t mrs, const struct ggml_tensor * tensor);

// return false if:
// - any src range overlaps with any existing dst range
// - the dst range overlaps with any existing range (src or dst)
bool ggml_mem_ranges_check(ggml_mem_ranges_t mrs, const struct ggml_tensor * tensor);

#ifdef __cplusplus
}
#endif
