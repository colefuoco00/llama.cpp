// helper functions for ggml-metal that are too difficult to implement in Objective-C

#pragma once

#include "ggml-mem-ranges.h"

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_cgraph;

// reorder the nodes in the graph to improve concurrency, while respecting fusion
//
// note: this implementation is generic and not specific to metal
//       if it proves to work well, we can start using it for other backends in the future
void ggml_graph_optimize(struct ggml_cgraph * gf);

#ifdef __cplusplus
}
#endif
