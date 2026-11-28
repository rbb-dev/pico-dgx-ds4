// Host side of the fused decode path's launches: cudaLaunchKernelEx with programmatic stream
// serialization (PDL) when asked, so a kernel's prologue overlaps the previous kernel in the stream.
#pragma once
#include <cuda_runtime.h>

#include "common/check.h"

namespace attn {

// carveout >= 0 also sets the launch's preferred shared-memory carveout (percent of the SM's unified
// memory kept as shared memory); -1 leaves the SM's split to the driver.
template <class... KArgs, class... Args>
inline void launch_co(void (*kernel)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t s, bool pdl,
                      int carveout, Args... args) {
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = smem;
  cfg.stream = s;
  cudaLaunchAttribute attr[2];
  int na = 0;
  if (pdl) {
    attr[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[na++].val.programmaticStreamSerializationAllowed = 1;
  }
  if (carveout >= 0) {
    attr[na].id = cudaLaunchAttributePreferredSharedMemoryCarveout;
    attr[na++].val.sharedMemCarveout = carveout;
  }
  cfg.attrs = attr;
  cfg.numAttrs = na;
  CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, static_cast<KArgs>(args)...));
}

template <class... KArgs, class... Args>
inline void launch(void (*kernel)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t s, bool pdl,
                   Args... args) {
  launch_co(kernel, grid, block, smem, s, pdl, -1, args...);
}

}  // namespace attn
