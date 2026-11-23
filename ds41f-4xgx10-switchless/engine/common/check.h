// Error checks shared by every part of the engine.
#pragma once
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include "common/die.h"

#define CUDA_CHECK(x)                                                                     \
  do {                                                                                    \
    cudaError_t e_ = (x);                                                                 \
    if (e_ != cudaSuccess) die("%s:%d: %s: %s", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
  } while (0)
