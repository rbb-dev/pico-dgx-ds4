// The runtime abliteration (--ablit FILE; tools/ablit/make_ablit.py): a low-rank edit of attn.wo_b's and the shared
// expert's outputs, applied to the rows a request marks instead of to the weights: y' = y - s Q (Q^T y), Q an
// orthonormal [5120][k] basis (k <= 2), s a strength per layer and output (StepSamp::ablit, 0 or 1, times the layer's s).
// Rows at strength 0 are not touched (their bits are unchanged). Each rank projects its own partial sum.
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>

#include <cuda_runtime.h>

#include "model/config.h"
#include "model/ops.h"

namespace model {
namespace ablit {

constexpr int kK = 2;  // basis columns the kernels read (a k = 1 file is padded with a zero column)

struct Pack {
  bool on = false;
  float s[kLayers][2] = {};  // [layer][0: wo_b, 1: shared expert]
  float* Q = nullptr;        // device [5120][kK]
  float* G[kLayers] = {};    // device [576][kK], this rank's rows; nullptr where s[L][1] is 0
  // prefill: the shared expert's dot products of the chunk's rows on this rank, and every rank's
  float *c_part = nullptr, *c_all = nullptr;
  int c_rows = 0;  // rows c_part holds (even: the all-gather moves multiples of 16 bytes)
};

// Reads and checks the file (magic, version, shape) and uploads Q and this rank's G slices. Dies on a bad file.
void load(Pack& p, const std::string& path, int rank);
// Room for a prefill chunk of M rows (c_part, c_all).
void reserve(Pack& p, int M);

// The rows of a step (decode, verify, a batch, a wide step): row r's strength from in->samp[in->sess[r]].ablit.
// y: R rows of ld elements (bf16 or fp32), the first 5120 projected.
void rows_bf16(uint16_t* y, int R, int ld, const Pack& p, int L, const ops::StepIn* in, cudaStream_t st);
// The shared expert's share of a step's MoE partial y (fp32 [R][ld]): hq / hs the shared expert's activations as
// the MoE kernels left them (e4m3 [R][576] with row stride hq_row; ue8m0 per 32 with row stride hs_row).
void shexp(float* y, int R, int ld, const uint8_t* hq, size_t hq_row, const uint8_t* hs, size_t hs_row, const Pack& p,
           int L, const ops::StepIn* in, cudaStream_t st);

// Prefill (one session a chunk, so one strength lam for every row; the caller skips all of it at lam 0):
void rows_bf16_lam(uint16_t* y, int n, int ld, const Pack& p, int L, float lam, cudaStream_t st);
// c_part [M][kK] = G^T h for the chunk's M rows (this rank's share of the dot products)
void shexp_dots(const Pack& p, int L, int M, const uint8_t* hq, size_t hq_row, const uint8_t* hs, size_t hs_row,
                cudaStream_t st);
// y (bf16 [n][ld], rows [r0, r0 + n) of the chunk): minus lam s Q c, c every rank's c_all rows summed in rank order
void shexp_apply_bf16(uint16_t* y, int n, int ld, int r0, int M, const Pack& p, int L, float lam, cudaStream_t st);

}  // namespace ablit
}  // namespace model
