// What feeds the tensor cores in the routed experts' prefill kernels: memory, the tensor cores themselves, or the
// CUDA cores that unpack and scale the weights.
//
// Each arm is a copy of the kernel's structure with every station but one taken out, run beside the unchanged
// kernel in the same interleaved rounds. The unchanged kernel is prefill.cu itself, included into this file, so
// that its launches, shapes and bucketing are the engine's; nothing in prefill.cu is touched.
//
//   up kernel (k_moe_up_ws, the default shape: 4 consumer warps, 64-k steps in pairs, 2 stages)
//     0 copy      the kernel as it is (the copy's own control: its outputs must equal the engine kernel's bit for bit)
//     1 memory    the producer as it is; the consumers read every byte of every stage and add the 32-bit words
//                 into a checksum; no unpack, no MMA, no epilogue. Result: what the copy pipeline delivers.
//     2 compute   the consumers as they are; the producer copies every stage from one small region that stays in L2
//                 (expert 0's weights, the first 64 rows of the activations; the epilogue writes the first 64 rows).
//                 Result: the consumers' rate with memory free, unpack included.
//     3 unpack    as 2, each MMA replaced by an exclusive-or of its operands into its accumulator registers (four
//                 three-input logic instructions where the MMA is one). Result: the CUDA-core work by itself. The
//                 exclusive-ors are extra instructions the kernel does not have, so this arm reads high by a few of them
//                 an MMA.
//   down kernel (k_moe_down_p, the separate-combine form, scales raw)
//     0 copy      as for up
//     1 memory    the loads as they are (h rows by cp.async, weights and scales into registers); every loaded word is
//                 added into the checksum; the bf16 output rows are stored in full, from the checksum (so that the
//                 stores are counted: a sentinel check finds any row not written); no unpack, no MMA
//     2 compute   weights and h rows from one small region in L2, the output rows stored over one 64-row window
//     3 unpack    as 2 with the MMAs replaced
//
// Reading the result: an arm at about the unchanged kernel's time says that station sets the time. Arm 2 near the full
// kernel's time says the consumers (tensor cores plus unpack) bind it; arm 1 near it says the loading dock does;
// arm 3 says how much of arm 2 the CUDA cores explain. The arms overlap in the kernel, so they need not add up.
// Controls: a pure MMA loop (the tensor ceiling, e4m3 x e2m1 block-scaled); the unchanged kernel in every round;
// arm 1's checksum against a host-side sum of every byte it should have read (a skipped copy changes it: --check
// shows that with copies skipped on purpose); arm 0 and arm 2 against the engine kernel and against each other.
// Not covered: the scales' 1-bit code (the weights here carry raw scales, so both the arms and the unchanged kernel
// run the raw-scale form), the fused combine (the down kernel runs separate), the shapes other than the default.
//
// usage:  bench_feed --check             tiny shapes, checksums and bit comparisons, exit 1 on any failure
//         bench_feed                     FEED_ROWS="2048 8192 16384" FEED_ROUNDS=11 (at least 11) FEED_E=384
//                                        FEED_ROUTE=uniform|skewed; FEED_ONLY=up|down
// The GPU must be idle. Rows are tokens a chunk (six picks each).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "kernels/prefill.cu"  // the unchanged kernels and their helpers, in this translation unit
namespace dsk {
void hc_boundary_split(int, int) {}  // prefill_tune_env calls it; this bench links none of the kernels library and never sets DSGB10_PF_TUNE
}

namespace dsk {
namespace {

// arms 4 and 5 are up-kernel arms: 4 the mover alone (consumers only wait and release), 5 compute with the weights used
// as loaded (no 4-bit expand), 6 arm 4's producer with arm 1's consumers (the check that arm 4's mover copies everything)
enum { kArmFull = 0, kArmMem = 1, kArmComp = 2, kArmUnpk = 3, kArmMov = 4, kArmNoExp = 5, kArmMovChk = 6 };
constexpr int kFC = 4, kFKU = 2, kFS = 2;  // the up kernel's default shape (PrefillTuning::upk 1)

// the operands of an MMA folded into its accumulators (arm 3): all four registers of A, both of B and both scale words
__device__ __forceinline__ void f_xor4(float d[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                       uint32_t b1, uint32_t sa, uint32_t sb) {
  d[0] = __uint_as_float(__float_as_uint(d[0]) ^ a0 ^ b0);
  d[1] = __uint_as_float(__float_as_uint(d[1]) ^ a1 ^ b1);
  d[2] = __uint_as_float(__float_as_uint(d[2]) ^ a2 ^ sa);
  d[3] = __uint_as_float(__float_as_uint(d[3]) ^ a3 ^ sb);
}

__device__ __forceinline__ uint32_t f_sum4(const uint4& v) { return v.x + v.y + v.z + v.w; }

__device__ __forceinline__ void f_block_add(uint32_t cs, uint32_t* chk) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) cs += __shfl_xor_sync(0xffffffffu, cs, o);
  if ((threadIdx.x & 31) == 0) atomicAdd(chk, cs);
}

// ---- The up kernel: k_moe_up_ws's structure (prefill.cu), raw scales, hinted accesses. skip: the chunk whose weights
// the producer does not copy (-1: none; the check's deliberate fault).
template <int ARM>
__global__ void __launch_bounds__(32 * (kFC + 1), UpWs<kFC, kFKU, kFS>::kMinBlocks)
    f_up(MoePArgs a, uint32_t* chk, int skip) {
  constexpr int C = kFC, KU = kFKU, S = kFS;
  using L = UpWs<C, KU, S>;
  constexpr int KC = L::KC, NCH = L::NCH;
  constexpr bool REDIR = ARM >= kArmComp;
  extern __shared__ __align__(16) uint8_t smem[];
  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + L::kRing);
  int* stok = reinterpret_cast<int*>(bars + 2 * S);
  float* srw = reinterpret_cast<float*>(stok + kTileRows);
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  pdl_launch_dependents();
  pdl_wait();
  const int4 te = a.tiles[blockIdx.y];
  if ((int)blockIdx.y >= *a.ntiles) return;
  const int e = te.x, base = te.y, rows = te.z, q = blockIdx.x;
  const int ew = REDIR ? 0 : e, hb = REDIR ? 0 : base;  // the weights' expert, the output rows' first row
  const int mt = (rows + 15) / 16;
  const uint64_t pol_w = l2_policy(!REDIR && (a.l2 & 1) && te.w <= kTileRows), pol_x = l2_policy(a.l2 & 4 ? 2 : 0);
  if (tid < kTileRows) {
    stok[tid] = tid < rows ? (REDIR ? tid : a.perm_tok[base + tid]) : 0;
    srw[tid] = tid < rows ? a.perm_w[base + tid] : 0.f;
  }
  const uint32_t s0 = smem_addr(smem), full0 = smem_addr(bars), empty0 = full0 + 8 * S;
  if (tid == 0)
    for (int s = 0; s < S; ++s) mbar_init(full0 + 8 * s, 32), mbar_init(empty0 + 8 * s, C);
  __syncthreads();

  if (warp == C) {  // the producer
    for (int c = 0; c < NCH; ++c) {
      const int s = c % S;
      if (c >= S) mbar_wait(empty0 + 8 * s, (uint32_t)(c / S - 1) & 1u);
      const uint32_t sA = s0 + s * L::kStage, sSA = sA + L::kA, sW = sSA + L::kSA, sWS = sW + L::kW;
#pragma unroll
      for (int j = 0; j < kTileRows * (KC / 16) / 32; ++j) {
        const int i = lane + 32 * j, r = i / (KC / 16), c16 = i % (KC / 16);
        cp_async16<true>(sA + r * KC + 32 * ((c16 >> 1) ^ (r & 3)) + 16 * (c16 & 1),
                         a.xq + (size_t)stok[r] * kDimP + c * KC + 16 * c16, r < rows, pol_x);
      }
#pragma unroll
      for (int r = lane; r < kTileRows; r += 32)
        cp_async_n<true, KC / 32>(sSA + r * (KC / 32), a.xs + (size_t)stok[r] * kKB1P + c * (KC / 32), r < rows, pol_x);
      if (c != skip) {
#pragma unroll
        for (int j = 0; j < C * 2 * KU; ++j) {  // j = (w * 2 + mat) * KU + u
          const int w = j / (2 * KU), mat = j / KU % 2, u = j % KU;
          const uint8_t* src = (mat ? a.w.w3 : a.w.w1) + (size_t)ew * kW13P + ((size_t)(q * C + w) * kP1P + c * KU + u) * 512;
          cp_async16<true>(sW + j * 512 + lane * 16, src + lane * 16, true, pol_w);
        }
        for (int i = lane; i < C * 2 * KU * 2; i += 32) {  // the scale words' 16-byte halves
          const int j = i >> 1, half = i & 1, w = j / (2 * KU), mat = j / KU % 2, u = j % KU, nq = q * C + w, p = c * KU + u;
          const uint8_t* raw = (mat ? a.w.s3 : a.w.s1) + (size_t)ew * kS13P + ((size_t)nq * kP1P + p) * 32 + 16 * half;
          cp_async16<true>(sWS + j * 32 + 16 * half, raw, true, pol_w);
        }
      }
      mbar_arrive_cp_async(full0 + 8 * s);
    }
    return;
  }

  // the consumers
  if constexpr (ARM == kArmMov) {
    for (int c = 0; c < NCH; ++c) {
      const int s = c % S;
      mbar_wait(full0 + 8 * s, (uint32_t)(c / S) & 1u);
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * s);
    }
    return;
  } else if constexpr (ARM == kArmMem || ARM == kArmMovChk) {
    uint32_t cs = 0;
    for (int c = 0; c < NCH; ++c) {
      const int s = c % S;
      mbar_wait(full0 + 8 * s, (uint32_t)(c / S) & 1u);
      const uint8_t* A = smem + s * L::kStage;
      const uint8_t* SA = A + L::kA;
      const uint4* W = reinterpret_cast<const uint4*>(SA + L::kSA) + warp * 2 * KU * 32;
      const uint32_t* WS = reinterpret_cast<const uint32_t*>(SA + L::kSA + L::kW) + warp * 2 * KU * 8;
#pragma unroll
      for (int j = 0; j < 2 * KU; ++j) {
        cs += f_sum4(W[j * 32 + lane]);
        if (lane < 8) cs += WS[j * 8 + lane];
      }
      const uint4* AS = reinterpret_cast<const uint4*>(A);  // the activations, then their scales: contiguous
      for (int i = warp * 32 + lane; i < (L::kA + L::kSA) / 16; i += C * 32) cs += f_sum4(AS[i]);
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * s);
    }
    f_block_add(cs, chk);
    return;
  } else {
    float acc[4][4][4];
#pragma unroll
    for (int mi = 0; mi < 4; ++mi)
#pragma unroll
      for (int j = 0; j < 4; ++j) acc[mi][j][0] = acc[mi][j][1] = acc[mi][j][2] = acc[mi][j][3] = 0.f;
    for (int c = 0; c < NCH; ++c) {
      const int s = c % S;
      mbar_wait(full0 + 8 * s, (uint32_t)(c / S) & 1u);
      const uint8_t* A = smem + s * L::kStage;
      const uint8_t* SA = A + L::kA;
      const uint4* W = reinterpret_cast<const uint4*>(SA + L::kSA) + warp * 2 * KU * 32;
      const uint32_t* WS = reinterpret_cast<const uint32_t*>(SA + L::kSA + L::kW) + warp * 2 * KU * 8;
      uint4 w[KU][2];
      uint32_t sw[KU][2];
#pragma unroll
      for (int u = 0; u < KU; ++u)
#pragma unroll
        for (int mat = 0; mat < 2; ++mat) {
          w[u][mat] = W[(mat * KU + u) * 32 + lane];
          sw[u][mat] = WS[(mat * KU + u) * 8 + g];
        }
      uint32_t sfaw[4][(KU + 1) / 2];
#pragma unroll
      for (int mi = 0; mi < 4; ++mi)
#pragma unroll
        for (int j = 0; j < (KU + 1) / 2; ++j)
          sfaw[mi][j] = reinterpret_cast<const uint32_t*>(SA + (mi * 16 + g + 8 * (lane & 1)) * (KC / 32))[j];
      auto chunk = [&](auto full) {
        constexpr bool FULL = decltype(full)::value;
#pragma unroll
        for (int u = 0; u < KU; ++u)
#pragma unroll
          for (int blk = 0; blk < 2; ++blk) {
            const int unit = 2 * u + blk;
            uint32_t af[4][4];
#pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
              if (!FULL && mi >= mt) break;
              const int r = mi * 16 + g;
              const uint2 lo = *reinterpret_cast<const uint2*>(A + r * KC + 32 * (unit ^ (r & 3)) + 8 * t);
              const uint2 hi = *reinterpret_cast<const uint2*>(A + (r + 8) * KC + 32 * (unit ^ (r & 3)) + 8 * t);
              af[mi][0] = lo.x, af[mi][1] = hi.x, af[mi][2] = lo.y, af[mi][3] = hi.y;
            }
#pragma unroll
            for (int mat = 0; mat < 2; ++mat)
#pragma unroll
              for (int h = 0; h < 2; ++h) {
                const uint32_t word = (&w[u][mat].x)[2 * h + blk];
                uint32_t b0, b1;
                if constexpr (ARM == kArmNoExp)
                  b0 = word, b1 = word;  // the loaded word as the MMA's operand: no expand
                else
                  b0 = e2m1x4_expand(word & 0xFFFFu), b1 = e2m1x4_expand(word >> 16);
#pragma unroll
                for (int mi = 0; mi < 4; ++mi)
                  if (FULL || mi < mt) {
                    if constexpr (ARM == kArmUnpk)
                      f_xor4(acc[mi][2 * mat + h], af[mi][0], af[mi][1], af[mi][2], af[mi][3], b0, b1,
                             sfaw[mi][unit >> 2], sw[u][mat]);
                    else
                      mma_e4m3_e2m1_sel(acc[mi][2 * mat + h], af[mi][0], af[mi][1], af[mi][2], af[mi][3], b0, b1,
                                        sfaw[mi][unit >> 2], (uint16_t)(unit & 3), sw[u][mat], (uint16_t)(2 * h + blk));
                  }
              }
          }
      };
      if (mt == 4)
        chunk(std::true_type{});
      else
        chunk(std::false_type{});
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * s);
    }

    // the epilogue as it is: SwiGLU, routing weight, bf16 rows over the ring, act_quant per 32 columns
    asm volatile("bar.sync 1, %0;" ::"n"(32 * C) : "memory");
    __nv_bfloat16* sH = reinterpret_cast<__nv_bfloat16*>(smem);
#pragma unroll
    for (int mi = 0; mi < 4; ++mi)
#pragma unroll
      for (int h = 0; h < 2; ++h)
#pragma unroll
        for (int hr = 0; hr < 2; ++hr) {
          const int r = mi * 16 + g + 8 * hr, col = 16 * warp + 8 * h + 2 * t;
          const float rw = srw[r];
          float v[2];
#pragma unroll
          for (int j = 0; j < 2; ++j) {
            const float gt = __bfloat162float(__float2bfloat16_rn(acc[mi][h][2 * hr + j]));
            const float up = __bfloat162float(__float2bfloat16_rn(acc[mi][2 + h][2 * hr + j]));
            v[j] = rw * swiglu(gt, up);
          }
          *reinterpret_cast<uint32_t*>(&sH[r * L::kHS + col]) = pack_bf16x2(v[0], v[1]);
        }
    asm volatile("bar.sync 1, %0;" ::"n"(32 * C) : "memory");
    constexpr int NB = C / 2;
    for (int u = warp; u < rows * NB; u += C) {
      const int r = u / NB, jb = u % NB;
      const float v = bf2f(sH[r * L::kHS + 32 * jb + lane]);
      const uint32_t ex = act_scale_byte(warp_max(fabsf(v)));
      a.hq[(size_t)(hb + r) * kInterP + q * 16 * C + 32 * jb + lane] = (uint8_t)quant_e4m3(v, ex);
      if (lane == 0) a.hs[(size_t)(hb + r) * kKB2P + q * NB + jb] = (uint8_t)ex;
    }
  }
}

// ---- The down kernel: k_moe_down_p<false, true, false>'s structure (separate combine, hinted accesses, raw scales).
// skip: 0 leaves the first batch of weights unloaded; 1000 stores no output rows (arm 1's deliberate faults; -1: none).
template <int ARM>
__global__ void __launch_bounds__(128) f_down(MoePArgs a, uint32_t* chk, int skip) {
  constexpr int KU = 3, LDA = kInterP;
  constexpr bool REDIR = ARM >= kArmComp;
  __shared__ __align__(16) uint8_t sA[kTileRows * LDA];
  __shared__ uint8_t sS[kTileRows * kKB2P];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
  pdl_launch_dependents();
  pdl_wait();
  const int tile = blockIdx.y;
  const int ng = a.cb0 + blockIdx.x;
  const int4 te = a.tiles[tile];
  if (tile >= *a.ntiles) return;
  const int e = te.x, base = te.y, rows = te.z;
  const int ew = REDIR ? 0 : e, sb = REDIR ? 0 : base, db = REDIR ? 0 : base;  // weights' expert, h rows' and outputs' first row
  const int mt = (rows + 15) / 16;
  const uint64_t pol_w = l2_policy(!REDIR && (a.l2 & 2) && te.w <= kTileRows), pol_h = l2_policy(a.l2 & 4 ? 2 : 0);
  const uint64_t pol_o = l2_policy(REDIR ? 0 : (a.l2 & 2 ? 1 : 0));
  const int nq0 = ng * 8 + 2 * warp;
  uint32_t cs = 0;
  const uint32_t sA0 = smem_addr(sA);
  for (int i = tid; i < kTileRows * (LDA / 16); i += 128) {
    const int r = i / (LDA / 16), c16 = i % (LDA / 16);
    cp_async16<true>(sA0 + r * LDA + 32 * ((c16 >> 1) ^ ((r >> 1) & 1)) + 16 * (c16 & 1),
                     a.hq + (size_t)(sb + (r < rows ? r : 0)) * kInterP + 16 * c16, r < rows, pol_h);
  }
  cp_async_commit();
  for (int i = tid; i < kTileRows * kKB2P; i += 128) {
    const int r = i / kKB2P;
    sS[i] = r < rows ? a.hs[(size_t)(sb + r) * kKB2P + i % kKB2P] : 127;
  }
  const uint8_t* wp[2];
  const uint8_t* sp[2];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    wp[i] = a.w.w2 + (size_t)ew * kW2P + ((size_t)(nq0 + i) * kP2P * 32 + lane) * 16;
    sp[i] = a.w.s2 + (size_t)ew * kS2P + ((size_t)(nq0 + i) * kP2P * 8 + g) * 4;
  }
  uint4 wa[KU][2] = {}, wb[KU][2] = {};
  uint32_t sa[KU][2] = {}, sbv[KU][2] = {};
  auto load_w = [&](uint4 (&w)[KU][2], uint32_t (&s)[KU][2], int p0) {
#pragma unroll
    for (int u = 0; u < KU; ++u)
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        w[u][i] = ldg128<true>(wp[i] + (size_t)(p0 + u) * 512, pol_w);
        s[u][i] = ldg32<true>(sp[i] + (size_t)(p0 + u) * 32, pol_w);
      }
  };
  float acc[4][4][4];
#pragma unroll
  for (int mi = 0; mi < 4; ++mi)
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[mi][j][0] = acc[mi][j][1] = acc[mi][j][2] = acc[mi][j][3] = 0.f;
  auto compute = [&](const uint4 (&w)[KU][2], const uint32_t (&s)[KU][2], int p0) {
    if constexpr (ARM == kArmMem) {
#pragma unroll
      for (int u = 0; u < KU; ++u)
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          cs += f_sum4(w[u][i]);
          if ((lane & 3) == 0) cs += s[u][i];  // the four lanes of a row group load the same word
        }
    } else {
#pragma unroll
      for (int u = 0; u < KU; ++u)
#pragma unroll
        for (int blk = 0; blk < 2; ++blk) {
          const int unit = 2 * (p0 + u) + blk;
          uint32_t b0[4], b1[4], sfb[4];
#pragma unroll
          for (int i = 0; i < 2; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
              const uint32_t word = (&w[u][i].x)[2 * h + blk];
              b0[2 * i + h] = e2m1x4_expand(word & 0xFFFFu);
              b1[2 * i + h] = e2m1x4_expand(word >> 16);
              sfb[2 * i + h] = (s[u][i] >> (8 * (2 * h + blk))) & 0xFFu;
            }
#pragma unroll
          for (int mi = 0; mi < 4; ++mi) {
            if (mi >= mt) break;
            const int r = mi * 16 + g;
            const int off = 32 * (unit ^ ((r >> 1) & 1)) + 8 * t;
            const uint8_t* pa = sA + r * LDA + off;
            const uint32_t a0 = *reinterpret_cast<const uint32_t*>(pa);
            const uint32_t a1 = *reinterpret_cast<const uint32_t*>(pa + 8 * LDA);
            const uint32_t a2 = *reinterpret_cast<const uint32_t*>(pa + 4);
            const uint32_t a3 = *reinterpret_cast<const uint32_t*>(pa + 8 * LDA + 4);
            const uint32_t sfa = sS[(mi * 16 + g + 8 * (lane & 1)) * kKB2P + unit];
#pragma unroll
            for (int n = 0; n < 4; ++n) {
              if constexpr (ARM == kArmUnpk)
                f_xor4(acc[mi][n], a0, a1, a2, a3, b0[n], b1[n], sfa, sfb[n]);
              else
                mma_e4m3_e2m1(acc[mi][n], a0, a1, a2, a3, b0[n], b1[n], sfa, sfb[n]);
            }
          }
        }
    }
  };
  if (skip != 0) load_w(wa, sa, 0);
  load_w(wb, sbv, KU);
  cp_async_wait<0>();
  __syncthreads();
  if constexpr (ARM == kArmMem) {  // the h rows and their scales, as loaded
    for (int i = tid; i < rows * (LDA / 16); i += 128) cs += f_sum4(reinterpret_cast<const uint4*>(sA)[i]);
    for (int i = tid; i < rows * kKB2P; i += 128) cs += sS[i];
  }
  compute(wa, sa, 0);
  load_w(wa, sa, 2 * KU);
  compute(wb, sbv, KU);
  compute(wa, sa, 2 * KU);
  __syncthreads();

  constexpr int LDO = 128 + 8;
  __nv_bfloat16* sO = reinterpret_cast<__nv_bfloat16*>(sA);
#pragma unroll
  for (int mi = 0; mi < 4; ++mi)
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int r = mi * 16 + g, c = 32 * warp + 8 * j + 2 * t;
      if constexpr (ARM == kArmMem) {  // rows the stores will carry: the checksum, never the sentinel's all-ones
        *reinterpret_cast<uint32_t*>(&sO[r * LDO + c]) = (cs * 2654435761u + r * 131u + c) & 0x7FFF7FFFu;
        *reinterpret_cast<uint32_t*>(&sO[(r + 8) * LDO + c]) = (cs * 2654435761u + (r + 8) * 131u + c) & 0x7FFF7FFFu;
      } else {
        *reinterpret_cast<uint32_t*>(&sO[r * LDO + c]) = pack_bf16x2(acc[mi][j][0], acc[mi][j][1]);
        *reinterpret_cast<uint32_t*>(&sO[(r + 8) * LDO + c]) = pack_bf16x2(acc[mi][j][2], acc[mi][j][3]);
      }
    }
  __syncthreads();
  if (skip != 1000)
    for (int i = tid; i < rows * 16; i += 128) {
      const int r = i >> 4, c8 = i & 15;
      stg128<true>(a.dout + (size_t)(db + r) * kDimP + ng * 128 + 8 * c8, *reinterpret_cast<const uint4*>(&sO[r * LDO + 8 * c8]),
                   pol_o);
    }
  if constexpr (ARM == kArmMem) f_block_add(cs, chk);
}

// rows [base, base + rows) of every tile, all 2,560 words of each: how many still hold the sentinel
__global__ void f_sentinel(const uint32_t* dout, const int4* tiles, const int* ntiles, unsigned long long* bad) {
  if ((int)blockIdx.x >= *ntiles) return;
  const int4 te = tiles[blockIdx.x];
  unsigned long long n = 0;
  for (int i = threadIdx.x; i < te.z * (kDimP / 2); i += blockDim.x)
    n += dout[(size_t)te.y * (kDimP / 2) + i] == 0xFFFFFFFFu;
  if (n) atomicAdd(bad, n);
}

// the tensor ceiling: eight independent chains a warp of the same block-scaled MMA the kernels issue
__global__ void __launch_bounds__(256) f_peak(float* out, int iters) {
  float acc[8][4] = {};
  const uint32_t a0 = threadIdx.x * 0x01010101u & 0x37373737u, a1 = a0 ^ 0x01000100u, a2 = a0 ^ 0x00010001u;
  const uint32_t a3 = a0 ^ 0x01010000u, b0 = a0 ^ 0x00000101u, b1 = a0 ^ 0x01000001u;
  for (int i = 0; i < iters; ++i)
#pragma unroll
    for (int c = 0; c < 8; ++c) mma_e4m3_e2m1(acc[c], a0, a1, a2, a3, b0, b1, 127, 127);
  float s = 0;
  for (int c = 0; c < 8; ++c) s += acc[c][0] + acc[c][1] + acc[c][2] + acc[c][3];
  if (s == 1.2345f) out[0] = s;
}

__global__ void f_fill(uint8_t* p, size_t n, uint32_t seed, uint8_t lo, uint8_t span, uint8_t sign) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint32_t h = (uint32_t)i * 2654435761u ^ seed;
    h ^= h >> 15, h *= 2246822519u, h ^= h >> 13;
    uint8_t v = (uint8_t)h;
    v = span ? (uint8_t)(lo + v % span) : (uint8_t)(v & 0x77);  // e4m3 and e2m1 without NaN
    if (sign) {
      uint32_t k = (uint32_t)i * 2654435761u ^ seed * 0x9E3779B9u;
      k ^= k >> 15, k *= 0x2C1B3C6Du, k ^= k >> 12;
      v ^= (uint8_t)(k & sign);
    }
    p[i] = v;
  }
}

uint8_t* f_dev(size_t n, uint32_t seed, uint8_t lo = 0, uint8_t span = 0, uint8_t sign = 0) {
  uint8_t* p;
  CUDA_CHECK(cudaMalloc(&p, n));
  f_fill<<<512, 256>>>(p, n, seed, lo, span, sign);
  CUDA_CHECK(cudaGetLastError());
  return p;
}

std::vector<int> f_routing(int M, int E, bool skewed, uint32_t seed) {  // six distinct experts a row
  std::vector<double> w(E, 1.0);
  std::vector<int> order(E);
  for (int e = 0; e < E; ++e) order[e] = e;
  uint64_t s = seed * 0x9E3779B97F4A7C15ull + 1;
  auto rnd = [&]() {
    s ^= s << 13, s ^= s >> 7, s ^= s << 17;
    return (s >> 11) * (1.0 / 9007199254740992.0);
  };
  for (int e = E - 1; e > 0; --e) std::swap(order[e], order[(int)(rnd() * (e + 1))]);
  if (skewed)
    for (int r = 0; r < E; ++r) w[order[r]] = 1.0 / std::sqrt(r + 1.0);
  std::vector<int> idx((size_t)M * 6);
  for (int m = 0; m < M; ++m) {
    std::vector<double> ww = w;
    for (int k = 0; k < 6; ++k) {
      double tot = 0;
      for (double v : ww) tot += v;
      double u = rnd() * tot;
      int e = 0;
      while (e < E - 1 && (u -= ww[e]) > 0) ++e;
      while (ww[e] == 0) e = (e + 1) % E;
      idx[(size_t)m * 6 + k] = e;
      ww[e] = 0;
    }
  }
  return idx;
}

// Several graph-captured functions timed round-robin, one replay each per round: the same method as bench_kernels'.
std::vector<std::vector<double>> f_time(const std::vector<std::function<void(cudaStream_t)>>& fns, int rounds) {
  cudaStream_t st;
  CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  std::vector<cudaGraphExec_t> execs;
  for (auto& fn : fns) {
    fn(st);  // warm-up (and lazy attributes)
    CUDA_CHECK(cudaStreamSynchronize(st));
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    CUDA_CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
    fn(st);
    CUDA_CHECK(cudaStreamEndCapture(st, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
    CUDA_CHECK(cudaGraphDestroy(graph));
    execs.push_back(exec);
  }
  cudaEvent_t a, b;
  CUDA_CHECK(cudaEventCreate(&a));
  CUDA_CHECK(cudaEventCreate(&b));
  std::vector<std::vector<double>> ts(fns.size());
  for (int r = 0; r < rounds; ++r)
    for (size_t i = 0; i < execs.size(); ++i) {
      CUDA_CHECK(cudaEventRecord(a, st));
      CUDA_CHECK(cudaGraphLaunch(execs[i], st));
      CUDA_CHECK(cudaEventRecord(b, st));
      CUDA_CHECK(cudaEventSynchronize(b));
      float ms;
      CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
      ts[i].push_back(ms);
    }
  for (auto& v : ts) std::sort(v.begin(), v.end());
  for (auto e : execs) CUDA_CHECK(cudaGraphExecDestroy(e));
  CUDA_CHECK(cudaEventDestroy(a));
  CUDA_CHECK(cudaEventDestroy(b));
  CUDA_CHECK(cudaStreamDestroy(st));
  return ts;
}

template <int ARM>
void f_launch_up(const MoePArgs& a, int maxt, uint32_t* chk, int skip, cudaStream_t st) {
  using L = UpWs<kFC, kFKU, kFS>;
  auto k = f_up<ARM>;
  static const bool once = [&] {
    set_smem(k, L::kSmem);
    CUDA_CHECK(cudaFuncSetAttribute(k, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    return true;
  }();
  (void)once;
  launch_pdl(k, dim3(kInterP / (16 * kFC), maxt), dim3(L::NT), L::kSmem, st, true, a, chk, skip);
}

template <int ARM>
void f_launch_down(const MoePArgs& a, int maxt, uint32_t* chk, int skip, cudaStream_t st) {
  launch_pdl(f_down<ARM>, dim3(a.ncb, maxt), dim3(128), 0, st, true, a, chk, skip);
}

using Launch = void (*)(const MoePArgs&, int, uint32_t*, int, cudaStream_t);
const Launch kUpArms[7] = {f_launch_up<0>, f_launch_up<1>, f_launch_up<2>, f_launch_up<3>,
                           f_launch_up<4>, f_launch_up<5>, f_launch_up<6>};
const Launch kDownArms[4] = {f_launch_down<0>, f_launch_down<1>, f_launch_down<2>, f_launch_down<3>};
const char* const kArmName[6] = {"0 copy (nothing removed)", "1 memory alone", "2 compute alone", "3 unpack alone",
                                 "4 mover alone (memory in cache)", "5 compute, no expand (up only)"};

// Sum mod 2^32 of the 32-bit words of each expert's slice, copied back piece by piece.
std::vector<uint32_t> f_expert_sums(const uint8_t* d, int E, size_t bytes) {
  std::vector<uint32_t> out(E), h(bytes / 4);
  for (int e = 0; e < E; ++e) {
    CUDA_CHECK(cudaMemcpy(h.data(), d + (size_t)e * bytes, bytes, cudaMemcpyDeviceToHost));
    uint64_t s = 0;
    for (uint32_t v : h) s += v;
    out[e] = (uint32_t)s;
  }
  return out;
}

struct Case {
  int M, E;
  bool skewed;
  MoeWeights mw;
  uint8_t *xq, *xs;
  int* idx;
  float* wts;
  void* ws;
  size_t wsb;
  MoePWs b;
  int P, maxt, nt;
  std::vector<int4> tiles;
  std::vector<int> perm_tok;
  uint8_t *hq_own, *hs_own;  // the up arms' outputs
  __nv_bfloat16* dout_own;   // the down arms' outputs
  uint32_t* chk;
  unsigned long long* bad;
  std::vector<uint32_t> w13sum, w2sum;  // per expert: w1 + s1 + w3 + s3 words, w2 + s2 words (lazy)
};

Case f_make(int M, int E, bool skewed) {
  Case c{};
  c.M = M, c.E = E, c.skewed = skewed, c.P = M * kTopK, c.maxt = E + c.P / kTileRows + 1;
  const size_t w13 = kW13P, s13 = kS13P, w2 = kW2P, s2 = kS2P;
  c.mw.w1 = f_dev(E * w13, 1, 0, 0, 0x88), c.mw.w3 = f_dev(E * w13, 3, 0, 0, 0x88), c.mw.w2 = f_dev(E * w2, 5, 0, 0, 0x88);
  c.mw.s1 = f_dev(E * s13, 21, 118, 16), c.mw.s3 = f_dev(E * s13, 22, 118, 16), c.mw.s2 = f_dev(E * s2, 23, 118, 16);
  c.xq = f_dev((size_t)M * kDimP, 13, 0, 0, 0x80), c.xs = f_dev((size_t)M * kKB1P, 14, 119, 4);
  CUDA_CHECK(cudaMalloc(&c.idx, (size_t)M * 24));
  CUDA_CHECK(cudaMalloc(&c.wts, (size_t)M * 24));
  {
    auto hi = f_routing(M, E, skewed, 1);
    std::vector<float> hw((size_t)M * 6);
    for (size_t i = 0; i < hw.size(); ++i) hw[i] = 0.05f + 0.01f * (float)(i % 17);
    CUDA_CHECK(cudaMemcpy(c.idx, hi.data(), hi.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(c.wts, hw.data(), hw.size() * 4, cudaMemcpyHostToDevice));
  }
  c.wsb = moe_prefill_workspace_bytes(M, E);
  CUDA_CHECK(cudaMalloc(&c.ws, c.wsb));
  c.b = carve(c.ws, M, E);
  CUDA_CHECK(cudaMalloc(&c.hq_own, (size_t)c.P * kInterP));
  CUDA_CHECK(cudaMalloc(&c.hs_own, (size_t)c.P * kKB2P));
  CUDA_CHECK(cudaMalloc(&c.dout_own, (size_t)c.P * kDimP * 2));
  CUDA_CHECK(cudaMalloc(&c.chk, 4));
  CUDA_CHECK(cudaMalloc(&c.bad, 8));
  CUDA_CHECK(cudaDeviceSynchronize());
  return c;
}

MoePArgs f_args(const Case& c, bool down) {
  MoePArgs a{c.xq, c.xs, c.idx, c.wts, c.M, c.E, c.mw, c.b.cnt, c.b.off, c.b.cursor, c.b.ntiles, c.b.tiles,
             c.b.perm_tok, c.b.perm_w, c.b.pos_of, down ? c.b.hq : c.hq_own, down ? c.b.hs : c.hs_own,
             c.dout_own, c.b.shout, nullptr, c.b.done, 0, kColBlocks, 7, 0, 0};
  return a;
}

void f_engine(const Case& c, int phases, int upk, cudaStream_t st) {
  PrefillTuning t;
  t.upk = upk, t.down = 1;  // the up kernel's shape; the separate combine, so that the down kernel is k_moe_down_p<false>
  prefill_tuning(t);
  MoePCfg cfg;
  cfg.phases = phases;
  moe_prefill(c.xq, c.xs, c.M, c.idx, c.wts, c.E, c.mw, c.ws, nullptr, st, cfg);
}

// The engine's bucketing, once; its up and down once (their outputs sit in the workspace); the tile list read back.
void f_prepare(Case& c) {
  f_engine(c, 1, 1, 0);
  f_engine(c, 4, 1, 0);
  f_engine(c, 8, 1, 0);
  CUDA_CHECK(cudaDeviceSynchronize());
  int nt;
  CUDA_CHECK(cudaMemcpy(&nt, c.b.ntiles, 4, cudaMemcpyDeviceToHost));
  c.nt = nt;
  c.tiles.resize(nt);
  c.perm_tok.resize(c.P);
  CUDA_CHECK(cudaMemcpy(c.tiles.data(), c.b.tiles, nt * sizeof(int4), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(c.perm_tok.data(), c.b.perm_tok, c.P * 4, cudaMemcpyDeviceToHost));
  long picks = 0;
  for (const int4& t : c.tiles) picks += t.z;
  if (picks != c.P) die("the workspace layout is not the one carve() describes: %ld picks in the tiles of %d", picks, c.P);
}

// ---- host-side sums of every byte the memory arms must read
uint32_t f_expect_up(Case& c, bool redir = false) {  // redir: arm 6's view (expert 0's weights, tokens 0..rows-1)
  if (c.w13sum.empty()) {
    auto a = f_expert_sums(c.mw.w1, c.E, kW13P), b = f_expert_sums(c.mw.w3, c.E, kW13P);
    auto s = f_expert_sums(c.mw.s1, c.E, kS13P), t = f_expert_sums(c.mw.s3, c.E, kS13P);
    c.w13sum.resize(c.E);
    for (int e = 0; e < c.E; ++e) c.w13sum[e] = a[e] + b[e] + s[e] + t[e];
  }
  std::vector<uint32_t> xq((size_t)c.M * kDimP / 4), xs((size_t)c.M * kKB1P / 4), R(c.M);
  CUDA_CHECK(cudaMemcpy(xq.data(), c.xq, xq.size() * 4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(xs.data(), c.xs, xs.size() * 4, cudaMemcpyDeviceToHost));
  for (int m = 0; m < c.M; ++m) {
    uint64_t s = 0;
    for (int i = 0; i < kDimP / 4; ++i) s += xq[(size_t)m * (kDimP / 4) + i];
    for (int i = 0; i < kKB1P / 4; ++i) s += xs[(size_t)m * (kKB1P / 4) + i];
    R[m] = (uint32_t)s;
  }
  uint64_t tot = 0;
  for (const int4& t : c.tiles) {  // each tile: its expert's w1, w3 and scales once, its rows' activations by every column CTA
    tot += c.w13sum[redir ? 0 : t.x];
    uint64_t rs = 0;
    for (int r = 0; r < t.z; ++r) rs += R[redir ? r : c.perm_tok[t.y + r]];
    tot += rs * (uint64_t)(kInterP / (16 * kFC));
  }
  return (uint32_t)tot;
}

uint32_t f_expect_down(Case& c) {
  if (c.w2sum.empty()) {
    auto a = f_expert_sums(c.mw.w2, c.E, kW2P), s = f_expert_sums(c.mw.s2, c.E, kS2P);
    c.w2sum.resize(c.E);
    for (int e = 0; e < c.E; ++e) c.w2sum[e] = a[e] + s[e];
  }
  std::vector<uint32_t> hq((size_t)c.P * kInterP / 4);
  std::vector<uint8_t> hs((size_t)c.P * kKB2P);
  CUDA_CHECK(cudaMemcpy(hq.data(), c.b.hq, hq.size() * 4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hs.data(), c.b.hs, hs.size(), cudaMemcpyDeviceToHost));
  uint64_t tot = 0;
  for (const int4& t : c.tiles) {  // each tile: its expert's w2 and scales once, its h rows and their scales by each of the 40 column CTAs
    tot += c.w2sum[t.x];
    uint64_t rs = 0;
    for (int r = 0; r < t.z; ++r) {
      for (int i = 0; i < kInterP / 4; ++i) rs += hq[(size_t)(t.y + r) * (kInterP / 4) + i];
      for (int i = 0; i < kKB2P; ++i) rs += hs[(size_t)(t.y + r) * kKB2P + i];
    }
    tot += rs * (uint64_t)kColBlocks;
  }
  return (uint32_t)tot;
}

uint32_t f_run_chk(const Case& c, bool down, int skip, cudaStream_t st, int arm = kArmMem) {
  CUDA_CHECK(cudaMemsetAsync(c.chk, 0, 4, st));
  (down ? kDownArms : kUpArms)[arm](f_args(c, down), c.maxt, c.chk, skip, st);
  uint32_t v;
  CUDA_CHECK(cudaMemcpyAsync(&v, c.chk, 4, cudaMemcpyDeviceToHost, st));
  CUDA_CHECK(cudaStreamSynchronize(st));
  return v;
}

unsigned long long f_sentinels(const Case& c, cudaStream_t st) {
  CUDA_CHECK(cudaMemsetAsync(c.bad, 0, 8, st));
  f_sentinel<<<c.nt, 256, 0, st>>>(reinterpret_cast<const uint32_t*>(c.dout_own), c.b.tiles, c.b.ntiles, c.bad);
  unsigned long long v;
  CUDA_CHECK(cudaMemcpyAsync(&v, c.bad, 8, cudaMemcpyDeviceToHost, st));
  CUDA_CHECK(cudaStreamSynchronize(st));
  return v;
}

bool f_same(const void* x, const void* y, size_t n) {
  std::vector<uint8_t> a(n), b(n);
  CUDA_CHECK(cudaMemcpy(a.data(), x, n, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(b.data(), y, n, cudaMemcpyDeviceToHost));
  return memcmp(a.data(), b.data(), n) == 0;
}

int g_fail = 0;
void f_report(bool ok, const char* what) {
  printf("  %-4s %s\n", ok ? "PASS" : "FAIL", what);
  if (!ok) ++g_fail;
}

// the checks every run makes: the memory arms' checksums against the host's sums, the stores' completeness
void f_checks(Case& c, bool deep) {
  cudaStream_t st = 0;
  const uint32_t eu = f_expect_up(c), ed = f_expect_down(c);
  char buf[160];
  uint32_t got = f_run_chk(c, false, -1, st);
  snprintf(buf, sizeof buf, "up arm 1 checksum %08x = host sum %08x", got, eu);
  f_report(got == eu, buf);
  CUDA_CHECK(cudaMemset(c.dout_own, 0xFF, (size_t)c.P * kDimP * 2));
  got = f_run_chk(c, true, -1, st);
  snprintf(buf, sizeof buf, "down arm 1 checksum %08x = host sum %08x", got, ed);
  f_report(got == ed, buf);
  unsigned long long bad = f_sentinels(c, st);
  snprintf(buf, sizeof buf, "down arm 1 stored every output row (%llu sentinel words left)", bad);
  f_report(bad == 0, buf);
  got = f_run_chk(c, false, -1, st, kArmMovChk);
  snprintf(buf, sizeof buf, "up arm 4's mover (checked by arm 6) copied every stage: checksum %08x = host sum %08x", got,
           f_expect_up(c, true));
  f_report(got == f_expect_up(c, true), buf);
  if (!deep) return;

  // RED: the same arms with a copy skipped on purpose must not pass
  for (int sk : {0, 3, 17}) {
    got = f_run_chk(c, false, sk, st);
    snprintf(buf, sizeof buf, "RED up arm 1, weights of chunk %d not copied: checksum %08x differs from %08x", sk, got, eu);
    f_report(got != eu, buf);
  }
  got = f_run_chk(c, true, 0, st);
  snprintf(buf, sizeof buf, "RED down arm 1, first weight batch not loaded: checksum %08x differs from %08x", got, ed);
  f_report(got != ed, buf);
  for (int sk : {0, 3, 17}) {
    got = f_run_chk(c, false, sk, st, kArmMovChk);
    snprintf(buf, sizeof buf, "RED up arm 6 (arm 4's mover), weights of chunk %d not copied: checksum %08x differs from %08x", sk,
             got, f_expect_up(c, true));
    f_report(got != f_expect_up(c, true), buf);
  }
  CUDA_CHECK(cudaMemset(c.dout_own, 0xFF, (size_t)c.P * kDimP * 2));
  f_run_chk(c, true, 1000, st);
  bad = f_sentinels(c, st);
  snprintf(buf, sizeof buf, "RED down arm 1, output stores skipped: %llu sentinel words left", bad);
  f_report(bad > 0, buf);

  // the copies against the engine's kernels, bit for bit
  CUDA_CHECK(cudaMemset(c.hq_own, 0, (size_t)c.P * kInterP));
  CUDA_CHECK(cudaMemset(c.hs_own, 0, (size_t)c.P * kKB2P));
  kUpArms[kArmFull](f_args(c, false), c.maxt, c.chk, -1, st);
  CUDA_CHECK(cudaStreamSynchronize(st));
  f_report(f_same(c.hq_own, c.b.hq, (size_t)c.P * kInterP) && f_same(c.hs_own, c.b.hs, (size_t)c.P * kKB2P),
           "up arm 0 (the copy) = k_moe_up_ws's hq and hs, bit for bit");
  CUDA_CHECK(cudaMemset(c.dout_own, 0, (size_t)c.P * kDimP * 2));
  kDownArms[kArmFull](f_args(c, true), c.maxt, c.chk, -1, st);
  CUDA_CHECK(cudaStreamSynchronize(st));
  f_report(f_same(c.dout_own, c.b.dout, (size_t)c.P * kDimP * 2), "down arm 0 (the copy) = k_moe_down_p's outputs, bit for bit");

  // the engine's up kernel is still what it was: shape 1 and the old kernel k_moe_up_p give the same bytes
  std::vector<uint8_t> hq1((size_t)c.P * kInterP), hq0(hq1.size());
  CUDA_CHECK(cudaMemcpy(hq1.data(), c.b.hq, hq1.size(), cudaMemcpyDeviceToHost));
  f_engine(c, 4, 0, st);
  CUDA_CHECK(cudaStreamSynchronize(st));
  CUDA_CHECK(cudaMemcpy(hq0.data(), c.b.hq, hq0.size(), cudaMemcpyDeviceToHost));
  f_report(hq0 == hq1, "engine: k_moe_up_ws (shape 1) hq = k_moe_up_p's hq");
  f_engine(c, 4, 1, st);
  CUDA_CHECK(cudaStreamSynchronize(st));
  CUDA_CHECK(cudaMemcpy(hq0.data(), c.b.hq, hq0.size(), cudaMemcpyDeviceToHost));
  f_report(hq0 == hq1, "engine: k_moe_up_ws's hq unchanged after the other kernel ran");

  // arm 2 against arm 0 where the redirect changes nothing: every tile on rows 0.., every expert holding expert 0's bytes
  {
    Case d = c;
    uint8_t* r[6];
    const size_t sz[6] = {kW13P, kW13P, kS13P, kS13P, kW2P, kS2P};
    const uint8_t* src[6] = {c.mw.w1, c.mw.w3, c.mw.s1, c.mw.s3, c.mw.w2, c.mw.s2};
    for (int i = 0; i < 6; ++i) {
      CUDA_CHECK(cudaMalloc(&r[i], c.E * sz[i]));
      for (int e = 0; e < c.E; ++e) CUDA_CHECK(cudaMemcpy(r[i] + e * sz[i], src[i], sz[i], cudaMemcpyDeviceToDevice));
    }
    d.mw.w1 = r[0], d.mw.w3 = r[1], d.mw.s1 = r[2], d.mw.s3 = r[3], d.mw.w2 = r[4], d.mw.s2 = r[5];
    std::vector<int4> t0 = c.tiles;
    for (auto& t : t0) t.y = 0;
    std::vector<int> p0(c.P);
    for (int i = 0; i < c.P; ++i) p0[i] = i % c.M;  // rows 0..63 of a tile read tokens 0..63
    int4* dt;
    int* dp;
    CUDA_CHECK(cudaMalloc(&dt, t0.size() * sizeof(int4)));
    CUDA_CHECK(cudaMalloc(&dp, c.P * 4));
    CUDA_CHECK(cudaMemcpy(dt, t0.data(), t0.size() * sizeof(int4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dp, p0.data(), c.P * 4, cudaMemcpyHostToDevice));
    MoePArgs au = f_args(d, false), ad = f_args(d, true);
    au.tiles = ad.tiles = dt, au.perm_tok = ad.perm_tok = dp;
    const size_t nq = (size_t)kTileRows * kInterP, ns = (size_t)kTileRows * kKB2P, no = (size_t)kTileRows * kDimP * 2;
    std::vector<uint8_t> refq(nq), refs(ns), ref_o(no), gq(nq), gs(ns), go(no);
    auto up_rows = [&](int arm, std::vector<uint8_t>& q, std::vector<uint8_t>& s) {
      CUDA_CHECK(cudaMemset(d.hq_own, 0, (size_t)c.P * kInterP));
      CUDA_CHECK(cudaMemset(d.hs_own, 0, (size_t)c.P * kKB2P));
      kUpArms[arm](au, c.maxt, c.chk, -1, st);
      CUDA_CHECK(cudaStreamSynchronize(st));
      CUDA_CHECK(cudaMemcpy(q.data(), d.hq_own, nq, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(s.data(), d.hs_own, ns, cudaMemcpyDeviceToHost));
    };
    up_rows(kArmFull, refq, refs);
    up_rows(kArmComp, gq, gs);
    f_report(refq == gq && refs == gs && std::any_of(refq.begin(), refq.end(), [](uint8_t v) { return v != 0; }),
             "up arm 2 = arm 0 where the redirect changes nothing (rows 0..63, non-trivial)");
    auto down_rows = [&](int arm, std::vector<uint8_t>& o) {
      CUDA_CHECK(cudaMemset(d.dout_own, 0, (size_t)c.P * kDimP * 2));
      kDownArms[arm](ad, c.maxt, c.chk, -1, st);
      CUDA_CHECK(cudaStreamSynchronize(st));
      CUDA_CHECK(cudaMemcpy(o.data(), d.dout_own, no, cudaMemcpyDeviceToHost));
    };
    down_rows(kArmFull, ref_o);
    down_rows(kArmComp, go);
    f_report(ref_o == go && std::any_of(ref_o.begin(), ref_o.end(), [](uint8_t v) { return v != 0; }),
             "down arm 2 = arm 0 where the redirect changes nothing (rows 0..63, non-trivial)");
    {  // arm 5 (on the same redirected tiles): reproducible, non-trivial, and dependent on every chunk's weights
      auto run5 = [&](int sk, std::vector<uint8_t>& q) {
        CUDA_CHECK(cudaMemset(d.hq_own, 0, (size_t)c.P * kInterP));
        CUDA_CHECK(cudaMemset(d.hs_own, 0, (size_t)c.P * kKB2P));
        kUpArms[kArmNoExp](au, c.maxt, c.chk, sk, st);
        CUDA_CHECK(cudaStreamSynchronize(st));
        CUDA_CHECK(cudaMemcpy(q.data(), d.hq_own, nq, cudaMemcpyDeviceToHost));
      };
      std::vector<uint8_t> q1(nq), q2(nq), q3(nq);
      run5(-1, q1), run5(-1, q2);
      f_report(q1 == q2 && std::any_of(q1.begin(), q1.end(), [](uint8_t v) { return v != 0; }),
               "up arm 5 (no expand) is reproducible and non-trivial (rows 0..63)");
      for (int sk : {0, 3, 17}) {
        run5(sk, q3);
        snprintf(buf, sizeof buf, "RED up arm 5, weights of chunk %d not copied: its output differs", sk);
        f_report(q3 != q1, buf);
      }
    }
    kUpArms[kArmUnpk](au, c.maxt, c.chk, -1, st);
    kDownArms[kArmUnpk](ad, c.maxt, c.chk, -1, st);
    CUDA_CHECK(cudaStreamSynchronize(st));
    f_report(true, "arm 3 (both kernels) ran to the end");
    for (auto p : r) CUDA_CHECK(cudaFree(p));
    CUDA_CHECK(cudaFree(dt));
    CUDA_CHECK(cudaFree(dp));
  }
}

void f_free(Case& c) {
  for (const void* p : {(const void*)c.mw.w1, (const void*)c.mw.w3, (const void*)c.mw.w2, (const void*)c.mw.s1,
                        (const void*)c.mw.s3, (const void*)c.mw.s2, (const void*)c.xq, (const void*)c.xs, (const void*)c.idx,
                        (const void*)c.wts, (const void*)c.ws, (const void*)c.hq_own, (const void*)c.hs_own,
                        (const void*)c.dout_own, (const void*)c.chk, (const void*)c.bad})
    CUDA_CHECK(cudaFree(const_cast<void*>(p)));
}

double f_mma_peak() {
  float* sink;
  CUDA_CHECK(cudaMalloc(&sink, 4));
  int sms;
  CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
  const int iters = 1024, grid = sms * 4;
  auto ts = f_time({[&](cudaStream_t st) { f_peak<<<grid, 256, 0, st>>>(sink, iters); }}, 7);
  CUDA_CHECK(cudaFree(sink));
  return (double)grid * 8 * iters * 8 * 8192 / (ts[0][ts[0].size() / 2] * 1e-3) / 1e12;
}

void f_time_case(Case& c, bool do_up, bool do_down, int rounds) {
  const char* route = c.skewed ? "skewed" : "uniform";
  const double fl_up = 2.0 * c.P * 2.0 * kInterP * kDimP, fl_dn = 2.0 * c.P * kDimP * kInterP;
  for (int pass = 0; pass < 2; ++pass) {
    const bool down = pass == 1;
    if (down ? !do_down : !do_up) continue;
    std::vector<std::function<void(cudaStream_t)>> fns;
    fns.push_back([&, down](cudaStream_t s) { f_engine(c, down ? 8 : 4, 1, s); });
    const int narm = down ? 4 : 6;
    for (int arm = 0; arm < narm; ++arm)
      fns.push_back([&, down, arm](cudaStream_t s) { (down ? kDownArms : kUpArms)[arm](f_args(c, down), c.maxt, c.chk, -1, s); });
    const auto ts = f_time(fns, rounds);
    const double t0 = ts[0][ts[0].size() / 2], fl = down ? fl_dn : fl_up;
    printf("\n%s kernel, %d rows (%d picks, %d tiles, %s routing), median of %d interleaved rounds\n", down ? "down" : "up", c.M,
           c.P, c.nt, route, rounds);
    printf("  %-34s %9s %9s %9s %8s\n", "", "median ms", "min ms", "vs engine", "TFLOPS");
    printf("  %-34s %9.3f %9.3f %9s %8.1f\n", down ? "engine k_moe_down_p (unchanged)" : "engine k_moe_up_ws (unchanged)", t0,
           ts[0][0], "1.000", fl / (t0 * 1e-3) / 1e12);
    for (int arm = 0; arm < narm; ++arm) {
      const double m = ts[arm + 1][ts[arm + 1].size() / 2];
      printf("  %-34s %9.3f %9.3f %9.3f %8.1f\n", kArmName[arm], m, ts[arm + 1][0], m / t0, fl / (m * 1e-3) / 1e12);
    }
  }
}

}  // namespace
}  // namespace dsk

int main(int argc, char** argv) {
  using namespace dsk;
  const bool check = argc > 1 && !strcmp(argv[1], "--check");
  cudaDeviceProp pr;
  CUDA_CHECK(cudaGetDeviceProperties(&pr, 0));
  printf("bench_feed on %s (sm_%d%d), %d SMs\n", pr.name, pr.major, pr.minor, pr.multiProcessorCount);
  if (check) {
    printf("--check: tiny shapes, 8 experts\n");
    for (const int M : {200, 70}) {
      printf("\nM=%d (uniform routing, then skewed)\n", M);
      for (const bool skewed : {false, true}) {
        Case c = f_make(M, 8, skewed);
        f_prepare(c);
        printf(" %s: %d tiles\n", skewed ? "skewed" : "uniform", c.nt);
        f_checks(c, true);
        f_free(c);
      }
    }
    printf("\nbench_feed --check: %s\n", g_fail ? "FAILED" : "every check passed");
    return g_fail ? 1 : 0;
  }
  std::vector<int> rows;
  if (const char* e = getenv("FEED_ROWS"))
    for (const char* p = e; *p;) {
      char* q;
      const long v = strtol(p, &q, 10);
      if (q == p) break;
      rows.push_back((int)v), p = q;
    }
  if (rows.empty()) rows = {2048, 8192, 16384};
  int rounds = getenv("FEED_ROUNDS") ? atoi(getenv("FEED_ROUNDS")) : 11;
  if (rounds < 11) rounds = 11, printf("FEED_ROUNDS raised to 11, the least a median is read from\n");
  const int E = getenv("FEED_E") ? atoi(getenv("FEED_E")) : 384;
  const bool skewed = getenv("FEED_ROUTE") && !strcmp(getenv("FEED_ROUTE"), "skewed");
  const std::string only = getenv("FEED_ONLY") ? getenv("FEED_ONLY") : "";
  printf("tensor ceiling (e4m3 x e2m1 block-scaled MMA, 8 chains a warp): %.0f TFLOPS\n", f_mma_peak());
  for (const int M : rows) {
    Case c = f_make(M, E, skewed);
    f_prepare(c);
    printf("\n=== %d rows ===\ndata checks:\n", M);
    f_checks(c, false);
    f_time_case(c, only != "down", only != "up", rounds);
    f_free(c);
  }
  printf("\nbench_feed: %s\n", g_fail ? "A CHECK FAILED: the timings above are not to be read" : "checks passed");
  return g_fail ? 1 : 0;
}
