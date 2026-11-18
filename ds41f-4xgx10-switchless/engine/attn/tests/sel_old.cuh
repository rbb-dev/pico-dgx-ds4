// Reference selection with a 17-bit block list (a uint16_t list plus a prefix split), valid up to 1,048,576 keys a row;
// the wide selection must equal it there. Included inside namespace attn::{anonymous} after indexer.cu.
#pragma once
// Row r's selection by a CTA's 1024 threads from the histogram of its keys' key>>5 (key 0, none, is not counted): the bin
// holding the k-th key, a 32-bin histogram inside it for the threshold key, then an ordered pass writing the keys above
// it plus the first `need` equal to it, in scan order. Candidates gather in ckey/cidx when they fit `cap`; bmax and
// blist narrow the scan to the blocks that reach the bin.
template <bool kL2>
__device__ __forceinline__ void select_bins_old(const SelectArgs& a, int r, int lo, int n, uint32_t top, uint32_t next,
                                            int cap = 0, uint16_t* ckey = nullptr, uint32_t* cidx = nullptr,
                                            const uint16_t* bmax = nullptr, uint16_t* blist = nullptr) {
  __shared__ uint32_t h2[32];
  __shared__ uint32_t tmp[64];
  __shared__ uint32_t info[5];  // threshold bin, keys above it, threshold key, ties to take, keys in the bin
  int tid = threadIdx.x;
  // a pick from the scan as attention reads it: a pool entry through the row's list (its first 256 chunks held here)
  __shared__ uint32_t xl[256];
  const bool xl_on = a.list && a.unit == 1 && !a.cand;
  const uint32_t* xrow = xl_on ? list_row(a.list, a.list_stride, a.slot, r) : nullptr;
  const int xn = xl_on ? min(256, a.list_stride) : 0;  // list_stride: a row's length
  const uint32_t xv = tid < xn ? __ldg(xrow + tid) : 0u;
  const uint16_t* row = a.keys + (size_t)r * a.keys_stride;
  if (tid < 32) h2[tid] = 0;
  uint32_t total, pre = block_excl_scan(top + next, tmp, &total);
  uint32_t K = min((uint32_t)a.kmax, total);
  if (K > 0) {
    if (pre < K && K <= pre + top) {
      info[0] = 2047 - 2 * tid;
      info[1] = pre;
      info[4] = top;
    } else if (pre + top < K && K <= pre + top + next) {
      info[0] = 2046 - 2 * tid;
      info[1] = pre + top;
      info[4] = next;
    }
  }
  if (tid < xn) xl[tid] = xv;
  __syncthreads();
  int* opos = a.out_pos + (size_t)r * a.out_stride;
  auto xlt = [&](int p) {
    if (!xl_on || p < 0) return p;
    uint32_t e = (uint32_t)p;
#ifdef DSGB10_POOL_CORRUPT  // the negative control: chunk 0's picks read the neighbouring entry
    if ((e >> kChunkShift) == 0) e ^= 1u;
#endif
    const uint32_t c = (int)(e >> kChunkShift) < xn ? xl[e >> kChunkShift] : __ldg(xrow + (e >> kChunkShift));
    return (int)((c << kChunkShift) | (e & (kChunkEntries - 1)));
  };
  uint16_t* okey = a.out_key + (size_t)r * a.out_stride;
  if (K > 0 && info[1] + info[4] <= (uint32_t)cap) {
    // the candidates, in scan order, and the threshold bin's histogram
    uint32_t bin = info[0], above = info[1], nc = 0;
    auto take = [&](uint32_t k, int i, uint32_t& o) {
      if (k && (k >> 5) >= bin && o < (uint32_t)cap) {
        ckey[o] = (uint16_t)k;
        cidx[o++] = i;
        if ((k >> 5) == bin) atomicAdd(&h2[k & 31], 1u);
      }
    };
    if (bmax) {  // the blocks that reach the bin, in order (each holds a candidate, so at most cap)
      const int nb = (n + 7) / 8;
      uint32_t nl = 0;
      __shared__ uint32_t nlow;  // listed blocks below 65536: blist is ascending, so they are its prefix
      if (tid == 0) nlow = 0;
      __syncthreads();
      for (int base = 0; base < nb; base += kSelChunk) {
        const int b0 = base + tid * kSelPer;
        uint32_t m[16], sel = 0;
        load16<kL2>(bmax, b0, nb, m);
#pragma unroll
        for (int j = 0; j < 16; ++j) sel |= (uint32_t)(m[j] && (m[j] >> 5) >= bin) << j;
        uint32_t tot, o = nl + block_excl_scan(__popc(sel), tmp, &tot);
        for (uint32_t t = sel; t; t &= t - 1)
          if (o < (uint32_t)cap) {
            const uint32_t blk = b0 + __ffs(t) - 1;
            if (blk < 65536u) atomicAdd(&nlow, 1u);
            blist[o++] = (uint16_t)(blk & 0xFFFFu);
          }
        nl += tot;
      }
      __syncthreads();
      nl = min(nl, (uint32_t)cap);
      for (uint32_t base = 0; base < nl; base += kSelThreads) {  // a thread: a listed block's keys
        const uint32_t i = base + tid;
        const int i0 = i < nl ? (int)((blist[i] | (i >= nlow ? 0x10000u : 0u)) * 8) : 0;
        uint32_t k[8] = {}, c = 0;
        if (i < nl) {
          load8<kL2>(row, i0, k);
#pragma unroll
          for (int j = 0; j < 8; ++j) c += k[j] && (k[j] >> 5) >= bin;
        }
        uint32_t tot, o = nc + block_excl_scan(c, tmp, &tot);
        if (i < nl) {
#pragma unroll
          for (int j = 0; j < 8; ++j) take(k[j], i0 + j, o);
        }
        nc += tot;
      }
    } else {
      for (int base = 0; base < n; base += kSelChunk) {
        int i0 = base + tid * kSelPer;
        uint32_t k[16], c = 0;
        load16<kL2>(row, i0, n, k);
#pragma unroll
        for (int j = 0; j < 16; ++j) c += k[j] && (k[j] >> 5) >= bin;
        uint32_t tot, o = nc + block_excl_scan(c, tmp, &tot);
#pragma unroll
        for (int j = 0; j < 16; ++j) take(k[j], i0 + j, o);
        nc += tot;
      }
    }
    __syncthreads();
    threshold_key(h2, bin, above, K, info);
    nc = min(nc, (uint32_t)cap);
    uint32_t theta = info[2], need = info[3], per = (nc + kSelThreads - 1) / kSelThreads;
    uint32_t c0 = min(nc, tid * per), c1 = min(nc, c0 + per), gt = 0, eq = 0;
    for (uint32_t i = c0; i < c1; ++i) {
      gt += ckey[i] > theta;
      eq += ckey[i] == theta;
    }
    uint32_t tot, p = block_excl_scan(gt << 16 | eq, tmp, &tot);
    uint32_t tie = p & 0xffff, o = (p >> 16) + min(tie, need);
    for (uint32_t i = c0; i < c1; ++i) {
      uint32_t k = ckey[i];
      if (k > theta || (k == theta && tie++ < need)) {
        opos[o] = xlt(scan_pos(a, r, cidx[i], lo));
        okey[o] = (uint16_t)k;
        ++o;
      }
    }
  } else if (K > 0) {
    uint32_t bin = info[0], above = info[1];
    for (int base = 0; base < n; base += kSelChunk) {
      uint32_t k[16];
      load16<kL2>(row, base + tid * kSelPer, n, k);
#pragma unroll
      for (int j = 0; j < 16; ++j)
        if (k[j] && (k[j] >> 5) == bin) atomicAdd(&h2[k[j] & 31], 1u);
    }
    __syncthreads();
    threshold_key(h2, bin, above, K, info);
    uint32_t theta = info[2], need = info[3], sel_base = 0, tie_base = 0;
    for (int base = 0; base < n; base += kSelChunk) {
      int i0 = base + tid * kSelPer;
      uint32_t k[16];
      load16<kL2>(row, i0, n, k);
      uint32_t gt = 0, eq = 0;
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        gt += k[j] > theta;
        eq += k[j] == theta;
      }
      uint32_t tot, p = block_excl_scan(gt << 16 | eq, tmp, &tot);
      uint32_t tie = tie_base + (p & 0xffff);
      uint32_t o = sel_base + (p >> 16) + min(tie, need);
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        bool take = k[j] > theta || (k[j] == theta && tie++ < need);
        if (take) {
          opos[o] = xlt(scan_pos(a, r, i0 + j, lo));
          okey[o] = (uint16_t)k[j];
          ++o;
        }
      }
      sel_base += tot >> 16;
      tie_base += tot & 0xffff;
    }
  }
  for (int i = K + tid; i < a.kmax; i += kSelThreads) {
    opos[i] = -1;
    okey[i] = 0;
  }
}

__device__ __forceinline__ void select_dense_old(const SelectArgs& a, int r, uint32_t* gh, int n, int cap, uint16_t* ckey,
                                                 uint32_t* cidx, const uint16_t* bmax, uint16_t* blist) {
  int tid = threadIdx.x;
  uint32_t top = __ldcg(gh + 2047 - 2 * tid), next = __ldcg(gh + 2046 - 2 * tid);
  gh[2047 - 2 * tid] = 0;
  gh[2046 - 2 * tid] = 0;
  select_bins_old<true>(a, r, 0, n, top, next, cap, ckey, cidx, bmax, blist);
}
