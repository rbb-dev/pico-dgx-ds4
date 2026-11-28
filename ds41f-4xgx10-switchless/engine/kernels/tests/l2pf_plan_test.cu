// The L2 prefetch's plan (kernels/l2pf.cu l2pf_plan, l2pf_draft_plan) on known inputs: each window's queue, where it
// starts and its budget, the draft pass's sites (DSK_HIDE2 part 8), and DSK_HIDE2's setting (hide2_parse). Host only: no
// kernel is launched, no GPU is touched; the pointers are never read.
#include <cstdint>
#include <cstdio>

#include "kernels/dsk.h"

namespace {

int g_fail = 0, g_ok = 0;

void check(bool ok, const char* what, long a = 0, long b = 0) {
  if (ok) {
    ++g_ok;
    return;
  }
  ++g_fail;
  printf("FAIL %s (%ld, %ld)\n", what, a, b);
}

constexpr uint32_t kChunk = 8192;
const void* fake(uintptr_t k) { return reinterpret_cast<const void*>(0x100000000ull + k * 0x10000000ull); }

dsk::L2pfTarget target(int k, size_t n0, size_t n1, size_t n2 = 0, size_t n3 = 0) {
  dsk::L2pfTarget t{};
  const size_t n[4] = {n0, n1, n2, n3};
  for (int j = 0; j < 4; ++j) t.p[j] = n[j] ? fake(k * 4 + j) : nullptr, t.n[j] = n[j];
  return t;
}

}  // namespace

int main() {
  using namespace dsk;
  // the sizes of one layer: wq_b 10.49 MB (+ scales), wo_a, wo_b, the router gate 3.93 MB (480 chunks)
  const L2pfTarget t[kL2pfTargets] = {target(0, 2293760, 2240),
                                      target(1, 10485760, 10240),
                                      target(2, 8388608, 8192),
                                      target(3, 5242880, 5120),
                                      target(4, 3932160, 0)};
  auto chunks = [](size_t n) { return (uint32_t)(((n & ~(size_t)15) + kChunk - 1) / kChunk); };
  const uint32_t woa = chunks(8388608) + chunks(8192), wob = chunks(5242880) + chunks(5120), gate = chunks(3932160);
  check(gate == 480, "the gate is 480 chunks", gate);

  auto plan = [&](int window, int R) {
    L2pfPlan p{};
    const bool ok = l2pf_plan(t, window, R, &p);
    check(ok, "a plan exists", window, R);
    return p;
  };
  auto same = [&](const L2pfPlan& a, const L2pfPlan& b, const char* what, long w, long R) {
    bool eq = a.count == b.count && a.first == b.first && a.budget == b.budget && a.ns_chunk == b.ns_chunk;
    for (int i = 0; eq && i < a.count; ++i) eq = a.p[i] == b.p[i] && a.n[i] == b.n[i] && a.end[i] == b.end[i];
    check(eq, what, w, R);
  };

  // the window plans: 3 rows, wo_a, wo_b, the gate; window 4 starts at the gate with 400 of its 480 chunks
  {
    const L2pfPlan w4 = plan(4, 3), w1 = plan(1, 3);
    check(w4.ns_chunk == 41, "ns_chunk 41", w4.ns_chunk);
    check(w4.end[w4.count - 1] == woa + wob + gate, "the queue ends at the gate", w4.end[w4.count - 1]);
    check(w4.first == woa + wob, "window 4 starts at the gate", w4.first);
    check(w4.budget == 400, "window 4 at 3 rows is 400 chunks", w4.budget);
    check(w1.budget == 687, "window 1 at 3 rows is 687 chunks", w1.budget);
    check(plan(4, 2).budget == 370 && plan(4, 8).budget == 546, "window 4 at 2 and 8 rows is 370 and 546 chunks", plan(4, 2).budget,
          plan(4, 8).budget);
    check(plan(1, 2).budget == 563 && plan(1, 8).budget == 1309, "window 1 at 2 and 8 rows is 563 and 1309 chunks", plan(1, 2).budget,
          plan(1, 8).budget);
  }
  // the windows' queues are the same at every row count from 2 up: wo_a, wo_b and the gate (wq_b joins at one row)
  for (int w = 1; w <= 4; ++w)
    for (int R : {2, 5, 8, 9, 16, 24}) {
      const L2pfPlan p = plan(w, R);
      check(p.count == 5 && p.end[p.count - 1] == woa + wob + gate, "windows plan wo_a, wo_b and the gate", w, R);
    }

  // the draft pass's sites: ranges in reading order, the chunk ends, the site's budget (32 us at the pace)
  {
    const void* const p[3] = {fake(1), nullptr, fake(2)};
    const size_t n[3] = {kChunk * 10, 4096, 100000};
    for (int site = 0; site < kL2pfDraftSites; ++site) {
      L2pfPlan d{};
      const bool ok = l2pf_draft_plan(p, n, 3, site, &d);
      check(ok, "a draft plan exists", site);
      if (!ok) continue;
      check(d.count == 2 && d.end[0] == 10 && d.end[1] == 10 + 13, "draft: ranges and chunk ends", d.count, d.end[1]);
      check(d.first == 0 && d.ns_chunk == 41 && d.budget == 780, "draft: starts at 0, 780 chunks", d.first, d.budget);
    }
    L2pfPlan d{};
    check(!l2pf_draft_plan(p, n, 3, kL2pfDraftSites, &d), "draft: a site past the last has no plan");
    check(!l2pf_draft_plan(p, n, 3, -1, &d), "draft: site -1 has no plan");
    const void* const none[1] = {nullptr};
    const size_t zero[1] = {0};
    check(!l2pf_draft_plan(none, zero, 1, 0, &d), "draft: no ranges, no plan");
  }
  // DSK_HIDE2's setting: unset is 10 (wq_b on two tiles and the draft pass's prefetch); 0, 2, 8 and 10 are the values, any
  // other stops the start
  {
    check(hide2_parse(nullptr) == 10, "DSK_HIDE2 unset reads 10", hide2_parse(nullptr));
    check(hide2_parse("0") == 0, "DSK_HIDE2=0 reads 0", hide2_parse("0"));
    check(hide2_parse("2") == kHide2Wqb && hide2_parse("8") == kHide2Draft && hide2_parse("10") == (kHide2Wqb | kHide2Draft),
          "DSK_HIDE2 2, 8 and 10 read as they are");
    for (const char* bad : {"1", "4", "15", "3", "12", "-1", "16", "", "x", "10x", " 10", "2.0"})
      check(hide2_parse(bad) == -1, bad, hide2_parse(bad));
  }
  printf("l2pf_plan_test: %d ok, %d FAIL\n", g_ok, g_fail);
  return g_fail ? 1 : 0;
}
