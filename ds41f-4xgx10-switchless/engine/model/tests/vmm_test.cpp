// The memory pool behind the chunk pool's arrays (model/vmm.h) against a fake backend, on the host (no GPU; the CMake
// target vmm_test): a touch of an unmapped chunk is detected, ensure() fails cleanly on an empty free list and maps
// nothing, release() returns every chunk, a failing driver leaves a consistent pool, commitments never exceed the
// pool, chunks are poisoned when mapped, and nothing is created or destroyed after the start.
#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

#include "model/vmm.h"

using namespace model::vmm;

namespace {

int g_fail = 0, g_checks = 0;
void check(bool ok, const char* what) {
  ++g_checks;
  if (ok) return;
  ++g_fail;
  std::printf("FAIL: %s\n", what);
}

struct Rig {
  FakeBackend* fb;
  std::unique_ptr<Pool> pool;
  explicit Rig(size_t chunks, size_t max_create = SIZE_MAX, size_t fail_map_at = 0) {
    auto b = std::make_unique<FakeBackend>(max_create, fail_map_at);
    fb = b.get();
    pool = std::make_unique<Pool>(std::move(b), chunks);
  }
};

// two arrays of 2 units of 4 chunks; slot 0 = unit 0, slot 1 = unit 1
void two_slots(Rig& r) {
  check(r.pool->add_array(4 * kChunk, 2) == 0, "array 0");
  check(r.pool->add_array(4 * kChunk, 2) == 1, "array 1");
  check(r.pool->add_slot(0, 1) == 0, "slot 0");
  check(r.pool->add_slot(1, 1) == 1, "slot 1");
}

void unmapped_touch_is_detected() {
  Rig r(8);
  two_slots(r);
  const uint64_t a0 = r.pool->base(0);
  check(!r.fb->touch(a0, 1), "RED: nothing is mapped before ensure, a touch must be detected");
  const size_t need[2] = {1, 1};  // one byte of each array: one chunk each
  check(r.pool->ensure(0, need), "ensure(1 byte, 1 byte)");
  check(r.fb->touch(a0, kChunk), "the first chunk of array 0 is touchable");
  check(r.fb->touch(r.pool->base(1), kChunk), "the first chunk of array 1 is touchable");
  check(!r.fb->touch(a0 + kChunk, 1), "the second chunk is not mapped: a touch is detected");
  check(!r.fb->touch(a0 + kChunk - 1, 2), "a touch that crosses into the second chunk is detected");
  check(!r.fb->touch(a0 + 4 * kChunk, 1), "slot 1's region is not mapped: a touch is detected");
  check(r.pool->mapped(0) == 2 && r.pool->mapped(0, 0) == 1 && r.pool->mapped(0, 1) == 1, "two chunks mapped in slot 0");
  check(r.pool->free_chunks() == 6, "six chunks free");
}

void growth_keeps_what_is_mapped() {
  Rig r(8);
  two_slots(r);
  const size_t two[2] = {2 * kChunk, 0};
  const size_t four[2] = {3 * kChunk + 1, kChunk};
  check(r.pool->ensure(0, two), "ensure two chunks");
  check(r.fb->touch(r.pool->base(0), 2 * kChunk), "two chunks touchable");
  check(r.pool->growth(0, four) == 3, "growth to 4 + 1 chunks takes 3 more");
  check(r.pool->ensure(0, four), "ensure grows");
  check(r.fb->touch(r.pool->base(0), 4 * kChunk), "four chunks touchable, the first two as before");
  check(r.pool->mapped(0, 0) == 4 && r.pool->mapped(0, 1) == 1, "prefix per array");
  const size_t smaller[2] = {kChunk, 0};
  check(r.pool->ensure(0, smaller) && r.pool->mapped(0) == 5, "a smaller request maps nothing and unmaps nothing");
  const size_t past[2] = {4 * kChunk + 1, 0};
  check(!r.pool->ensure(0, past), "a request past the slot's region fails");
  check(r.pool->mapped(0) == 5, "and changes nothing");
}

void empty_free_list_fails_cleanly() {
  Rig r(3);
  two_slots(r);
  const size_t need[2] = {2 * kChunk, 2 * kChunk};  // four chunks, three exist
  check(!r.pool->ensure(0, need), "RED: ensure past the free list must fail");
  check(r.pool->mapped(0) == 0 && r.fb->mapped_chunks() == 0, "and map nothing (all or nothing across the arrays)");
  check(r.pool->free_chunks() == 3, "the free list is whole");
  check(!r.pool->error().empty(), "with a reason");
  const size_t fits[2] = {2 * kChunk, kChunk};  // three chunks
  check(r.pool->ensure(0, fits), "what fits still maps");
  check(r.pool->free_chunks() == 0, "the free list is empty");
  const size_t more[2] = {2 * kChunk, 2 * kChunk};
  check(!r.pool->ensure(0, more), "one more chunk fails on the empty list");
  const size_t other[2] = {1, 0};
  check(!r.pool->ensure(1, other), "another slot fails on the empty list");
  check(r.fb->mapped_chunks() == 3, "nothing else was mapped");
}

void release_returns_chunks() {
  Rig r(8);
  two_slots(r);
  const size_t a[2] = {3 * kChunk, 2 * kChunk}, b[2] = {kChunk, kChunk};
  check(r.pool->ensure(0, a) && r.pool->ensure(1, b), "two slots mapped");
  check(r.pool->free_chunks() == 1 && r.pool->used_chunks() == 7 && r.pool->peak_used() == 7, "seven used");
  const uint64_t a0 = r.pool->base(0);
  r.pool->release(0);
  check(r.pool->free_chunks() == 6 && r.pool->mapped(0) == 0, "release gives slot 0's five chunks back");
  check(!r.fb->touch(a0, 1) && !r.fb->touch(a0 + 2 * kChunk, 1), "RED: a released slot's chunks are unmapped, a touch must be detected");
  check(r.fb->touch(r.pool->base(0) + 4 * kChunk, kChunk), "slot 1 is untouched");
  check(r.pool->peak_used() == 7, "the peak stays");
  check(r.pool->ensure(0, a), "the freed chunks map again");
  check(r.fb->touch(a0, 3 * kChunk), "and are touchable at slot 0's addresses");
  check(r.fb->live_handles() == 8 && r.fb->created_total() == 8, "no chunk was created or destroyed after the start");
}

void creation_failure_is_reported() {
  Rig r(8, 5);
  check(!r.pool->ok() && r.pool->created() == 5, "an allocation failure at the sixth chunk: 5 made, not ok");
  check(r.pool->free_chunks() == 5, "the five are free");
}

void driver_failure_keeps_the_pool_consistent() {
  Rig r(8, SIZE_MAX, 2);  // the second map call fails
  two_slots(r);
  const size_t need[2] = {3 * kChunk, 0};
  check(!r.pool->ensure(0, need), "a map that fails fails the ensure");
  check(r.pool->mapped(0, 0) == 1 && r.pool->free_chunks() == 7, "the first chunk stays mapped, the failed one is still free");
  check(r.fb->touch(r.pool->base(0), kChunk) && !r.fb->touch(r.pool->base(0) + kChunk, 1), "touchable as far as it got");
  check(r.pool->ensure(0, need), "the next try maps the rest");
  check(r.fb->touch(r.pool->base(0), 3 * kChunk), "three chunks");
}

void commitments_never_exceed_the_pool() {
  Rig r(10);
  two_slots(r);
  check(r.pool->reserve(0, 6), "slot 0 reserves 6 of 10");
  check(!r.pool->reserve(1, 5), "slot 1 cannot reserve 5 more");
  check(r.pool->reserve(1, 4), "slot 1 reserves 4");
  check(r.pool->uncommitted() == 0 && r.pool->committed_total() == 10, "the pool is committed");
  const size_t need[2] = {kChunk, kChunk};
  check(r.pool->ensure(0, need), "slot 0 maps inside its reservation");
  check(r.pool->committed(0) == 6 && r.pool->committed_total() == 10, "a slot below its reservation holds the reservation");
  check(r.pool->reserve(0, 1) && r.pool->committed(0) == 2, "a smaller reservation commits what is mapped");
  check(r.pool->uncommitted() == 4, "the rest is uncommitted");
  r.pool->release(1);
  check(r.pool->reserved(1) == 0 && r.pool->uncommitted() == 8, "release drops the reservation");
  // within the reservations, mapping cannot fail
  check(r.pool->reserve(1, 8) && r.pool->reserve(0, 2), "reserve 8 + 2");
  const size_t big[2] = {4 * kChunk, 4 * kChunk};  // slot 1: 8 chunks
  check(r.pool->ensure(1, big), "slot 1 maps all 8 of its reservation with slot 0's two mapped");
  check(r.pool->free_chunks() == 0, "the pool is full, as committed");
}

void chunks_are_poisoned_when_mapped() {
  Rig r(8);
  two_slots(r);
  r.pool->set_poison(true);
  const size_t need[2] = {2 * kChunk, kChunk};
  check(r.pool->ensure(0, need) && r.fb->poisoned_total() == 3, "every mapped chunk is poisoned");
  r.pool->release(0);
  check(r.fb->poisoned_total() == 0, "an unmapped chunk holds no poison record");
  check(r.pool->ensure(0, need) && r.fb->poisoned_total() == 3, "and a chunk mapped again is poisoned again");
}

void regions_do_not_overlap() {
  Rig r(8);
  two_slots(r);
  check(r.pool->base(0) != r.pool->base(1), "arrays have ranges of their own");
  const size_t need[2] = {kChunk, 0};
  check(r.pool->ensure(1, need), "slot 1 maps");
  check(r.fb->touch(r.pool->base(0) + 4 * kChunk, kChunk), "at its unit's offset in array 0");
  check(!r.fb->touch(r.pool->base(0), 1), "and not at slot 0's");
  check(r.pool->slot_capacity(1, 0) == 4 * kChunk, "a slot's capacity is its units' strides");
  check(r.pool->add_array(kChunk, 1) == -1, "arrays cannot be added after the slots");
}


// 32 and 64 slots on one pool (--slots 32 / 64): every slot owns its unit's range, the commitments of all of them
// stay inside the pool, and a slot far from the first maps and releases like any other
void many_slots_share_one_pool() {
  for (const int n : {32, 64}) {
    Rig r(100);  // 100 chunks for n slots of 2 arrays x 4 chunks whole: far less than they would take
    check(r.pool->add_array(4 * kChunk, n) == 0 && r.pool->add_array(4 * kChunk, n) == 1, "two arrays of n units");
    bool ok = true;
    for (int s = 0; s < n; ++s) ok &= r.pool->add_slot(s, 1) == s;
    check(ok && r.pool->slots() == n, "n slots, one unit each");
    int reserved = 0;
    for (int s = 0; s < n; ++s) reserved += r.pool->reserve(s, 4) ? 1 : 0;
    check(reserved == 25 && r.pool->uncommitted() == 0, "25 slots of 4 chunks fill the pool, the others are refused");
    const int last = n - 1;
    check(!r.pool->reserve(last, 1), "the last slot cannot commit one more");
    r.pool->release(0);
    r.pool->release(1);
    check(r.pool->reserve(last, 4) && r.pool->committed(last) == 4, "slots released: the last slot's 4 chunks commit");
    const size_t need[2] = {2 * kChunk, 2 * kChunk};
    check(r.pool->ensure(last, need), "the last slot maps its 4 chunks");
    check(r.fb->touch(r.pool->base(0) + (size_t)last * 4 * kChunk, 2 * kChunk), "at its unit's offset in array 0");
    check(!r.fb->touch(r.pool->base(0) + (size_t)(last - 1) * 4 * kChunk, 1), "and not at the slot before it");
    check(r.pool->slot_capacity(last, 1) == 4 * kChunk, "its capacity is its unit's stride");
    r.pool->release(last);
    check(r.pool->mapped(last) == 0 && r.pool->committed_total() <= 100, "released: nothing mapped in it");
  }
}

void nothing_is_created_or_destroyed_after_the_start() {
  Rig r(8);
  two_slots(r);
  check(r.pool->add_chunks(4) == 4, "more chunks, at the start");
  const size_t made = r.fb->created_total();
  check(made == 12 && r.fb->live_handles() == 12, "twelve chunks in all");
  const size_t need[2] = {2 * kChunk, 2 * kChunk};
  for (int round = 0; round < 20; ++round) {  // the slots take turns on the same chunks
    check(r.pool->ensure(round % 2, need), "a slot maps");
    r.pool->release(round % 2);
  }
  check(r.fb->created_total() == made && r.fb->live_handles() == made, "no chunk was created or destroyed while it ran");
  check(r.pool->free_chunks() == 12 && r.fb->mapped_chunks() == 0, "and every chunk is free at the end");
}

}  // namespace

int main() {
  unmapped_touch_is_detected();
  growth_keeps_what_is_mapped();
  empty_free_list_fails_cleanly();
  release_returns_chunks();
  creation_failure_is_reported();
  driver_failure_keeps_the_pool_consistent();
  commitments_never_exceed_the_pool();
  chunks_are_poisoned_when_mapped();
  regions_do_not_overlap();
  many_slots_share_one_pool();
  nothing_is_created_or_destroyed_after_the_start();
  std::printf("%s: %d checks, %d failed\n", g_fail ? "FAIL" : "PASS", g_checks, g_fail);
  return g_fail ? 1 : 0;
}
