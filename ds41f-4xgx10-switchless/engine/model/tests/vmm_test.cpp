// The slot memory pool (model/vmm.h) against a fake backend, on the host (no GPU; the CMake target vmm_test):
// a touch of an unmapped chunk is detected, ensure() fails cleanly on an empty free list and maps nothing,
// release() returns every chunk, a failing driver leaves a consistent pool, commitments never exceed the
// pool, chunks are poisoned when mapped. The store's regions on the same pool (the shared budget): a block maps the
// chunks it overlaps under a reference count, an idle chunk stays mapped until a trim, the store's chunks and the
// slots' commitments never exceed the pool together, a commitment that finds the pool short fits once the store has
// trimmed, and nothing is created or destroyed after the start.
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

// ---- regions: the session store on the slot pool -----------------------------------------------------

// two slots of two arrays (4 chunks each) on 8 chunks, then 4 more for the store, and a region over all 12
struct Shared {
  Rig r;
  int region = -1;
  Shared() : r(8) {
    two_slots(r);
    check(r.pool->add_chunks(4) == 4 && r.pool->ok() && r.pool->total() == 12, "the store's 4 chunks are made at the start");
    region = r.pool->add_region(r.pool->total());
    check(region >= 0 && r.pool->region_chunks(region) == 12, "a region over every chunk of the pool");
  }
};

void store_blocks_map_the_chunks_they_overlap() {
  Shared sh;
  Pool& p = *sh.r.pool;
  const uint64_t base = p.region_base(sh.region);
  check(!sh.r.fb->touch(base, 1), "RED: nothing of the region is mapped before a block: a touch must be detected");
  check(p.region_need(sh.region, kChunk / 2, kChunk) == 2, "a block across a chunk boundary needs two chunks");
  check(p.region_acquire(sh.region, kChunk / 2, kChunk), "block A at half a chunk, a chunk long");
  check(p.region_mapped(sh.region) == 2 && p.regions_mapped() == 2 && p.free_chunks() == 10, "chunks 0 and 1 are mapped");
  check(sh.r.fb->touch(base + kChunk / 2, kChunk) && !sh.r.fb->touch(base + 2 * kChunk, 1), "touchable over the block, not past it");
  check(p.region_refs(sh.region, 0) == 1 && p.region_refs(sh.region, 1) == 1, "one block on each");
  check(p.region_acquire(sh.region, kChunk + kChunk / 4, kChunk / 4), "block B inside chunk 1");
  check(p.region_mapped(sh.region) == 2 && p.region_refs(sh.region, 1) == 2, "chunk 1 has two blocks, nothing new is mapped");
  check(p.uncommitted() == 12 - 2, "the store's chunks leave ten uncommitted");
  check(!p.region_acquire(sh.region, 11 * kChunk + kChunk / 2, kChunk), "a block past the region fails");
  check(p.region_mapped(sh.region) == 2, "and maps nothing");
}

void an_idle_chunk_stays_mapped_until_the_trim() {
  Shared sh;
  Pool& p = *sh.r.pool;
  const uint64_t base = p.region_base(sh.region);
  check(p.region_acquire(sh.region, kChunk / 2, kChunk) && p.region_acquire(sh.region, kChunk + kChunk / 4, kChunk / 4), "A and B");
  p.region_drop(sh.region, kChunk / 2, kChunk);  // A leaves: chunk 0 has no block, chunk 1 still has B
  check(p.region_refs(sh.region, 0) == 0 && p.region_refs(sh.region, 1) == 1, "chunk 0 has none, chunk 1 has B");
  check(p.region_idle(sh.region) == 1 && p.region_mapped(sh.region) == 2, "chunk 0 is idle and still mapped");
  check(sh.r.fb->touch(base, kChunk), "an idle chunk is still touchable (a queued copy may be reading it)");
  check(p.uncommitted() == 10, "an idle chunk still counts against the slots' budget");
  check(p.region_acquire(sh.region, 0, 4096) && p.region_idle(sh.region) == 0, "a block on the idle chunk takes it again, as it is");
  check(p.region_mapped(sh.region) == 2 && sh.r.fb->mapped_chunks() == 2, "with no map call");
  p.region_drop(sh.region, 0, 4096);
  p.region_drop(sh.region, kChunk + kChunk / 4, kChunk / 4);
  check(p.region_idle(sh.region) == 2 && p.region_mapped(sh.region) == 2, "both idle");
  check(p.region_trim(sh.region) == 2, "the trim unmaps both");
  check(p.region_mapped(sh.region) == 0 && p.free_chunks() == 12 && p.uncommitted() == 12, "every chunk is free again");
  check(!sh.r.fb->touch(base, 1) && !sh.r.fb->touch(base + kChunk, 1), "RED: a trimmed chunk is unmapped, a touch must be detected");
  check(p.region_trim(sh.region) == 0, "a trim with nothing idle does nothing");
  check(p.region_acquire(sh.region, 5 * kChunk, 3 * kChunk) && p.region_mapped(sh.region) == 3 && p.region_refs(sh.region, 6) == 1,
        "a block of three whole chunks maps three");
  check(p.region_need(sh.region, 5 * kChunk, 3 * kChunk) == 0, "and needs nothing more");
  check(p.region_need(sh.region, 4 * kChunk + kChunk - 1, 2) == 1, "two bytes across a boundary need the chunk below it");
}

void the_store_maps_only_what_no_slot_committed() {
  Shared sh;
  Pool& p = *sh.r.pool;
  check(p.reserve(0, 3), "slot 0 commits 3 of 12");
  check(p.region_acquire(sh.region, 0, 9 * kChunk), "the store maps the nine chunks left");
  check(p.uncommitted() == 0 && p.regions_mapped() == 9 && p.free_chunks() == 3, "the pool is committed and mapped to the last chunk");
  check(!p.region_acquire(sh.region, 9 * kChunk, 1), "RED: one chunk more than the slots left uncommitted must fail");
  check(p.region_mapped(sh.region) == 9 && !p.error().empty(), "nothing was mapped, with a reason");
  check(!p.reserve(1, 1), "a commitment on top of the store's chunks fails too");
  // slot 0 stays within its reservation: it maps its 3 chunks although the store holds the rest
  const size_t need[2] = {2 * kChunk, kChunk};
  check(p.ensure(0, need) && p.mapped(0) == 3, "slot 0 maps what it committed");
  check(p.free_chunks() == 0 && sh.r.fb->mapped_chunks() == 12, "all twelve are mapped, none twice");
  check(p.committed_total() + p.regions_mapped() <= p.total(), "slots' commitments + the store's chunks <= the pool");
}

void a_commitment_after_store_growth_fits_once_the_store_gives_back() {
  Shared sh;
  Pool& p = *sh.r.pool;
  check(p.region_acquire(sh.region, 0, 10 * kChunk), "the store grows into ten chunks while the slots are idle");
  check(!p.reserve(0, 6), "slot 0 cannot commit 6: ten of twelve are the store's");
  check(p.reserve(0, 2), "but it can commit 2");
  // the store frees the blocks over chunks 4..9 (a session spilled): idle, still mapped, still counted
  p.region_drop(sh.region, 4 * kChunk, 6 * kChunk);
  check(p.region_idle(sh.region) == 6, "six chunks idle");
  check(!p.reserve(0, 6), "RED: idle chunks are still mapped: the reservation still fails before the trim");
  check(p.region_trim(sh.region) == 6, "the trim gives six back");
  check(p.reserve(0, 6), "the commitment fits now");
  check(p.uncommitted() == 12 - 6 - 4, "the store's four chunks and slot 0's six leave two");
  const size_t big[2] = {3 * kChunk, 3 * kChunk};  // six chunks
  check(p.ensure(0, big) && p.mapped(0) == 6, "and slot 0 maps all six");
  check(p.free_chunks() == 2, "two chunks free: the store's four and slot 0's six are mapped");
  p.release(0);
  check(p.uncommitted() == 8 && p.region_mapped(sh.region) == 4, "a released slot's chunks are uncommitted again; the store's four stay");
  check(p.region_acquire(sh.region, 4 * kChunk, 8 * kChunk), "and the store grows into them");
  check(p.regions_mapped() == 12 && p.uncommitted() == 0, "all twelve");
}

void a_driver_failure_in_a_block_maps_nothing() {
  Rig r(8, SIZE_MAX, 2);  // the second map call fails
  check(r.pool->add_chunks(4) == 4, "four more chunks");
  const int g = r.pool->add_region(12);
  check(g >= 0, "a region");
  check(!r.pool->region_acquire(g, 0, 3 * kChunk), "a block of three chunks whose second map fails");
  check(r.pool->region_mapped(g) == 0 && r.fb->mapped_chunks() == 0 && r.pool->free_chunks() == 12, "all or nothing: the first chunk was given back");
  check(r.pool->region_refs(g, 0) == 0 && r.pool->region_idle(g) == 0, "no reference counted");
  check(r.pool->region_acquire(g, 0, 3 * kChunk) && r.pool->region_mapped(g) == 3, "the next try maps all three");
}

void nothing_is_created_or_destroyed_after_the_start() {
  Rig r(8);
  two_slots(r);
  check(r.pool->add_chunks(4) == 4, "the store's chunks, at the start");
  const int g = r.pool->add_region(r.pool->total());
  const size_t made = r.fb->created_total();
  check(made == 12 && r.fb->live_handles() == 12, "twelve chunks in all");
  const size_t need[2] = {2 * kChunk, 2 * kChunk};
  for (int round = 0; round < 20; ++round) {  // the slots and the store take turns on the same chunks
    check(r.pool->ensure(round % 2, need), "a slot maps");
    check(r.pool->region_acquire(g, (size_t)(round % 5) * kChunk, 3 * kChunk), "the store maps three chunks beside it");
    r.pool->release(round % 2);
    r.pool->region_drop(g, (size_t)(round % 5) * kChunk, 3 * kChunk);
    r.pool->region_trim(g);
  }
  check(r.fb->created_total() == made && r.fb->live_handles() == made, "no chunk was created or destroyed while it ran");
  check(r.pool->free_chunks() == 12 && r.fb->mapped_chunks() == 0, "and every chunk is free at the end");
}

void a_trim_is_the_only_way_a_store_chunk_leaves() {
  // blocks over many chunks come and go in a different order: the mapped set is always exactly the chunks with a block
  // or idle, and a trim leaves exactly the chunks with a block
  Shared sh;
  Pool& p = *sh.r.pool;
  struct B {
    size_t off, n;
  };
  const B blocks[] = {{0, 100 * 1024}, {kChunk - 4096, 8192}, {3 * kChunk + 5, 3 * kChunk}, {7 * kChunk, kChunk}, {7 * kChunk + 10, 10}};
  for (const B& b : blocks) check(p.region_acquire(sh.region, b.off, b.n), "block acquired");
  // chunks: 0 (A, B), 1 (B), 3-6 (C), 7 (D, E)
  check(p.region_mapped(sh.region) == 7, "chunks 0, 1, 3, 4, 5, 6 and 7");
  check(p.region_refs(sh.region, 0) == 2 && p.region_refs(sh.region, 7) == 2 && p.region_refs(sh.region, 2) == 0, "counts");
  p.region_drop(sh.region, blocks[2].off, blocks[2].n);
  p.region_drop(sh.region, blocks[4].off, blocks[4].n);
  check(p.region_idle(sh.region) == 4 && p.region_mapped(sh.region) == 7, "C's four chunks are idle; D still holds chunk 7");
  check(p.region_trim(sh.region) == 4 && p.region_mapped(sh.region) == 3, "the trim leaves the chunks a block is on");
  p.region_drop(sh.region, blocks[0].off, blocks[0].n);
  p.region_drop(sh.region, blocks[1].off, blocks[1].n);
  p.region_drop(sh.region, blocks[3].off, blocks[3].n);
  check(p.region_idle(sh.region) == 3 && p.region_trim(sh.region) == 3 && p.regions_mapped() == 0, "and the last trim takes the rest");
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
  store_blocks_map_the_chunks_they_overlap();
  an_idle_chunk_stays_mapped_until_the_trim();
  the_store_maps_only_what_no_slot_committed();
  a_commitment_after_store_growth_fits_once_the_store_gives_back();
  a_driver_failure_in_a_block_maps_nothing();
  nothing_is_created_or_destroyed_after_the_start();
  a_trim_is_the_only_way_a_store_chunk_leaves();
  std::printf("%s: %d checks, %d failed\n", g_fail ? "FAIL" : "PASS", g_checks, g_fail);
  return g_fail ? 1 : 0;
}
