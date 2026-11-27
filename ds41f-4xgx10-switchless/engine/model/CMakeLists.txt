add_library(dsmodel STATIC model.cu ablit.cu ops.cu fused.cu sample.cu prefill.cu state.cu spec.cu window.cu tenant.cu wave_probe.cu spill.cpp vmm.cpp vmm.cu chunks.cpp copy.cu store_trace.cpp store_core.cpp quarter.cu vit_area.cu vit_cmd.cu rows_lend.cu)
# an old-rules build (CMAKE_CXX_FLAGS holds -DPICO_OLD_RULES) gives the kernels' sources the same definition, which nvcc does not take from it
if(CMAKE_CXX_FLAGS MATCHES "-DPICO_OLD_RULES")
  target_compile_definitions(dsmodel PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:PICO_OLD_RULES>)
endif()
target_include_directories(dsmodel PUBLIC ${CMAKE_SOURCE_DIR})
find_package(CUDAToolkit REQUIRED)
target_link_libraries(dsmodel PUBLIC dsattn dsk_kernels dsgb10_comm engram dsw_loader dspark vit_kernels CUDA::cuda_driver)

add_executable(coll_tests tests/coll_tests.cu)
target_include_directories(coll_tests PRIVATE ${CMAKE_SOURCE_DIR}/attn/tests)
target_link_libraries(coll_tests dsmodel)

# the runtime abliteration's kernels against a host reference (one GB10, no ring, no weights)
add_executable(ablit_test tests/ablit_test.cu)
target_link_libraries(ablit_test dsmodel)

# B3: the batch spec step's layout and verdicts (spec_batch.h) against a host reference; host only, no GPU
add_executable(spec_batch_test tests/spec_batch_test.cu)
target_include_directories(spec_batch_test PRIVATE ${CMAKE_SOURCE_DIR})

# The sampler's shared rules (samp_rule.h: the Gumbel path's flag, the certified head's row test, the noise and the Gumbel-max pick, the
# settings' parsers) and the draft module's copy of the noise (dspark/noise.h) against their own oracles, each with its wrong-rule
# control; host only, no GPU
add_executable(samp_rule_test tests/samp_rule_test.cpp)
target_include_directories(samp_rule_test PRIVATE ${CMAKE_SOURCE_DIR})

# The exact shared steps' settings (exact_rule.h: DSGB10_NORM_EXACT, DSGB10_WIDE_EXACT, _MT, _GQ, DSGB10_R16_EXACT: their ranges and the
# refusals of a value outside them), each with its wrong-rule control; host only, no GPU
add_executable(exact_rule_test tests/exact_rule_test.cpp)
target_include_directories(exact_rule_test PRIVATE ${CMAKE_SOURCE_DIR})

# Copy drafting's matcher (copy.h) against a plain reference; host only, no GPU
add_executable(copy_match_test tests/copy_match_test.cpp)
target_include_directories(copy_match_test PRIVATE ${CMAKE_SOURCE_DIR})

# Long copy steps (copy_run.h: the cursor, the ladder, copy_fit, DSGB10_COPY_ROWS / _ENTER, the 7-draft block's rule) against a plain
# reference on planted streams, each rule with its wrong-rule control; host only, no GPU
add_executable(copy_run_test tests/copy_run_test.cpp)
target_include_directories(copy_run_test PRIVATE ${CMAKE_SOURCE_DIR})

# Copy drafting's learner (copy.h: outcome, the shared cells, a conversation's ratio, the gate's row) against its cases and a
# simulated stream; host only, no GPU. COPY_ADAPT_TEST_RULES adds the wrong-rule switches its controls use
add_executable(copy_adapt_test tests/copy_adapt_test.cpp)
target_include_directories(copy_adapt_test PRIVATE ${CMAKE_SOURCE_DIR})

# Copy drafting on a GB10 (small allocations only, so it can run beside an engine): the device's index and lookups
# against the host reference, bit for bit, on random and repetitive token streams (tests/copy_kernel_test.cu)
add_executable(copy_kernel_test tests/copy_kernel_test.cu)
target_include_directories(copy_kernel_test PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(copy_kernel_test dsmodel)

# Copy drafting's learner inside the two gate kernels, on a GB10 (not run by the build): the kernels' state against the host reference,
# entries that join and leave a batch step, the attach reset, and the kernels' cost with and without the learner (tests/copy_gate_test.cu)
add_executable(copy_gate_test tests/copy_gate_test.cu)
target_include_directories(copy_gate_test PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(copy_gate_test dsmodel)

# The slot memory pool (vmm.h) against a fake backend; host only, no GPU
add_executable(vmm_test tests/vmm_test.cpp vmm.cpp)
target_include_directories(vmm_test PRIVATE ${CMAKE_SOURCE_DIR})
add_executable(chunks_test tests/chunks_test.cpp chunks.cpp)
target_include_directories(chunks_test PRIVATE ${CMAKE_SOURCE_DIR})

# A piece's quarters (quarter_map.h): the ownership maps and the quarter layout against an oracle written from the entry's
# arithmetic, the exact sums, the trailer's round trip and its refusals; host only, no GPU
add_executable(quarter_map_test tests/quarter_map_test.cpp)
target_include_directories(quarter_map_test PRIVATE ${CMAKE_SOURCE_DIR})

# The quartering kernels (quarter.cu) on a GB10 (small allocations only): k_quarter, k_unquarter and the exact sums against the
# oracle's bytes and sums, a flip in a quarter's last word
add_executable(quarter_test tests/quarter_test.cu)
target_include_directories(quarter_test PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(quarter_test dsmodel)

# The store's trace (store_trace.h): the format written from a scripted sequence of entries and read back, the comparison's
# controls, the refusals; host only, no GPU
add_executable(store_trace_test tests/store_trace_test.cpp store_trace.cpp)
target_include_directories(store_trace_test PRIVATE ${CMAKE_SOURCE_DIR})

# The store's bookkeeping (store_core.h) over a fake device: unit rows with their controls, the replay of recorded traces
# (tests/traces) with the invariants after every record, the coverage of entry and answer kinds; host only, no GPU, no CUDA header
# on the include path of what it builds. STORE_CORE_TEST_RULES adds the wrong-rule switches the controls use to the core
add_executable(store_core_test tests/store_core_test.cpp store_core.cpp chunks.cpp store_trace.cpp)
target_include_directories(store_core_test PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_definitions(store_core_test PRIVATE STORE_CORE_TEST_RULES STORE_TRACES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/tests/traces\")

# Four cores in step over four fake devices whose collectives meet at a barrier (host only)
add_executable(four_rank_test tests/four_rank_test.cpp store_core.cpp chunks.cpp store_trace.cpp)
target_include_directories(four_rank_test PRIVATE ${CMAKE_SOURCE_DIR})
target_compile_definitions(four_rank_test PRIVATE STORE_CORE_TEST_RULES)

# A request's image rows as store numbers: geometry, runs, the flights' ledger against the allocator, the reservation (host only)
add_executable(img_rows_test tests/img_rows_test.cpp chunks.cpp)
target_include_directories(img_rows_test PRIVATE ${CMAKE_SOURCE_DIR})

# The lent groups' ledger, the hand-over proof, the cache's sizes and resize plan (lend.h) over the real allocator and the VMM
# pool on a fake driver; host only, no GPU. Built with -DPICO_OLD_RULES its red cases fail
add_executable(lend_test tests/lend_test.cpp chunks.cpp vmm.cpp)
target_include_directories(lend_test PRIVATE ${CMAKE_SOURCE_DIR})

# The tenants' table (lend.h) and the three steps of a tenant's change (tenant.h) with fake bodies over the same fake pool: the
# invariant, every tenant's room, a body failing on one rank, the one queue of store changes, the order tenants give way in, the
# digest's tenant terms; host only, no GPU. Built with -DPICO_OLD_RULES its red cases fail
add_executable(tenant_test tests/tenant_test.cpp chunks.cpp vmm.cpp store_core.cpp store_trace.cpp)
target_include_directories(tenant_test PRIVATE ${CMAKE_SOURCE_DIR})

# The pool on a GB10 (stop the engine first): a graph captured over a reserved range replays after more is mapped;
# maps and unmaps beside a replaying graph (tests/vmm_probe.cu)
add_executable(vmm_probe tests/vmm_probe.cu)
target_link_libraries(vmm_probe CUDA::cuda_driver CUDA::cudart)

# A window's memory leaving the chunk pool and coming back (vmm.h shrink / trim / add_chunks) on a GB10: free device
# memory follows, a graph over the arrays replays after the give-back (tests/vmm_giveback.cu)
add_executable(vmm_giveback tests/vmm_giveback.cu)
target_include_directories(vmm_giveback PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(vmm_giveback dsmodel)

# Profiling builds (-DDSGB10_NVTX=ON): NVTX ranges per turn, chunk, layer and part for nsys (model/nvtx.h),
# and DSGB10_ROUTE_DUMP (model/prefill.cu). The ranges cost nothing without a profiler attached.
# Fault injection for the tenants' hand-over tests (model/tenant.cu): DSGB10_LEND_FAIL=rank:R,step:B|C|back[,tenant:N], and
# tenant 15, a dummy device block driven by CONFIG's test_tenant_groups
option(DSGB10_TEST_FAULTS "DSGB10_LEND_FAIL fault injection and the dummy tenant in the pool's hand-over" OFF)
if(DSGB10_TEST_FAULTS)
  target_compile_definitions(dsmodel PUBLIC DSGB10_TEST_FAULTS)
endif()

option(DSGB10_NVTX "NVTX ranges in the model and the server, for nsys" OFF)
if(DSGB10_NVTX)
  target_compile_definitions(dsmodel PUBLIC DSGB10_NVTX)
endif()

# Development builds (-DDSGB10_DEV=ON): upstream's measurement instruments, none of them in the serving engine: the step log
# (DSGB10_STEP_LOG, DSGB10_FORCE_NTOP: the confidence network's training data, tools/stack4), the hidden-state dump
# (DSGB10_MH_DUMP, DSGB10_MH_RANK: the drafter's training data) and the certified head's check against the full head
# (DSGB10_CERT_CHECK, DSGB10_CERT_LOG, DSGB10_CERT_FORCE: the proof that it picks the full head's token)
option(DSGB10_DEV "development instruments: step log, hidden-state dump, certified-head check" OFF)
if(DSGB10_DEV)
  target_compile_definitions(dsmodel PUBLIC DSGB10_DEV)
endif()

# The windowed token history (tokwin.h): Engram hashes from a trimmed window against the full vector, and memory over
# 32 units of 2M-token prompts; host only
add_executable(tokwin_test tests/tokwin_test.cpp)
target_include_directories(tokwin_test PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(tokwin_test PRIVATE engram)

# The embedding's vocabulary cut and its exchange (common/embed_shard.h) against the replicated row, all 129,280 ids, four
# slots simulated; host only, no GPU. Built with -DPICO_OLD_RULES the other ranks' slots hold +0.0 and its red rows fail
add_executable(embed_shard_test tests/embed_shard_test.cpp)
target_include_directories(embed_shard_test PRIVATE ${CMAKE_SOURCE_DIR})

# The embedding lookup's kernels on a GB10 (stop the engine first: four shards of 331 MB): embed_part over four ranks' shards summed
# like the ring, embed_fan, and prefill's embed_send / embed_pick, for all 129,280 ids against the replicated row
# (tests/embed_shard_dev_test.cu)
add_executable(embed_shard_dev_test tests/embed_shard_dev_test.cu)
target_include_directories(embed_shard_dev_test PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(embed_shard_dev_test dsmodel)
