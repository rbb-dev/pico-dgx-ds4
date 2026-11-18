add_library(dsmodel STATIC model.cu ablit.cu ops.cu fused.cu sample.cu prefill.cu state.cu spec.cu window.cu lend.cu wave_probe.cu spill.cpp vmm.cpp vmm.cu chunks.cpp)
target_include_directories(dsmodel PUBLIC ${CMAKE_SOURCE_DIR})
find_package(CUDAToolkit REQUIRED)
target_link_libraries(dsmodel PUBLIC dsattn dsk_kernels dsgb10_comm engram dsw_loader dspark CUDA::cuda_driver)

add_executable(coll_tests tests/coll_tests.cu)
target_include_directories(coll_tests PRIVATE ${CMAKE_SOURCE_DIR}/attn/tests)
target_link_libraries(coll_tests dsmodel)

# the runtime abliteration's kernels against a host reference (one GB10, no ring, no weights)
add_executable(ablit_test tests/ablit_test.cu)
target_link_libraries(ablit_test dsmodel)

# B3: the batch spec step's layout and verdicts (spec_batch.h) against a host reference; host only, no GPU
add_executable(spec_batch_test tests/spec_batch_test.cu)
target_include_directories(spec_batch_test PRIVATE ${CMAKE_SOURCE_DIR})

# The slot memory pool (vmm.h) against a fake backend; host only, no GPU
add_executable(vmm_test tests/vmm_test.cpp vmm.cpp)
target_include_directories(vmm_test PRIVATE ${CMAKE_SOURCE_DIR})
add_executable(chunks_test tests/chunks_test.cpp chunks.cpp)
target_include_directories(chunks_test PRIVATE ${CMAKE_SOURCE_DIR})

# A request's image rows as store numbers: geometry, runs, the flights' ledger against the allocator, the reservation (host only)
add_executable(img_rows_test tests/img_rows_test.cpp chunks.cpp)
target_include_directories(img_rows_test PRIVATE ${CMAKE_SOURCE_DIR})

# The lent groups' ledger, the hand-over proof, the cache's sizes and resize plan (lend.h) over the real allocator and the VMM
# pool on a fake driver; host only, no GPU. Built with -DPICO_OLD_RULES its red cases fail
add_executable(lend_test tests/lend_test.cpp chunks.cpp vmm.cpp)
target_include_directories(lend_test PRIVATE ${CMAKE_SOURCE_DIR})

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
# Fault injection for the Engram hand-over tests (model/lend.cu): DSGB10_LEND_FAIL=rank:R,step:B|C|back
option(DSGB10_TEST_FAULTS "DSGB10_LEND_FAIL fault injection in the Engram hand-over" OFF)
if(DSGB10_TEST_FAULTS)
  target_compile_definitions(dsmodel PUBLIC DSGB10_TEST_FAULTS)
endif()

option(DSGB10_NVTX "NVTX ranges in the model and the server, for nsys" OFF)
if(DSGB10_NVTX)
  target_compile_definitions(dsmodel PUBLIC DSGB10_NVTX)
endif()

# The windowed token history (tokwin.h): Engram hashes from a trimmed window against the full vector, and memory over
# 32 units of 2M-token prompts; host only
add_executable(tokwin_test tests/tokwin_test.cpp)
target_include_directories(tokwin_test PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(tokwin_test PRIVATE engram)
