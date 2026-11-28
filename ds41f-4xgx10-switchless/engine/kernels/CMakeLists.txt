add_library(dsk_kernels STATIC
  act_quant.cu
  gemv_fp8.cu
  moe.cu
  gemv16.cu
  gemv_fp4.cu
  l2pf.cu
  norm_router.cu
  prefill.cu
  prefill_attn.cu
  prefill_index.cu
  prefill_comp.cu
  prefill_mhc.cu
)
target_include_directories(dsk_kernels PUBLIC ${CMAKE_SOURCE_DIR})

add_executable(test_kernels tests/test_kernels.cu)
target_link_libraries(test_kernels dsk_kernels)

add_executable(bench_kernels tests/bench_kernels.cu)
target_link_libraries(bench_kernels dsk_kernels)

# what feeds the expert kernels' tensor cores (memory, MMA or unpack: arms of the up and down kernels beside the unchanged ones);
# it includes prefill.cu itself, so it links nothing of ours; built on request only
add_executable(bench_feed EXCLUDE_FROM_ALL tests/bench_feed.cu)

# the prefill attention side (M5, part 2), checked against engine/attn's kernels and reference cases
add_executable(test_pattn tests/test_pattn.cu)
target_link_libraries(test_pattn dsk_kernels dsattn)
add_executable(bench_pattn tests/bench_pattn.cu)
target_link_libraries(bench_pattn dsk_kernels dsattn)
add_executable(bench_chain tests/bench_chain.cu)
target_link_libraries(bench_chain dsk_kernels dsattn)

# lossless weight compression benches (docs/engine.md; stack5.md): bench only
add_executable(bench_wcomp tests/bench_wcomp.cu)
target_link_libraries(bench_wcomp dsk_kernels)

# every YaRN-rotating kernel: the table and the computed rotation give the same bits (tests/rope_kernels_test.cu)
add_executable(rope_kernels_test tests/rope_kernels_test.cu)
target_include_directories(rope_kernels_test PRIVATE ${CMAKE_SOURCE_DIR}/attn/tests)
target_link_libraries(rope_kernels_test dsk_kernels dsattn)

# the L2 prefetch's plan on known inputs (l2pf_plan, l2pf_draft_plan); host only, no GPU
add_executable(l2pf_plan_test tests/l2pf_plan_test.cu)
target_link_libraries(l2pf_plan_test dsk_kernels)
