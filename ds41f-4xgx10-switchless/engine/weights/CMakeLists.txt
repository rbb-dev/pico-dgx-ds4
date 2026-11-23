# Per-rank weight files: partition plan, converter (tiles on the GPU with engine/kernels' own
# functions), verifier (host only), the device loader and the kernel cross-check.
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

add_library(dsw_host STATIC safetensors.cpp plan.cpp)
target_include_directories(dsw_host PUBLIC ${CMAKE_SOURCE_DIR} ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(dsw_host PUBLIC Threads::Threads)

add_executable(dsw_convert convert.cpp)
target_link_libraries(dsw_convert PRIVATE dsw_host dsk_kernels CUDA::cudart_static)

add_executable(dsw_verify verify.cpp)
target_link_libraries(dsw_verify PRIVATE dsw_host)

add_library(dsw_loader STATIC loader.cpp)
target_link_libraries(dsw_loader PUBLIC dsw_host CUDA::cudart_static)

add_executable(dsw_load load_main.cpp)
target_link_libraries(dsw_load PRIVATE dsw_loader)

add_executable(dsw_kcheck kcheck.cu)
target_link_libraries(dsw_kcheck PRIVATE dsw_loader dsk_kernels)

# the loader refuses another rank's file (host only: tests/rank_test.cpp)
add_executable(dsw_rank_test tests/rank_test.cpp)
target_link_libraries(dsw_rank_test PRIVATE dsw_loader)

# LoadOptions::dest: a tensor read straight into a caller's device buffer (needs one GPU: tests/dest_test.cpp)
add_executable(dsw_dest_test tests/dest_test.cpp)
target_link_libraries(dsw_dest_test PRIVATE dsw_loader)

# what the loader holds on the host while it loads and after: staging sized to the reads, all freed (needs one GPU: tests/stage_test.cpp)
add_executable(dsw_stage_test tests/stage_test.cpp)
target_link_libraries(dsw_stage_test PRIVATE dsw_loader)

# the plan cuts embed.weight and the Markov embedding by vocabulary rows, four ranks (host only: tests/embed_plan_test.cpp;
# built with -DPICO_OLD_RULES the tables stay whole and it fails)
add_executable(dsw_embed_plan_test tests/embed_plan_test.cpp)
target_link_libraries(dsw_embed_plan_test PRIVATE dsw_host)
