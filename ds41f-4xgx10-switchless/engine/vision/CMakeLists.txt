# The image encoder (docs/vision.md): host side (weights, slices), kernels, the unsplit and four-rank runs.
find_package(CUDAToolkit REQUIRED)

add_library(vit_host STATIC vit_weights.cpp vit_file.cpp)
target_include_directories(vit_host PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(vit_host PUBLIC dsw_host dsw_loader)

# this rank's vision weight file on the device (O_DIRECT into one arena, the file's table of contents checked)
add_library(vit_device STATIC vit_device.cpp)
target_include_directories(vit_device PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(vit_device PUBLIC vit_host CUDA::cudart_static)

# host only: the vision weight files (dsw_convert --vision) against the checkpoint, the file check and the digest
add_executable(vision_dsw_roundtrip tests/vision_dsw_roundtrip.cpp)
add_dependencies(vision_dsw_roundtrip dsw_convert)
target_compile_definitions(vision_dsw_roundtrip PRIVATE DSW_CONVERT_PATH="$<TARGET_FILE:dsw_convert>")
target_link_libraries(vision_dsw_roundtrip PRIVATE vit_host)

add_library(vit_kernels STATIC vit_kernels.cu vit_tower.cu vit_rank.cu)
target_include_directories(vit_kernels PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(vit_kernels PUBLIC vit_host CUDA::cudart_static)

# host only: slices, grids, the host references of the copy kernels, a checkpoint round trip
add_executable(vit_host_test tests/vit_host_test.cpp)
target_compile_options(vit_host_test PRIVATE -ffp-contract=off)
target_link_libraries(vit_host_test PRIVATE vit_host)

# GPU: every kernel and the tower, split against unsplit in bytes; the deliberately wrong builds must differ
add_executable(vit_test tests/vit_test.cu)
target_compile_options(vit_test PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off> $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffp-contract=off>)
target_link_libraries(vit_test PRIVATE vit_kernels)

# GPU: time of every kernel at the encoder's shapes, a rank's share of a block, the unsplit tower
add_executable(bench_vit tests/bench_vit.cu)
target_link_libraries(bench_vit PRIVATE vit_kernels)

# host only: the .npy reader and writer of the exactness harness, against bytes numpy wrote
add_executable(vit_npy_test tests/vit_npy_test.cpp)
target_include_directories(vit_npy_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR})

# GPU: each vit:: operation on the reference's own tensors, the unsplit tower free-running with taps, the copy-type
# weights, vit::gemm on random data: what tools/vision/vit_compare.py judges
add_executable(vit_teacher tests/vit_teacher.cu)
target_include_directories(vit_teacher PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(vit_teacher PRIVATE vit_kernels)

# four Sparks: a rank's own run of the split tower over the ring (start it with engine/comm/run4-vit.sh)
add_executable(vit_ring_test tests/vit_ring_test.cu)
target_link_libraries(vit_ring_test PRIVATE vit_kernels vit_device dsgb10_comm)

# host only: vit_ring_test's chunked host all-gather against a fake that refuses what the ring's refuses
add_executable(vit_gather_test tests/vit_gather_test.cpp)
