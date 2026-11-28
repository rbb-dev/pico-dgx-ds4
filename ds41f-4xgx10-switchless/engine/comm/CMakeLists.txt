find_package(CUDAToolkit REQUIRED)

add_library(dsgb10_comm STATIC ring.cpp fleet.cpp collectives.cu)
target_link_libraries(dsgb10_comm PUBLIC ibverbs CUDA::cuda_driver pthread)

# ring_test's copy, with the test hooks (collectives.cu: a stalled block for the race test); the engine links
# dsgb10_comm, which has none
add_library(dsgb10_comm_test STATIC ring.cpp fleet.cpp collectives.cu)
target_compile_definitions(dsgb10_comm_test PUBLIC DSGB10_COMM_TEST_HOOKS)
target_link_libraries(dsgb10_comm_test PUBLIC ibverbs CUDA::cuda_driver pthread)

add_executable(ring_test tests/ring_test.cu)
target_link_libraries(ring_test PRIVATE dsgb10_comm_test)

# argmax_rows_row's block layout, winner rule and carried row against the whole table (host only: tests/argmax_row_test.cpp);
# built with -DPICO_OLD_RULES ties go the other way and it fails
add_executable(argmax_row_test tests/argmax_row_test.cpp)
target_include_directories(argmax_row_test PRIVATE ${CMAKE_SOURCE_DIR})

# comm/host_gather.h on four threads over socket pairs (host only): the framing of the host all-gather (call id and size on every hop)
add_executable(host_gather_test tests/host_gather_test.cpp)
target_include_directories(host_gather_test PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(host_gather_test PRIVATE pthread)

# comm/gid_pick.h (host only): which GID table entry a link uses for its fleet address; its control, the rule before
# (the fleet's GID_INDEX or nothing, any type), must fail
add_executable(gid_pick_test tests/gid_pick_test.cpp)
target_include_directories(gid_pick_test PRIVATE ${CMAKE_SOURCE_DIR})
