# Engram: hashing, the NVMe row store (io_uring, raw syscalls) and its tools. Host only.
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

add_library(engram STATIC hash.cpp store.cpp display.cpp)
# Fault injection for the identity and hand-over tests (DSGB10_ENGRAM_BREAK here, DSGB10_LEND_FAIL in model/lend.cu)
option(DSGB10_TEST_FAULTS "fault injection for the Engram cache tests" OFF)
if(DSGB10_TEST_FAULTS)
  target_compile_definitions(engram PUBLIC DSGB10_TEST_FAULTS)
endif()
target_include_directories(engram PUBLIC ${CMAKE_SOURCE_DIR} ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(engram PUBLIC dsw_host Threads::Threads)

# the row cache split over the display reservation and ordinary memory against an all-ordinary one, and the reservation's
# fallbacks (tools/rowcache_test.cpp; host only, uses /dev/dri/card0 when it opens)
add_executable(engram_rowcache_test tools/rowcache_test.cpp)
target_link_libraries(engram_rowcache_test PRIVATE engram)

# the growable block under the row cache (tools/block_test.cpp; host only)
add_executable(engram_block_test tools/block_test.cpp)
target_link_libraries(engram_block_test PRIVATE engram)

foreach(tool hashcheck iobench rowcheck storecheck gathercheck cachesim e2e pack)
  add_executable(engram_${tool} tools/${tool}.cpp)
  target_link_libraries(engram_${tool} PRIVATE engram)
endforeach()
