# The engine: C++20 + CUDA for GB10 (sm_121a) only.
# Each line of work owns one subdirectory with its own CMakeLists.txt; it is picked up when present.
cmake_minimum_required(VERSION 3.28)
if(NOT DEFINED CMAKE_CUDA_COMPILER)
  set(CMAKE_CUDA_COMPILER /usr/local/cuda-13.0/bin/nvcc)
endif()
# Block-scaled MMA needs the arch-specific target: compute_121a/sm_121a (see docs/findings.md#tensor-cores).
set(CMAKE_CUDA_ARCHITECTURES OFF)
project(dsgb10 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_121a,code=sm_121a --expt-relaxed-constexpr -lineinfo")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

foreach(dir common weights comm kernels attn vision dspark engram model server tests)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${dir}/CMakeLists.txt)
    add_subdirectory(${dir})
  endif()
endforeach()

# tools/alloc_audit.py against alloc_sites.txt: the tripwire for new allocation sites, and its own selftest (host only; needs
# python3). `alloc_audit` is the check itself: it fails the build target when the sources and the list differ.
add_executable(alloc_audit_test tools/alloc_audit_test.cpp)
target_compile_definitions(alloc_audit_test PRIVATE ENGINE_SRC_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\")
# tools/store_trace_check.py `same` on a real failed-give-back trace (a rank's rolled-back attempt is its own; a real difference fails)
add_executable(store_trace_check_test tools/store_trace_check_test.cpp)
target_compile_definitions(store_trace_check_test PRIVATE ENGINE_SRC_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\")
# Copy drafting replayed offline on recorded token streams (tools/copy_replay.cpp): the engine's own host matcher and gate
# functions (model/copy.h) on conversations' ids; host only, no GPU. With no arguments it runs its self-test.
add_executable(copy_replay tools/copy_replay.cpp)
target_include_directories(copy_replay PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
add_custom_target(alloc_audit COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tools/alloc_audit.py --src ${CMAKE_CURRENT_SOURCE_DIR}
                  COMMENT "alloc_audit: allocation call sites against alloc_sites.txt")

# This project's fleet file beside the binaries that read one (dsgb10_rank, ring_test, coll_tests), as
# fleet.conf, where they look by default; run4.sh and service.sh install their FLEET beside the copies
# they run, and --fleet FILE reads another.
foreach(dir server comm model)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../fleet/homelab.conf ${CMAKE_BINARY_DIR}/${dir}/fleet.conf COPYONLY)
endforeach()
