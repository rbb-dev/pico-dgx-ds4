add_executable(dsgb10_rank main.cpp cmd.cpp serve.cpp node_stats.cpp ../common/alloc_wrap.cpp)
target_link_libraries(dsgb10_rank PRIVATE dsmodel vit_device)
# The allocation seal (common/alloc_seal.h): a counting function in front of every call of tools/alloc_audit.py's list made by the
# engine's objects, on this target only (common/alloc_wrap.cpp)
foreach(sym cudaMalloc cudaMallocHost cudaHostAlloc cudaGraphInstantiate cuMemCreate cuMemAddressReserve mmap posix_memalign ibv_reg_mr)
  target_link_options(dsgb10_rank PRIVATE "LINKER:--wrap=${sym}")
endforeach()

# common/alloc_seal.h: before and after the sizing read, the scopes, outside calls printed once a site (host only)
add_executable(alloc_seal_test tests/alloc_seal_test.cpp)
target_include_directories(alloc_seal_test PRIVATE ${CMAKE_SOURCE_DIR})

# interleave.h decisions against a table of cases (host only)
add_executable(interleave_test tests/interleave_test.cpp)

# split_pace.h: when a split command is due and what it leaves on the pending list (host only)
add_executable(split_pace_test tests/split_pace_test.cpp)

# turns.h: which waiting request leads the next run (host only)
add_executable(turns_test tests/turns_test.cpp)

# reclaim.h: which lines give way, the window plan's actions, the window close, the live set (host only)
add_executable(reclaim_test tests/reclaim_test.cpp)

# cancel.h: a client that left, the queues swept for it (host only)
add_executable(cancel_test tests/cancel_test.cpp)

# the admission arithmetic with and without a finished request's reservation (host only)
add_executable(resv_test tests/resv_test.cpp)

# the admission count for a request whose conversation is partly in the disk tier, and the read-back's budget (host only)
add_executable(admit_test tests/admit_test.cpp)
target_include_directories(admit_test PRIVATE ${CMAKE_SOURCE_DIR})

# model/pool_rules.h: the chunk pool's rules (host only)
add_executable(pool_rules_test tests/pool_rules_test.cpp)

# bounds.h: the output queue's cap and the prompts' byte budget (host only)
add_executable(bounds_test tests/bounds_test.cpp)

# served.h: the flags and DSGB10_* settings a `--mode serve` start takes by default, against the served list (host only)
add_executable(served_test tests/served_test.cpp)
target_compile_definitions(served_test PRIVATE SERVED_ENGINE_DIR="${CMAKE_SOURCE_DIR}" SERVED_MAIN_CPP="${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")

# common/heap.h: giving the free pages of the heap back (host only)
add_executable(heap_test tests/heap_test.cpp)

# img_hold.h: a flight's image rows in the store, taken once, held across resumes, freed at every end (host only)
add_executable(img_hold_test tests/img_hold_test.cpp ../model/chunks.cpp)
target_include_directories(img_hold_test PRIVATE ${CMAKE_SOURCE_DIR})

# ids_file.h: the server's token ids past their cap go to files and come back (host only)
add_executable(ids_file_test tests/ids_file_test.cpp)

# common/byte_account.h: the byte account at ready: meminfo and status texts, the remainder, its bound (host only)
add_executable(byte_account_test tests/byte_account_test.cpp)
target_include_directories(byte_account_test PRIVATE ${CMAKE_SOURCE_DIR})

# What publishing STATS from the decode loops costs on the host (lend_pace.h's publish_due and the frame's host part); prints ns
add_executable(publish_bench tests/publish_bench.cpp)
target_include_directories(publish_bench PRIVATE ${CMAKE_SOURCE_DIR})

# mem_watch.h: the reservation's watch: when a reading is a new low worth a line (host only)
add_executable(mem_watch_test tests/mem_watch_test.cpp)

# rows_loan.h: when the lent prefill row set is asked for and given back (host only)
add_executable(rows_loan_test tests/rows_loan_test.cpp)

# lend_pace.h: what pauses a decoding run (the memory watch does not: it rides the run) (host only)
add_executable(lend_pace_test tests/lend_pace_test.cpp)

# prefix.h: a request that carries only its suffix: the prompt built from the server's history, the epoch rule, the kept index's record (host only)
add_executable(prefix_test tests/prefix_test.cpp)

# pixels.h: a SUBMIT carrying pixels: the layout, the two kinds of wrong frame (host only)
add_executable(pixels_test tests/pixels_test.cpp)

# vit_sched.h: what an image costs on the four ranks, the next unit of a pixel request (encode piece or rows), the one-image mark (host only)
add_executable(vit_sched_test tests/vit_sched_test.cpp)

# ride_wire.h: the imager's ride on the wire: one decision on four ranks, readings that must be equal (host only)
add_executable(ride_wire_test tests/ride_wire_test.cpp)
