add_executable(dsgb10_rank main.cpp cmd.cpp serve.cpp node_stats.cpp)
target_link_libraries(dsgb10_rank PRIVATE dsmodel vit_device)

# interleave.h decisions against a table of cases (host only)
add_executable(interleave_test tests/interleave_test.cpp)

# turns.h: which waiting request leads the next run (host only)
add_executable(turns_test tests/turns_test.cpp)

# reclaim.h: which lines give way, the window plan's actions, the window close, the live set (host only)
add_executable(reclaim_test tests/reclaim_test.cpp)

# cancel.h: a client that left, the queues swept for it (host only)
add_executable(cancel_test tests/cancel_test.cpp)

# the admission arithmetic with and without a finished request's reservation (host only)
add_executable(resv_test tests/resv_test.cpp)

# model/pool_rules.h: the chunk pool's rules (host only)
add_executable(pool_rules_test tests/pool_rules_test.cpp)

# bounds.h: the output queue's cap and the prompts' byte budget (host only)
add_executable(bounds_test tests/bounds_test.cpp)

# common/heap.h: giving the free pages of the heap back (host only)
add_executable(heap_test tests/heap_test.cpp)

# img_hold.h: a flight's image rows in the store, taken once, held across resumes, freed at every end (host only)
add_executable(img_hold_test tests/img_hold_test.cpp ../model/chunks.cpp)
target_include_directories(img_hold_test PRIVATE ${CMAKE_SOURCE_DIR})
