add_executable(dsgb10_rank main.cpp cmd.cpp serve.cpp node_stats.cpp)
target_link_libraries(dsgb10_rank PRIVATE dsmodel)

# interleave.h decisions against a table of cases (host only)
add_executable(interleave_test tests/interleave_test.cpp)

# reclaim.h: which lines give way, the live set (host only)
add_executable(reclaim_test tests/reclaim_test.cpp)

# cancel.h: a client that left, the queues swept for it (host only)
add_executable(cancel_test tests/cancel_test.cpp)

# model/pool_rules.h: the chunk pool's rules (host only)
add_executable(pool_rules_test tests/pool_rules_test.cpp)
