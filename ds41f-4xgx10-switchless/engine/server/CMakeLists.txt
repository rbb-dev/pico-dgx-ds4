add_executable(dsgb10_rank main.cpp cmd.cpp serve.cpp node_stats.cpp)
target_link_libraries(dsgb10_rank PRIVATE dsmodel)
