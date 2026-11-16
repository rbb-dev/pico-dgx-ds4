// This Spark's health for the dashboard (docs/dashboard.md, "The node sampler"): a thread asleep in accept() until rank 0's
// front-end sends "GET\n", then one JSON line from /proc, /sys, NVML, the NVMe's SMART page and the rails' counters.
// It never touches the engine's threads, a step or a CUDA call (NVML allocates no GPU memory and launches nothing) and runs on the small cores.
#pragma once
#include <cstdint>
#include <string>

namespace node_stats {

struct Memory {  // this rank's device memory at the start of serving: the dashboard's weights / work / KV bar
  uint64_t device_bytes = 0, weights_bytes = 0, work_bytes = 0, kv_bytes = 0;
};

// Listens on addr:port (port 0: no sampler) and starts the thread; rails: the two RDMA devices whose counters the page
// shows (the fleet's NEXT_DEV0 and NEXT_DEV1). A port it cannot bind is reported and left alone: serving goes on.
void start(int rank, const std::string& addr, int port, const Memory& mem, const std::string rails[2]);

}  // namespace node_stats
