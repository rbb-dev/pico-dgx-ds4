// Bounds on what the server queues in memory (no engine state; tested by tests/bounds_test.cpp).
//
// A connection's output queue holds the frames waiting for the writer; one that stops reading never drains it, so the
// queue has a byte cap and a connection that passes it is dropped.
//
// PICO_OLD_RULES (never defined by the engine) leaves the queue unbounded, so the test can show it fails on it.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace bounds {

constexpr size_t kOutCap = 64ull << 20;  // bytes queued for one connection's writer

// The bytes queued for one connection's writer.
class OutQueue {
 public:
  // A frame of n bytes is queued: false when the queue would pass the cap (the connection is dropped; nothing was added).
  bool add(size_t n) {
#ifndef PICO_OLD_RULES
    if (queued_.load() + n > kOutCap) return false;
#endif
    queued_ += n;
    return true;
  }
  // The writer has written or dropped a frame of n bytes.
  void done(size_t n) { queued_ -= n; }
  size_t queued() const { return queued_.load(); }

 private:
  std::atomic<size_t> queued_{0};
};

}  // namespace bounds
