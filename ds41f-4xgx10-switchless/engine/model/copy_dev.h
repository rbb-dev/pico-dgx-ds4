// Copy drafting on the device (copy.h is the matcher; copy.cu the memory and the kernels that fill it).
#pragma once
#include <cuda_runtime.h>

#include <vector>

#include "model/copy.h"
#include "model/copy_run.h"
#include "model/ops.h"

namespace model::copy {

static_assert(kMaxDrafts == ops::kCopyDrafts, "ops.h's kCopyDrafts is copy.h's");
static_assert(kMaxUnitsCopy == 128, "config.h's kMaxUnits");

// Device memory, one: every unit's view, the gate's table, the running digest. A kernel takes a unit by its index.
struct Dev {
  Unit unit[kMaxUnitsCopy];  // cap 0: the unit has no copy drafting
  Table table;               // T[bucket][row]
  uint32_t digest;           // the fold of what every step chose, in step order
  uint32_t pad;
  Learn learn;               // the chances that follow the traffic (copy.h; read and written only with kAdapt)
  UnitLearn ul[kMaxUnitsCopy];  // each unit's conversation counts and its last proposal
  Run run[kMaxUnitsCopy];       // each unit's cursor and ladder rung (copy_run.h; read and written only with DSGB10_COPY_ROWS above 8)
};

// What a spec step's gate kernels take: null dev = copy drafting off (the kernels are then as without it).
struct Args {
  Dev* dev = nullptr;
  const int* row0 = nullptr;  // batch steps: entry k's first row of the step (spec.cu's sp.row0)
};

// Test hooks (tests/copy_gate_test.cu): the two gate kernels of spec.cu launched on their own, with the arguments a spec step gives them
// (k_spec_next; k_spec_gate_batch_all with no calibration, dump or trims past the row budget).
namespace test {
void gate_one(const ops::StepIn* in, ops::SpecState* ss, const int* drafts, const float* surv, ops::SampleOut* outs, Args cp, cudaStream_t s);
void gate_batch(const ops::StepIn* in, ops::SpecState* ss, const int* tok0, int tok_stride, const float* surv0, int surv_stride,
                ops::SampleOut* outs, Args cp, cudaStream_t s);
}  // namespace test

class Index {
 public:
  // ctx[u]: positions unit u covers (0: unused). The arena is cut into mirrors and tables by plan_units.
  Index(const int* ctx, int nunits, size_t arena_bytes, cudaStream_t s);
  ~Index();
  Index(const Index&) = delete;
  Index& operator=(const Index&) = delete;
  Dev* dev() const { return dev_; }
  size_t bytes() const { return used_; }  // of the arena actually cut
  const std::vector<Plan>& plan() const { return plan_; }
  // The unit's history from pos0 on is these n tokens (host memory): written to the mirror and indexed. A rewrite below
  // the history's length (a new request on the unit) makes the table again from the kept tokens; positions past the
  // mirror's cap are ignored. Ordered on s, so after the steps already queued.
  void set_tokens(int unit, int pos0, int n, const int32_t* tokens, cudaStream_t s);
  // Test hooks: the unit's view and table copied to the host, and the matcher run on the device.
  Unit view(int unit) const;
  std::vector<uint32_t> table_copy(int unit) const;
  std::vector<int32_t> tokens_copy(int unit, int n) const;
  // propose() on the device for each query: out[q * kMaxDrafts ..], counts[q], matches[q] = {end, len, bucket}
  struct Query {
    int unit, P, k, maxm, minlen;
    int32_t d[8];
  };
  void propose_on_device(const std::vector<Query>& q, std::vector<int32_t>& out, std::vector<int32_t>& counts,
                         std::vector<int32_t>& matches, cudaStream_t s);
  // append_step() on the device (the decode step's append), one thread
  void append_on_device(int unit, int pos0, int count, const int32_t* rows, int32_t bonus, cudaStream_t s);
  // The unit now holds conversation `handle` (Model::slot_attach, in command order on every rank): its counts and its last
  // proposal are cleared when the handle differs from the one it held. Without kAdapt nothing is queued. Ordered on s.
  void attach(int unit, int handle, cudaStream_t s);
  // Test hook: the learner's state and the units' counts copied to the host.
  void learn_copy(Learn* L, std::vector<UnitLearn>* units) const;

 private:
  Dev* dev_ = nullptr;
  uint8_t* arena_ = nullptr;
  size_t used_ = 0;
  std::vector<Plan> plan_;
  std::vector<Unit> host_;
  std::vector<int> attached_;  // the conversation each unit last held (-1: none)
  bool adapt_ = false;
  int32_t* flag_ = nullptr;  // device scratch: set_tokens' rewrite flag
  int32_t* stage_ = nullptr;
  size_t stage_cap_ = 0;
};

}  // namespace model::copy
