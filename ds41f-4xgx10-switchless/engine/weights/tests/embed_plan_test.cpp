// dsw_embed_plan_test: the plan cuts embed.weight and the Markov embedding by vocabulary rows (plan.cpp), four ranks, 32,320
// rows each, the same rows the head's slice holds; nothing of either is replicated and the cut covers each table exactly.
// Host only (a plan reads shapes, never bytes). Built with -DPICO_OLD_RULES the tables stay whole and the rows fail.
//   dsw_embed_plan_test          exit 0 when every check passes
#include <cstdio>
#include <string>

#include "weights/plan.h"

using namespace dsw;

namespace {

int fails = 0;
void expect(bool ok, const std::string& what) {
  std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
  if (!ok) ++fails;
}

SrcTensor src(const std::string& name, std::vector<uint64_t> shape) {
  SrcTensor t;
  t.name = name;
  t.shard = 0;
  t.dtype = "BF16";
  t.shape = shape;
  uint64_t n = 2;
  for (uint64_t d : shape) n *= d;
  t.bytes = n;
  return t;
}

const OutTensor* find(const Plan& p, const std::string& name) {
  for (const OutTensor& o : p.tensors)
    if (o.name == name) return &o;
  return nullptr;
}

}  // namespace

int main() {
  Checkpoint ck;
  ck.dir = "synthetic";
  ck.shards = {"synthetic"};
  ck.shard_bytes = {0};
  for (const SrcTensor& t : {src("embed.weight", {129280, 5120}), src("head.weight", {129280, 5120}),
                             src("norm.weight", {5120}), src("mtp.2.markov_head.embed.weight", {129280, 256}),
                             src("mtp.2.markov_head.head.weight", {129280, 256})})
    ck.tensors[t.name] = t;
  const std::string names[2] = {"embed.weight", "mtp.2.markov_head.embed.weight"};
  const uint64_t cols[2] = {5120, 256};
  for (int r = 0; r < 4; r++) {
    const Plan p = make_plan(ck, r, 4);
    expect(p.tensors.size() == 5, "rank " + std::to_string(r) + ": five tensors planned");
    for (int k = 0; k < 2; k++) {
      const OutTensor* o = find(p, names[k]);
      if (!o) {
        expect(false, names[k] + " planned");
        continue;
      }
      const std::string at = "rank " + std::to_string(r) + " " + names[k];
      expect(o->kind == SliceKind::Rows, at + " is cut by rows (" + o->slice + ")");
      expect(o->shape.size() == 2 && o->shape[0] == 32320 && o->shape[1] == cols[k], at + " has 32,320 rows");
      expect(o->bytes == 32320ull * cols[k] * 2, at + " holds " + std::to_string(o->bytes) + " bytes");
      expect(o->pieces.size() == 1 && o->pieces[0].row0 == 32320ull * r && o->pieces[0].nrows == 32320,
             at + " takes rows [32320 * rank, +32320): the head's rows");
      expect(o->layout == Layout::Natural, at + " stays natural");
    }
    const OutTensor* head = find(p, "head.weight");
    expect(head && head->pieces.size() == 1 && head->pieces[0].row0 == 32320ull * r, "the head's rows are the embedding's");
  }
  check_coverage(ck, 4);  // dies unless the four cuts cover each source tensor exactly, whole ones whole
  expect(true, "the four ranks' cuts cover every table exactly once");
  std::printf("%s\n", fails ? "FAILED" : "all ok");
  return fails ? 1 : 0;
}
