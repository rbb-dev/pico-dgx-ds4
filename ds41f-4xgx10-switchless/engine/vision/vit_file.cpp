#include "vision/vit_file.h"

#include <algorithm>
#include <unordered_map>

namespace vit {

namespace {

std::string shape_str(const std::vector<uint64_t>& s) {
  std::string o = "[";
  for (size_t i = 0; i < s.size(); i++) o += (i ? "," : "") + std::to_string(s[i]);
  return o + "]";
}

}  // namespace

std::vector<FileTensor> expected_file(int blocks) {
  constexpr uint64_t hq = kDim / kRanks, mq = kInter / kRanks, aq = kLm / kRanks;
  std::vector<FileTensor> v;
  for (int i = 0; i < blocks; i++) {
    const std::string p = "vision.blocks." + std::to_string(i) + ".";
    v.push_back({p + "attn.wo.bias", {hq}, false});
    v.push_back({p + "attn.wo.weight", {hq, kDim}, false});
    v.push_back({p + "attn.wqkv.bias", {3 * hq}, false});
    v.push_back({p + "attn.wqkv.weight", {3 * hq, kDim}, false});
    v.push_back({p + "mlp.w1.weight", {2 * mq, kDim}, false});
    v.push_back({p + "mlp.w2.weight", {hq, kInter}, false});
    v.push_back({p + "norm1.weight", {kDim}, true});
    v.push_back({p + "norm2.weight", {kDim}, true});
  }
  v.push_back({"aligner.w1.bias", {aq}, false});
  v.push_back({"aligner.w1.weight", {aq, kMergeIn}, false});
  v.push_back({"aligner.w2.bias", {aq}, false});
  v.push_back({"aligner.w2.weight", {aq, kLm}, false});
  v.push_back({"image_end", {kLm}, true});
  v.push_back({"image_newline", {kLm}, true});
  v.push_back({"image_start", {kLm}, true});
  v.push_back({"vision.norm.weight", {kDim}, true});
  v.push_back({"vision.patch_embed.proj.bias", {kDim}, true});
  v.push_back({"vision.patch_embed.proj.weight", {kDim, kPatchK}, true});
  // the plan sorts a group of tensors by name (vision.blocks.10 before vision.blocks.2)
  std::sort(v.begin(), v.end(), [](const FileTensor& a, const FileTensor& b) { return a.name < b.name; });
  return v;
}

std::string check_vision_toc(const std::vector<dsw::DeviceTensor>& tensors, int blocks) {
#ifdef PICO_OLD_RULES
  (void)tensors, (void)blocks;
  return "";  // without the feature nothing is checked: any file is taken as the vision weights
#else
  const std::vector<FileTensor> want = expected_file(blocks);
  std::unordered_map<std::string, const dsw::DeviceTensor*> have;
  for (const auto& t : tensors)
    if (!have.emplace(t.name, &t).second) return "tensor " + t.name + " appears twice";
  for (const FileTensor& w : want) {
    auto it = have.find(w.name);
    if (it == have.end()) return "tensor " + w.name + " is missing";
    const dsw::DeviceTensor& t = *it->second;
    std::vector<uint64_t> shape;
    for (int d = 0; d < t.ndim; d++) shape.push_back((uint64_t)t.shape[d]);
    if (t.dtype != dsw::DType::BF16) return w.name + " is " + dsw::dtype_name(t.dtype) + ", wanted BF16";
    if (t.layout != dsw::Layout::Natural) return w.name + " is stored in layout " + dsw::layout_name(t.layout) + ", wanted natural";
    if (shape != w.shape) return w.name + " has shape " + shape_str(shape) + ", this rank's slice is " + shape_str(w.shape);
    uint64_t n = 2;
    for (uint64_t d : shape) n *= d;
    if (t.bytes != n) return w.name + " holds " + std::to_string(t.bytes) + " bytes, its shape needs " + std::to_string(n);
  }
  if (tensors.size() != want.size()) {
    for (const auto& t : tensors) {
      bool known = false;
      for (const FileTensor& w : want) known = known || w.name == t.name;
      if (!known) return "tensor " + t.name + " is not part of the vision weights";
    }
    return "the file holds " + std::to_string(tensors.size()) + " tensors, " + std::to_string(want.size()) + " expected";
  }
  return "";
#endif
}

namespace {
struct Fnv {
  uint64_t h = 1469598103934665603ull;
  void bytes(const void* p, size_t n) {
    const auto* b = static_cast<const uint8_t*>(p);
    for (size_t i = 0; i < n; i++) h = (h ^ b[i]) * 1099511628211ull;
  }
  template <class T>
  void val(T v) { bytes(&v, sizeof v); }
};
}  // namespace

uint64_t file_set_digest(const dsw::Header& hd, const std::vector<dsw::DeviceTensor>& tensors, int blocks) {
#ifdef PICO_OLD_RULES
  (void)hd, (void)tensors, (void)blocks;
  return 0;  // without the feature the four files are never compared
#else
  std::unordered_map<std::string, bool> whole;
  for (const FileTensor& w : expected_file(blocks)) whole[w.name] = w.replicated;
  Fnv f;
  f.val<uint32_t>(hd.world);
  f.val<uint64_t>(hd.file_bytes);
  f.val<uint32_t>((uint32_t)tensors.size());
  for (const auto& t : tensors) {
    f.bytes(t.name.data(), t.name.size());
    f.val<uint8_t>(0);
    f.val<uint32_t>((uint32_t)t.dtype);
    f.val<int32_t>(t.ndim);
    for (int d = 0; d < t.ndim; d++) f.val<int64_t>(t.shape[d]);
    f.val<uint64_t>(t.bytes);
    auto it = whole.find(t.name);
    if (it != whole.end() && it->second) f.val<uint32_t>(t.crc32c);
  }
  return f.h;
#endif
}

}  // namespace vit
