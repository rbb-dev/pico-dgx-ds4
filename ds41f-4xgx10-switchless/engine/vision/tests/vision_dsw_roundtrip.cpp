// vision_dsw_roundtrip: the vision weight files (dsw_convert --vision, docs/vision.md) against the checkpoint they were
// cut from. Host only: dsw_convert's vision path takes no GPU. Built with -DPICO_OLD_RULES (no vision plan, no check of a
// file's contents, no digest) the cases marked "red" fail.
//
//   vision_dsw_roundtrip [--blocks N] [--dir DIR] [--convert PATH] [--keep]
//
// Writes a synthetic checkpoint of N blocks (default 1; 32 is the real model's 266 tensors) in two shards, with the
// image delimiters in the second shard as in the real checkpoint and a text tensor beside them, under DIR/vdr.<pid>
// (default .; on a file system that does the O_DIRECT reads of the converter, not tmpfs). Runs dsw_convert --vision
// for ranks 0-3 on it, then:
//   - the four files, put together by the slice kind each TOC entry declares (replicated: equal on every rank; rows:
//     the ranks' rows one after another; rows-seg: segment by segment), equal the checkpoint's tensors byte for byte:
//     the patch embedding whole at K = 588 and every 1-D bias slice included, no text tensor in a file;
//   - each file equals vit::slice_rank of the checkpoint's weights (the cut the kernels and their tests use);
//   - the TOC of each file passes check_vision_toc, and a TOC with a tensor missing, extra, twice, of another shape, type
//     or layout does not; the digest of the four files is one value and a changed shape, a changed replicated CRC or a
//     changed size gives another; a file cut for another rank is refused by the loader.
// --keep leaves DIR/vdr.<pid> (checkpoint with its index, the four files) for dsw_verify --vision and check_dsw.py.
// Exit 0 when every check passes.
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "vision/vit_file.h"
#include "vision/vit_weights.h"
#include "weights/io.h"
#include "weights/loader.h"
#include "weights/plan.h"
#include "weights/safetensors.h"

#ifndef DSW_CONVERT_PATH
#define DSW_CONVERT_PATH "dsw_convert"
#endif

using namespace vit;

namespace {

int fails = 0;
void expect(bool ok, const std::string& what) {
  std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
  if (!ok) ++fails;
}

struct Src {
  std::string name;
  const std::vector<u16>* v;
  std::vector<uint64_t> shape;
};

std::vector<Src> sources(const Weights& w) {
  std::vector<Src> ts;
  for (size_t i = 0; i < w.blocks.size(); i++) {
    const std::string p = "vision.blocks." + std::to_string(i) + ".";
    const BlockW& b = w.blocks[i];
    ts.push_back({p + "attn.wqkv.weight", &b.wqkv, {3072, 1024}}), ts.push_back({p + "attn.wqkv.bias", &b.bqkv, {3072}});
    ts.push_back({p + "attn.wo.weight", &b.wo, {1024, 1024}}), ts.push_back({p + "attn.wo.bias", &b.bo, {1024}});
    ts.push_back({p + "mlp.w1.weight", &b.w1, {5632, 1024}}), ts.push_back({p + "mlp.w2.weight", &b.w2, {1024, 2816}});
    ts.push_back({p + "norm1.weight", &b.norm1, {1024}}), ts.push_back({p + "norm2.weight", &b.norm2, {1024}});
  }
  ts.push_back({"vision.patch_embed.proj.weight", &w.patch_w, {1024, 588}});
  ts.push_back({"vision.patch_embed.proj.bias", &w.patch_b, {1024}});
  ts.push_back({"vision.norm.weight", &w.norm, {1024}});
  ts.push_back({"aligner.w1.weight", &w.a1w, {5120, 9216}}), ts.push_back({"aligner.w1.bias", &w.a1b, {5120}});
  ts.push_back({"aligner.w2.weight", &w.a2w, {5120, 5120}}), ts.push_back({"aligner.w2.bias", &w.a2b, {5120}});
  ts.push_back({"image_start", &w.start, {5120}}), ts.push_back({"image_newline", &w.newline, {5120}});
  ts.push_back({"image_end", &w.end, {5120}});
  return ts;
}

void write_all(FILE* f, const void* p, size_t n) {
  if (std::fwrite(p, 1, n, f) != n) die("write: %s", std::strerror(errno));
}

// One shard holding `ts` (BF16).
void write_shard(const std::string& path, const std::vector<Src>& ts) {
  std::string hdr = "{";
  uint64_t off = 0;
  for (size_t i = 0; i < ts.size(); i++) {
    std::string shape = "[";
    for (size_t j = 0; j < ts[i].shape.size(); j++) shape += (j ? "," : "") + std::to_string(ts[i].shape[j]);
    shape += "]";
    const uint64_t nb = ts[i].v->size() * 2;
    hdr += (i ? "," : "") + std::string("\"") + ts[i].name + "\":{\"dtype\":\"BF16\",\"shape\":" + shape +
           ",\"data_offsets\":[" + std::to_string(off) + "," + std::to_string(off + nb) + "]}";
    off += nb;
  }
  hdr += "}";
  while (hdr.size() % 8) hdr += ' ';
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) die("open %s", path.c_str());
  const uint64_t hl = hdr.size();
  write_all(f, &hl, 8);
  write_all(f, hdr.data(), hdr.size());
  for (auto& t : ts) write_all(f, t.v->data(), t.v->size() * 2);
  std::fflush(f);
  ::fsync(fileno(f));
  std::fclose(f);
}

struct RankFile {
  dsw::Header h;
  std::vector<dsw::TocEntry> toc;
  std::string path;
  FILE* f = nullptr;
  std::vector<uint8_t> read(const dsw::TocEntry& e) const {
    std::vector<uint8_t> b(e.bytes);
    if (std::fseek(f, (long)e.offset, SEEK_SET) != 0 || std::fread(b.data(), 1, b.size(), f) != b.size()) die("read %s", e.name);
    return b;
  }
  const dsw::TocEntry* find(const std::string& n) const {
    for (const auto& e : toc)
      if (n == e.name) return &e;
    return nullptr;
  }
};

RankFile open_rank(const std::string& path) {
  RankFile r;
  r.path = path;
  r.f = std::fopen(path.c_str(), "rb");
  if (!r.f) die("open %s", path.c_str());
  if (std::fread(&r.h, sizeof r.h, 1, r.f) != 1) die("read header %s", path.c_str());
  r.toc.resize(r.h.n_tensors);
  std::fseek(r.f, (long)r.h.toc_offset, SEEK_SET);
  if (r.h.n_tensors && std::fread(r.toc.data(), sizeof(dsw::TocEntry), r.toc.size(), r.f) != r.toc.size()) die("read toc");
  return r;
}

std::vector<uint8_t> bytes_of(const std::vector<u16>& v) {
  const auto* p = reinterpret_cast<const uint8_t*>(v.data());
  return std::vector<uint8_t>(p, p + v.size() * 2);
}

bool zero_padding(const RankFile& r, const dsw::TocEntry& e) {
  const uint64_t pad = dsw::align_up(e.bytes) - e.bytes;
  std::vector<uint8_t> b(pad);
  std::fseek(r.f, (long)(e.offset + e.bytes), SEEK_SET);
  if (pad && std::fread(b.data(), 1, pad, r.f) != pad) return false;
  for (uint8_t x : b)
    if (x) return false;
  return true;
}

// the loader refuses a file cut for another rank (it aborts)
bool refused_as(const std::string& path, int rank) {
  std::fflush(stdout);
  const pid_t c = fork();
  if (c < 0) die("fork");
  if (c == 0) {
    std::freopen("/dev/null", "w", stderr);
    dsw::WeightFile wf(path, rank, 4);
    std::_Exit(0);
  }
  int st = 0;
  waitpid(c, &st, 0);
  return WIFSIGNALED(st) && WTERMSIG(st) == SIGABRT;
}

dsw::DeviceTensor dt(const std::string& name, std::vector<int64_t> shape, uint32_t crc = 7) {
  dsw::DeviceTensor t;
  t.name = name, t.dtype = dsw::DType::BF16, t.ndim = (int)shape.size();
  uint64_t n = 2;
  for (size_t i = 0; i < shape.size(); i++) t.shape[i] = shape[i], n *= (uint64_t)shape[i];
  t.bytes = n, t.crc32c = crc;
  return t;
}

// a table of contents as a correct file has it, from the expected list
std::vector<dsw::DeviceTensor> toc_for(int blocks) {
  std::vector<dsw::DeviceTensor> v;
  for (const FileTensor& f : expected_file(blocks)) {
    std::vector<int64_t> s(f.shape.begin(), f.shape.end());
    v.push_back(dt(f.name, s, f.replicated ? 1234u : 99u));
  }
  return v;
}

void test_toc_and_digest(int blocks) {
  const std::vector<dsw::DeviceTensor> good = toc_for(blocks);
  expect(check_vision_toc(good, blocks).empty(), "a table of contents of the expected tensors passes: " + check_vision_toc(good, blocks));
  expect(good.size() == (size_t)(8 * blocks + 10), "the expected file holds 8 per block and 10 more (266 for 32 blocks)");
  auto refused = [&](const char* what, std::vector<dsw::DeviceTensor> t) {
    const std::string why = check_vision_toc(t, blocks);
    expect(!why.empty(), std::string("refused: ") + what + (why.empty() ? "" : " (" + why + ")"));
  };
  {
    auto t = good;
    t.erase(t.begin() + 3);
    refused("a tensor missing", t);
  }
  {
    auto t = good;
    t.push_back(dt("layers.0.attn_norm.weight", {1024}));
    refused("a text tensor beside them", t);
  }
  {
    auto t = good;
    t.push_back(t[0]);
    refused("a tensor twice", t);
  }
  {
    auto t = good;
    for (auto& x : t)
      if (x.name == "aligner.w1.weight") x.shape[0] = 5120, x.bytes = 5120ull * 9216 * 2;
    refused("the whole aligner matrix where a quarter belongs (a file not cut)", t);
  }
  {
    auto t = good;
    for (auto& x : t)
      if (x.name == "vision.patch_embed.proj.weight") x.shape[1] = 640, x.bytes = 1024ull * 640 * 2;
    refused("the patch embedding padded in the file", t);
  }
  {
    auto t = good;
    t[1].dtype = dsw::DType::F32;
    refused("another type", t);
  }
  {
    auto t = good;
    t[1].layout = dsw::Layout::L1;
    refused("a tiled layout", t);
  }
  {
    auto t = good;
    t[2].bytes += 2;
    refused("a size that does not fit the shape", t);
  }
  dsw::Header h{};
  h.world = 4, h.file_bytes = 123456;
  const uint64_t d0 = file_set_digest(h, good, blocks);
  expect(d0 == file_set_digest(h, good, blocks), "the digest is the same on every call");
  expect(d0 != 0, "the digest is not zero");
  {
    h.rank = 2;  // the rank is not in the digest
    expect(d0 == file_set_digest(h, good, blocks), "the digest does not depend on the rank");
    h.rank = 0;
  }
  {
    auto t = good;
    for (auto& x : t)
      if (x.name == "vision.norm.weight") x.crc32c ^= 1;
    expect(d0 != file_set_digest(h, t, blocks), "a changed CRC of a replicated tensor changes the digest");
  }
  {
    auto t = good;
    for (auto& x : t)
      if (x.name == "aligner.w2.weight") x.crc32c ^= 1;
    expect(d0 == file_set_digest(h, t, blocks), "the CRC of a rank's own slice is not in it (it differs per rank)");
  }
  {
    auto t = good;
    t[0].bytes += 2;
    expect(d0 != file_set_digest(h, t, blocks), "a changed size changes the digest");
    dsw::Header h2 = h;
    h2.file_bytes += kRanks * 4096;
    expect(d0 != file_set_digest(h2, good, blocks), "a changed file size changes the digest");
    h2 = h, h2.world = 8;
    expect(d0 != file_set_digest(h2, good, blocks), "a changed world changes the digest");
  }
}

}  // namespace

int main(int argc, char** argv) {
  int blocks = 1;
  std::string dir = ".", convert = DSW_CONVERT_PATH;
  bool keep = false;
  for (int i = 1; i < argc; i++) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) die("%s needs a value", a.c_str());
      return argv[++i];
    };
    if (a == "--blocks") blocks = std::atoi(next().c_str());
    else if (a == "--dir") dir = next();
    else if (a == "--convert") convert = next();
    else if (a == "--keep") keep = true;
    else die("usage: vision_dsw_roundtrip [--blocks N] [--dir DIR] [--convert PATH] [--keep]");
  }
  if (blocks < 1 || blocks > kLayers) die("--blocks 1..32");

  test_toc_and_digest(blocks);

  const std::string work = dir + "/vdr." + std::to_string(getpid());
  ::mkdir(work.c_str(), 0755);
  const std::string ck = work + "/ckpt";
  ::mkdir(ck.c_str(), 0755);
  const Weights w = make_synthetic(23, blocks);
  const std::vector<Src> all = sources(w);
  std::vector<Src> shard1, shard2;
  for (const Src& s : all) (s.name.rfind("image_", 0) == 0 ? shard2 : shard1).push_back(s);
  const std::vector<u16> decoy(16, to_bf16(1.0f));
  shard1.push_back({"layers.0.attn_norm.weight", &decoy, {16}});  // a text tensor: no vision file may hold it
  write_shard(ck + "/model-00001-of-00002.safetensors", shard1);
  write_shard(ck + "/model-00002-of-00002.safetensors", shard2);
  {  // the index, as the real checkpoint has one (the reader cross-checks it against the shards)
    std::string idx = "{\"metadata\":{},\"weight_map\":{";
    bool first = true;
    for (const auto* sh : {&shard1, &shard2})
      for (const Src& t : *sh) {
        idx += std::string(first ? "" : ",") + "\"" + t.name + "\":\"model-0000" + (sh == &shard1 ? "1" : "2") + "-of-00002.safetensors\"";
        first = false;
      }
    idx += "}}";
    FILE* f = std::fopen((ck + "/model.safetensors.index.json").c_str(), "wb");
    if (!f) die("open index");
    write_all(f, idx.data(), idx.size());
    std::fclose(f);
  }

  std::string files[4];
  for (int r = 0; r < 4; r++) {
    files[r] = work + "/vision.rank" + std::to_string(r) + ".dsw";
    const std::string cmd = convert + " --vision --ckpt " + ck + " --rank " + std::to_string(r) + " --out " + files[r] +
                            " --threads 2 --mem-gib 1 > " + work + "/convert" + std::to_string(r) + ".log 2>&1";
    const int rc = std::system(cmd.c_str());
    expect(rc == 0, "dsw_convert --vision rank " + std::to_string(r) + " (exit " + std::to_string(rc) + ")");
    if (rc != 0) {
      std::string tail = "tail -5 " + work + "/convert" + std::to_string(r) + ".log";
      std::system(tail.c_str());
    }
  }
  bool made = true;
  for (int r = 0; r < 4; r++) made = made && ::access(files[r].c_str(), R_OK) == 0;
  if (!made) {
    std::printf("FAILED\n");
    std::string rm = "rm -rf " + work;
    std::system(rm.c_str());
    return 1;
  }

  RankFile rf[4];
  for (int r = 0; r < 4; r++) rf[r] = open_rank(files[r]);
  const size_t want_n = all.size();
  for (int r = 0; r < 4; r++) {
    expect(rf[r].h.rank == (uint32_t)r && rf[r].h.world == 4, "file " + std::to_string(r) + " says rank " + std::to_string(r) + " of 4");
    expect(rf[r].toc.size() == want_n, "file " + std::to_string(r) + " holds " + std::to_string(rf[r].toc.size()) + " tensors, the checkpoint has " + std::to_string(want_n) + " of the vision tower");
    bool text = false;
    for (const auto& e : rf[r].toc) text = text || std::string(e.name).rfind("layers.", 0) == 0;
    expect(!text, "file " + std::to_string(r) + " holds no text tensor");
  }

  // the four files put together = the checkpoint, by the kinds the TOC declares
  size_t compared = 0, bad = 0, one_d = 0, seg = 0;
  uint64_t rank_bytes[4] = {0, 0, 0, 0};
  for (const Src& s : all) {
    const std::vector<uint8_t> whole = bytes_of(*s.v);
    const dsw::TocEntry* e0 = rf[0].find(s.name);
    if (!e0) {
      std::printf("FAIL no tensor %s in file 0\n", s.name.c_str());
      bad++;
      continue;
    }
    bool ok = true;
    std::vector<std::vector<uint8_t>> part(4);
    for (int r = 0; r < 4; r++) {
      const dsw::TocEntry* e = rf[r].find(s.name);
      ok = ok && e && e->kind == e0->kind && e->nseg == e0->nseg && e->dtype == (uint32_t)dsw::DType::BF16 && e->layout == 0 &&
           e->src_ndim == s.shape.size() && zero_padding(rf[r], *e);
      if (!e) {
        ok = false;
        break;
      }
      for (size_t d = 0; d < s.shape.size(); d++) ok = ok && e->src_shape[d] == s.shape[d];
      part[r] = rf[r].read(*e);
      rank_bytes[r] += e->bytes;
      ok = ok && dsw::crc32c(part[r].data(), part[r].size()) == e->crc32c;
    }
    if (ok) {
      std::vector<uint8_t> got;
      const uint64_t rows = s.shape[0], rb = whole.size() / rows;
      switch (dsw::SliceKind(e0->kind)) {
        case dsw::SliceKind::Replicated:
          for (int r = 0; r < 4; r++) ok = ok && part[r] == whole;
          got = whole;
          break;
        case dsw::SliceKind::Rows:
          for (int r = 0; r < 4; r++) got.insert(got.end(), part[r].begin(), part[r].end());
          break;
        case dsw::SliceKind::RowsSeg: {
          seg++;
          const uint64_t ns = e0->nseg, per = part[0].size() / ns;  // one rank's piece of one segment
          for (uint64_t g = 0; g < ns; g++)
            for (int r = 0; r < 4; r++) got.insert(got.end(), part[r].begin() + g * per, part[r].begin() + (g + 1) * per);
          ok = ok && rows % ns == 0 && (rows / ns) % 4 == 0;
          break;
        }
        default: ok = false;
      }
      ok = ok && got == whole;
    }
    if (s.shape.size() == 1 && dsw::SliceKind(e0->kind) != dsw::SliceKind::Replicated) one_d++;
    compared++;
    if (!ok) {
      bad++;
      std::printf("FAIL %s: the four files put together differ from the checkpoint\n", s.name.c_str());
    }
  }
  expect(compared == want_n && bad == 0, "the four files put together equal the checkpoint's " + std::to_string(compared) + " tensors, bytes (" + std::to_string(bad) + " bad)");
  expect(one_d == (size_t)(2 * blocks + 2), "1-D slices (wqkv, wo, aligner biases) compared: " + std::to_string(one_d));
  expect(seg == (size_t)(3 * blocks), "segment-cut tensors (wqkv weight and bias, w1) compared: " + std::to_string(seg));
  {
    const dsw::TocEntry* e = rf[3].find("vision.patch_embed.proj.weight");
    expect(e && e->bytes == 1204224 && e->shape[0] == 1024 && e->shape[1] == 588 && e->kind == (uint32_t)dsw::SliceKind::Replicated,
           "the patch embedding is stored whole as [1024][588] on every rank");
  }
  const uint64_t per_rank = (uint64_t)blocks * 6428672 + 23592960 + 2560 + 13107200 + 2560 + 1239040;
  for (int r = 0; r < 4; r++)
    expect(rank_bytes[r] == per_rank, "rank " + std::to_string(r) + "'s tensor bytes: " + std::to_string(rank_bytes[r]) + " (6,428,672 a block + 37,944,320)");

  // each file = slice_rank of the checkpoint's weights
  for (int r = 0; r < 4; r++) {
    const Weights s = slice_rank(w, r);
    const std::vector<Src> mine = sources(s);
    bool ok = true;
    for (const Src& m : mine) {
      const dsw::TocEntry* e = rf[r].find(m.name);
      ok = ok && e && rf[r].read(*e) == bytes_of(*m.v);
    }
    expect(ok, "file " + std::to_string(r) + " equals vit::slice_rank(checkpoint, " + std::to_string(r) + ")");
  }

  // the loader's view: the table of contents the engine checks, and the refusal of another rank's file
  for (int r = 0; r < 4; r++) {
    dsw::WeightFile wf(files[r], r, 4);
    const std::string why = check_vision_toc(wf.tensors(), blocks);
    expect(why.empty(), "file " + std::to_string(r) + " passes check_vision_toc" + (why.empty() ? "" : ": " + why));
  }
  {
    uint64_t d[4];
    for (int r = 0; r < 4; r++) {
      dsw::WeightFile wf(files[r], r, 4);
      d[r] = file_set_digest(wf.header(), wf.tensors(), blocks);
    }
    expect(d[0] != 0 && d[0] == d[1] && d[1] == d[2] && d[2] == d[3], "the four files give one digest");
  }
  for (int r = 0; r < 4; r++) expect(refused_as(files[r], (r + 1) % 4), "file " + std::to_string(r) + " is refused as rank " + std::to_string((r + 1) % 4));

  for (int r = 0; r < 4; r++) std::fclose(rf[r].f);
  if (keep) {
    std::printf("kept %s\n", work.c_str());
  } else {
    std::string rm = "rm -rf " + work;
    if (std::system(rm.c_str()) != 0) std::printf("note: could not remove %s\n", work.c_str());
  }
  std::printf("%s\n", fails ? "FAILED" : "all ok");
  return fails ? 1 : 0;
}
