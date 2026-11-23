// The image encoder's command on a rank: one range of blocks of one image through vit::RankRun over the ring (the in-place
// all-gather of comm::allgather_multi, on the model's stream), the pixels of rank 0 through the pinned staging buffer into block 0
// of the pixel slot, and at the last block the image's span written with vit::span_rows straight into the flight's store numbers
// (in runs that do not cross a number's edge: no row crosses the fabric), then one vote: every rank sums its rows' bytes and the four
// sums are compared in one host all-gather.
//
// One image at a time: from an image's block 0 to its last block the work area is that image's. A range that is not the next of the
// image between its blocks, or an image that starts while one is, stops the engine. A mark whose flight's rows are gone (its
// request ended between two ranges) is stale: the next image starts at block 0 over it.
//
// DSGB10_VIT_FAIL=rank:R,at:vote (fault builds): rank R's sum is off by one. DSGB10_VIT_DUMP=DIR (fault builds): every rank writes an
// encoded image's span rows to DIR/rank<R>-key<K>-span<S>.bin, for the tests that compare them with the unsplit run.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>

#include "model/impl.h"
#include "vision/vit_rank.h"

namespace model {

using namespace detail;

#ifdef DSGB10_TEST_FAULTS
namespace {
bool vit_vote_fault(int rank) {
  static bool fired = false;
  const char* env = std::getenv("DSGB10_VIT_FAIL");
  int r = -1;
  char w[16] = {};
  if (fired || !env || std::sscanf(env, "rank:%d,at:%15s", &r, w) != 2 || r != rank || std::strcmp(w, "vote") != 0) return false;
  fired = true;
  std::printf("rank %d: vision fault injected at vote\n", rank);
  return true;
}
}  // namespace
#endif

void Model::Impl::vit_prune() {
  for (auto it = vit.written.begin(); it != vit.written.end();) it = core->img_find(it->first) ? std::next(it) : vit.written.erase(it);
}

bool Model::Impl::vit_span_written(int key, size_t span) const {
  const auto it = vit.written.find(key);
  return it == vit.written.end() || span >= it->second.size() || it->second[span] != 0;
}

void Model::vision_register(int key, int n) {
  d_->vit_prune();
  d_->vit.written[key].assign((size_t)std::max(0, n), 0);
}

int Model::vision_run(int key, int span, int row0, int nh, int nw, int b0, int b1, const uint8_t* pixels) {
  return d_->vit_run(key, span, row0, nh, nw, b0, b1, pixels);
}

int Model::Impl::vit_run(int key, int span, int row0, int nh, int nw, int b0, int b1, const uint8_t* pixels) {
  Vit& V = vit;
  if (!V.w || !V.area) die("model: an image command on rank %d, which holds no image encoder work area", rank);
  const vit::Grid g = vit::make_grid(nh, nw);
  const int nb = (int)V.w->blocks.size();
  if (nh < 1 || nw < 1 || g.patches() > vit::kMaxPatches || g.rows() > 1024 || b0 < 0 || b0 >= b1 || b1 > nb)
    die("model: image %d of flight %d: %d x %d patches, blocks [%d, %d) of %d", span, key, nh, nw, b0, b1, nb);
  const imgrows::Ledger::Rows* fr = core->img_find(key);
  if (!fr || row0 < 0 || (size_t)row0 + (size_t)g.rows() > fr->rows)
    die("model: image %d of flight %d: its %d rows from row %d are not among the flight's %zu", span, key, g.rows(), row0, fr ? fr->rows : 0);
  vit_prune();
  if (V.mid_key && !core->img_find(V.mid_key)) V.mid_key = 0, V.mid_span = -1, V.mid_next = 0;  // its request ended between two ranges
  if (V.mid_key) {
    if (key != V.mid_key || span != V.mid_span || b0 != V.mid_next)
      die("model: image %d of flight %d, blocks [%d, %d), while image %d of flight %d is between its blocks (its next is block %d)", span, key, b0,
          b1, V.mid_span, V.mid_key, V.mid_next);
  } else if (b0 != 0) {
    die("model: image %d of flight %d, blocks [%d, %d): no image is between its blocks, so a range starts at block 0", span, key, b0, b1);
  }
  const int n = g.patches();
  uint8_t* const pix = vit::pixel_slot(V.area, n, g.merged());
#ifdef DSGB10_TEST_FAULTS
  if (std::getenv("DSGB10_VIT_POISON")) {  // every buffer a range does not carry over is NaN, so a read of a stale or unused one shows in the bytes
    if (b0 == 0) CUDA_CHECK(cudaMemsetAsync(V.area, 0xFF, Model::vision_area_bytes(), st));
    else vit::RankRun::poison_all_but_state(V.area, g, st);
  }
#endif
  if (b0 == 0 && rank == 0) {  // the padded image into block 0 of the pixel slot, through the pinned staging buffer
    const size_t bytes = (size_t)n * vit::kPatchK;
    if (!pixels || bytes > Model::kVitStageBytes) die("model: image %d of flight %d: %zu bytes of pixels (%s), the staging buffer holds %zu", span, key, bytes,
                                              pixels ? "given" : "none", Model::kVitStageBytes);
    CUDA_CHECK(cudaEventSynchronize(V.stage_ev));  // the last image's copy out of it
    std::memcpy(V.stage, pixels, bytes);
    CUDA_CHECK(cudaMemsetAsync(pix, 0, vit::pixel_block_bytes(n), st));  // the padding up to 16 bytes
    CUDA_CHECK(cudaMemcpyAsync(pix, V.stage, bytes, cudaMemcpyHostToDevice, st));
    CUDA_CHECK(cudaEventRecord(V.stage_ev, st));
  }
  vit::RankRunConfig cfg;
  cfg.w = V.w, cfg.rank = rank, cfg.grid = g, cfg.work = V.area;
  cfg.pixels = b0 == 0 && rank == 0 ? pix : nullptr;
  cfg.st = st, cfg.b0 = b0, cfg.b1 = b1;
  vit::RankRun run(cfg);
  vit::GatherPoint gp;
  while (run.step(&gp)) {
    const comm::AgSeg seg{static_cast<const char*>(gp.buf) + (size_t)rank * gp.block_bytes, gp.buf, gp.block_bytes};
    comm::allgather_multi(ch, &seg, 1, st);
  }
  bool ok = true;
  if (b1 == nb) {
    uint64_t* const scratch = reinterpret_cast<uint64_t*>(static_cast<char*>(V.area) + (Model::vision_area_bytes() - vit::digest_scratch_bytes(1024)));
    uint64_t vote = 0;
    uint64_t at = 0;
    imgrows::runs((size_t)row0, (size_t)row0 + (size_t)g.rows(), [&](size_t r, size_t cnt) {
      bf16* dst = reinterpret_cast<bf16*>(const_cast<uint8_t*>(img_row(key, r)));
      vit::write_span_rows(*V.w, V.area, g, (int)(r - (size_t)row0), (int)cnt, dst, st);
      const uint64_t d = vit::digest(vit::rows_view(dst, vit::kLm), (int)cnt, vit::kLm, scratch, st);  // syncs the stream
      vote = (vote ^ d) * 0x9E3779B97F4A7C15ull + (++at);
    });
#ifdef DSGB10_TEST_FAULTS
    if (vit_vote_fault(rank)) ++vote;
    if (const char* dir = std::getenv("DSGB10_VIT_DUMP")) {
      char path[512];
      std::snprintf(path, sizeof path, "%s/rank%d-key%d-span%d.bin", dir, rank, key, span);
      if (FILE* f = std::fopen(path, "wb")) {
        std::vector<uint8_t> h(imgrows::kRowBytes);
        for (int r = 0; r < g.rows(); ++r) {
          CUDA_CHECK(cudaMemcpy(h.data(), img_row(key, (size_t)row0 + (size_t)r), h.size(), cudaMemcpyDeviceToHost));
          std::fwrite(h.data(), 1, h.size(), f);
        }
        std::fclose(f);
      }
    }
#endif
    uint64_t all[4] = {};
    host_ring->allgather_host(&vote, all, sizeof vote, comm::kCallVit);
    ok = all[0] == all[1] && all[1] == all[2] && all[2] == all[3];
    if (!ok && rank == 0)
      std::printf("model: image %d of flight %d: the four ranks' rows differ (sums %016llx %016llx %016llx %016llx)\n", span, key,
                  (unsigned long long)all[0], (unsigned long long)all[1], (unsigned long long)all[2], (unsigned long long)all[3]);
  }
  CUDA_CHECK(cudaStreamSynchronize(st));
  if (b1 < nb) {
    V.mid_key = key, V.mid_span = span, V.mid_next = b1;
  } else {
    V.mid_key = 0, V.mid_span = -1, V.mid_next = 0;
    const auto it = V.written.find(key);
    if (ok && it != V.written.end() && (size_t)span < it->second.size()) it->second[(size_t)span] = 1;
  }
  return ok ? Model::kVitOk : Model::kVitVote;
}

}  // namespace model
