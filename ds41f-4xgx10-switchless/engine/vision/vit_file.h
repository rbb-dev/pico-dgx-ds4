// The vision weight file of one rank (dsw_convert --vision): what it must hold, and the digest the four ranks compare
// at start. Host only: it reads the table of contents the loader already parsed, no CUDA and no device memory.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "vision/vit_weights.h"
#include "weights/loader.h"

namespace vit {

// One tensor a rank's file holds: its name, shape (the rank's slice) and whether it is whole on every rank.
struct FileTensor {
  std::string name;
  std::vector<uint64_t> shape;
  bool replicated;
};

// The 266 tensors of rank `rank`'s file for `blocks` blocks of the tower (32 in the real model; fewer in tests), in
// the order dsw_convert writes them (sorted by name). Every tensor is BF16 and Natural layout.
std::vector<FileTensor> expected_file(int blocks = kLayers);

// "" when `tensors` (the file's table of contents, in file order) is exactly expected_file(blocks): the same names,
// BF16, Natural layout, the rank's slice shapes, each stored in `bytes` = elements x 2; else the first difference in
// plain words. A file cut for another model, another split or with a tensor missing is refused here.
std::string check_vision_toc(const std::vector<dsw::DeviceTensor>& tensors, int blocks = kLayers);

// A 64-bit digest of what makes four files one set: the header's world and size, every tensor's name, type, shape and
// size, and the CRC of every tensor that is whole on every rank (so the replicated tensors of the four files agree
// byte for byte). Equal for the four files of one conversion, different when one file comes from another checkpoint or
// another cut. The rank is not in it.
uint64_t file_set_digest(const dsw::Header& h, const std::vector<dsw::DeviceTensor>& tensors, int blocks = kLayers);

}  // namespace vit
