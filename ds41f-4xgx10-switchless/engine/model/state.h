// Sessions' saved state (state.cu): what a session needs to continue from a position without recomputing. Its
// compressed caches are its own chunk lists in the chunk pool; a line holding it points its rows at them, so no cache is
// copied. Window rings and compressor carries are snapshotted into bookmarks (window bookmark: the 128 positions before a
// restart point, 2 numbers; end bookmark: whole rings and carries, 3). Details: docs/engine.md "The chunk pool", "Kept sessions".
#pragma once
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "model/impl.h"
#include "model/img_rows.h"
#include "model/prefill.h"

namespace model::detail {

constexpr size_t kRingBytes = (size_t)attn::kRing * (kHead + kHead / 32);
constexpr size_t kCarryBytes = (size_t)16 * kHead * 4 * 2;
constexpr size_t kEndBytes = kLayers * kRingBytes + kKvSlots * kCarryBytes;

}  // namespace model::detail
