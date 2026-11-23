// The frames whose layout depends on the connection's protocol version (HelloFe.version): the SUBMIT's head and the DONE (host only;
// tested by tests/prefix_test.cpp). A connection of fe::kVersion sees the base frames: DONE is the 56-byte header and the error text, and
// a SUBMIT with kFlagPrefix is malformed (the reader closes the connection, as it does for any malformed frame). A connection of
// fe::kVersionPrefix gets the 16-byte DoneTail on every DONE of a request that ran, and may send kFlagPrefix.
//
// PICO_OLD_RULES (never defined by the engine) builds the policy without the rule: every connection gets the tail and may send the flag.
#pragma once
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "../common/frontend_protocol.h"

namespace frames {

// The versions the engine accepts at HELLO
inline bool version_ok(uint32_t v) { return v == fe::kVersion || v == fe::kVersionPrefix; }

// The head of a SUBMIT payload `p` (n bytes) on a connection of version `ver`: the header, the SubmitPrefix (zeros without the flag),
// and `*base`, the payload's size up to the end of the prompt ids (stop ids and prompt included; an image part follows). False: a
// malformed frame: shorter than the header, kFlagPrefix on a connection that did not say kVersionPrefix, or a length the header does
// not account for (an image request, rows or pixels, is longer by its image part, at least 4 bytes; both flags together are malformed).
inline bool parse_submit_head(const char* p, size_t n, uint32_t ver, fe::SubmitHeader* h, fe::SubmitPrefix* px, size_t* base) {
  if (n < sizeof *h) return false;
  std::memcpy(h, p, sizeof *h);
  const bool pfx = h->flags & fe::kFlagPrefix;
#ifndef PICO_OLD_RULES
  if (pfx && ver != fe::kVersionPrefix) return false;
#else
  (void)ver;
#endif
  *px = fe::SubmitPrefix{};
  *base = sizeof *h + (pfx ? sizeof *px : 0) + 4 * (size_t(h->n_stop_ids) + h->n_prompt);
#ifndef PICO_OLD_RULES
  if ((h->flags & fe::kFlagImages) && (h->flags & fe::kFlagPixels)) return false;
  if (!(h->flags & (fe::kFlagImages | fe::kFlagPixels)) ? n != *base : n < *base + 4) return false;
#else
  if (!(h->flags & fe::kFlagImages) ? n != *base : n < *base + 4) return false;
#endif
  if (pfx) std::memcpy(px, p + sizeof *h, sizeof *px);
  return true;
}

// A DONE frame (its 8-byte header included) for a connection of version `ver`: `tail` goes in only on a connection of kVersionPrefix.
inline std::vector<char> done_frame(uint32_t ver, const fe::DoneHeader& d, const fe::DoneTail* tail, const std::string& err) {
  std::vector<char> b(sizeof(fe::FrameHeader));
  auto add = [&](const void* p, size_t n) {
    const char* c = static_cast<const char*>(p);
    b.insert(b.end(), c, c + n);
  };
  add(&d, sizeof d);
#ifndef PICO_OLD_RULES
  if (tail && ver == fe::kVersionPrefix) add(tail, sizeof *tail);
#else
  if (tail) add(tail, sizeof *tail), (void)ver;
#endif
  add(err.data(), err.size());
  const fe::FrameHeader fh{uint32_t(b.size() - sizeof(fe::FrameHeader)), fe::kDone, 0};
  std::memcpy(b.data(), &fh, sizeof fh);
  return b;
}

}  // namespace frames
