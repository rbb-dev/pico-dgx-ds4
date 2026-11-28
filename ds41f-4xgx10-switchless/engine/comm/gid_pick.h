// Which entry of a port's GID table carries a link's address (comm/ring.cpp open_link), without a device: the rule alone,
// so that comm/tests/gid_pick_test.cpp holds it on the host.
#pragma once
#include <cstdint>
#include <cstring>

namespace comm {

// One entry of a port's GID table as the device reports it: empty or not, its type, its 16 bytes.
struct GidEntry {
  bool present = false;
  bool roce_v2 = false;
  uint8_t raw[16] = {};
};

// The index of `want` (an IPv4-mapped address) among the port's RoCE v2 GIDs: `pref`, the fleet file's GID_INDEX, when
// that entry is it, else the first entry that is, else -1. The address is what the fleet file pins; the index is the
// kernel's and can differ from box to box or after a link goes down and up (e.g. 4 instead of 3), and it is used on
// this side only (sgid_index): the peer gets the GID itself. An entry of another type (RoCE v1) is never taken.
// `entry(i)` reads entry i.
template <class Entry>
int pick_gid(int pref, const uint8_t* want, int tbl_len, Entry entry) {
  auto is = [&](int i) {
    const GidEntry e = entry(i);
    return e.present && e.roce_v2 && std::memcmp(e.raw, want, 16) == 0;
  };
  if (pref >= 0 && pref < tbl_len && is(pref)) return pref;
  for (int i = 0; i < tbl_len; i++)
    if (i != pref && is(i)) return i;
  return -1;
}

}  // namespace comm
