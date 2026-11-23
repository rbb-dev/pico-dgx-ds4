// The byte a test break serves (engram.h, break_byte): for every possible first byte of a row it differs from the row's own
// and is never an e4m3 NaN (0x7f, 0xff), so a break changes what a request receives and nothing a step could stall on. The
// plain flip of bit 0 is checked to reach a NaN from 0x7e and 0xfe: that is what the function exists to avoid. Host only.
//   engram_break_test
#include <cstdio>

#include "engram/engram.h"

using namespace engram;

static int fails = 0;
#define CHECK(c, ...) do { const bool ok_ = (c); std::printf("%s ", ok_ ? "ok  " : "FAIL"); std::printf(__VA_ARGS__); std::printf("\n"); fails += !ok_; } while (0)

static bool nan_e4m3(unsigned b) { return (b & 0x7f) == 0x7f; }

int main() {
  int same = 0, to_nan = 0, plain_nan = 0;
  for (unsigned b = 0; b < 256; ++b) {
    const unsigned v = break_byte(uint8_t(b));
    same += v == b;
    to_nan += !nan_e4m3(b) && nan_e4m3(v);
    plain_nan += !nan_e4m3(b) && nan_e4m3(b ^ 1);
  }
  CHECK(plain_nan == 2, "the plain flip of bit 0 makes a NaN from %d bytes (0x7e, 0xfe)", plain_nan);
  CHECK(same == 0, "the break's byte differs from the row's for every byte value (%d the same)", same);
  CHECK(to_nan == 0, "the break's byte is a NaN for %d byte values that were not", to_nan);
  CHECK(break_byte(0x7e) == 0x7c && break_byte(0xfe) == 0xfc && break_byte(0x00) == 0x01, "the bytes at the edges");
  std::printf("%s\n", fails ? "FAILED" : "all passed");
  return fails ? 1 : 0;
}
