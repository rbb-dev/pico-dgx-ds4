// Host only: the .npy reader and writer vit_teacher uses, against bytes numpy itself wrote.
//   vit_npy_test
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "vision/tests/npy.h"

namespace {

int fails = 0;
void expect(bool ok, const std::string& what) {
  if (!ok) fails++;
  std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
}

// np.save of: arange(6, uint16).reshape(2, 3) + 0x3f80; [1.5, -2, 0.25] float32; arange(8, uint8).reshape(2, 2, 2)
#include "vit_npy_fixtures.inc"

std::vector<unsigned char> slurp(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  return std::vector<unsigned char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

template <size_t N>
void write_file(const std::string& p, const unsigned char (&b)[N]) {
  std::ofstream f(p, std::ios::binary);
  f.write(reinterpret_cast<const char*>(b), N);
}

}  // namespace

int main() {
  char tmpl[] = "/tmp/vit_npy_testXXXXXX";
  const std::string dir = mkdtemp(tmpl);

  // reading what numpy wrote
  write_file(dir + "/u16.npy", kU16);
  const npy::Array a = npy::read(dir + "/u16.npy");
  expect(a.descr == "<u2" && a.shape == std::vector<size_t>({2, 3}), "reads a uint16 [2][3] header");
  const std::vector<uint16_t> u = npy::as_u16(a);
  expect(u.size() == 6 && u[0] == 0x3f80 && u[5] == 0x3f85, "reads its values");
  write_file(dir + "/f32.npy", kF32);
  const std::vector<float> f = npy::as_f32(npy::read(dir + "/f32.npy"));
  expect(f.size() == 3 && f[0] == 1.5f && f[1] == -2.0f && f[2] == 0.25f, "reads a float32 [3,] (one-element shape tuple)");
  write_file(dir + "/u8.npy", kU8);
  const npy::Array b = npy::read(dir + "/u8.npy");
  expect(b.descr == "|u1" && b.shape == std::vector<size_t>({2, 2, 2}) && npy::as_u8(b)[7] == 7, "reads a uint8 [2][2][2]");

  // writing: the same bytes numpy writes
  npy::write(dir + "/w16.npy", "<u2", {2, 3}, u.data(), u.size() * 2);
  const std::vector<unsigned char> w16 = slurp(dir + "/w16.npy");
  expect(w16.size() == sizeof kU16 && std::memcmp(w16.data(), kU16, sizeof kU16) == 0, "writes numpy's bytes (uint16, 2-D)");
  npy::write(dir + "/wf.npy", "<f4", {3}, f.data(), f.size() * 4);
  const std::vector<unsigned char> wf = slurp(dir + "/wf.npy");
  expect(wf.size() == sizeof kF32 && std::memcmp(wf.data(), kF32, sizeof kF32) == 0, "writes numpy's bytes (float32, 1-D)");
  const std::vector<uint8_t> u8 = npy::as_u8(b);
  npy::write(dir + "/w8.npy", "|u1", {2, 2, 2}, u8.data(), u8.size());
  const std::vector<unsigned char> w8 = slurp(dir + "/w8.npy");
  expect(w8.size() == sizeof kU8 && std::memcmp(w8.data(), kU8, sizeof kU8) == 0, "writes numpy's bytes (uint8, 3-D)");

  // refusals: wrong type asked for, a size that does not match, not an npy file, Fortran order
  bool threw = false;
  try { npy::as_f32(a); } catch (const std::exception&) { threw = true; }
  expect(threw, "asking a uint16 array for float32 is refused");
  threw = false;
  try { npy::write(dir + "/bad.npy", "<u2", {2, 3}, u.data(), 10); } catch (const std::exception&) { threw = true; }
  expect(threw, "a byte count that does not match the shape is refused");
  { std::ofstream g(dir + "/junk.npy", std::ios::binary); g << "not numpy at all, just text of some length........"; }
  threw = false;
  try { npy::read(dir + "/junk.npy"); } catch (const std::exception&) { threw = true; }
  expect(threw, "a file that is not .npy is refused");
  std::vector<unsigned char> fo(kU16, kU16 + sizeof kU16);
  const std::string needle = "False";
  for (size_t i = 0; i + 5 < fo.size(); i++)
    if (std::memcmp(&fo[i], needle.data(), 5) == 0) { std::memcpy(&fo[i], "True ", 5); break; }
  { std::ofstream g(dir + "/fortran.npy", std::ios::binary); g.write(reinterpret_cast<const char*>(fo.data()), fo.size()); }
  threw = false;
  try { npy::read(dir + "/fortran.npy"); } catch (const std::exception&) { threw = true; }
  expect(threw, "Fortran order is refused");

  // write creates the folders on the way
  npy::write(dir + "/a/b/c.npy", "<u2", {1}, u.data(), 2);
  expect(npy::read(dir + "/a/b/c.npy").shape == std::vector<size_t>({1}), "writing creates missing folders");

  std::string cmd = "rm -rf " + dir;
  if (std::system(cmd.c_str()) != 0) {}
  std::printf(fails ? "FAILED %d\n" : "ALL OK\n", fails);
  return fails ? 1 : 0;
}
