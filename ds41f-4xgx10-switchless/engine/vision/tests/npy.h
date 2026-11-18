// A minimal .npy reader and writer (format 1.0 and 2.0, C order, little endian) for the vision test tools: the dumps
// tools/vision/dump_ref.py writes are read here and the outputs vit_compare.py reads are written here. Header-only,
// host only. Every malformed input throws std::runtime_error with the path in the message.
#pragma once
#include <errno.h>
#include <sys/stat.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace npy {

struct Array {
  std::string descr;           // "<u2" (bf16 bits), "<f4", "|u1"
  std::vector<size_t> shape;
  std::vector<uint8_t> data;   // C order
  size_t count() const {
    size_t n = 1;
    for (size_t s : shape) n *= s;
    return n;
  }
};

inline size_t item_bytes(const std::string& descr) {
  if (descr == "<u2") return 2;
  if (descr == "<f4") return 4;
  if (descr == "|u1") return 1;
  throw std::runtime_error("npy: unsupported dtype " + descr);
}

// Creates `path`'s folders (all of them), as mkdir -p.
inline void make_parents(const std::string& path) {
  for (size_t i = 1; i < path.size(); i++)
    if (path[i] == '/') {
      const std::string d = path.substr(0, i);
      if (mkdir(d.c_str(), 0755) != 0 && errno != EEXIST) throw std::runtime_error("npy: cannot create " + d);
    }
}

inline Array read(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("npy: cannot open " + path);
  unsigned char pre[10];
  f.read(reinterpret_cast<char*>(pre), 10);
  if (f.gcount() != 10 || std::memcmp(pre, "\x93NUMPY", 6) != 0) throw std::runtime_error("npy: not a .npy file: " + path);
  size_t hlen;
  if (pre[6] == 1) {
    hlen = pre[8] | (size_t)pre[9] << 8;
  } else if (pre[6] == 2) {
    unsigned char more[2];
    f.read(reinterpret_cast<char*>(more), 2);
    hlen = pre[8] | (size_t)pre[9] << 8 | (size_t)more[0] << 16 | (size_t)more[1] << 24;
  } else {
    throw std::runtime_error("npy: unsupported version in " + path);
  }
  std::string h(hlen, '\0');
  f.read(&h[0], (std::streamsize)hlen);
  if ((size_t)f.gcount() != hlen) throw std::runtime_error("npy: short header in " + path);
  auto field = [&](const char* key) -> std::string {
    const size_t k = h.find(key);
    if (k == std::string::npos) throw std::runtime_error(std::string("npy: no ") + key + " in the header of " + path);
    return h.substr(k + std::strlen(key));
  };
  Array a;
  {
    const std::string d = field("'descr':");
    const size_t q = d.find('\''), e = d.find('\'', q + 1);
    if (q == std::string::npos || e == std::string::npos) throw std::runtime_error("npy: bad descr in " + path);
    a.descr = d.substr(q + 1, e - q - 1);
  }
  if (field("'fortran_order':").find("False") == std::string::npos) throw std::runtime_error("npy: Fortran order in " + path);
  {
    const std::string s = field("'shape':");
    const size_t o = s.find('('), c = s.find(')');
    if (o == std::string::npos || c == std::string::npos) throw std::runtime_error("npy: bad shape in " + path);
    size_t v = 0;
    bool in = false;
    for (size_t i = o + 1; i < c; i++) {
      if (s[i] >= '0' && s[i] <= '9') v = v * 10 + (size_t)(s[i] - '0'), in = true;
      else if (in) a.shape.push_back(v), v = 0, in = false;
    }
    if (in) a.shape.push_back(v);
  }
  const size_t bytes = a.count() * item_bytes(a.descr);
  a.data.resize(bytes);
  f.read(reinterpret_cast<char*>(a.data.data()), (std::streamsize)bytes);
  if ((size_t)f.gcount() != bytes) throw std::runtime_error("npy: short data in " + path);
  return a;
}

// Writes format 1.0 as numpy does (header padded with spaces to a multiple of 64 bytes in all, a newline last).
inline void write(const std::string& path, const std::string& descr, const std::vector<size_t>& shape, const void* data,
                  size_t bytes) {
  size_t n = 1;
  for (size_t s : shape) n *= s;
  if (n * item_bytes(descr) != bytes) throw std::runtime_error("npy: " + std::to_string(bytes) + " bytes do not fill the shape of " + path);
  std::string shp = "(";
  for (size_t i = 0; i < shape.size(); i++) shp += (i ? ", " : "") + std::to_string(shape[i]);
  if (shape.size() == 1) shp += ",";
  shp += ")";
  std::string h = "{'descr': '" + descr + "', 'fortran_order': False, 'shape': " + shp + ", }";
  const size_t total = (10 + h.size() + 1 + 63) / 64 * 64;
  h.append(total - 10 - h.size() - 1, ' ');
  h += '\n';
  make_parents(path);
  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("npy: cannot write " + path);
  const unsigned char pre[10] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0, (unsigned char)(h.size() & 255), (unsigned char)(h.size() >> 8)};
  f.write(reinterpret_cast<const char*>(pre), 10);
  f.write(h.data(), (std::streamsize)h.size());
  f.write(static_cast<const char*>(data), (std::streamsize)bytes);
  if (!f) throw std::runtime_error("npy: write failed for " + path);
}

inline std::vector<uint16_t> as_u16(const Array& a) {
  if (a.descr != "<u2") throw std::runtime_error("npy: " + a.descr + " is not uint16");
  std::vector<uint16_t> v(a.count());
  std::memcpy(v.data(), a.data.data(), a.data.size());
  return v;
}
inline std::vector<float> as_f32(const Array& a) {
  if (a.descr != "<f4") throw std::runtime_error("npy: " + a.descr + " is not float32");
  std::vector<float> v(a.count());
  std::memcpy(v.data(), a.data.data(), a.data.size());
  return v;
}
inline std::vector<uint8_t> as_u8(const Array& a) {
  if (a.descr != "|u1") throw std::runtime_error("npy: " + a.descr + " is not uint8");
  return a.data;
}

}  // namespace npy
