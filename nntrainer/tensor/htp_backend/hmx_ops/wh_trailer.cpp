// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   wh_trailer.cpp
 * @brief  WH-weight trailer parser implementation
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs
 */
#ifdef ENABLE_HEXKL

#include <wh_trailer.h>

#include <cstring>
#include <istream>
#include <ostream>

namespace nntrainer {

static bool g_wh_bake_requested = false;
bool isWHBakeRequested() { return g_wh_bake_requested; }
void setWHBakeRequested(bool v) { g_wh_bake_requested = v; }

namespace hmx {

std::vector<WHTrailerEntry> *g_wh_collector = nullptr;

namespace {
void put_u32(std::ostream &os, uint32_t v) {
  os.write(reinterpret_cast<const char *>(&v), sizeof(v));
}
void put_u64(std::ostream &os, uint64_t v) {
  os.write(reinterpret_cast<const char *>(&v), sizeof(v));
}
bool get_u32(std::istream &is, uint32_t &v) {
  is.read(reinterpret_cast<char *>(&v), sizeof(v));
  return static_cast<bool>(is);
}
bool get_u64(std::istream &is, uint64_t &v) {
  is.read(reinterpret_cast<char *>(&v), sizeof(v));
  return static_cast<bool>(is);
}
} // namespace

void writeWHTrailer(std::ostream &os,
                    const std::vector<WHTrailerEntry> &entries) {
  const std::streamoff start = os.tellp();
  for (const auto &e : entries) {
    put_u32(os, static_cast<uint32_t>(e.name.size()));
    os.write(e.name.data(), static_cast<std::streamsize>(e.name.size()));
    put_u32(os, e.N);
    put_u32(os, e.K);
    put_u64(os, static_cast<uint64_t>(e.wh_bytes.size()));
    os.write(e.wh_bytes.data(),
             static_cast<std::streamsize>(e.wh_bytes.size()));
  }
  const std::streamoff end = os.tellp();
  const uint64_t trailer_bytes = static_cast<uint64_t>(end - start);
  put_u64(os, trailer_bytes);
  put_u32(os, static_cast<uint32_t>(entries.size()));
  os.write(WH_TRAILER_MAGIC, 4);
}

bool readWHTrailer(std::istream &is, std::vector<WHTrailerEntry> &out) {
  out.clear();
  is.clear();
  is.seekg(0, std::ios::end);
  const std::streamoff file_end = is.tellg();
  if (file_end < 16)
    return false;
  is.seekg(file_end - 16, std::ios::beg);
  uint64_t trailer_bytes = 0;
  uint32_t count = 0;
  char magic[4];
  if (!get_u64(is, trailer_bytes) || !get_u32(is, count))
    return false;
  is.read(magic, 4);
  if (!is || std::memcmp(magic, WH_TRAILER_MAGIC, 4) != 0)
    return false;

  const std::streamoff trailer_start =
    file_end - 16 - static_cast<std::streamoff>(trailer_bytes);
  if (trailer_start < 0)
    return false;
  is.seekg(trailer_start, std::ios::beg);

  out.reserve(count);
  for (uint32_t i = 0; i < count; ++i) {
    uint32_t name_len = 0;
    if (!get_u32(is, name_len))
      return (out.clear(), false);
    WHTrailerEntry e;
    e.name.resize(name_len);
    if (name_len) {
      is.read(&e.name[0], name_len);
      if (!is)
        return (out.clear(), false);
    }
    uint32_t N = 0, K = 0;
    uint64_t blen = 0;
    if (!get_u32(is, N) || !get_u32(is, K) || !get_u64(is, blen))
      return (out.clear(), false);
    e.N = N;
    e.K = K;
    e.wh_bytes.resize(blen);
    if (blen) {
      is.read(e.wh_bytes.data(), static_cast<std::streamsize>(blen));
      if (!is)
        return (out.clear(), false);
    }
    out.push_back(std::move(e));
  }
  return true;
}

} // namespace hmx
} // namespace nntrainer

#endif // ENABLE_HEXKL
