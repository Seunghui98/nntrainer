// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   htp_weight_cache.h
 * @date   14 Sep 2026
 * @brief  The converted model: one file per baked WH weight, read into the
 *         DSP-mapped arena
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * Registration costs 2297 ms for one layer's 64 weights (doc 46 section
 * 20), most of it the RM->WH bake on the DSP -- about 25 ms a weight -- and
 * the baked bytes then sit on DSP heap, which tops out at 1.89 GB against
 * the 3.9 the whole model needs (doc 45 Gate 0). The bake is deterministic
 * (doc 43 section 7), so a run that pays it writes the bytes out, and every
 * later run reads them into an rpcmem arena the DSP maps once and DMAs
 * from at the same rate as its own heap (doc 46 Gate 0c). Nothing is
 * copied to the DSP and nothing lives on its heap.
 *
 * The directory named by NNTR_HTP_WEIGHT_CACHE is that converted model. Off
 * when unset: where a process writes files is the operator's call. One
 * model per directory -- everything in it is loaded into the arena, so a
 * second model's files there cost memory for nothing. Conversion is not a
 * separate mode: a weight with no file is baked, exported, written, and
 * placed in the arena in the same call, so a first run converts what it
 * touches and later runs find it.
 *
 * The file records a hash of the SOURCE weight bytes. That is what lets a
 * hit skip htp_qs4cx_from_packed as well as the bake -- the scale and
 * colsum arrays come out of the file too -- and it is what makes a stale
 * file a miss rather than a silently wrong matmul. Everything read back is
 * checked: magic, version, K, N, byte counts, file length, source hash.
 * These bytes go straight into a weight the DSP multiplies by.
 */

#ifndef __NNTRAINER_HTP_WEIGHT_CACHE_H__
#define __NNTRAINER_HTP_WEIGHT_CACHE_H__
#ifdef __cplusplus
#ifdef ENABLE_HEXKL

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <string>
#include <vector>

#include <htp_wh_layout.h>

namespace nntrainer {

/** @brief FNV-1a over a byte range. Not a security hash -- it identifies
 *  which weight a cache file holds, and a collision would be read as a hit
 *  on the wrong weight, which the K/N and length checks then have to catch.
 */
inline uint64_t htpWeightHash(const void *data, size_t bytes) {
  const uint8_t *p = static_cast<const uint8_t *>(data);
  uint64_t h = 1469598103934665603ULL;
  for (size_t i = 0; i < bytes; ++i) {
    h ^= p[i];
    h *= 1099511628211ULL;
  }
  return h;
}

/** @brief Bytes on disk ahead of the payload. Fixed width and fixed order,
 *  so a file written by one build is readable by the next or rejected by
 *  the version, never misread. */
struct HtpWeightCacheHeader {
  char magic[8];    /**< kMagic: 7 chars plus its terminator */
  uint32_t version; /**< 1 */
  uint32_t K;
  uint32_t N;
  uint32_t wh_len;   /**< (K/32)*(N/32)*512 */
  uint64_t src_hash; /**< htpWeightHash over the source weight bytes */
};

/** @brief The whole cache, which is just a directory and an on/off state. */
class HtpWeightCache {
public:
  static HtpWeightCache &global() {
    static HtpWeightCache instance;
    return instance;
  }

  bool enabled() const { return !dir_.empty(); }

  /** @brief Where this weight's file would be. Shape and source hash are
   *  both in the name so two weights never share one, and a shape change
   *  lands on a different path rather than overwriting. */
  std::string path(uint32_t K, uint32_t N, uint64_t src_hash) const {
    char buf[96];
    std::snprintf(buf, sizeof(buf), "/wh_%ux%u_%016llx.bin", K, N,
                  static_cast<unsigned long long>(src_hash));
    return dir_ + buf;
  }

  /** @brief Every weight file in the directory, in readdir order. The
   *  arena is sized and filled from this list, so the order is only a
   *  placement order. */
  std::vector<std::string> listFiles() const {
    std::vector<std::string> out;
    DIR *d = ::opendir(dir_.c_str());
    if (d == nullptr)
      return out;
    while (dirent *e = ::readdir(d)) {
      const std::string name = e->d_name;
      if (name.size() > 7 && name.compare(0, 3, "wh_") == 0 &&
          name.compare(name.size() - 4, 4, ".bin") == 0)
        out.push_back(dir_ + "/" + name);
    }
    ::closedir(d);
    return out;
  }

  /** @brief Reads and validates one file's header. false for anything
   *  short, foreign, or from another version -- an ordinary miss. */
  bool readHeader(std::FILE *f, HtpWeightCacheHeader &h) const {
    return std::fread(&h, sizeof(h), 1, f) == 1 &&
           std::memcmp(h.magic, kMagic, sizeof(h.magic)) == 0 &&
           h.version == kVersion && h.wh_len == whBytes(h.K, h.N);
  }

  /**
   * @brief Reads the payload that follows a header straight into @a wh --
   *        the arena, so the bytes are never held anywhere else -- and the
   *        three N-sized arrays into vectors.
   */
  bool readPayload(std::FILE *f, const HtpWeightCacheHeader &h, uint8_t *wh,
                   std::vector<float> &w_scale, std::vector<int32_t> &colsum_w,
                   std::vector<float> &bias) const {
    const uint32_t N = h.N;
    w_scale.resize(N);
    colsum_w.resize(N);
    bias.resize(N);
    return std::fread(wh, 1, h.wh_len, f) == h.wh_len &&
           std::fread(w_scale.data(), sizeof(float), N, f) == N &&
           std::fread(colsum_w.data(), sizeof(int32_t), N, f) == N &&
           std::fread(bias.data(), sizeof(float), N, f) == N;
  }

  /**
   * @brief Writes a bake to the cache, via a temporary and a rename.
   *
   * The rename is what keeps a killed process from leaving a half-written
   * file that the length checks alone would not catch on every truncation
   * boundary. A failed write is not an error worth failing the model load
   * over -- the run simply does not get a cache -- so this reports nothing.
   */
  void store(const std::string &file, uint32_t K, uint32_t N, uint64_t src_hash,
             const uint8_t *wh, const float *w_scale, const int32_t *colsum_w,
             const float *bias) const {
    const std::string tmp = file + ".tmp";
    std::FILE *f = std::fopen(tmp.c_str(), "wb");
    if (f == nullptr)
      return;

    HtpWeightCacheHeader h{};
    std::memcpy(h.magic, kMagic, sizeof(h.magic));
    h.version = kVersion;
    h.K = K;
    h.N = N;
    h.wh_len = whBytes(K, N);
    h.src_hash = src_hash;

    const bool ok = std::fwrite(&h, sizeof(h), 1, f) == 1 &&
                    std::fwrite(wh, 1, h.wh_len, f) == h.wh_len &&
                    std::fwrite(w_scale, sizeof(float), N, f) == N &&
                    std::fwrite(colsum_w, sizeof(int32_t), N, f) == N &&
                    std::fwrite(bias, sizeof(float), N, f) == N;
    const bool closed = std::fclose(f) == 0;
    if (ok && closed) {
      std::rename(tmp.c_str(), file.c_str());
    } else {
      std::remove(tmp.c_str());
    }
  }

  /** @brief WH byte count for a shape. Forwards to htp_wh_layout.h so the
   *  file format and the packer cannot drift apart. */
  static uint32_t whBytes(uint32_t K, uint32_t N) {
    return static_cast<uint32_t>(nntrainer::whBytes(K, N));
  }

private:
  HtpWeightCache() {
    const char *dir = std::getenv("NNTR_HTP_WEIGHT_CACHE");
    if (dir != nullptr && dir[0] != '\0')
      dir_ = dir;
  }

  /** Exactly 8 bytes with its terminator, matching magic[8]. */
  static constexpr const char *kMagic = "NTRWHCA";
  static constexpr uint32_t kVersion = 1;
  std::string dir_;
};

} // namespace nntrainer

#endif // ENABLE_HEXKL
#endif // __cplusplus
#endif // __NNTRAINER_HTP_WEIGHT_CACHE_H__
