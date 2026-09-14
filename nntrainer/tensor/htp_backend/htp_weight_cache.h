// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   htp_weight_cache.h
 * @date   14 Sep 2026
 * @brief  On-disk cache of baked WH weight bytes, to skip the DSP bake
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * Registration costs 2297 ms for this model's 64 weights (doc 46 section
 * 20), and the larger half of that is the RM->WH bake on the DSP -- about
 * 25 ms a weight, 7168 independent tile rearrangements for a gate_up. The
 * bake is deterministic (doc 43 section 7 checked it with FNV-1a), so a run
 * that has paid for it can write the bytes out and every later run reads
 * them instead, through weight_register_u8i4_baked.
 *
 * Off unless NNTR_HTP_WEIGHT_CACHE names a directory. A cache is a place
 * the process writes files, and where those files go is the operator's
 * call, not a default this code should invent.
 *
 * The WH bytes are read straight into the caller's buffer rather than a
 * vector it would then have to copy: that buffer is rpcmem/ION, so FastRPC
 * hands it to the DSP without a copy of its own, and the alternative was
 * paying for two. Reading the cache was 583 ms of a 975 ms registration --
 * 178 MB at 0.31 GB/s -- so the copies are not a rounding error.
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
#include <string>
#include <vector>

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

  /**
   * @brief Reads a cached bake, if one matching this exact weight exists.
   *
   * @return true with all four vectors filled; false for any mismatch,
   *         short read, or missing file -- all of which are ordinary misses
   *         and leave the caller to bake normally.
   */
  bool load(const std::string &file, uint32_t K, uint32_t N, uint64_t src_hash,
            uint8_t *wh, uint32_t wh_cap, std::vector<float> &w_scale,
            std::vector<int32_t> &colsum_w, std::vector<float> &bias) const {
    std::FILE *f = std::fopen(file.c_str(), "rb");
    if (f == nullptr)
      return false;

    HtpWeightCacheHeader h{};
    bool ok = std::fread(&h, sizeof(h), 1, f) == 1 &&
              std::memcmp(h.magic, kMagic, sizeof(h.magic)) == 0 &&
              h.version == kVersion && h.K == K && h.N == N &&
              h.src_hash == src_hash && h.wh_len == whBytes(K, N) &&
              h.wh_len <= wh_cap;
    if (ok) {
      w_scale.resize(N);
      colsum_w.resize(N);
      bias.resize(N);
      ok = std::fread(wh, 1, h.wh_len, f) == h.wh_len &&
           std::fread(w_scale.data(), sizeof(float), N, f) == N &&
           std::fread(colsum_w.data(), sizeof(int32_t), N, f) == N &&
           std::fread(bias.data(), sizeof(float), N, f) == N;
    }
    std::fclose(f);
    return ok;
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

  /** @brief WH byte count for a shape: one 512-byte tile per (k,n) tile
   *  pair, HEXKL_HMX_INT8_BLOCK_N_INNER by _N_COL, both 32. Written in tile
   *  form rather than the K*N/2 it reduces to, because that is the form the
   *  DSP checks it against and the two must not drift. */
  static uint32_t whBytes(uint32_t K, uint32_t N) {
    return (K / 32u) * (N / 32u) * 512u;
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
