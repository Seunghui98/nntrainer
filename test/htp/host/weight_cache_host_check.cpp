// SPDX-License-Identifier: Apache-2.0
/* Host check for htp_weight_cache.h. No device, no Hexagon SDK.
 *
 * The cache is a trust boundary: its bytes become a weight the DSP
 * multiplies by, and it is read from a file that another process, an
 * interrupted run, or a stale model could have written. So what is checked
 * here is mostly the rejections -- a round trip that works is the easy
 * half. */

#define ENABLE_HEXKL
#include "htp_weight_cache.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using nntrainer::HtpWeightCache;
using nntrainer::htpWeightHash;

static int failures = 0;

static void check(const char *what, bool ok) {
  std::printf("%-34s: %s\n", what, ok ? "ok" : "FAIL");
  if (!ok)
    ++failures;
}

int main() {
  const uint32_t K = 64, N = 64; // small but tile-legal: 2 x 2 tiles
  const uint32_t wh_len = HtpWeightCache::whBytes(K, N);
  check("whBytes matches the tile form", wh_len == (K / 32) * (N / 32) * 512);

  char tmpl[] = "/tmp/nntr_wc_XXXXXX";
  const char *dir = mkdtemp(tmpl);
  if (dir == nullptr) {
    std::printf("mkdtemp failed\n");
    return 1;
  }
  setenv("NNTR_HTP_WEIGHT_CACHE", dir, 1);
  const HtpWeightCache &wc = HtpWeightCache::global();
  check("enabled when the env var is set", wc.enabled());

  std::vector<uint8_t> wh(wh_len);
  for (uint32_t i = 0; i < wh_len; ++i)
    wh[i] = static_cast<uint8_t>(i * 31u + 7u);
  std::vector<float> w_scale(N), bias(N);
  std::vector<int32_t> colsum(N);
  for (uint32_t n = 0; n < N; ++n) {
    w_scale[n] = 0.001f * static_cast<float>(n + 1);
    colsum[n] = static_cast<int32_t>(n) - 32;
    bias[n] = 0.0f;
  }

  const uint64_t hash = htpWeightHash(wh.data(), wh_len);
  const std::string path = wc.path(K, N, hash);
  wc.store(path, K, N, hash, wh.data(), w_scale.data(), colsum.data(),
           bias.data());

  std::vector<uint8_t> r_wh;
  std::vector<float> r_scale, r_bias;
  std::vector<int32_t> r_colsum;
  check("round trip loads",
        wc.load(path, K, N, hash, r_wh, r_scale, r_colsum, r_bias));
  check("round trip is byte-identical",
        r_wh == wh && r_scale == w_scale && r_colsum == colsum &&
          r_bias == bias);

  // Every one of these must be a miss, not a load of the wrong bytes.
  check("wrong source hash rejected",
        !wc.load(path, K, N, hash ^ 1ull, r_wh, r_scale, r_colsum, r_bias));
  check("wrong K rejected",
        !wc.load(path, K * 2, N, hash, r_wh, r_scale, r_colsum, r_bias));
  check("wrong N rejected",
        !wc.load(path, K, N * 2, hash, r_wh, r_scale, r_colsum, r_bias));
  check("missing file rejected",
        !wc.load(wc.path(K, N, hash + 1), K, N, hash + 1, r_wh, r_scale,
                 r_colsum, r_bias));

  // A run killed mid-write must not leave something that loads. store()
  // writes through a temporary and renames, so the half-file is the .tmp
  // and never the path load() reads -- truncating the real file stands in
  // for the failure mode a non-atomic writer would have had.
  {
    std::FILE *f = std::fopen(path.c_str(), "rb");
    std::vector<uint8_t> whole;
    if (f != nullptr) {
      std::fseek(f, 0, SEEK_END);
      whole.resize(static_cast<size_t>(std::ftell(f)));
      std::fseek(f, 0, SEEK_SET);
      if (std::fread(whole.data(), 1, whole.size(), f) != whole.size())
        whole.clear();
      std::fclose(f);
    }
    check("file is header + payload",
          whole.size() ==
            sizeof(nntrainer::HtpWeightCacheHeader) + wh_len +
              static_cast<size_t>(N) * (sizeof(float) * 2 + sizeof(int32_t)));

    const std::string cut = std::string(dir) + "/truncated.bin";
    std::FILE *o = std::fopen(cut.c_str(), "wb");
    std::fwrite(whole.data(), 1, whole.size() / 2, o);
    std::fclose(o);
    check("truncated file rejected",
          !wc.load(cut, K, N, hash, r_wh, r_scale, r_colsum, r_bias));
  }

  std::printf(failures == 0 ? "\nWEIGHT CACHE CHECKS PASS\n"
                            : "\nWEIGHT CACHE CHECKS FAILED\n");
  return failures != 0;
}
