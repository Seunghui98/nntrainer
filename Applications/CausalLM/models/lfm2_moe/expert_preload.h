// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 * @file expert_preload.h
 * @brief Independent load-time budget for the shared expert cache, and
 *        the decode pool's warm-up from the prefill's routing.
 * @author Codex <noreply@openai.com>
 * @bug No known bugs
 */
#ifndef __CAUSALLM_EXPERT_PRELOAD_H__
#define __CAUSALLM_EXPERT_PRELOAD_H__

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <vector>

namespace causallm {
/**
 * @brief Total experts to read at startup, across all layers.
 *
 * Unset defaults to 128, capped at the cache capacity. Zero reserves the
 * arena but leaves all expert weight reads to prefill. An explicit value
 * at least as large as the capacity restores a full-cache preload.
 */
inline size_t expertPreloadLimit(size_t capacity, const char *value) {
  if (!value)
    return std::min(capacity, size_t{128});
  if (*value < '0' || *value > '9')
    throw std::invalid_argument("NNTR_MOE_PRELOAD_EXPERTS must be nonnegative");
  char *end = nullptr;
  errno = 0;
  const auto count = std::strtoull(value, &end, 10);
  if (errno || *end || count > std::numeric_limits<size_t>::max())
    throw std::invalid_argument("Invalid NNTR_MOE_PRELOAD_EXPERTS");
  return std::min(capacity, static_cast<size_t>(count));
}

/**
 * @brief NNTR_MOE_WARM=<n>: experts per layer the decode pool reads in
 *        while the first token's setup runs (graph init, seed), from the
 *        prefill's routing (warmList). Unset or 0: off.
 */
inline size_t expertWarmPerLayer(const char *value) {
  if (!value || !*value)
    return 0;
  if (*value < '0' || *value > '9')
    throw std::invalid_argument("NNTR_MOE_WARM must be nonnegative");
  char *end = nullptr;
  errno = 0;
  const auto count = std::strtoull(value, &end, 10);
  if (errno || *end || count > 4096)
    throw std::invalid_argument("Invalid NNTR_MOE_WARM");
  return static_cast<size_t>(count);
}

/**
 * @brief One layer's warm list from a prefill call: the last token's
 *        extended top-k in rank order (@a ranked, @a per_token ids a token,
 *        token after token), then the experts the call routed most rows
 *        to (@a rows, by expert; ties to the lower id). At most @a n,
 *        distinct.
 */
inline std::vector<unsigned> warmList(const std::vector<int> &ranked,
                                      size_t per_token,
                                      const std::vector<size_t> &rows,
                                      size_t n) {
  std::vector<unsigned> out;
  std::vector<char> taken(rows.size(), 0);
  const size_t from =
    per_token && ranked.size() >= per_token ? ranked.size() - per_token : 0;
  for (size_t i = from; i < ranked.size() && out.size() < n; ++i) {
    const int e = ranked[i];
    if (e >= 0 && static_cast<size_t>(e) < rows.size() && !taken[e]) {
      taken[e] = 1;
      out.push_back(static_cast<unsigned>(e));
    }
  }
  std::vector<unsigned> by_rows;
  for (size_t e = 0; e < rows.size(); ++e)
    if (!taken[e] && rows[e] != 0)
      by_rows.push_back(static_cast<unsigned>(e));
  std::stable_sort(by_rows.begin(), by_rows.end(),
                   [&](unsigned a, unsigned b) { return rows[a] > rows[b]; });
  for (size_t i = 0; i < by_rows.size() && out.size() < n; ++i)
    out.push_back(by_rows[i]);
  return out;
}
} // namespace causallm
#endif
