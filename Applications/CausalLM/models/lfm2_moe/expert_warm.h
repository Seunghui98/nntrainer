// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 * @file   expert_warm.h
 * @brief  NNTR_MOE_WARM: the decode expert pool's warm-up from the
 *         prefill's routing, for both MoE layers with an expert LRU
 *         (Lfm2MoELayer, Gemma 4's, and Lfm2MoePoolLayer).
 * @see    https://github.com/nnstreamer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */
#ifndef __CAUSALLM_EXPERT_WARM_H__
#define __CAUSALLM_EXPERT_WARM_H__

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#include <compute_ops.h>
#include <expert_lru.h>
#include <expert_preload.h>

namespace causallm {

/** @brief One layer file's warm state: each layer slot's list from its
 *         last prefill call, and the LRU slots a queued warm-up holds. */
class ExpertWarm {
public:
  using Desc = nntrainer::ComputeOps::ExpertFileDesc;

  /** @param n experts per layer; by default NNTR_MOE_WARM's, read at the
   *         first use rather than at static initialization (0: off) */
  explicit ExpertWarm(size_t n = kFromEnv) : n_(n) {}

  /** @brief NNTR_MOE_WARM's experts per layer, read once (0: off). */
  static size_t perLayer() {
    static const size_t n = expertWarmPerLayer(std::getenv("NNTR_MOE_WARM"));
    return n;
  }

  /** @brief A prefill call of layer @a slot: its warm list from the
   *         routing (@a ranked, the extended top-k token by token) and
   *         each expert's rows. A no-op when off. */
  void note(int slot, const std::vector<int> &ranked,
            const std::vector<std::vector<std::pair<unsigned, float>>> &rows,
            unsigned int total_tokens) {
    if (count() == 0 || slot < 0 || total_tokens < 2)
      return;
    if (lists_.size() <= static_cast<size_t>(slot))
      lists_.resize(slot + 1);
    std::vector<size_t> n(rows.size());
    for (size_t e = 0; e < rows.size(); ++e)
      n[e] = rows[e].size();
    lists_[slot] = warmList(ranked, ranked.size() / total_tokens, n, count());
  }

  /** @brief Hands the warm-up to the backend (set_decode_pool_warm), once
   *         per process, when on. @a layers: every layer slot's experts,
   *         by expert. */
  void install(nntrainer::ComputeOps *ops, ExpertLru &lru,
               const std::vector<std::vector<Desc>> &layers) {
    if (count() == 0)
      return;
    ops->set_decode_pool_warm(
      [this, ops, &lru, &layers] { return begin(ops, lru, layers); },
      [this, &lru](const std::vector<const void *> &keys) {
        lru.unhold(held_);
        held_ = 0;
        lru.acquire(
          keys, [](ExpertLru::Key) {},
          [](ExpertLru::Key) {
            throw std::logic_error("expert warm-up overfilled the LRU");
          });
      });
  }

  /** @brief Forget an unloaded model's lists (its pool is reset too). */
  void reset() {
    lists_.clear();
    held_ = 0;
  }

private:
  static constexpr size_t kFromEnv = static_cast<size_t>(-1);

  size_t count() {
    if (n_ == kFromEnv)
      n_ = perLayer();
    return n_;
  }

  /** Room for the warm experts not resident, rank-major over the layers
   *  (a short room cuts every layer's tail, not the last layers), the
   *  slots held, the experts to read. */
  std::vector<Desc> begin(nntrainer::ComputeOps *ops, ExpertLru &lru,
                          const std::vector<std::vector<Desc>> &layers) {
    std::vector<Desc> want;
    std::vector<ExpertLru::Key> pinned;
    size_t depth = 0;
    for (const auto &w : lists_)
      depth = std::max(depth, w.size());
    for (size_t r = 0; r < depth; ++r)
      for (size_t l = 0; l < lists_.size() && l < layers.size(); ++l) {
        if (r >= lists_[l].size() || lists_[l][r] >= layers[l].size())
          continue;
        const Desc &d = layers[l][lists_[l][r]];
        if (lru.resident(d.key_gu))
          pinned.push_back(d.key_gu);
        else
          want.push_back(d);
      }
    want.resize(std::min(want.size(), lru.room(pinned)));
    auto release = [ops](ExpertLru::Key k) {
      if (!ops->release_qs4cx_wh_expert(k))
        throw std::logic_error(
          "expert warm-up evicted an expert the accelerator does not hold");
    };
    if (want.empty() || !lru.makeRoom(want.size(), pinned, release))
      return {};
    lru.hold(want.size());
    held_ = want.size();
    return want;
  }

  size_t n_;
  std::vector<std::vector<unsigned>> lists_;
  size_t held_ = 0;
};

} // namespace causallm

#endif // __CAUSALLM_EXPERT_WARM_H__
