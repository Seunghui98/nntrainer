// SPDX-License-Identifier: Apache-2.0
/**
 * @file htp_decode_hook.h
 * @brief Bind fused CausalLM layers to the native HTP decode hooks.
 * @author NNTrainer contributors
 * @bug See the project issue tracker.
 */
#pragma once
#include <cstdlib>

#include <nntr_htp_decode_hook.h>
namespace causallm {
using nntrainer::htpDecodeAdd;
using nntrainer::htpDecodeAttn;
using nntrainer::htpDecodeBan;
using nntrainer::htpDecodeConvGate;
using nntrainer::htpDecodeHasTokenId;
using nntrainer::htpDecodeKvAmaxFloor;
using nntrainer::htpDecodeKvSeed;
using nntrainer::htpDecodeKvSeedF16;
using nntrainer::htpDecodeKvSeedKvq;
using nntrainer::htpDecodeKvSeedKvqTiles;
using nntrainer::htpDecodeKvWindow;
using nntrainer::htpDecodeLmHead;
using nntrainer::htpDecodeOp;
using nntrainer::htpDecodeQkNorm;
using nntrainer::htpDecodeRmsNorm;
using nntrainer::htpDecodeRouter;
using nntrainer::htpDecodeRowResident;
using nntrainer::htpDecodeTokenId;
using nntrainer::htpDecodeWantLogits;

/**
 * @brief [C] NNTR_HTP_KV_ADOPT=1: each HTP attention layer keeps its int8
 *        prefill cache past its last prefill call -- trimmed to the masters,
 *        its CPU rows freed -- and seeds its decode cache from it. Implies
 *        NNTR_HTP_Q2_KDIM (K's prefill scale per (head, dim)). Read once.
 */
inline bool htpKvAdopt() {
  static const bool on = [] {
    const char *v = std::getenv("NNTR_HTP_KV_ADOPT");
    return v != nullptr && v[0] == '1' && v[1] == '\0';
  }();
  return on;
}

/** @brief [C] The absolute position the prompt being prefilled ends at, set
 *  by the model around its prefill (0: none, or a handoff the model cannot
 *  take: KV-shared layers, a KV snapshot to save): a layer's call ending
 *  there is its last. */
inline unsigned &htpPrefillEnd() {
  static unsigned end = 0;
  return end;
}
} // namespace causallm
