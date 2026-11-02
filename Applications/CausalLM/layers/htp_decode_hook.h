// SPDX-License-Identifier: Apache-2.0
/**
 * @file htp_decode_hook.h
 * @brief Bind fused CausalLM layers to the native HTP decode hooks.
 * @author NNTrainer contributors
 * @bug See the project issue tracker.
 */
#pragma once
#include <cstdlib>
#include <cstring>

#include <nntr_htp_decode_hook.h>
namespace causallm {
using nntrainer::htpDecodeAdd;
using nntrainer::htpDecodeAttn;
using nntrainer::htpDecodeBan;
using nntrainer::htpDecodeConvGate;
using nntrainer::htpDecodeHasTokenId;
using nntrainer::htpDecodeKvAmaxFloor;
using nntrainer::htpDecodeKvPrepare;
using nntrainer::htpDecodeKvSeed;
using nntrainer::htpDecodeKvSeedF16;
using nntrainer::htpDecodeKvSeedKvq;
using nntrainer::htpDecodeKvSeedKvqLayer;
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
 * @brief [C] The KV adoption, on unless NNTR_HTP_KV_ADOPT=0: each HTP
 *        attention layer's last prefill call seeds its decode cache from
 *        its int8 prefill cache and frees its CPU rows. Implies
 *        NNTR_HTP_Q2_KDIM (K's prefill scale per (head, dim)). Read once.
 *        The default since S26 Ultra round 10 (Gemma 4 26B-A4B, p1k /
 *        p2k): the first token 436 / 501 -> 174 / 183 ms with the
 *        warm-up, the peak 20 MiB lower, the prefill within noise.
 */
inline bool htpKvAdopt() {
  static const bool on = [] {
    const char *v = std::getenv("NNTR_HTP_KV_ADOPT");
    return v == nullptr || std::strcmp(v, "0") != 0;
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

/** @brief NNTR_MOE_WARM: true while the model runs the last block of a
 *  prefill, the one the first decode row follows (set around it by the
 *  model's prefill; false elsewhere, and in any other prefill path). */
inline bool &htpPrefillLastBlock() {
  static bool last = false;
  return last;
}
} // namespace causallm
