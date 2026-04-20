// SPDX-License-Identifier: Apache-2.0
/**
 *
 * @file   llm_util.cpp
 * @brief  util functions for llm (refactored from main.cpp)
 * @date   21 August 2024
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <llm_util.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <vector>

std::vector<unsigned int> generate_multi_tokens(
  float *logits, unsigned int NUM_VOCAB, unsigned int NUM_TARGET_TOKENS,
  float repetition_penalty, unsigned int *input_ids, unsigned int NUM_INPUT_IDS,
  unsigned int *bad_words_ids, unsigned int NUM_BAD_WORDS_IDS) {

  std::vector<unsigned int> outputs;

  // apply repetition penalty
  if (repetition_penalty != 1 && input_ids != nullptr && NUM_INPUT_IDS != 0) {
    applyRepetitionPenalty(logits, input_ids, NUM_INPUT_IDS,
                           repetition_penalty);
  }

  // apply bad words penalty
  if (bad_words_ids != nullptr && NUM_BAD_WORDS_IDS != 0)
    applyBadWordsPenalty(logits, bad_words_ids, NUM_BAD_WORDS_IDS);

  // Sort and generate multiple tokens
  std::vector<std::pair<unsigned int, float>> top_indices_and_logits;
  for (unsigned int i = 0; i < NUM_VOCAB; ++i) {
    top_indices_and_logits.push_back({i, logits[i]});
  }
  sort(top_indices_and_logits.begin(), top_indices_and_logits.end(),
       [](auto &a, auto &b) { return a.second > b.second; });

  // add sampled words
  for (unsigned int i = 0; i < NUM_TARGET_TOKENS; ++i) {
    outputs.push_back(top_indices_and_logits[i].first);
  }

  return outputs;
}

void applyRepetitionPenalty(float *logits, unsigned int *input_ids,
                            unsigned int NUM_INPUT_IDS,
                            float repetition_penalty,
                            const std::vector<unsigned int> &exempt_ids) {
  for (unsigned int i = 0; i < NUM_INPUT_IDS; ++i) {
    unsigned int tok = input_ids[i];
    // Never penalise EOS-like tokens — otherwise the model can't end turns
    // once they've appeared in history (e.g. <|im_end|> in the chat template).
    if (std::find(exempt_ids.begin(), exempt_ids.end(), tok) !=
        exempt_ids.end()) {
      continue;
    }
    if (logits[tok] < 0) {
      logits[tok] *= repetition_penalty;
    } else {
      logits[tok] /= repetition_penalty;
    }
  }
}

void applyBadWordsPenalty(float *logits, unsigned int *bad_words_ids,
                          unsigned int NUM_BAD_WORDS_IDS) {
  for (unsigned int i = 0; i < NUM_BAD_WORDS_IDS; ++i) {
    logits[bad_words_ids[i]] = -INFINITY;
  }
}

/**
 * @brief Apply temperature & top-k & top-p to logits, matching the HF
 *        transformers pipeline used to tune Qwen3 / Llama / etc.
 *
 * Steps: temperature -> top-k mask -> top-p (nucleus) mask. Masked logits are
 * set to -inf so the caller's softmax naturally zeros their probability.
 *
 * @return Max remaining logit, for numerically stable softmax in the caller.
 */
float applyTKP(float *logits, int len, float temperature, unsigned int top_k,
               float top_p) {

  const float NEG_INF = -std::numeric_limits<float>::infinity();

  // 1. Temperature scaling.
  if (temperature > 1e-5f) {
    const float inv_t = 1.0f / temperature;
    for (int i = 0; i < len; ++i) {
      logits[i] *= inv_t;
    }
  }

  // 2. top-k: keep only the k largest logits, mask the rest to -inf.
  if (top_k > 0 && top_k < static_cast<unsigned int>(len)) {
    std::vector<float> scratch(logits, logits + len);
    std::nth_element(scratch.begin(), scratch.begin() + (top_k - 1),
                     scratch.end(), std::greater<float>());
    const float kth = scratch[top_k - 1];
    for (int i = 0; i < len; ++i) {
      if (logits[i] < kth) {
        logits[i] = NEG_INF;
      }
    }
  }

  // 3. top-p (nucleus): keep the smallest prefix of sorted-descending logits
  //    whose softmax cumulative probability first reaches top_p, mask rest.
  if (top_p > 0.0f && top_p < 1.0f) {
    std::vector<std::pair<float, int>> pairs;
    pairs.reserve(len);
    for (int i = 0; i < len; ++i) {
      pairs.emplace_back(logits[i], i);
    }
    std::sort(pairs.begin(), pairs.end(),
              [](const std::pair<float, int> &a,
                 const std::pair<float, int> &b) {
                return a.first > b.first;
              });

    float mx = pairs[0].first;
    double sum = 0.0;
    std::vector<double> probs(len);
    for (int i = 0; i < len; ++i) {
      probs[i] = pairs[i].first == NEG_INF ? 0.0
                                           : std::exp(pairs[i].first - mx);
      sum += probs[i];
    }
    if (sum > 0.0) {
      double cum = 0.0;
      int cutoff = len;
      for (int i = 0; i < len; ++i) {
        cum += probs[i] / sum;
        if (cum >= top_p) {
          cutoff = i + 1;
          break;
        }
      }
      for (int j = cutoff; j < len; ++j) {
        logits[pairs[j].second] = NEG_INF;
      }
    }
  }

  // 4. Return max of the remaining logits for stable softmax in the caller.
  float mx = NEG_INF;
  for (int i = 0; i < len; ++i) {
    if (logits[i] > mx) {
      mx = logits[i];
    }
  }
  return mx;
}
