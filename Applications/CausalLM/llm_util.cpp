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
  std::partial_sort(top_indices_and_logits.begin(),
                    top_indices_and_logits.begin() + NUM_TARGET_TOKENS,
                    top_indices_and_logits.end(),
                    [](auto &a, auto &b) { return a.second > b.second; });

  // add sampled words
  for (unsigned int i = 0; i < NUM_TARGET_TOKENS; ++i) {
    outputs.push_back(top_indices_and_logits[i].first);
  }

  return outputs;
}

void applyRepetitionPenalty(float *logits, unsigned int *input_ids,
                            unsigned int NUM_INPUT_IDS,
                            float repetition_penalty) {
  for (unsigned int i = 0; i < NUM_INPUT_IDS; ++i) {
    if (logits[input_ids[i]] < 0) {
      logits[input_ids[i]] *= repetition_penalty;
    } else {
      logits[input_ids[i]] /= repetition_penalty;
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
 * @brief Apply temperature & top-k & top-p to logits
 * @details Applies temperature scaling, then top-k filtering, then top-p
 *          (nucleus) filtering. Non-selected logits are set to -INFINITY so
 *          the subsequent softmax+sampling is restricted to the chosen tokens.
 * @return Max logit of the surviving tokens (for numerically stable softmax)
 */
float applyTKP(float *logits, int len, float temperature, unsigned int top_k,
               float top_p) {

  // 1. Apply temperature scaling
  if (temperature > 1e-5) {
    for (int i = 0; i < len; ++i)
      logits[i] /= temperature;
  }

  // 2. Clamp top_k to valid range
  if (top_k == 0 || top_k > static_cast<unsigned int>(len))
    top_k = static_cast<unsigned int>(len);

  // 3. Build index-logit pairs and partial-sort to find top-k
  std::vector<std::pair<int, float>> pairs;
  pairs.reserve(len);
  for (int i = 0; i < len; ++i)
    pairs.push_back({i, logits[i]});

  std::partial_sort(pairs.begin(), pairs.begin() + top_k, pairs.end(),
                    [](const auto &a, const auto &b) {
                      return a.second > b.second;
                    });

  // 4. Apply top-p (nucleus) filtering within the top-k candidates.
  //    Convert to probabilities (softmax over top-k) then accumulate.
  float max_logit = pairs[0].second;
  float sum_exp = 0.0f;
  for (unsigned int i = 0; i < top_k; ++i)
    sum_exp += std::exp(pairs[i].second - max_logit);

  float cum_prob = 0.0f;
  unsigned int nucleus_size = 0;
  for (unsigned int i = 0; i < top_k; ++i) {
    cum_prob += std::exp(pairs[i].second - max_logit) / sum_exp;
    ++nucleus_size;
    if (cum_prob >= top_p)
      break;
  }

  // 5. Mask all logits outside the nucleus to -INFINITY
  std::fill(logits, logits + len, -std::numeric_limits<float>::infinity());
  for (unsigned int i = 0; i < nucleus_size; ++i)
    logits[pairs[i].first] = pairs[i].second;

  return max_logit;
}
