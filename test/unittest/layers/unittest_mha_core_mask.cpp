// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   unittest_mha_core_mask.cpp
 * @date   09 June 2026
 * @brief  Unit test for MHACoreLayer::add_mask_and_softmax_full (DDTree
 *         additive-mask, non-causal full-softmax attention path).
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <climits>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <vector>

#include <mha_core.h>
#include <tensor.h>
#include <tensor_dim.h>

using nntrainer::Tensor;

// Reference: out = softmax(score + mask) over keys [0, to), full
// (non-triangular) per (query, head). qk layout: rows = step_size*to, width =
// num_head; (query i, key j, head h) -> qk[(i*to + j)*num_head + h]. mask
// row-major [step_size, kv_len], mask(i,j) = mask[i*kv_len + j].
TEST(MHACoreMask, MaskedSoftmaxMatchesReference) {
  const size_t step_size = 2; // 2 query rows
  const unsigned int to = 3;  // 3 keys
  const size_t num_head = 1;  // single head
  const unsigned int kv_len = to;

  Tensor qk(1, 1, static_cast<unsigned int>(step_size * to),
            static_cast<unsigned int>(num_head));
  float *q = qk.getData<float>();
  // query 0 raw scores over keys: 1,2,3 ; query 1: 0,0,0
  const float scores[2][3] = {{1.0f, 2.0f, 3.0f}, {0.0f, 0.0f, 0.0f}};
  for (size_t i = 0; i < step_size; ++i)
    for (unsigned int j = 0; j < to; ++j)
      q[(i * to + j) * num_head + 0] = scores[i][j];

  Tensor mask(1, 1, static_cast<unsigned int>(step_size), kv_len);
  float *m = mask.getData<float>();
  const float NEG = -1e30f;
  // query 0 hides key 1 ; query 1 hides key 0
  const float maskv[2][3] = {{0.0f, NEG, 0.0f}, {NEG, 0.0f, 0.0f}};
  for (size_t i = 0; i < step_size; ++i)
    for (unsigned int j = 0; j < kv_len; ++j)
      m[i * kv_len + j] = maskv[i][j];

  causallm::MHACoreLayer::add_mask_and_softmax_full(qk, mask, step_size, to,
                                                    num_head);

  // query 0: softmax over {key0=1, key2=3} (key1 hidden)
  const float e1 = std::exp(1.0f), e3 = std::exp(3.0f);
  const float d0 = e1 + e3;
  EXPECT_NEAR(q[0], e1 / d0, 1e-5);
  EXPECT_NEAR(q[1], 0.0f, 1e-6);
  EXPECT_NEAR(q[2], e3 / d0, 1e-5);
  EXPECT_NEAR(q[0] + q[1] + q[2], 1.0f, 1e-5);

  // query 1: softmax over {key1, key2} equal (key0 hidden) -> 0, 0.5, 0.5
  EXPECT_NEAR(q[3], 0.0f, 1e-6);
  EXPECT_NEAR(q[4], 0.5f, 1e-5);
  EXPECT_NEAR(q[5], 0.5f, 1e-5);
  EXPECT_NEAR(q[3] + q[4] + q[5], 1.0f, 1e-5);
}

// Per-layer full/sliding verify-mask selection (gemma4). Drives the layer's
// own selectVerifyMask() (the exact predicate used inside forwarding()) plus
// its own add_mask_and_softmax_full(): a full-attention layer (window ==
// UINT_MAX) must consume the FULL mask, a sliding layer (finite window) must
// consume the SLIDING mask. We prove this by feeding the *selected* mask into
// the real softmax and checking the output equals the hand-computed masked
// softmax for the mask that layer should have used -- and differs from the
// distribution the other mask would have produced.
TEST(MHACoreMask, PerLayerFullVsSlidingSelection) {
  const size_t step_size = 1; // 1 query row keeps the math tiny
  const unsigned int to = 2;  // 2 keys
  const size_t num_head = 1;
  const unsigned int kv_len = to;
  const float NEG = -1e30f;

  // Two DISTINCT global masks. Full hides key1; sliding hides key0. With equal
  // raw scores this forces visibly different softmax outputs, so consuming the
  // wrong mask cannot accidentally pass.
  Tensor fullMask(1, 1, static_cast<unsigned int>(step_size), kv_len);
  Tensor slidingMask(1, 1, static_cast<unsigned int>(step_size), kv_len);
  fullMask.getData<float>()[0] = 0.0f;    // key0 visible
  fullMask.getData<float>()[1] = NEG;     // key1 hidden
  slidingMask.getData<float>()[0] = NEG;  // key0 hidden
  slidingMask.getData<float>()[1] = 0.0f; // key1 visible

  // --- selection: full-attention layer (window == kNoSlidingWindow) -> full.
  // This locks the sentinel semantics: only the named constant selects full.
  // ---
  Tensor *sel_full = causallm::MHACoreLayer::selectVerifyMask(
    &fullMask, &slidingMask, causallm::kNoSlidingWindow);
  EXPECT_EQ(sel_full, &fullMask);

  // --- selection: sliding layer (finite window, e.g. 2) -> slidingMask ---
  Tensor *sel_sliding =
    causallm::MHACoreLayer::selectVerifyMask(&fullMask, &slidingMask, 2);
  EXPECT_EQ(sel_sliding, &slidingMask);

  // A large finite window is still a sliding layer (only kNoSlidingWindow is
  // "full"); it must select the sliding mask.
  EXPECT_EQ(
    causallm::MHACoreLayer::selectVerifyMask(&fullMask, &slidingMask, 4096),
    &slidingMask);

  // Full-attention-only model (no sliding mask): every layer uses full.
  EXPECT_EQ(causallm::MHACoreLayer::selectVerifyMask(&fullMask, nullptr, 2),
            &fullMask);
  // No verify pass active -> nullptr.
  EXPECT_EQ(causallm::MHACoreLayer::selectVerifyMask(nullptr, &slidingMask, 2),
            nullptr);

  // Equal raw scores so the only thing that differs is the applied mask.
  auto make_scores = []() {
    Tensor qk(1, 1, static_cast<unsigned int>(step_size * to),
              static_cast<unsigned int>(num_head));
    qk.getData<float>()[0] = 1.0f; // key0 score
    qk.getData<float>()[1] = 1.0f; // key1 score
    return qk;
  };

  // Full layer applies its selected (full) mask -> key1 hidden -> [1, 0].
  Tensor qk_full = make_scores();
  causallm::MHACoreLayer::add_mask_and_softmax_full(qk_full, *sel_full,
                                                    step_size, to, num_head);
  EXPECT_NEAR(qk_full.getData<float>()[0], 1.0f, 1e-6);
  EXPECT_NEAR(qk_full.getData<float>()[1], 0.0f, 1e-6);

  // Sliding layer applies its selected (sliding) mask -> key0 hidden -> [0, 1].
  Tensor qk_sliding = make_scores();
  causallm::MHACoreLayer::add_mask_and_softmax_full(qk_sliding, *sel_sliding,
                                                    step_size, to, num_head);
  EXPECT_NEAR(qk_sliding.getData<float>()[0], 0.0f, 1e-6);
  EXPECT_NEAR(qk_sliding.getData<float>()[1], 1.0f, 1e-6);

  // The two layers genuinely produced different attention -> selection matters.
  EXPECT_GT(
    std::abs(qk_full.getData<float>()[0] - qk_sliding.getData<float>()[0]),
    0.5f);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
