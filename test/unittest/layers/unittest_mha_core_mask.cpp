// SPDX-License-Identifier: Apache-2.0
/**
 * @file   unittest_mha_core_mask.cpp
 * @brief  Unit test for MHACoreLayer::add_mask_and_softmax_full (DDTree
 *         additive-mask, non-causal full-softmax attention path).
 */

#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <vector>

#include <mha_core.h>
#include <tensor.h>
#include <tensor_dim.h>

using nntrainer::Tensor;

// Reference: out = softmax(score + mask) over keys [0, to), full (non-triangular)
// per (query, head). qk layout: rows = step_size*to, width = num_head;
// (query i, key j, head h) -> qk[(i*to + j)*num_head + h]. mask row-major
// [step_size, kv_len], mask(i,j) = mask[i*kv_len + j].
TEST(MHACoreMask, MaskedSoftmaxMatchesReference) {
  const size_t step_size = 2;  // 2 query rows
  const unsigned int to = 3;   // 3 keys
  const size_t num_head = 1;   // single head
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

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
