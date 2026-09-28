// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   unittest_hvx_int_epilogue.cpp
 * @date   28 Sep 2026
 * @brief  The integer MoE epilogue's HVX version is bit-identical to its
 *         portable C reference on the device (doc 53 section 9.5)
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * The comparison runs on the DSP (nntr_hvx_int_epilogue.c): the reference
 * is the same C the host checks run, compiled for Hexagon, so this test
 * has no model of its own to drift. What it asserts is that every
 * mismatch counter comes back 0, for a seed without a bias and one with.
 */

#include <gtest/gtest.h>

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <AEEStdErr.h>
#include <remote.h>

#include "nntr_hvx.h"

namespace {

std::string hex(int err) {
  std::ostringstream os;
  os << "0x" << std::hex << std::setw(8) << std::setfill('0')
     << static_cast<unsigned>(err);
  return os.str();
}

class HvxIntEpilogue : public ::testing::Test {
protected:
  void SetUp() override {
    remote_rpc_control_unsigned_module unsigned_pd = {CDSP_DOMAIN_ID, 1};
    int err = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE,
                                     &unsigned_pd, sizeof(unsigned_pd));
    ASSERT_EQ(err, AEE_SUCCESS) << "enabling unsigned PD failed: " << hex(err);
    const std::string uri = std::string(nntr_hvx_URI) + "&_dom=cdsp";
    err = nntr_hvx_open(uri.c_str(), &handle_);
    ASSERT_EQ(err, AEE_SUCCESS)
      << "nntr_hvx_open failed: " << hex(err)
      << " -- is libnntr_hvx_skel.so on ADSP_LIBRARY_PATH?";
  }
  void TearDown() override {
    if (handle_) {
      nntr_hvx_close(handle_);
    }
  }
  remote_handle64 handle_ = 0;
};

void expect_identical(remote_handle64 h, unsigned seed) {
  std::vector<unsigned> counts(8, 0xFFFFFFFFu);
  const int err = nntr_hvx_int_epilogue_selfcheck(
    h, seed, counts.data(), static_cast<int>(counts.size()));
  ASSERT_EQ(err, AEE_SUCCESS) << "selfcheck failed: " << hex(err);
  std::cout << "INT_EPILOGUE seed=" << seed << " elements=" << counts[5]
            << " batches=" << counts[7] << " checksum=0x" << std::hex
            << counts[6] << std::dec << "\n";
  EXPECT_GT(counts[5], 0u);
  EXPECT_EQ(counts[7], 4u) << "expected 4 staged batches (16,16,16,8 pairs)";
  EXPECT_EQ(counts[0], 0u) << "SwiGLU mantissas differ";
  EXPECT_EQ(counts[1], 0u) << "batch exponents differ";
  EXPECT_EQ(counts[2], 0u) << "requant bytes differ";
  EXPECT_EQ(counts[3], 0u) << "requant scales differ";
  EXPECT_EQ(counts[4], 0u) << "requant zero points differ";
}

} // namespace

TEST_F(HvxIntEpilogue, MatchesReferenceNoBias) {
  expect_identical(handle_, 2u);
}
TEST_F(HvxIntEpilogue, MatchesReferenceWithBias) {
  expect_identical(handle_, 7u);
}
TEST_F(HvxIntEpilogue, RejectsShortCounts) {
  unsigned c[4] = {0, 0, 0, 0};
  EXPECT_EQ(nntr_hvx_int_epilogue_selfcheck(handle_, 1u, c, 4), AEE_EBADPARM);
}

int main(int argc, char **argv) {
  int result = -1;
  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error during InitGoogleTest" << std::endl;
    return 0;
  }
  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error during RUN_ALL_TESTS()" << std::endl;
  }
  return result;
}
