// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   unittest_hvx_rope.cpp
 * @brief  Device tests for the HVX uint8 RoPE endpoint.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <AEEStdErr.h>
#include <remote.h>

#include "mha_htp_host_model.h"
#include "nntr_hvx.h"

namespace {

constexpr int kDspOffset = 0x80000400;

std::string hex(int err) {
  std::ostringstream os;
  os << "0x" << std::hex << std::setw(8) << std::setfill('0')
     << static_cast<unsigned>(err);
  return os.str();
}

class HtpSession : public ::testing::Test {
protected:
  void SetUp() override {
    remote_rpc_control_unsigned_module unsigned_pd = {CDSP_DOMAIN_ID, 1};
    int err = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE,
                                     &unsigned_pd, sizeof(unsigned_pd));
    ASSERT_EQ(err, AEE_SUCCESS) << "enabling unsigned PD failed: " << hex(err);

    const std::string uri = std::string(nntr_hvx_URI) + "&_dom=cdsp";
    err = nntr_hvx_open(uri.c_str(), &handle_);
    ASSERT_EQ(err, AEE_SUCCESS) << "nntr_hvx_open failed: " << hex(err);
  }

  void TearDown() override {
    if (handle_) {
      nntr_hvx_close(handle_);
    }
  }

  remote_handle64 handle_ = 0;
};

class HvxRope : public HtpSession {};

struct Case {
  uint32_t rows;
  uint32_t width;
  uint32_t dim;
};

TEST_F(HvxRope, RejectsBadLengths) {
  std::vector<uint8_t> x(64, 17), y(64);
  std::vector<float> scale(1, 1.0f);
  std::vector<int32_t> zp(1, 0);
  std::vector<int16_t> cos(32, 32767), sin(32, 0);
  uint32_t saturated = 0;

  int err = nntr_hvx_rope_u8(handle_, 1, 64, 64, x.data(), 63, scale.data(), 1,
                             zp.data(), 1, cos.data(), (int)cos.size(),
                             sin.data(), (int)sin.size(), 1.0f, 0, y.data(),
                             (int)y.size(), &saturated);
  EXPECT_EQ(err, AEE_EBADPARM + kDspOffset) << "got " << hex(err);
}

TEST_F(HvxRope, MatchesReferenceWithinOneLsb) {
  for (const Case c :
       std::vector<Case>{{1, 64, 64}, {3, 96, 64}, {2, 128, 128}}) {
    const uint32_t half = c.dim / 2u;
    std::vector<uint8_t> x((size_t)c.rows * c.width);
    std::vector<uint8_t> out(x.size()), ref(x.size());
    std::vector<float> scale(c.rows);
    std::vector<int32_t> zp(c.rows);
    std::vector<int16_t> cos((size_t)c.rows * half);
    std::vector<int16_t> sin(cos.size());

    for (uint32_t m = 0; m < c.rows; ++m) {
      scale[m] = 0.25f + 0.125f * (float)m;
      zp[m] = 17 + (int32_t)m * 23;
      for (uint32_t k = 0; k < half; ++k) {
        cos[(size_t)m * half + k] = (int16_t)(32000 - (int)(k * 73u));
        sin[(size_t)m * half + k] = (int16_t)((int)(k * 41u) - 500);
      }
    }
    for (uint32_t i = 0; i < x.size(); ++i) {
      x[i] = (uint8_t)((i * 29u + 7u) & 255u);
    }

    uint32_t ref_sat = 0;
    rope_u8_ref(x.data(), ref.data(), c.rows, c.width, c.dim, scale.data(),
                zp.data(), cos.data(), sin.data(), 0.5f, 123, &ref_sat);
    uint32_t dsp_sat = 0;
    int err = nntr_hvx_rope_u8(
      handle_, c.rows, c.width, c.dim, x.data(), (int)x.size(), scale.data(),
      (int)scale.size(), zp.data(), (int)zp.size(), cos.data(), (int)cos.size(),
      sin.data(), (int)sin.size(), 0.5f, 123, out.data(), (int)out.size(),
      &dsp_sat);
    ASSERT_EQ(err, AEE_SUCCESS) << "shape " << c.rows << "x" << c.width
                                << " dim=" << c.dim << ": " << hex(err);
    EXPECT_EQ(dsp_sat, ref_sat);

    uint32_t mismatch = 0;
    int max_err = 0;
    for (size_t i = 0; i < out.size(); ++i) {
      const int e = std::abs((int)out[i] - (int)ref[i]);
      max_err = std::max(max_err, e);
      mismatch += e != 0;
      EXPECT_LE(e, 1) << "element " << i;
    }
    std::cout << "ROPE_FIELD rows=" << c.rows << " width=" << c.width
              << " dim=" << c.dim << " max_lsb=" << max_err
              << " mismatch_ratio=" << (double)mismatch / (double)out.size()
              << " saturated=" << dsp_sat << std::endl;
  }
}

TEST_F(HvxRope, LaneOrderAndSaturationAreObservable) {
  const uint32_t rows = 1, width = 64, dim = 64, half = dim / 2u;
  std::vector<uint8_t> x(width, 200), out(width), ref(width);
  std::vector<float> scale(1, 1.0f);
  std::vector<int32_t> zp(1, 0);
  std::vector<int16_t> cos(half), sin(half, 0);
  for (uint32_t k = 0; k < half; ++k) {
    cos[k] = (int16_t)(k * 256u);
  }
  uint32_t ref_sat = 0;
  rope_u8_ref(x.data(), ref.data(), rows, width, dim, scale.data(), zp.data(),
              cos.data(), sin.data(), 1.0f, 0, &ref_sat);
  uint32_t dsp_sat = 0;
  ASSERT_EQ(nntr_hvx_rope_u8(handle_, rows, width, dim, x.data(), width,
                             scale.data(), 1, zp.data(), 1, cos.data(),
                             (int)cos.size(), sin.data(), (int)sin.size(), 1.0f,
                             0, out.data(), width, &dsp_sat),
            AEE_SUCCESS);
  EXPECT_EQ(dsp_sat, ref_sat);
  EXPECT_EQ(out, ref);
  EXPECT_EQ(out[0], 0u);
  EXPECT_EQ(out[1], 2u);

  std::fill(cos.begin(), cos.end(), 32767);
  std::fill(x.begin(), x.end(), 255);
  ASSERT_EQ(nntr_hvx_rope_u8(handle_, rows, width, dim, x.data(), width,
                             scale.data(), 1, zp.data(), 1, cos.data(),
                             (int)cos.size(), sin.data(), (int)sin.size(),
                             0.125f, 0, out.data(), width, &dsp_sat),
            AEE_SUCCESS);
  EXPECT_GT(dsp_sat, 0u);
}

} // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
