// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   unittest_pelang_bicubic_resize.cpp
 * @date   18 Aug 2026
 * @brief  Bit-exactness test for pelangResizeImageFloat() (G7) against a
 *         frozen PIL Image.BICUBIC reference.
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * kSrcPixels / kPilBicubicRef were generated once via:
 *   python3 -c "
 *   from PIL import Image; import numpy as np
 *   np.random.seed(123)
 *   arr = (np.random.rand(4, 5, 3) * 255).astype(np.uint8)
 *   img = Image.fromarray(arr, mode='RGB')
 *   out = np.asarray(img.resize((7, 5), Image.BICUBIC), dtype=np.uint8)
 *   ..."
 * and frozen here so this test has no runtime Python/PIL dependency. A
 * larger, file-based bit-exactness sweep (multiple shapes, including an
 * exact-size no-resize path and a degenerate 1x1 source) was run manually
 * against real PIL during development -- see PELANG_NEXT_AGENT_HANDOFF.md.
 * This test is the checked-in, dependency-free regression guard for that.
 */

#include <pelang_bicubic_resize.h>

#include <gtest/gtest.h>

#include <vector>

namespace {

// src 5x4 (WxH), interleaved HWC uint8.
static const unsigned char kSrcPixels[] = {
  177, 72,  57,  140, 183, 107, 250, 174, 122, 99,  87,  185,
  111, 15,  101, 188, 46,  44,  135, 135, 161, 216, 184, 155,
  184, 82,  92,  58,  74,  160, 23,  110, 109, 125, 108, 79,
  108, 227, 240, 127, 159, 29,  80,  105, 220, 63,  123, 251,
  132, 156, 30,  210, 153, 138, 87,  77,  106, 173, 223, 130,
};

// PIL `Image.fromarray(kSrcPixels, "RGB").resize((7, 5), Image.BICUBIC)`,
// interleaved HWC uint8.
static const unsigned char kPilBicubicRef[] = {
  178, 67,  55,  149, 137, 83,  169, 193, 109, 251, 174, 121,
  135, 113, 179, 93,  49,  153, 115, 8,   93,  200, 43,  36,
  157, 104, 105, 158, 167, 157, 233, 178, 140, 190, 105, 125,
  118, 62,  129, 66,  54,  140, 102, 73,  64,  116, 92,  95,
  137, 144, 155, 154, 211, 206, 169, 152, 79,  118, 103, 105,
  54,  84,  208, 16,  118, 157, 80,  113, 91,  135, 144, 96,
  129, 209, 217, 116, 161, 81,  109, 137, 102, 107, 140, 208,
  61,  121, 255, 97,  144, 122, 160, 161, 37,  214, 150, 134,
  112, 81,  117, 117, 136, 115, 182, 238, 127,
};

} // namespace

TEST(PelangBicubicResize, MatchesPilBicubicPixelForPixel) {
  constexpr int kSrcW = 5, kSrcH = 4, kDstW = 7, kDstH = 5, kChannels = 3;
  ASSERT_EQ(sizeof(kSrcPixels), static_cast<size_t>(kSrcW * kSrcH * kChannels));
  ASSERT_EQ(sizeof(kPilBicubicRef),
           static_cast<size_t>(kDstW * kDstH * kChannels));

  std::vector<float> out;
  causallm::pelangResizeImageFloat(kSrcPixels, kSrcW, kSrcH, kChannels, kDstW,
                                   kDstH, out);

  ASSERT_EQ(out.size(), sizeof(kPilBicubicRef));
  for (size_t i = 0; i < out.size(); ++i) {
    // pelangResizeImageFloat's clip8() rounds to a whole uint8 value already;
    // the +/-0.5f window just tolerates float representation, not rounding.
    EXPECT_NEAR(out[i], static_cast<float>(kPilBicubicRef[i]), 0.5f)
      << "pixel index " << i;
  }
}

TEST(PelangBicubicResize, IdentitySizeIsAPixelForPixelCopy) {
  // pelangLoadAndPreprocessImage's caller takes the "no resize" branch when
  // src size == target size; pelangResizeImageFloat itself is only ever
  // called with mismatched sizes in that path, so this exercises the
  // "1-tap, weight=1.0" degenerate case of the coefficient math instead
  // (upscaling a 1x1 source, where every output pixel must equal the single
  // source pixel exactly -- no filter can introduce ringing/error there).
  constexpr int kChannels = 3;
  const unsigned char src[kChannels] = {42, 200, 7};

  std::vector<float> out;
  causallm::pelangResizeImageFloat(src, 1, 1, kChannels, 4, 4, out);

  ASSERT_EQ(out.size(), static_cast<size_t>(4 * 4 * kChannels));
  for (int i = 0; i < 4 * 4; ++i) {
    EXPECT_FLOAT_EQ(out[i * kChannels + 0], 42.f);
    EXPECT_FLOAT_EQ(out[i * kChannels + 1], 200.f);
    EXPECT_FLOAT_EQ(out[i * kChannels + 2], 7.f);
  }
}
