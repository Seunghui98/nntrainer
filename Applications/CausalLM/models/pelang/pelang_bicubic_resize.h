// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   pelang_bicubic_resize.h
 * @date   18 Aug 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  PE-Lang-L14-448 image preprocessing (G7 — plan §1.5, §3.5):
 *         Pillow-BICUBIC-bit-exact resize + [-1,1] normalization.
 *
 * PE-Lang uses Pillow BICUBIC (a=-0.5, support=2.0) resize, NOT the BILINEAR
 * path SigLIP2 reproduces (siglip2_vision_encoder.cpp). This mirrors that
 * implementation's structure exactly (same separable horizontal/vertical
 * passes, same PIL `Resample.c` PRECISION_BITS=22 fixed-point accumulation +
 * clip8 rounding) with only the filter kernel and its support radius swapped
 * for the bicubic (a=-0.5) convolution kernel PIL uses for `Image.BICUBIC`.
 * It is intentionally NOT shared with SigLIP2's copy: each vision-encoder
 * model owns its preprocessing so a future change to one filter can't
 * regress the other (same reasoning as keeping `pe_rope` separate from
 * mha_core's RoPE path instead of extending it).
 *
 * Split into its own header (rather than living inline in
 * pelang_vision_encoder.cpp) so it can be exercised directly by
 * test/unittest/pelang/unittest_pelang_bicubic_resize.cpp, which diffs it
 * pixel-for-pixel against Pillow's actual `Image.BICUBIC` -- see that file
 * for how to regenerate the comparison fixtures.
 */

#ifndef __PELANG_BICUBIC_RESIZE_H__
#define __PELANG_BICUBIC_RESIZE_H__

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace causallm {

/**
 * @brief PIL's bicubic (a=-0.5) convolution kernel (Resample.c
 *        `bicubic_filter`). Support radius is 2.0.
 */
inline double pelangBicubicFilter(double x) {
  constexpr double a = -0.5;
  if (x < 0.0)
    x = -x;
  if (x < 1.0)
    return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
  if (x < 2.0)
    return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
  return 0.0;
}

/**
 * @brief Precompute Pillow's resample coefficients for one axis, BICUBIC
 *        variant. Identical structure to
 *        siglip2PrecomputeCoeffs()/ImagingPrecomputeCoeffs (Resample.c),
 *        differing only in filter() and support (2.0 instead of BILINEAR's
 *        1.0):
 *   scale       = in_size / out_size
 *   filterscale = max(scale, 1)
 *   support     = 2.0 * filterscale            (BICUBIC support = 2.0)
 *   center      = (xx + 0.5) * scale
 *   xmin        = max(0, floor(center - support))
 *   xmax        = min(in_size, ceil(center + support)) - xmin
 *   weight[k]   = bicubic_filter((xmin + k + 0.5 - center) / filterscale)
 *   weights normalized to sum to 1.
 *
 * @param in_size  source dimension
 * @param out_size destination dimension
 * @param bounds   out: 2*out_size ints (xmin, xmax per output pixel)
 * @param coeffs   out: out_size * ksize doubles (row-major)
 * @return ksize   max number of source samples contributing per output pixel
 */
inline int pelangPrecomputeCoeffs(int in_size, int out_size,
                                  std::vector<int> &bounds,
                                  std::vector<double> &coeffs) {
  const double scale =
    static_cast<double>(in_size) / static_cast<double>(out_size);
  const double filterscale = scale < 1.0 ? 1.0 : scale;
  const double support = 2.0 * filterscale; // BICUBIC filter support == 2.0
  const int ksize = static_cast<int>(std::ceil(support)) * 2 + 1;

  bounds.assign(static_cast<size_t>(out_size) * 2, 0);
  coeffs.assign(static_cast<size_t>(out_size) * ksize, 0.0);

  for (int xx = 0; xx < out_size; ++xx) {
    const double center = (xx + 0.5) * scale;
    const double ss = 1.0 / filterscale;
    int xmin = static_cast<int>(center - support + 0.5);
    if (xmin < 0)
      xmin = 0;
    int xmax = static_cast<int>(center + support + 0.5);
    if (xmax > in_size)
      xmax = in_size;
    xmax -= xmin;

    double *k = &coeffs[static_cast<size_t>(xx) * ksize];
    double wsum = 0.0;
    int x = 0;
    for (; x < xmax; ++x) {
      double w = pelangBicubicFilter((x + xmin - center + 0.5) * ss);
      k[x] = w;
      wsum += w;
    }
    for (x = 0; x < xmax; ++x) {
      if (wsum != 0.0)
        k[x] /= wsum;
    }
    for (; x < ksize; ++x)
      k[x] = 0.0;

    bounds[static_cast<size_t>(xx) * 2 + 0] = xmin;
    bounds[static_cast<size_t>(xx) * 2 + 1] = xmax;
  }
  return ksize;
}

/**
 * @brief Resize an image buffer reproducing Pillow's Image.BICUBIC resample
 *        via PIL's 8-bit fixed-point separable pipeline (horizontal pass
 *        then vertical pass, each accumulating int64 products of quantized
 *        PRECISION_BITS=22 coefficients and rounding back to uint8 range via
 *        clip8 between/after passes). Matching PIL's *integer* arithmetic
 *        (not float coefficients) matters: siglip2_vision_encoder.cpp's
 *        BILINEAR port found ~212/150528 pixels differing by a single uint8
 *        LSB with float coefficients, which downstream flipped a generated
 *        token -- the fixed-point path removed all pixel differences there.
 *
 * @param src Source image data (interleaved HWC, uint8).
 * @param src_w, src_h Source dimensions.
 * @param channels Number of channels (must be 3).
 * @param dst_w, dst_h Destination dimensions.
 * @param dst_float Output buffer of size dst_w * dst_h * channels (float,
 *        holding the exact uint8 result values in [0,255]).
 */
inline void pelangResizeImageFloat(const unsigned char *src, int src_w,
                                   int src_h, int channels, int dst_w,
                                   int dst_h, std::vector<float> &dst_float) {
  constexpr int PRECISION_BITS = 32 - 8 - 2;
  const int64_t ROUND = static_cast<int64_t>(1) << (PRECISION_BITS - 1);

  auto quantize = [](const std::vector<double> &coeffs, int n, int ksize,
                     std::vector<int> &icoeffs) {
    icoeffs.assign(static_cast<size_t>(n) * ksize, 0);
    for (size_t i = 0; i < static_cast<size_t>(n) * ksize; ++i)
      icoeffs[i] = static_cast<int>(
        std::lround(coeffs[i] * static_cast<double>(1 << PRECISION_BITS)));
  };

  auto clip8 = [ROUND](int64_t acc) -> float {
    int64_t r = (acc + ROUND) >> PRECISION_BITS;
    if (r < 0)
      r = 0;
    if (r > 255)
      r = 255;
    return static_cast<float>(r);
  };

  // ----- Horizontal pass: [src_h, src_w] -> [src_h, dst_w] -----
  std::vector<int> h_bounds;
  std::vector<double> h_coeffs;
  const int h_ksize = pelangPrecomputeCoeffs(src_w, dst_w, h_bounds, h_coeffs);
  std::vector<int> h_icoeffs;
  quantize(h_coeffs, dst_w, h_ksize, h_icoeffs);

  std::vector<float> tmp(static_cast<size_t>(src_h) * dst_w * channels);
  for (int y = 0; y < src_h; ++y) {
    for (int x = 0; x < dst_w; ++x) {
      const int xmin = h_bounds[static_cast<size_t>(x) * 2 + 0];
      const int xmax = h_bounds[static_cast<size_t>(x) * 2 + 1];
      const int *k = &h_icoeffs[static_cast<size_t>(x) * h_ksize];
      int64_t acc[3] = {0, 0, 0};
      for (int i = 0; i < xmax; ++i) {
        const int sx = xmin + i;
        const int idx = (y * src_w + sx) * channels;
        for (int c = 0; c < channels; ++c)
          acc[c] += static_cast<int64_t>(src[idx + c]) * k[i];
      }
      const int out_idx = (y * dst_w + x) * channels;
      for (int c = 0; c < channels; ++c)
        tmp[out_idx + c] = clip8(acc[c]);
    }
  }

  // ----- Vertical pass: [src_h, dst_w] -> [dst_h, dst_w] -----
  std::vector<int> v_bounds;
  std::vector<double> v_coeffs;
  const int v_ksize = pelangPrecomputeCoeffs(src_h, dst_h, v_bounds, v_coeffs);
  std::vector<int> v_icoeffs;
  quantize(v_coeffs, dst_h, v_ksize, v_icoeffs);

  dst_float.resize(static_cast<size_t>(dst_w) * dst_h * channels);
  for (int y = 0; y < dst_h; ++y) {
    const int ymin = v_bounds[static_cast<size_t>(y) * 2 + 0];
    const int ymax = v_bounds[static_cast<size_t>(y) * 2 + 1];
    const int *k = &v_icoeffs[static_cast<size_t>(y) * v_ksize];
    for (int x = 0; x < dst_w; ++x) {
      int64_t acc[3] = {0, 0, 0};
      for (int i = 0; i < ymax; ++i) {
        const int sy = ymin + i;
        const int idx = (sy * dst_w + x) * channels;
        for (int c = 0; c < channels; ++c)
          acc[c] += static_cast<int64_t>(tmp[idx + c]) * k[i];
      }
      const int out_idx = (y * dst_w + x) * channels;
      for (int c = 0; c < channels; ++c)
        dst_float[out_idx + c] = clip8(acc[c]);
    }
  }
}

/**
 * @brief Load, BICUBIC-resize, and normalize an image into CHW float data
 *        (`(pixel/255 - 0.5) / 0.5`, plan §1.5).
 *
 * @param stbi_load_fn A callable matching stbi_load's signature, injected so
 *        this header doesn't need to pull in stb_image.h/.inc itself; pass
 *        `::stbi_load` (declared by the includer's own stb_image.inc, in
 *        pelang_vision_encoder.cpp's case) or a fake loader in tests.
 * @param stbi_free_fn A callable matching stbi_image_free's signature.
 */
template <typename LoadFn, typename FreeFn>
std::vector<float> pelangLoadAndPreprocessImage(const std::string &filepath,
                                                int target_width,
                                                int target_height,
                                                LoadFn &&stbi_load_fn,
                                                FreeFn &&stbi_free_fn) {
  int width, height, channels;
  unsigned char *image =
    stbi_load_fn(filepath.c_str(), &width, &height, &channels, 0);
  if (!image) {
    throw std::runtime_error("Failed to load image: " + filepath);
  }

  std::vector<unsigned char> rgb_buf;
  const unsigned char *rgb_ptr = image;
  int rgb_channels = channels;
  if (channels == 1) {
    rgb_buf.resize(static_cast<size_t>(width * height * 3));
    for (int i = 0; i < width * height; ++i) {
      rgb_buf[i * 3] = image[i];
      rgb_buf[i * 3 + 1] = image[i];
      rgb_buf[i * 3 + 2] = image[i];
    }
    rgb_ptr = rgb_buf.data();
    rgb_channels = 3;
  } else if (channels == 4) {
    rgb_buf.resize(static_cast<size_t>(width * height * 3));
    for (int i = 0; i < width * height; ++i) {
      rgb_buf[i * 3] = image[i * 4];
      rgb_buf[i * 3 + 1] = image[i * 4 + 1];
      rgb_buf[i * 3 + 2] = image[i * 4 + 2];
    }
    rgb_ptr = rgb_buf.data();
    rgb_channels = 3;
  } else if (channels != 3) {
    stbi_free_fn(image);
    throw std::runtime_error("Unsupported number of channels: " +
                             std::to_string(channels));
  }

  std::vector<float> resized_float;
  if (width != target_width || height != target_height) {
    pelangResizeImageFloat(rgb_ptr, width, height, rgb_channels, target_width,
                           target_height, resized_float);
  } else {
    resized_float.resize(
      static_cast<size_t>(target_width * target_height * rgb_channels));
    for (size_t i = 0; i < resized_float.size(); ++i)
      resized_float[i] = static_cast<float>(rgb_ptr[i]);
  }

  stbi_free_fn(image);

  // Convert HWC float [0,255] to CHW float [-1,1] (PE-Lang normalization,
  // mean=std=0.5 -- plan §1.5, identical formula to SigLIP2's).
  std::vector<float> output(3 * target_height * target_width);
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < target_height; ++y) {
      for (int x = 0; x < target_width; ++x) {
        float val = resized_float[(y * target_width + x) * 3 + c];
        float pixel = (val / 255.0f - 0.5f) / 0.5f;
        output[c * target_height * target_width + y * target_width + x] =
          pixel;
      }
    }
  }

  return output;
}

} // namespace causallm

#endif /* __PELANG_BICUBIC_RESIZE_H__ */
