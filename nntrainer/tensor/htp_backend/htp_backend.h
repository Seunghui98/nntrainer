// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   htp_backend.h
 * @date   18 Jun 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 * @brief  HTP (Hexagon Tensor Processor) backend lifecycle.
 *
 * Process-wide singleton, always disabled today: every
 * HtpComputeOps::supports_*() returns false so callers transparently fall
 * back to the CPU path. It used to own a Qualcomm HexKL CPU Macro API
 * (sdkl.h / libsdkl.so) session, but that session had no caller left --
 * docs/htp_attention/10_mha_htp_plan.md section 6 established there is no
 * macro-API FC dispatch left on this lineage to migrate, so the session
 * existed only to print a version string. Removed instead of kept: per
 * the documented macro/micro one-way door, a macro-API session opened
 * after any HexKL micro-API FastRPC session (the kernels this backend
 * will dispatch to) fails permanently, so an unused macro session was a
 * standing hazard, not a convenience.
 *
 * docs/htp_attention/40_moe_ffn_htp_task.md Stage 2 replaces this
 * constructor with the real HexKL micro-API FastRPC session (nntr_hvx_open
 * against the skel PR #4256 already ships and device-verified).
 *
 * Compiled only when ENABLE_HEXKL is defined (meson: -Denable-htp=true).
 */

#ifndef __HTP_BACKEND_H__
#define __HTP_BACKEND_H__
#ifdef __cplusplus
#ifdef ENABLE_HEXKL

namespace nntrainer {

/**
 * @class HtpBackend
 * @brief Process-wide owner of the HTP (HexKL micro-API/FastRPC) session.
 */
class HtpBackend {
public:
  /**
   * @brief Access the process-wide singleton.
   */
  static HtpBackend &global();

  /**
   * @brief Whether the HTP session is initialized and usable. Always
   *        false until Stage 2 wires the real FastRPC session; when
   *        false, all HTP ops must defer to the CPU fallback.
   */
  bool enabled() const { return enabled_; }

  ~HtpBackend() = default;

  HtpBackend(const HtpBackend &) = delete;
  HtpBackend &operator=(const HtpBackend &) = delete;

private:
  HtpBackend() = default;

  bool enabled_ = false;
};

} // namespace nntrainer

#endif // ENABLE_HEXKL
#endif // __cplusplus
#endif // __HTP_BACKEND_H__
