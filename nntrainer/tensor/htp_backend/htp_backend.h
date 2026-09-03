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
 * Process-wide singleton owning the one HexKL micro-API FastRPC session a
 * process opens (nntr_hvx_open against the skel PR #4256 device-verified --
 * test/htp/nntr_hvx.idl is the interface, and this file's meson.build
 * generates the same client stub from it that test/htp's gtests use).
 * If the skel isn't reachable (missing from ADSP_LIBRARY_PATH, no device,
 * driver error, ...) construction leaves the backend DISABLED and every
 * HtpComputeOps::supports_*() reports false, so callers transparently fall
 * back to the CPU path.
 *
 * This used to own a Qualcomm HexKL CPU Macro API (sdkl.h / libsdkl.so)
 * session instead. That session had no caller left -- docs/htp_attention/
 * 10_mha_htp_plan.md section 6 established there is no macro-API FC
 * dispatch left on this lineage to migrate, so it existed only to print a
 * version string -- and removing it was not optional: per the documented
 * macro/micro one-way door, a macro-API session opened after any HexKL
 * micro-API FastRPC session (this file's session, now) fails permanently.
 *
 * Compiled only when ENABLE_HEXKL is defined (meson: -Denable-htp=true).
 */

#ifndef __HTP_BACKEND_H__
#define __HTP_BACKEND_H__
#ifdef __cplusplus
#ifdef ENABLE_HEXKL

#include <cstdint>

namespace nntrainer {

/**
 * @class HtpBackend
 * @brief Process-wide owner of the HTP (HexKL micro-API/FastRPC) session.
 */
class HtpBackend {
public:
  /**
   * @brief Access the process-wide singleton. The first call attempts
   *        nntr_hvx_open() exactly once (thread-safe).
   */
  static HtpBackend &global();

  /**
   * @brief Whether the HTP session is initialized and usable. When false,
   *        all HTP ops must defer to the CPU fallback.
   */
  bool enabled() const { return enabled_; }

  /**
   * @brief The FastRPC session handle every nntr_hvx_* call dispatches
   *        through (HtpComputeOps, from the first accelerated kernel
   *        onward). Only meaningful when enabled() is true.
   */
  uint64_t handle() const { return handle_; }

  ~HtpBackend();

  HtpBackend(const HtpBackend &) = delete;
  HtpBackend &operator=(const HtpBackend &) = delete;

private:
  HtpBackend();

  bool enabled_ = false;
  uint64_t handle_ = 0; ///< remote_handle64 from nntr_hvx_open; opaque here
                        ///< so this header does not need <remote.h>.
};

} // namespace nntrainer

#endif // ENABLE_HEXKL
#endif // __cplusplus
#endif // __HTP_BACKEND_H__
