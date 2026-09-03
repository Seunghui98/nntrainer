// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   htp_backend.cpp
 * @date   18 Jun 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 * @brief  HTP backend lifecycle implementation.
 *
 * Disabled placeholder: docs/htp_attention/40_moe_ffn_htp_task.md Stage 2
 * replaces this constructor with the real HexKL micro-API FastRPC session.
 * See htp_backend.h for why the HexKL CPU Macro API (sdkl.h / libsdkl.so)
 * session this file used to own was removed rather than kept.
 */

#ifdef ENABLE_HEXKL

#include <htp_backend.h>

namespace nntrainer {

HtpBackend &HtpBackend::global() {
  static HtpBackend instance;
  return instance;
}

} // namespace nntrainer

#endif // ENABLE_HEXKL
