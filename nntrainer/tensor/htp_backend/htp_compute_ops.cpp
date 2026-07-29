// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   htp_compute_ops.cpp
 * @date   18 Jun 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 * @brief  HTP (Hexagon/HMX) ComputeOps subclass — NPU dispatch entry point.
 *
 * Parallels ClComputeOps: overrides only the accelerated ops and their
 * supports_*() predicates; everything else falls through to the base
 * ComputeOps defaults. Actual HexKL calls live in hmx_ops/hexkl_mm.
 *
 * Compiled only when ENABLE_HEXKL is defined.
 *
 * This commit adds the table itself and its registration. The accelerated
 * overrides land with the kernels they dispatch to; until then every op
 * falls through to the CPU implementation, so an engine="htp" model runs
 * correctly, just without acceleration.
 */

#ifdef ENABLE_HEXKL

#include <compute_ops.h>
#include <cpu_ops_table.h>
#include <htp_backend.h>

namespace nntrainer {

/**
 * @brief ComputeOps subclass for delegating operations to the HTP NPU backend.
 */
class HtpComputeOps : public CpuComputeOps {};

ComputeOps *get_htp_ops() {
  static HtpComputeOps instance;
  return &instance;
}

} // namespace nntrainer

#endif // ENABLE_HEXKL
