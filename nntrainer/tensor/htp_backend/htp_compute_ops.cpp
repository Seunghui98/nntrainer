// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   htp_compute_ops.cpp
 * @date   18 Jun 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  HTP (Hexagon/HMX) ComputeOps entry point.
 *
 * Compiled only when ENABLE_HEXKL is defined.
 *
 * HtpComputeOps overrides exactly the ops it accelerates and leaves
 * everything else to the ComputeOps base (default-throw) via inheritance --
 * mirrors ClComputeOps (nntrainer/tensor/cl_operations/cl_compute_ops.cpp),
 * which is the template this was copied from, not a design invented here.
 *
 * gemm_q4_0_accel_fp32 is the first kernel: one FastRPC call per Q4_0 FC
 * dot(), single weight, single activation -- hexkl_mm_u8i4_layer_run with
 * n_handles=1 underneath. It is Tier 1 of docs/htp_attention/
 * 40_moe_ffn_htp_task.md section 3 and benefits every FC layer under
 * engine=htp, not just the LFM2 MoE FFN that motivated this task.
 */

#ifdef ENABLE_HEXKL

#include <compute_ops.h>
#include <htp_backend.h>
#include <htp_q4_0_convert.h>

#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <nntr_hvx.h>

namespace nntrainer {

class HtpComputeOps : public ComputeOps {
public:
  bool supports_gemm_q4_0_accel_fp32() const override { return true; }

  // decode (M == 1) is the shape this kernel exists for, not an edge case
  // to avoid: the M=1 padding tax is ~40us against 113us of DSP-only work
  // (156us at M=64 -- 64x the rows for 1.38x the cost, same 64-row-wide
  // accumulator either way), so the GEMV-instead-of-HMX alternative is
  // already rejected -- see docs/htp_attention/34_fc_measured.md section5.1
  // and 41_moe_ffn_e2e_and_perf_task.md sectionB3.
  bool accelerates_q4_0_at_m1() const override { return true; }

  // matAdata: Q4_0x4-repacked weight bytes, identity-cached across calls --
  // the same pointer for the lifetime of a loaded model (inference does not
  // move weight tensors), which is what makes registering once and keying
  // the HexKL handle off it safe.
  void gemm_q4_0_accel_fp32(void *matAdata, float *matBdata, float *matCdata,
                            unsigned int M, unsigned int N,
                            unsigned int K) override {
    const remote_handle64 session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    const uint32_t handles[1] = {get_or_register(matAdata, session, K, N)};

    const int act_len = static_cast<int>(M) * static_cast<int>(K);
    const int out_len = static_cast<int>(M) * static_cast<int>(N);
    const int err = nntr_hvx_mm_u8i4_layer(session, M, K, handles, 1, matBdata,
                                           act_len, matCdata, out_len);
    if (err != AEE_SUCCESS) {
      throw std::runtime_error("nntr_hvx_mm_u8i4_layer failed: err=" +
                               std::to_string(err));
    }
  }

private:
  uint32_t get_or_register(void *matAdata, remote_handle64 session, uint32_t K,
                           uint32_t N) {
    std::lock_guard<std::mutex> lock(handle_mutex_);
    auto it = handle_cache_.find(matAdata);
    if (it != handle_cache_.end())
      return it->second;

    std::vector<int8_t> q_w4_i8(static_cast<size_t>(K) * N);
    std::vector<float> w_scale(N);
    std::vector<int32_t> colsum_w(N);
    std::vector<float> bias(N, 0.0f); // Q4_0 FC weights carry no bias tensor
    htp_qs4cx_from_q4_0x4(matAdata, K, N, q_w4_i8.data(), w_scale.data(),
                          colsum_w.data());

    uint32_t handle = 0;
    const int err = nntr_hvx_weight_register_u8i4(
      session, K, N, q_w4_i8.data(), static_cast<int>(q_w4_i8.size()),
      w_scale.data(), static_cast<int>(N), colsum_w.data(), static_cast<int>(N),
      bias.data(), static_cast<int>(N), &handle);
    if (err != AEE_SUCCESS) {
      throw std::runtime_error("nntr_hvx_weight_register_u8i4 failed: err=" +
                               std::to_string(err));
    }

    // Kept resident for the process lifetime -- Stage 6 (residency, see
    // 40_moe_ffn_htp_task.md) is what has to bound this table's size and
    // add release for the full-checkpoint case; not needed yet.
    handle_cache_.emplace(matAdata, handle);
    return handle;
  }

  std::mutex handle_mutex_;
  std::unordered_map<const void *, uint32_t> handle_cache_;
};

ComputeOps *get_htp_ops() {
  static HtpComputeOps instance;
  return &instance;
}

} // namespace nntrainer

#endif // ENABLE_HEXKL
