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
 * HtpComputeOps overrides exactly the ops it accelerates and inherits the
 * CPU implementation of everything else from CpuComputeOps.
 *
 * It deliberately does NOT derive from the abstract ComputeOps base the way
 * ClComputeOps does. A layer's engine covers every tensor in it, not just
 * the ones this backend has a kernel for: Lfm2MoELayer's router gate is
 * FP32 by construction (lfm2_moe_layer.cpp keeps gate and expert_bias
 * unquantized), so with engine=htp its dot() reaches sgemm_fp32, which has
 * no supports_* guard and no fallback of its own in FloatTensor::dot. Off
 * the abstract base that threw "ComputeOps::sgemm_fp32 not implemented by
 * this backend" on the first prefill. CpuComputeOps is stateless, so
 * inheriting it costs nothing and makes every unaccelerated op behave the
 * way it does with engine=cpu.
 *
 * gemm_q4_0_accel_fp32 is the first kernel: one FastRPC call per Q4_0 FC
 * dot(), single weight, single activation -- hexkl_mm_u8i4_layer_run with
 * n_handles=1 underneath. It is Tier 1 of docs/htp_attention/
 * 40_moe_ffn_htp_task.md section 3 and benefits every FC layer under
 * engine=htp, not just the LFM2 MoE FFN that motivated this task.
 */

#ifdef ENABLE_HEXKL

#include <compute_ops.h>
#include <cpu_ops_table.h>
#include <htp_backend.h>
#include <htp_q4_0_convert.h>
#include <htp_rpcmem.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <nntr_hvx.h>

namespace nntrainer {

namespace {

/**
 * @brief Slots mm_u8i4_layer_timed fills.
 *
 * Restated here because the ARM side cannot include the DSP source that
 * declares them (nntr_hvx_mm_u8i4.c). The entry point rejects a stale count
 * with AEE_EBADPARM rather than truncating, so a drift shows up as a failed
 * call, not as wrong numbers.
 */
enum {
  HTP_T_DSP_TOTAL = 0,
  HTP_T_QUANT,
  HTP_T_DEQUANT,
  HTP_T_ACC_READ,
  HTP_T_ACC_COPY,
  HTP_T_DRAIN,
  HTP_T_ACC_STRIDE,
  HTP_N_STAGES
};

/**
 * @brief Per-stage timing for the HTP path. Off unless NNTR_HTP_PROFILE is set.
 *
 * NNTR_HTP_PROFILE=1 times the three host-side stages a weight goes through:
 * the Q4_0/QS4CX -> HexKL-registry conversion, the weight_register FastRPC,
 * and the per-dot() layer call. =2 additionally routes the layer call through
 * mm_u8i4_layer_timed, so the DSP reports its own microseconds and the
 * FastRPC transport share becomes host_us - dsp_us rather than a guess
 * (mobile_e2e_run_guide.md section 6, evidence step 2).
 *
 * Registration is accounted separately from the layer call on purpose: it
 * runs once per weight pointer and therefore lands entirely in the first
 * forward pass, so "prefill got slower but decode did not" and "every call
 * got slower" are different findings and have to be separable.
 */
class HtpProfile {
public:
  static HtpProfile &global() {
    static HtpProfile instance;
    return instance;
  }

  int level() const { return level_; }

  static uint64_t nowUs() {
    return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch())
        .count());
  }

  void addRegister(uint64_t total_us, uint64_t convert_us, uint64_t rpc_us,
                   bool ion) {
    std::lock_guard<std::mutex> lock(mutex_);
    ++reg_calls_;
    reg_total_us_ += total_us;
    convert_us_ += convert_us;
    rpc_us_ += rpc_us;
    if (ion)
      ++reg_ion_calls_;
  }

  void addInvoke(unsigned M, unsigned K, unsigned N, uint64_t host_us,
                 const uint32_t *stage_us) {
    std::lock_guard<std::mutex> lock(mutex_);
    Bucket &b = buckets_[std::make_tuple(K, N, M == 1)];
    ++b.calls;
    b.rows += M;
    b.host_us += host_us;
    if (stage_us != nullptr) {
      b.dsp_us += stage_us[HTP_T_DSP_TOTAL];
      b.quant_us += stage_us[HTP_T_QUANT];
      b.dequant_us += stage_us[HTP_T_DEQUANT];
      b.acc_us += stage_us[HTP_T_ACC_READ] + stage_us[HTP_T_ACC_COPY];
      b.drain_us += stage_us[HTP_T_DRAIN];
    }
  }

  ~HtpProfile() {
    if (level_ != 0)
      dump();
  }

private:
  struct Bucket {
    uint64_t calls = 0;
    uint64_t rows = 0; /**< summed M, so prefill batching is visible */
    uint64_t host_us = 0;
    uint64_t dsp_us = 0;
    uint64_t quant_us = 0;
    uint64_t dequant_us = 0;
    uint64_t acc_us = 0;
    uint64_t drain_us = 0;
  };

  HtpProfile() {
    const char *env = std::getenv("NNTR_HTP_PROFILE");
    level_ = (env != nullptr) ? std::atoi(env) : 0;
  }

  static double ms(uint64_t us) { return static_cast<double>(us) / 1000.0; }

  void dump() {
    // stderr, not the logger: this runs at static destruction, and the run
    // is driven from an adb shell where stderr is what the operator sees.
    std::fprintf(stderr, "\n[HTP-PROFILE] level=%d\n", level_);

    uint64_t invoke_us = 0;
    for (const auto &entry : buckets_)
      invoke_us += entry.second.host_us;

    std::fprintf(stderr,
                 "[HTP-PROFILE] weight registration (once per weight, so all "
                 "of it lands in the first forward)\n");
    std::fprintf(stderr,
                 "[HTP-PROFILE]   weights registered : %llu\n"
                 "[HTP-PROFILE]   convert to registry: %10.1f ms  (%.2f "
                 "ms/weight)\n"
                 "[HTP-PROFILE]   register FastRPC   : %10.1f ms  (%.2f "
                 "ms/weight)\n"
                 "[HTP-PROFILE]   alloc + other      : %10.1f ms\n"
                 "[HTP-PROFILE]   registration total : %10.1f ms\n"
                 "[HTP-PROFILE]   rpcmem/ION buffer  : %llu/%llu weights "
                 "(the rest used plain heap -- pinned+mapped per call)\n",
                 (unsigned long long)reg_calls_, ms(convert_us_),
                 reg_calls_ ? ms(convert_us_) / reg_calls_ : 0.0, ms(rpc_us_),
                 reg_calls_ ? ms(rpc_us_) / reg_calls_ : 0.0,
                 ms(reg_total_us_ - convert_us_ - rpc_us_), ms(reg_total_us_),
                 (unsigned long long)reg_ion_calls_,
                 (unsigned long long)reg_calls_);

    // M==1 is decode's shape, but an expert that drew a single token during
    // prefill lands in the same row -- read the two as shapes, not phases.
    std::fprintf(stderr,
                 "[HTP-PROFILE] layer calls (M==1 is decode's shape)\n");
    for (const auto &entry : buckets_) {
      const unsigned k = std::get<0>(entry.first);
      const unsigned n = std::get<1>(entry.first);
      const bool decode = std::get<2>(entry.first);
      const Bucket &b = entry.second;
      std::fprintf(stderr,
                   "[HTP-PROFILE]   K=%-5u N=%-5u %-7s calls=%-7llu "
                   "rows=%-8llu host=%9.1f ms (%7.1f us/call)",
                   k, n, decode ? "M==1" : "M>1", (unsigned long long)b.calls,
                   (unsigned long long)b.rows, ms(b.host_us),
                   b.calls ? static_cast<double>(b.host_us) / b.calls : 0.0);
      if (level_ >= 2 && b.calls != 0) {
        const double dsp_per = static_cast<double>(b.dsp_us) / b.calls;
        const double host_per = static_cast<double>(b.host_us) / b.calls;
        std::fprintf(stderr,
                     "  dsp=%7.1f us/call (%4.1f%%) transport=%7.1f us/call"
                     "  [quant %.1f dequant %.1f acc %.1f drain %.1f]",
                     dsp_per, host_per > 0.0 ? 100.0 * dsp_per / host_per : 0.0,
                     host_per - dsp_per,
                     static_cast<double>(b.quant_us) / b.calls,
                     static_cast<double>(b.dequant_us) / b.calls,
                     static_cast<double>(b.acc_us) / b.calls,
                     static_cast<double>(b.drain_us) / b.calls);
      }
      std::fprintf(stderr, "\n");
    }

    std::fprintf(stderr,
                 "[HTP-PROFILE] layer calls total : %10.1f ms\n"
                 "[HTP-PROFILE] HTP host time     : %10.1f ms "
                 "(registration + layer calls)\n\n",
                 ms(invoke_us), ms(reg_total_us_ + invoke_us));
  }

  int level_ = 0;
  std::mutex mutex_;
  uint64_t reg_calls_ = 0;
  uint64_t reg_ion_calls_ = 0;
  uint64_t reg_total_us_ = 0;
  uint64_t convert_us_ = 0;
  uint64_t rpc_us_ = 0;
  std::map<std::tuple<unsigned, unsigned, bool>, Bucket> buckets_;
};

} // namespace

class HtpComputeOps : public CpuComputeOps {
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

    invokeLayer(session, handles, matBdata, matCdata, M, N, K);
  }

  // A QS4CX weight was quantized once, straight from FP32, and already
  // holds the int4 values this registry wants -- so the seam is
  // htp_qs4cx_from_packed's bit rearrangement plus a colsum, not a second
  // quantization. That is both more accurate (measured on this branch:
  // mean_abs_err 0.0333 vs 0.0451 through Q4_0, a 26% cut) and cheaper to
  // load (95.9 -> 59.9 ms for a 2048x3584 gate_up, 77.2 -> 22.2 ms for a
  // 1792x2048 down, x86 host). Prefer feeding the FFN QS4CX; the Q4_0
  // entry above stays for weights shared with the CPU path.
  bool supports_gemm_qs4cx_accel_fp32() const override { return true; }

  void gemm_qs4cx_accel_fp32(void *matAdata, float *matAscale, float *matBdata,
                             float *matCdata, unsigned int M, unsigned int N,
                             unsigned int K) override {
    const remote_handle64 session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    const uint32_t handles[1] = {
      get_or_register_qs4cx(matAdata, matAscale, session, K, N)};

    invokeLayer(session, handles, matBdata, matCdata, M, N, K);
  }

private:
  /** @brief The one FastRPC layer call both accelerated entries make.
   *
   *  Under NNTR_HTP_PROFILE >= 2 it goes through mm_u8i4_layer_timed so the
   *  DSP's own microseconds come back alongside the host wall clock; the
   *  difference between the two is the FastRPC transport. The production
   *  path (profile off) still calls the untimed entry, which is why the
   *  DSP probes cost nothing when nobody is measuring.
   */
  static void invokeLayer(remote_handle64 session, const uint32_t *handles,
                          float *matBdata, float *matCdata, unsigned int M,
                          unsigned int N, unsigned int K) {
    const int act_len = static_cast<int>(M) * static_cast<int>(K);
    const int out_len = static_cast<int>(M) * static_cast<int>(N);

    HtpProfile &profile = HtpProfile::global();
    if (profile.level() == 0) {
      const int err = nntr_hvx_mm_u8i4_layer(
        session, M, K, handles, 1, matBdata, act_len, matCdata, out_len);
      if (err != AEE_SUCCESS) {
        throw std::runtime_error("nntr_hvx_mm_u8i4_layer failed: err=" +
                                 std::to_string(err));
      }
      return;
    }

    uint32_t stage_us[HTP_N_STAGES] = {0};
    const bool timed = profile.level() >= 2;
    const uint64_t t0 = HtpProfile::nowUs();
    const int err =
      timed ? nntr_hvx_mm_u8i4_layer_timed(session, M, K, handles, 1, matBdata,
                                           act_len, matCdata, out_len, stage_us,
                                           HTP_N_STAGES)
            : nntr_hvx_mm_u8i4_layer(session, M, K, handles, 1, matBdata,
                                     act_len, matCdata, out_len);
    const uint64_t elapsed = HtpProfile::nowUs() - t0;
    if (err != AEE_SUCCESS) {
      throw std::runtime_error(std::string(timed
                                             ? "nntr_hvx_mm_u8i4_layer_timed"
                                             : "nntr_hvx_mm_u8i4_layer") +
                               " failed: err=" + std::to_string(err));
    }
    profile.addInvoke(M, K, N, elapsed, timed ? stage_us : nullptr);
  }

  uint32_t get_or_register(void *matAdata, remote_handle64 session, uint32_t K,
                           uint32_t N) {
    std::lock_guard<std::mutex> lock(handle_mutex_);
    auto it = handle_cache_.find(matAdata);
    if (it != handle_cache_.end())
      return it->second;

    const uint64_t t_begin = HtpProfile::nowUs();
    HtpRpcBuffer q_w4_i8(static_cast<size_t>(K) * N);
    std::vector<float> w_scale(N);
    std::vector<int32_t> colsum_w(N);
    const uint64_t t_convert = HtpProfile::nowUs();
    htp_qs4cx_from_q4_0x4(matAdata, K, N,
                          reinterpret_cast<int8_t *>(q_w4_i8.data()),
                          w_scale.data(), colsum_w.data());
    const uint64_t convert_us = HtpProfile::nowUs() - t_convert;

    return register_locked(matAdata, session, K, N, q_w4_i8, w_scale, colsum_w,
                           t_begin, convert_us);
  }

  uint32_t get_or_register_qs4cx(void *matAdata, const float *matAscale,
                                 remote_handle64 session, uint32_t K,
                                 uint32_t N) {
    std::lock_guard<std::mutex> lock(handle_mutex_);
    auto it = handle_cache_.find(matAdata);
    if (it != handle_cache_.end())
      return it->second;

    const uint64_t t_begin = HtpProfile::nowUs();
    HtpRpcBuffer q_w4_i8(static_cast<size_t>(K) * N);
    std::vector<float> w_scale(N);
    std::vector<int32_t> colsum_w(N);
    const uint64_t t_convert = HtpProfile::nowUs();
    htp_qs4cx_from_packed(matAdata, matAscale, K, N,
                          reinterpret_cast<int8_t *>(q_w4_i8.data()),
                          w_scale.data(), colsum_w.data());
    const uint64_t convert_us = HtpProfile::nowUs() - t_convert;

    return register_locked(matAdata, session, K, N, q_w4_i8, w_scale, colsum_w,
                           t_begin, convert_us);
  }

  /** @brief Register a converted weight and cache its handle.
   *  @note  Call with handle_mutex_ already held.
   *  @param t_begin   caller's start timestamp, for the profile's
   *                   registration total (unused when profiling is off)
   *  @param convert_us microseconds the caller spent in htp_qs4cx_from_* */
  uint32_t register_locked(void *key, remote_handle64 session, uint32_t K,
                           uint32_t N, HtpRpcBuffer &q_w4_i8,
                           std::vector<float> &w_scale,
                           std::vector<int32_t> &colsum_w, uint64_t t_begin,
                           uint64_t convert_us) {
    std::vector<float> bias(N, 0.0f); // FC weights carry no bias tensor

    uint32_t handle = 0;
    const uint64_t t_rpc = HtpProfile::nowUs();
    const int err = nntr_hvx_weight_register_u8i4(
      session, K, N, reinterpret_cast<int8_t *>(q_w4_i8.data()),
      static_cast<int>(q_w4_i8.size()), w_scale.data(), static_cast<int>(N),
      colsum_w.data(), static_cast<int>(N), bias.data(), static_cast<int>(N),
      &handle);
    const uint64_t rpc_us = HtpProfile::nowUs() - t_rpc;
    if (err != AEE_SUCCESS) {
      throw std::runtime_error("nntr_hvx_weight_register_u8i4 failed: err=" +
                               std::to_string(err));
    }
    HtpProfile &profile = HtpProfile::global();
    if (profile.level() != 0) {
      profile.addRegister(HtpProfile::nowUs() - t_begin, convert_us, rpc_us,
                          q_w4_i8.isIon());
    }

    // Kept resident for the process lifetime -- Stage 6 (residency, see
    // 40_moe_ffn_htp_task.md) is what has to bound this table's size and
    // add release for the full-checkpoint case; not needed yet.
    handle_cache_.emplace(key, handle);
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
