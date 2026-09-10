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
 *
 * gemm_q4_0_batch_fp32 is the same call with n_handles > 1: several
 * weights sharing one activation (LFM2-MoE decode's selected experts'
 * gate_up projections against the single routed token) go out as one
 * FastRPC call so hexkl_mm_u8i4_layer_run can prefetch the next handle's
 * weight while the current one computes.
 */

#ifdef ENABLE_HEXKL

#include <compute_ops.h>
#include <cpu_ops_table.h>
#include <htp_act_quant.h>
#include <htp_backend.h>
#include <htp_q4_0_convert.h>
#include <htp_rpcmem.h>
#include <swiglu_det.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
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
 * @brief Slots mm_u8i4_layer_fused_timed fills.
 *
 * Restated for the same reason as HTP_T_* (the ARM side cannot include the
 * DSP source). NOT the HTP_T_* layout: the fused call runs a second QUANT
 * pass (the SwiGLU output's requant) and a SwiGLU stage the unfused call
 * does not have, so it carries its own slot list and its own Bucket
 * counter -- folding SwiGLU into an existing slot would put its time in a
 * column named after something else.
 */
enum {
  HTP_FU_T_DSP_TOTAL = 0,
  HTP_FU_T_QUANT,
  HTP_FU_T_SWIGLU,
  HTP_FU_T_DEQUANT,
  HTP_FU_T_ACC_READ,
  HTP_FU_T_ACC_COPY,
  HTP_FU_T_DRAIN,
  HTP_FU_T_ACC_STRIDE,
  HTP_FU_N_STAGES
};

/**
 * @brief Slots mm_u8i4_gate_up_swiglu_timed fills, in order -- test/htp/
 * nntr_hvx_mm_u8i4.c's GU_T_* enum restated (the ARM side cannot include
 * that DSP source). NOT the same layout as HTP_FU_T_*: this call has no
 * down-side accumulator at all, so there is no ACC_COPY slot -- reusing
 * HTP_FU_T_*'s indices against this array would silently read DRAIN's
 * value out of the slot ACC_STRIDE actually lives in.
 */
enum {
  HTP_GU_T_DSP_TOTAL = 0,
  HTP_GU_T_QUANT,
  HTP_GU_T_SWIGLU,
  HTP_GU_T_DEQUANT,
  HTP_GU_T_ACC_READ,
  HTP_GU_T_DRAIN,
  HTP_GU_T_ACC_STRIDE,
  HTP_GU_N_STAGES
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

  /** Same buckets as addInvoke -- the fused call has its own K/N shape (the
   * down matmul's output width, not gate_up's), so it lands in its own row
   * and the two-dot path's numbers stay directly comparable across builds. */
  void addInvokeFused(unsigned M, unsigned K, unsigned N, uint64_t host_us,
                      const uint32_t *stage_us) {
    std::lock_guard<std::mutex> lock(mutex_);
    Bucket &b = buckets_[std::make_tuple(K, N, M == 1)];
    ++b.calls;
    b.rows += M;
    b.host_us += host_us;
    if (stage_us != nullptr) {
      b.dsp_us += stage_us[HTP_FU_T_DSP_TOTAL];
      b.quant_us += stage_us[HTP_FU_T_QUANT];
      b.swiglu_us += stage_us[HTP_FU_T_SWIGLU];
      b.dequant_us += stage_us[HTP_FU_T_DEQUANT];
      b.acc_us += stage_us[HTP_FU_T_ACC_READ] + stage_us[HTP_FU_T_ACC_COPY];
      b.drain_us += stage_us[HTP_FU_T_DRAIN];
    }
  }

  /** Same buckets again, this call's own (smaller) stage layout -- see
   * HTP_GU_T_*'s doc comment for why it is not HTP_FU_T_*. Bucketed under
   * (K, 2*inter, M==1) -- gate_up's own real shape -- not the down matmul's,
   * so this row is directly comparable to what a plain (unfused) gate_up
   * dot() would have shown for the same layer. */
  void addInvokeGateUpSwiglu(unsigned M, unsigned K, unsigned N_gate_up,
                             uint64_t host_us, const uint32_t *stage_us) {
    std::lock_guard<std::mutex> lock(mutex_);
    Bucket &b = buckets_[std::make_tuple(K, N_gate_up, M == 1)];
    ++b.calls;
    b.rows += M;
    b.host_us += host_us;
    if (stage_us != nullptr) {
      b.dsp_us += stage_us[HTP_GU_T_DSP_TOTAL];
      b.quant_us += stage_us[HTP_GU_T_QUANT];
      b.swiglu_us += stage_us[HTP_GU_T_SWIGLU];
      b.dequant_us += stage_us[HTP_GU_T_DEQUANT];
      b.acc_us += stage_us[HTP_GU_T_ACC_READ];
      b.drain_us += stage_us[HTP_GU_T_DRAIN];
    }
  }

  /** ARM-side staging: the memcpy of the activation into the rpcmem buffer
   *  and of the result back out. It sits OUTSIDE every host_us window
   *  above (those start after the copy, so that transport = host - dsp
   *  stays a FastRPC number), which meant it was invisible -- and at this
   *  model's shapes it is ~0.9 MB per expert, 64 times per layer. Counted
   *  here so "HTP host time" stops understating what the path costs. */
  void addStaging(uint64_t us, uint64_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    staging_us_ += us;
    staging_bytes_ += bytes;
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
    uint64_t swiglu_us = 0; /**< fused calls only; 0 elsewhere */
    uint64_t dequant_us = 0;
    uint64_t acc_us = 0;
    uint64_t drain_us = 0;
  };

  HtpProfile() {
    const char *env = std::getenv("NNTR_HTP_PROFILE");
    level_ = (env != nullptr) ? std::atoi(env) : 0;
    // Captured at construction, not at static-destruction time in dump():
    // HtpProfile::global() is always reached through a HtpBackend::global()
    // call first (every accelerated entry point fetches the session handle
    // before it ever touches the profile), so construction order is safe;
    // destruction order is not something to lean on for a second singleton.
    qos_mode_ = HtpBackend::global().qosMode();
  }

  static double ms(uint64_t us) { return static_cast<double>(us) / 1000.0; }

  void dump() {
    // stderr, not the logger: this runs at static destruction, and the run
    // is driven from an adb shell where stderr is what the operator sees.
    std::fprintf(
      stderr,
      "\n[HTP-PROFILE] level=%d qos_mode=%d (2=poll 1=PM "
      "0=interrupt-driven -- every transport number below is only "
      "comparable to 34_fc_measured.md's 326us/call at qos_mode=2)\n",
      level_, qos_mode_);

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
        const double quant_per = static_cast<double>(b.quant_us) / b.calls;
        const double swiglu_per = static_cast<double>(b.swiglu_us) / b.calls;
        const double dequant_per = static_cast<double>(b.dequant_us) / b.calls;
        const double acc_per = static_cast<double>(b.acc_us) / b.calls;
        const double drain_per = static_cast<double>(b.drain_us) / b.calls;
        // What the accelerator actually exists for, by subtraction: the DSP
        // clock minus every stage that is a format change or a wait. Nothing
        // on the DSP times the HMX issue loop directly, and adding a probe
        // inside it would perturb what it measures -- this residue is the
        // honest number, and it is the one to compare against the layer's
        // MAC count. It also absorbs whatever the probes do not name, so
        // treat it as an upper bound on the matmul, not an exact figure.
        const double mm_per = dsp_per - (quant_per + swiglu_per + dequant_per +
                                         acc_per + drain_per);
        std::fprintf(stderr,
                     "  dsp=%7.1f us/call (%4.1f%%) transport=%7.1f us/call"
                     "  [quant %.1f swiglu %.1f dequant %.1f acc %.1f "
                     "drain %.1f | mm<=%.1f (%.1f%% of host)]",
                     dsp_per, host_per > 0.0 ? 100.0 * dsp_per / host_per : 0.0,
                     host_per - dsp_per, quant_per, swiglu_per, dequant_per,
                     acc_per, drain_per, mm_per,
                     host_per > 0.0 ? 100.0 * mm_per / host_per : 0.0);
      }
      std::fprintf(stderr, "\n");
    }

    std::fprintf(
      stderr,
      "[HTP-PROFILE] layer calls total : %10.1f ms\n"
      "[HTP-PROFILE] arm staging memcpy: %10.1f ms  (%.1f MB in+out, "
      "%.1f GB/s) -- outside every host= above\n"
      "[HTP-PROFILE] HTP host time     : %10.1f ms "
      "(registration + layer calls + staging)\n\n",
      ms(invoke_us), ms(staging_us_),
      static_cast<double>(staging_bytes_) / (1024.0 * 1024.0),
      staging_us_ ? static_cast<double>(staging_bytes_) / staging_us_ / 1000.0
                  : 0.0,
      ms(reg_total_us_ + invoke_us + staging_us_));
  }

  int level_ = 0;
  int qos_mode_ = 0;
  std::mutex mutex_;
  uint64_t reg_calls_ = 0;
  uint64_t reg_ion_calls_ = 0;
  uint64_t reg_total_us_ = 0;
  uint64_t staging_us_ = 0;
  uint64_t staging_bytes_ = 0;
  uint64_t convert_us_ = 0;
  uint64_t rpc_us_ = 0;
  std::map<std::tuple<unsigned, unsigned, bool>, Bucket> buckets_;
};

/**
 * @brief memcpy that charges itself to HtpProfile's staging line.
 *
 * Every host_us window in this file starts AFTER the activation is copied
 * into the rpcmem buffer and ends BEFORE the result is copied back out, on
 * purpose: that is what keeps `transport = host - dsp` a FastRPC number
 * rather than a FastRPC-plus-memcpy number. The consequence was that the
 * copies appeared nowhere at all, and on this model's MoE they are not
 * small -- ~450 KB in and ~450 KB out per expert call, 64 calls per layer.
 * Same copy, one accumulator, so the profile's bottom line stops
 * understating what the path costs.
 */
/**
 * @brief NNTR_L2_CHECK: does the DSP still hand back non-finite values?
 *
 * The two L2 failures in docs/htp_attention/43_moe_ffn_measured_next_
 * levers.md section 7 were diagnosed as hvx_recip_qf32 returning NaN for
 * any gate at or below hvx_swiglu_row_f32's exp clamp. That diagnosis is
 * only worth what a device can confirm, and the model's own output cannot
 * confirm it: wrong text is equally consistent with a dozen other causes.
 * This is the discriminator. If it counts zero and the text is still
 * wrong, the SwiGLU clamp is NOT the (whole) bug and the search moves
 * elsewhere; if it counts non-zero, whatever skel is on the device does
 * not have the fix in it.
 *
 * Cheap where it matters most: a NaN SwiGLU lane lands in its row's
 * requantization scale (hvx_quant_rows_u8_params scans the whole row for
 * min/max), so scanning m_pad floats -- 64 of them -- catches it before
 * the value has spread anywhere. Off unless the env var is set.
 */
inline bool l2CheckEnabled() {
  static const bool on = std::getenv("NNTR_L2_CHECK") != nullptr;
  return on;
}

/** @brief NNTR_L2_DIFF: run both MoE FFN paths and report their SNR. See
 *         HtpComputeOps::l2Diff for what the number separates. */
inline bool l2DiffEnabled() {
  static const bool on = std::getenv("NNTR_L2_DIFF") != nullptr;
  return on;
}

/**
 * @brief NNTR_L2_SHADOW: run the fused path, then hand the model the
 *        REFERENCE result instead of the fused one.
 *
 * Splits "the fused path computes wrong values" from "the fused path has a
 * side effect" -- the two remaining stories, and no SNR number can tell
 * them apart. Under this flag both paths execute in full, touching every
 * buffer, taking every lock, leaving every piece of state exactly as a
 * normal fused run would; only the floats the layer goes on to use are
 * swapped for the reference's.
 *
 *   text becomes correct -> the values are the fault, and the 67-80 dB
 *                           calls are worth chasing;
 *   text still broken    -> the values are NOT the fault (they match the
 *                           working path at 142 dB on 84% of calls
 *                           anyway), and the bug is a side effect --
 *                           aliasing, a clobbered workspace, retained
 *                           state -- that differencing output values can
 *                           never surface.
 *
 * Implies NNTR_L2_DIFF: the reference it hands over is the one l2Diff
 * already computes.
 */
inline bool l2ShadowEnabled() {
  static const bool on = std::getenv("NNTR_L2_SHADOW") != nullptr;
  return on;
}

/** @brief Reports the first non-finite element of @a v, once per call. */
inline void l2CheckFinite(const char *what, const float *v, size_t n,
                          unsigned M, unsigned K, unsigned N) {
  size_t bad = 0;
  size_t first = 0;
  for (size_t i = 0; i < n; ++i) {
    if (!std::isfinite(v[i])) {
      if (bad == 0)
        first = i;
      ++bad;
    }
  }
  if (bad != 0) {
    std::fprintf(stderr,
                 "[L2-CHECK] %s: %llu/%llu non-finite (first idx %llu = %g) "
                 "M=%u K=%u N=%u\n",
                 what, (unsigned long long)bad, (unsigned long long)n,
                 (unsigned long long)first, static_cast<double>(v[first]), M, K,
                 N);
  }
}

inline void stagedMemcpy(void *dst, const void *src, size_t bytes) {
  HtpProfile &p = HtpProfile::global();
  if (p.level() == 0) {
    std::memcpy(dst, src, bytes);
    return;
  }
  const uint64_t t0 = HtpProfile::nowUs();
  std::memcpy(dst, src, bytes);
  p.addStaging(HtpProfile::nowUs() - t0, bytes);
}

} // namespace

class HtpComputeOps : public CpuComputeOps {
public:
  bool supports_gemm_q4_0_accel_fp32() const override { return true; }

  // ponytail: decode-shaped (M == 1) calls are declined for the MoE FFN,
  // not accelerated -- reversing 34_fc_measured.md section5.1's own
  // conclusion for the isolated single-FC comparison it was measured on.
  // That comparison is still correct in isolation (M=1 padding tax ~40us
  // against 113us of DSP-only work, so one call is cheap) -- what it does
  // not cover is 22 MoE layers' worth of M==1 calls stacked into one
  // decode token. Device-measured on this branch: gate_up decode's DMA is
  // fully exposed at M=1 (drain 419.2us for a 14.7MB weight group = 35
  // GB/s, no different from one handle -- cross-matmul prefetch has
  // nothing to hide behind at this width, unlike the FC benchmark's
  // narrower weights), so the real per-layer cost is ~2.6ms/token, and
  // 22 layers' worth is a ~57ms/token transport floor alone (17.5 TPS
  // ceiling) against a CPU decode baseline measured at ~25.9 TPS on the
  // same device. HTP cannot win decode until the call is fused across
  // projections or the DMA-exposure problem above is fixed -- ceiling:
  // revisit if 41_moe_ffn_e2e_and_perf_task.md's P1.3 (fused gate_up+
  // swiglu+down, one call per layer) or P2 (u8 in/out, cuts payload 4x)
  // lands, since either changes this arithmetic. Until then declining M=1
  // costs nothing on the FC path this predicate was written for (single
  // dot(), not a MoE stack) and saves the one MoE FFN caller from a
  // documented loss.
  bool accelerates_q4_0_at_m1() const override { return false; }

  // matAdata: Q4_0x4-repacked weight bytes, identity-cached across calls --
  // the same pointer for the lifetime of a loaded model (inference does not
  // move weight tensors), which is what makes registering once and keying
  // the HexKL handle off it safe.
  void gemm_q4_0_accel_fp32(void *matAdata, float *matBdata, float *matCdata,
                            unsigned int M, unsigned int N,
                            unsigned int K) override {
    const remote_handle64 session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    const uint32_t handle = get_or_register(matAdata, session, K, N);

    // ponytail: NOT invokeLayerU8In. Quantizing the activation on ARM/NEON
    // moved that work off HVX (already vectorized, already writes straight
    // into VTCM for HMX to read -- no DRAM detour ever existed for this
    // step) onto ARM scalar/NEON code paying its own cost outside every
    // number this file's profiler measures. Device-measured: DSP time did
    // drop 19-26%, but wall-clock barely moved because the ARM-side
    // quantize (300-420us unvectorized, ~200us with a magic-number round)
    // ate the transport savings it was supposed to create. Reverted to
    // the HVX-quantizes-into-VTCM path. invokeLayerU8In/htp_act_quant.*
    // stay in the tree, unused, in case a caller that is not this one
    // ever legitimately arrives with pre-quantized bytes already in hand.
    invokeLayer(session, &handle, 1, matBdata, matCdata, M, N, K);
  }

  // Several Q4_0 weights that share ONE activation -- LFM2-MoE decode's
  // top-K selected experts' gate_up projections, all against the single
  // token just routed -- go out as ONE FastRPC call across their handles
  // instead of one call per weight. hexkl_mm_u8i4_layer_run's own doc
  // (test/htp/nntr_hvx.idl mm_u8i4_layer) says what that call buys beyond
  // saving round trips: it double-buffers each handle's weight into VTCM
  // with the next handle prefetched while the current one computes, which
  // only happens with more than one handle in the call -- measured 1.7-2x
  // over one-call-per-weight (docs/htp_attention/34_fc_measured.md
  // section4 items C and E). Nothing here changes CPU: FloatTensor::dot's
  // vector overload falls back to the same per-weight loop this replaces
  // when supports_gemm_q4_0_batch_fp32() is false, so this is additive.
  //
  // NOTE: this override was written and syntax-checked in an earlier
  // session but never actually landed in a commit -- Lfm2MoELayer's
  // caller-side grouping shipped without it, so every "grouped" decode
  // call silently fell through Tensor::dot's un-accelerated per-weight
  // loop (ComputeOps::gemm_q4_0_fp32, plain CPU dequant+GEMM) instead of
  // reaching HTP at all. Confirmed by NNTR_HTP_PROFILE=2 output showing
  // zero calls of any shape for gate_up at M==1 after the caller-side
  // change landed. This commit is that missing override.
  bool supports_gemm_q4_0_batch_fp32() const override { return true; }

  void gemm_q4_0_batch_fp32(std::vector<void *> matAdata, float *matBdata,
                            std::vector<float *> matCdata, unsigned int M,
                            std::vector<unsigned int> N,
                            unsigned int K) override {
    const remote_handle64 session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    const size_t n = matAdata.size();
    std::vector<uint32_t> handles(n);
    unsigned int n_total = 0;
    for (size_t i = 0; i < n; ++i) {
      handles[i] = get_or_register(matAdata[i], session, K, N[i]);
      n_total += N[i];
    }

    // mm_u8i4_layer returns one contiguous [M, sum(N)] block (doc order:
    // "one contiguous M x handle[i].N block per handle in call order"), but
    // matCdata is one pointer per weight -- stage into scratch, then scatter
    // each weight's columns into its own output. M is small at this call's
    // one real shape (decode, M==1), so this scratch and the extra copy are
    // a handful of KB, not a hidden cost.
    std::vector<float> out_cat(static_cast<size_t>(M) * n_total);
    invokeLayer(session, handles.data(), static_cast<int>(n), matBdata,
                out_cat.data(), M, n_total, K);

    unsigned int col_offset = 0;
    for (size_t i = 0; i < n; ++i) {
      for (unsigned int row = 0; row < M; ++row) {
        std::memcpy(matCdata[i] + static_cast<size_t>(row) * N[i],
                    out_cat.data() + static_cast<size_t>(row) * n_total +
                      col_offset,
                    N[i] * sizeof(float));
      }
      col_offset += N[i];
    }
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
    const uint32_t handle =
      get_or_register_qs4cx(matAdata, matAscale, session, K, N);

    // ponytail: see gemm_q4_0_accel_fp32's identical comment -- reverted
    // from invokeLayerU8In for the same reason.
    invokeLayer(session, &handle, 1, matBdata, matCdata, M, N, K);
  }

  // Same grouping as gemm_q4_0_batch_fp32, for QS4CX weights -- see the
  // comment on the base declaration (compute_ops.h). Without this,
  // FloatTensor::dot's vector overload had no accelerated path for QS4CX at
  // all: every "grouped" decode call under --moe_dtype QS4CX silently fell
  // through to one dotQs4cx() -- one FastRPC call -- per expert, the same
  // shape of miss gemm_q4_0_batch_fp32's own commit fixed for Q4_0.
  bool supports_gemm_qs4cx_batch_fp32() const override { return true; }

  void gemm_qs4cx_batch_fp32(std::vector<void *> matAdata,
                             std::vector<float *> matAscale, float *matBdata,
                             std::vector<float *> matCdata, unsigned int M,
                             std::vector<unsigned int> N,
                             unsigned int K) override {
    const remote_handle64 session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    const size_t n = matAdata.size();
    std::vector<uint32_t> handles(n);
    unsigned int n_total = 0;
    for (size_t i = 0; i < n; ++i) {
      handles[i] =
        get_or_register_qs4cx(matAdata[i], matAscale[i], session, K, N[i]);
      n_total += N[i];
    }

    // Same stage-then-scatter shape as gemm_q4_0_batch_fp32 -- see that
    // function's comment for why: mm_u8i4_layer returns one contiguous
    // block per handle, matCdata is one pointer per weight.
    std::vector<float> out_cat(static_cast<size_t>(M) * n_total);
    invokeLayer(session, handles.data(), static_cast<int>(n), matBdata,
                out_cat.data(), M, n_total, K);

    unsigned int col_offset = 0;
    for (size_t i = 0; i < n; ++i) {
      for (unsigned int row = 0; row < M; ++row) {
        std::memcpy(matCdata[i] + static_cast<size_t>(row) * N[i],
                    out_cat.data() + static_cast<size_t>(row) * n_total +
                      col_offset,
                    N[i] * sizeof(float));
      }
      col_offset += N[i];
    }
  }

  // The fused expert FFN (doc 43 §[L2]): one FastRPC call per expert per
  // layer instead of two, with the M x 2I + M x I intermediates never
  // leaving the DSP. The caller (Lfm2MoELayer) gates on M > 1 itself --
  // decode's single token cannot amortize the fused call's 64-row pad
  // tax, the same reasoning as accelerates_q4_0_at_m1() being false.
  bool supports_gemm_qs4cx_fused_swiglu_fp32() const override { return true; }

  void gemm_qs4cx_fused_swiglu_fp32(std::vector<void *> matAdata,
                                    std::vector<float *> matAscale,
                                    float *matBdata, float *matCdata,
                                    unsigned int M, std::vector<unsigned int> N,
                                    unsigned int K) override {
    if (matAdata.size() != 2 || matAscale.size() != 2 || N.size() != 2) {
      throw std::invalid_argument(
        "gemm_qs4cx_fused_swiglu_fp32 needs exactly [gate_up, down]");
    }
    const remote_handle64 session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    const unsigned int inter = N[0] / 2;
    const uint32_t h_gu =
      get_or_register_qs4cx(matAdata[0], matAscale[0], session, K, N[0]);
    const uint32_t h_dn =
      get_or_register_qs4cx(matAdata[1], matAscale[1], session, inter, N[1]);
    // ponytail: split-call path (invokeGateUpSwiglu + invokeLayerU8In), NOT
    // invokeFused. The one-call fused kernel (still in the tree, dormant --
    // see invokeFused's own comment) produced wrong output on this model's
    // real weights; root cause never found despite passing its own
    // synthetic-data unit test. This is the smaller-surface alternative
    // doc 43 §7 describes: gate_up+SwiGLU+requant in one call (one weight,
    // one VTCM region set), feeding the requantized u8 intermediate into
    // the ALREADY-device-verified u8in path for down instead of
    // reimplementing the down matmul. Verified stage-by-stage on device
    // (unittest_hvx_mm_u8i4's GateUpSwiglu* tests) before being wired here.
    invokeGateUpSwiglu(session, h_gu, matBdata, M, K, inter);
    invokeLayerU8InRaw(session, &h_dn, 1, htp_act_m_pad(M), matCdata, M, N[1],
                       inter);
    if (l2DiffEnabled() || l2ShadowEnabled())
      l2Diff(session, h_gu, h_dn, matBdata, matCdata, M, K, inter, N[1]);
  }

  /**
   * @brief NNTR_L2_DIFF: the one control neither L2 attempt ever applied.
   *
   * Both attempts were gated on unittest_hvx_mm_u8i4's SNR against
   * fill_deterministic weights and activations, passed at 138-141 dB, and
   * still broke the real model -- so the variable nobody has held fixed is
   * the DATA, not the kernel. This runs the unit test's exact reference
   * (mm_u8i4_layer on gate_up, SwiGLU on the host with expf, mm_u8i4_layer
   * on down) against the split-call path, on THIS model's registered weight
   * bytes and THIS forward's real activation, and reports the SNR per call.
   *
   * It bisects the search in one run:
   *   high SNR -> the kernels agree on real data too, so the fault is on the
   *               ARM side of the call (what is fed in, what is done with
   *               what comes out), not inside the DSP;
   *   low SNR  -> the kernels genuinely disagree on real data, and the
   *               synthetic distribution was hiding it -- the dB number to
   *               chase, with the offending M printed next to it.
   *
   * Debug path: plain heap buffers, three extra FastRPC round trips per
   * expert. Never on unless the env var is.
   */
  void l2Diff(remote_handle64 session, uint32_t h_gu, uint32_t h_dn,
              const float *matBdata, float *got, unsigned int M, unsigned int K,
              unsigned int inter, unsigned int N_out) {
    std::vector<float> act(static_cast<size_t>(M) * K);
    std::memcpy(act.data(), matBdata, act.size() * sizeof(float));

    std::vector<float> gu_out(static_cast<size_t>(M) * 2 * inter, 0.0f);
    int err = nntr_hvx_mm_u8i4_layer(
      session, M, K, &h_gu, 1, act.data(), static_cast<int>(act.size()),
      gu_out.data(), static_cast<int>(gu_out.size()));
    if (err != AEE_SUCCESS) {
      std::fprintf(stderr, "[L2-DIFF] reference gate_up failed: %d\n", err);
      return;
    }

    // [A1] swiglu_det, the same specification Lfm2MoELayer's two-dot path
    // and the DSP's hvx_swiglu_det.h both run. This used std::exp, which
    // made total_flips measure the wrong thing: a nonzero count could mean
    // either that the fused kernel disagreed with the host path the model
    // actually uses, or merely that both disagreed with libm. Against this
    // reference, total_flips == 0 is exactly the property the fused path
    // needs, and anything else is a real divergence.
    std::vector<float> mid(static_cast<size_t>(M) * inter);
    for (unsigned int m = 0; m < M; ++m) {
      for (unsigned int j = 0; j < inter; ++j) {
        const float g = gu_out[static_cast<size_t>(m) * 2 * inter + j];
        const float u = gu_out[static_cast<size_t>(m) * 2 * inter + inter + j];
        mid[static_cast<size_t>(m) * inter + j] = swiglu_det_one(g, u);
      }
    }

    std::vector<float> ref(static_cast<size_t>(M) * N_out, 0.0f);
    err = nntr_hvx_mm_u8i4_layer(session, M, inter, &h_dn, 1, mid.data(),
                                 static_cast<int>(mid.size()), ref.data(),
                                 static_cast<int>(ref.size()));
    if (err != AEE_SUCCESS) {
      std::fprintf(stderr, "[L2-DIFF] reference down failed: %d\n", err);
      return;
    }

    double sig = 0.0, noise = 0.0, max_abs_err = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
      const double r = ref[i], d = static_cast<double>(got[i]) - r;
      sig += r * r;
      noise += d * d;
      if (std::fabs(d) > max_abs_err)
        max_abs_err = std::fabs(d);
    }
    const double snr = (noise == 0.0) ? 999.0 : 10.0 * std::log10(sig / noise);

    // Outlier-row hypothesis: hvx_quant_rows_u8_params sets ONE scale per
    // row from that row's own min/max (the same scan the NaN bug poisoned
    // -- doc 43 section 5). A real trained model's SwiGLU intermediate can
    // have an outlier value fill_deterministic's bounded uniform fill never
    // produces; if one lane dominates a row's range, the rest of that row's
    // 255 levels spread thinner and the row's requant noise rises without
    // ever going non-finite. `mid` is exactly that intermediate (pre-
    // quantization, host f32), already computed above for the reference --
    // this reuses it rather than adding a DSP round trip.
    //
    // Printed for EVERY call, not just suspect ones: the first round only
    // ever showed the bad calls' own spans (0.05-0.24), with nothing to
    // compare them against, so the hypothesis was untestable -- a narrow
    // span could mean "outlier-free" (hypothesis holds) or "this model's
    // rows are always this narrow" (hypothesis is irrelevant). This line
    // is the missing baseline.
    float call_max_span = 0.0f;
    size_t call_max_span_row = 0;
    for (unsigned int m = 0; m < M; ++m) {
      float row_min = mid[static_cast<size_t>(m) * inter];
      float row_max = row_min;
      for (unsigned int j = 0; j < inter; ++j) {
        const float v = mid[static_cast<size_t>(m) * inter + j];
        row_min = std::min(row_min, v);
        row_max = std::max(row_max, v);
      }
      if (row_max - row_min > call_max_span) {
        call_max_span = row_max - row_min;
        call_max_span_row = m;
      }
    }

    if (snr < 100.0) {
      size_t worst_row = 0;
      double worst_row_err = -1.0;
      for (unsigned int m = 0; m < M; ++m) {
        double row_err = 0.0;
        for (unsigned int n = 0; n < N_out; ++n) {
          const double d =
            static_cast<double>(got[static_cast<size_t>(m) * N_out + n]) -
            ref[static_cast<size_t>(m) * N_out + n];
          row_err += d * d;
        }
        if (row_err > worst_row_err) {
          worst_row_err = row_err;
          worst_row = m;
        }
      }
      float row_min = mid[worst_row * inter], row_max = row_min;
      for (unsigned int j = 0; j < inter; ++j) {
        const float v = mid[static_cast<size_t>(worst_row) * inter + j];
        row_min = std::min(row_min, v);
        row_max = std::max(row_max, v);
      }

      // Bin-flip test, take 1 (dev_scale held fixed): is this a rounding-
      // boundary sensitivity given a SHARED scale? hvx_quant_rows_u8_params/
      // hvx_quant_pack_u8_ah are the SAME function on both paths -- verified
      // by inspection, not assumed -- so a difference cannot come from the
      // quantizer having two implementations. What CAN differ is the float
      // value handed to it: the reference quantizes the host's exact
      // expf(); the device quantized its own hvx_exp_sf/hvx_recip_qf32
      // approximation (~1e-6 relative error, well within spec -- doc 43's
      // L2 fix candidate). act_ah_buf_/act_scale_scratch_/act_zp_scratch_
      // still hold the device's own gate_up_swiglu output for this exact
      // forward's worst row -- invokeLayerU8InRaw only reads them, never
      // clears them -- so this first pass re-quantizes `mid` with the
      // DEVICE's OWN scale/zp and counts per-element disagreements.
      //
      // Take 2 (below, scale_diff/total_flips): the first round's result
      // (0 flips on the two WORST calls, 1 flip on the three milder ones)
      // means take 1 tests the wrong thing when it comes back clean --
      // hvx_quant_rows_u8_params derives scale/zp from THIS row's own
      // min/max, so if the approximation nudges the row's extreme value
      // even slightly, the REFERENCE path (which quantizes `mid` with a
      // scale/zp it computes fresh from `mid`'s own min/max, independent of
      // the device's) uses a DIFFERENT scale than the device did -- not a
      // sparse per-element flip but a systematic per-row bias that shifts
      // every one of the row's `inter` dequantized values in the same
      // direction, which a down matmul's summation does not cancel out.
      // Reimplements hvx_quant_rows_u8_params' exact formula (source read,
      // not guessed) on `mid` alone to get that independent scale/zp, then
      // diffs against the device's actual bytes the same way take 1 did.
      constexpr uint32_t kTileRow = 64, kTileInner = 32, kActTileBytes = 2048;
      const uint32_t n_ktiles = inter / kTileInner;
      const uint32_t rb = static_cast<uint32_t>(worst_row) / kTileRow;
      const uint32_t r = static_cast<uint32_t>(worst_row) % kTileRow;
      const float dev_scale = act_scale_scratch_[worst_row];
      const int32_t dev_zp = act_zp_scratch_[worst_row];
      const uint8_t *ah = act_ah_buf_->data();

      const float rmin = std::min(row_min, 0.0f);
      const float rmax = std::max(row_max, 0.0f);
      const float host_scale = (rmax > rmin) ? (rmax - rmin) / 255.0f : 1.0f;
      const int32_t host_zp =
        (rmax > rmin)
          ? std::max(0, std::min(255, static_cast<int32_t>(
                                        std::nearbyint(-rmin / host_scale))))
          : 0;

      // Flip count over the WHOLE block, not just worst_row. The previous
      // round counted only the worst row and its result ("1 flip") was
      // reported as if it described the call -- it did not. This is the
      // number that can honestly be held against row_sq_err and the SNR.
      uint32_t block_flips = 0;
      for (uint32_t m = 0; m < M; ++m) {
        const float s_m = act_scale_scratch_[m];
        const int32_t z_m = act_zp_scratch_[m];
        if (s_m == 0.0f)
          continue;
        const uint32_t rb_m = m / kTileRow, r_m = m % kTileRow;
        for (uint32_t j = 0; j < inter; ++j) {
          const uint32_t kt = j / kTileInner, c = j % kTileInner;
          const size_t idx =
            (static_cast<size_t>(rb_m) * n_ktiles + kt) * kActTileBytes +
            r_m * kTileInner + c;
          int32_t hb = static_cast<int32_t>(std::nearbyint(
                         mid[static_cast<size_t>(m) * inter + j] / s_m)) +
                       z_m;
          hb = std::max(0, std::min(255, hb));
          if (static_cast<int32_t>(ah[idx]) != hb)
            ++block_flips;
        }
      }

      uint32_t flips = 0, total_flips = 0;
      int32_t max_flip = 0, max_total_flip = 0;
      for (uint32_t j = 0; j < inter; ++j) {
        const uint32_t kt = j / kTileInner, c = j % kTileInner;
        const size_t idx =
          (static_cast<size_t>(rb) * n_ktiles + kt) * kActTileBytes +
          r * kTileInner + c;
        const int32_t dev_byte = ah[idx];
        const float v = mid[static_cast<size_t>(worst_row) * inter + j];

        int32_t byte_dev_scale =
          dev_scale != 0.0f
            ? static_cast<int32_t>(std::nearbyint(v / dev_scale)) + dev_zp
            : dev_zp;
        byte_dev_scale = std::max(0, std::min(255, byte_dev_scale));
        const int32_t d1 = dev_byte - byte_dev_scale;
        if (d1 != 0) {
          ++flips;
          if (std::abs(d1) > std::abs(max_flip))
            max_flip = d1;
        }

        int32_t byte_host_scale =
          static_cast<int32_t>(std::nearbyint(v / host_scale)) + host_zp;
        byte_host_scale = std::max(0, std::min(255, byte_host_scale));
        const int32_t d2 = dev_byte - byte_host_scale;
        if (d2 != 0) {
          ++total_flips;
          if (std::abs(d2) > std::abs(max_total_flip))
            max_total_flip = d2;
        }
      }
      const double scale_diff_pct =
        host_scale != 0.0f
          ? 100.0 * (static_cast<double>(dev_scale) - host_scale) / host_scale
          : 0.0;

      std::fprintf(
        stderr,
        "[L2-DIFF] M=%-4u K=%u inter=%u N=%u  snr=%8.2f dB  "
        "max_abs_err=%g  worst_row=%zu row_sq_err=%g "
        "mid_range=[%.4f, %.4f] mid_span=%.4f  call_max_span=%.4f@row%zu  "
        "bin_flips=%u/%u max_flip=%d  dev_scale=%.9g host_scale=%.9g "
        "scale_diff=%.6f%%  total_flips=%u/%u max_total_flip=%d  "
        "block_flips=%u/%u\n",
        M, K, inter, N_out, snr, max_abs_err, worst_row, worst_row_err, row_min,
        row_max, row_max - row_min, call_max_span, call_max_span_row, flips,
        inter, max_flip, dev_scale, host_scale, scale_diff_pct, total_flips,
        inter, max_total_flip, block_flips, static_cast<uint32_t>(M) * inter);
      if (l2ShadowEnabled())
        std::memcpy(got, ref.data(), ref.size() * sizeof(float));
      return;
    }
    std::fprintf(stderr,
                 "[L2-DIFF] M=%-4u K=%u inter=%u N=%u  snr=%8.2f dB  "
                 "max_abs_err=%g  call_max_span=%.4f@row%zu\n",
                 M, K, inter, N_out, snr, max_abs_err, call_max_span,
                 call_max_span_row);
    if (l2ShadowEnabled())
      std::memcpy(got, ref.data(), ref.size() * sizeof(float));
  }

private:
  /** @brief Grows @a buf to at least @a bytes, reusing it otherwise.
   *
   *  One buffer for activation, one for output, reused across every call
   *  instead of one rpcmem alloc/free per matmul -- the buffers this
   *  replaces (the tensor pool's plain heap pointers, passed straight
   *  through) are pinned and mapped by the FastRPC driver on EVERY call
   *  (htp_rpcmem.h's own doc comment; measured at ~155 MB/s vs ION's
   *  44.6 GB/s, docs/htp_attention/34_fc_measured.md section4 item F). The
   *  memcpy this adds is the trade: cheap relative to a per-call pin+map,
   *  but that trade is exactly what NNTR_HTP_PROFILE's transport column is
   *  for confirming on device -- do not assume the win without it. */
  static void ensureCapacity(std::unique_ptr<HtpRpcBuffer> &buf, size_t bytes) {
    if (!buf || buf->size() < bytes) {
      buf = std::make_unique<HtpRpcBuffer>(bytes);
    }
  }

  /** @brief The one FastRPC layer call both accelerated entries make.
   *
   *  Under NNTR_HTP_PROFILE >= 2 it goes through mm_u8i4_layer_timed so the
   *  DSP's own microseconds come back alongside the host wall clock; the
   *  difference between the two is the FastRPC transport. The production
   *  path (profile off) still calls the untimed entry, which is why the
   *  DSP probes cost nothing when nobody is measuring.
   *
   *  Not static any more: it now owns act_buf_/out_buf_, a pair of
   *  rpcmem-backed scratch buffers reused across calls (ensureCapacity
   *  above). invoke_mutex_ guards them -- required once they are shared
   *  mutable state, and consistent with the single-owner assumption
   *  test/htp/nntr_hvx_session.h already documents for the one HTP session
   *  a process opens (one VTCM arena, one HMX lock).
   */
  void invokeLayer(remote_handle64 session, const uint32_t *handles,
                   int num_handles, float *matBdata, float *matCdata,
                   unsigned int M, unsigned int N, unsigned int K) {
    const int act_len = static_cast<int>(M) * static_cast<int>(K);
    const int out_len = static_cast<int>(M) * static_cast<int>(N);

    std::lock_guard<std::mutex> lock(invoke_mutex_);
    ensureCapacity(act_buf_, static_cast<size_t>(act_len) * sizeof(float));
    ensureCapacity(out_buf_, static_cast<size_t>(out_len) * sizeof(float));
    float *act_f32 = reinterpret_cast<float *>(act_buf_->data());
    float *out_cat = reinterpret_cast<float *>(out_buf_->data());
    stagedMemcpy(act_f32, matBdata,
                 static_cast<size_t>(act_len) * sizeof(float));

    HtpProfile &profile = HtpProfile::global();
    if (profile.level() == 0) {
      const int err =
        nntr_hvx_mm_u8i4_layer(session, M, K, handles, num_handles, act_f32,
                               act_len, out_cat, out_len);
      if (err != AEE_SUCCESS) {
        throw std::runtime_error("nntr_hvx_mm_u8i4_layer failed: err=" +
                                 std::to_string(err));
      }
      stagedMemcpy(matCdata, out_cat,
                   static_cast<size_t>(out_len) * sizeof(float));
      return;
    }

    uint32_t stage_us[HTP_N_STAGES] = {0};
    const bool timed = profile.level() >= 2;
    const uint64_t t0 = HtpProfile::nowUs();
    const int err =
      timed ? nntr_hvx_mm_u8i4_layer_timed(session, M, K, handles, num_handles,
                                           act_f32, act_len, out_cat, out_len,
                                           stage_us, HTP_N_STAGES)
            : nntr_hvx_mm_u8i4_layer(session, M, K, handles, num_handles,
                                     act_f32, act_len, out_cat, out_len);
    const uint64_t elapsed = HtpProfile::nowUs() - t0;
    if (err != AEE_SUCCESS) {
      throw std::runtime_error(std::string(timed
                                             ? "nntr_hvx_mm_u8i4_layer_timed"
                                             : "nntr_hvx_mm_u8i4_layer") +
                               " failed: err=" + std::to_string(err));
    }
    stagedMemcpy(matCdata, out_cat,
                 static_cast<size_t>(out_len) * sizeof(float));
    profile.addInvoke(M, K, N, elapsed, timed ? stage_us : nullptr);
  }

  /** @brief Same call as invokeLayer, but the activation is quantized and
   *  AH-tile-packed on the ARM side first (htp_act_quant.h) and sent as u8
   *  instead of f32 -- see mm_u8i4_layer_u8in's IDL doc for what this buys.
   *  Used by both accel entries, which the M > 1 gate means only ever run
   *  at prefill shapes (accelerates_q4_0_at_m1() is false), matching this
   *  path's scope: 41_moe_ffn_e2e_and_perf_task.md's P2. */
  void invokeLayerU8In(remote_handle64 session, const uint32_t *handles,
                       int num_handles, const float *matBdata, float *matCdata,
                       unsigned int M, unsigned int N, unsigned int K) {
    const uint32_t m_pad = htp_act_m_pad(M);
    const size_t act_ah_bytes = static_cast<size_t>(m_pad) * K;
    const int out_len = static_cast<int>(M) * static_cast<int>(N);

    std::lock_guard<std::mutex> lock(invoke_mutex_);
    ensureCapacity(act_ah_buf_, act_ah_bytes);
    ensureCapacity(out_buf_, static_cast<size_t>(out_len) * sizeof(float));
    uint8_t *act_ah = act_ah_buf_->data();
    float *out_cat = reinterpret_cast<float *>(out_buf_->data());

    if (act_scale_scratch_.size() < m_pad) {
      act_scale_scratch_.resize(m_pad);
      act_zp_scratch_.resize(m_pad);
    }
    htp_quant_pack_u8_ah(matBdata, M, K, act_ah, act_scale_scratch_.data(),
                         act_zp_scratch_.data());

    HtpProfile &profile = HtpProfile::global();
    if (profile.level() == 0) {
      const int err = nntr_hvx_mm_u8i4_layer_u8in(
        session, M, K, handles, num_handles, act_ah,
        static_cast<int>(act_ah_bytes), act_scale_scratch_.data(),
        static_cast<int>(m_pad), act_zp_scratch_.data(),
        static_cast<int>(m_pad), out_cat, out_len);
      if (err != AEE_SUCCESS) {
        throw std::runtime_error("nntr_hvx_mm_u8i4_layer_u8in failed: err=" +
                                 std::to_string(err));
      }
      stagedMemcpy(matCdata, out_cat,
                   static_cast<size_t>(out_len) * sizeof(float));
      return;
    }

    uint32_t stage_us[HTP_N_STAGES] = {0};
    const bool timed = profile.level() >= 2;
    const uint64_t t0 = HtpProfile::nowUs();
    const int err =
      timed
        ? nntr_hvx_mm_u8i4_layer_u8in_timed(
            session, M, K, handles, num_handles, act_ah,
            static_cast<int>(act_ah_bytes), act_scale_scratch_.data(),
            static_cast<int>(m_pad), act_zp_scratch_.data(),
            static_cast<int>(m_pad), out_cat, out_len, stage_us, HTP_N_STAGES)
        : nntr_hvx_mm_u8i4_layer_u8in(
            session, M, K, handles, num_handles, act_ah,
            static_cast<int>(act_ah_bytes), act_scale_scratch_.data(),
            static_cast<int>(m_pad), act_zp_scratch_.data(),
            static_cast<int>(m_pad), out_cat, out_len);
    const uint64_t elapsed = HtpProfile::nowUs() - t0;
    if (err != AEE_SUCCESS) {
      throw std::runtime_error(
        std::string(timed ? "nntr_hvx_mm_u8i4_layer_u8in_timed"
                          : "nntr_hvx_mm_u8i4_layer_u8in") +
        " failed: err=" + std::to_string(err));
    }
    stagedMemcpy(matCdata, out_cat,
                 static_cast<size_t>(out_len) * sizeof(float));
    profile.addInvoke(M, K, N, elapsed, timed ? stage_us : nullptr);
  }

  /** @brief The fused MoE expert FFN call: gate_up -> SwiGLU -> down in one
   *  FastRPC round trip (doc 43 §[L2]). Same scratch-buffer reuse and
   *  timed-entry split as invokeLayer; the SwiGLU stage gets its own
   *  profile bucket via addInvokeFused. */
  void invokeFused(remote_handle64 session, const uint32_t *handles,
                   float *matBdata, float *matCdata, unsigned int M,
                   unsigned int N, unsigned int K) {
    const int act_len = static_cast<int>(M) * static_cast<int>(K);
    const int out_len = static_cast<int>(M) * static_cast<int>(N);

    std::lock_guard<std::mutex> lock(invoke_mutex_);
    ensureCapacity(act_buf_, static_cast<size_t>(act_len) * sizeof(float));
    ensureCapacity(out_buf_, static_cast<size_t>(out_len) * sizeof(float));
    float *act_f32 = reinterpret_cast<float *>(act_buf_->data());
    float *out_f32 = reinterpret_cast<float *>(out_buf_->data());
    stagedMemcpy(act_f32, matBdata,
                 static_cast<size_t>(act_len) * sizeof(float));

    HtpProfile &profile = HtpProfile::global();
    if (profile.level() == 0) {
      const int err = nntr_hvx_mm_u8i4_layer_fused(
        session, M, K, handles, 2, act_f32, act_len, out_f32, out_len);
      if (err != AEE_SUCCESS) {
        throw std::runtime_error("nntr_hvx_mm_u8i4_layer_fused failed: err=" +
                                 std::to_string(err));
      }
      stagedMemcpy(matCdata, out_f32,
                   static_cast<size_t>(out_len) * sizeof(float));
      return;
    }

    uint32_t stage_us[HTP_FU_N_STAGES] = {0};
    const bool timed = profile.level() >= 2;
    const uint64_t t0 = HtpProfile::nowUs();
    const int err =
      timed
        ? nntr_hvx_mm_u8i4_layer_fused_timed(session, M, K, handles, 2, act_f32,
                                             act_len, out_f32, out_len,
                                             stage_us, HTP_FU_N_STAGES)
        : nntr_hvx_mm_u8i4_layer_fused(session, M, K, handles, 2, act_f32,
                                       act_len, out_f32, out_len);
    const uint64_t elapsed = HtpProfile::nowUs() - t0;
    if (err != AEE_SUCCESS) {
      throw std::runtime_error(
        std::string(timed ? "nntr_hvx_mm_u8i4_layer_fused_timed"
                          : "nntr_hvx_mm_u8i4_layer_fused") +
        " failed: err=" + std::to_string(err));
    }
    stagedMemcpy(matCdata, out_f32,
                 static_cast<size_t>(out_len) * sizeof(float));
    profile.addInvokeFused(M, K, N, elapsed, timed ? stage_us : nullptr);
  }

  /** @brief [L2, split-call variant] gate_up matmul -> SwiGLU -> requantize
   *  to u8 AH, ONE weight -- doc 43 §7's smaller-surface alternative to
   *  invokeFused. Leaves its result in act_ah_buf_/act_scale_scratch_/
   *  act_zp_scratch_ for invokeLayerU8InRaw (below) to consume immediately
   *  after -- both run under gemm_qs4cx_fused_swiglu_fp32's call, so there
   *  is no other caller between them to race with. */
  void invokeGateUpSwiglu(remote_handle64 session, uint32_t handle_gate_up,
                          const float *matBdata, unsigned int M, unsigned int K,
                          unsigned int inter) {
    const int act_len = static_cast<int>(M) * static_cast<int>(K);
    const uint32_t m_pad = htp_act_m_pad(M);
    const size_t out_ah_bytes = static_cast<size_t>(m_pad) * inter;

    std::lock_guard<std::mutex> lock(invoke_mutex_);
    ensureCapacity(act_buf_, static_cast<size_t>(act_len) * sizeof(float));
    ensureCapacity(act_ah_buf_, out_ah_bytes);
    if (act_scale_scratch_.size() < m_pad) {
      act_scale_scratch_.resize(m_pad);
      act_zp_scratch_.resize(m_pad);
    }
    float *act_f32 = reinterpret_cast<float *>(act_buf_->data());
    uint8_t *out_ah = act_ah_buf_->data();
    stagedMemcpy(act_f32, matBdata,
                 static_cast<size_t>(act_len) * sizeof(float));

    HtpProfile &profile = HtpProfile::global();
    if (profile.level() == 0) {
      const int err = nntr_hvx_mm_u8i4_gate_up_swiglu(
        session, M, K, handle_gate_up, act_f32, act_len, out_ah,
        static_cast<int>(out_ah_bytes), act_scale_scratch_.data(),
        static_cast<int>(m_pad), act_zp_scratch_.data(),
        static_cast<int>(m_pad));
      if (err != AEE_SUCCESS) {
        throw std::runtime_error(
          "nntr_hvx_mm_u8i4_gate_up_swiglu failed: err=" + std::to_string(err));
      }
      if (l2CheckEnabled())
        l2CheckFinite("gate_up_swiglu out_scale", act_scale_scratch_.data(),
                      m_pad, M, K, 2 * inter);
      return;
    }

    uint32_t stage_us[HTP_GU_N_STAGES] = {0};
    const bool timed = profile.level() >= 2;
    const uint64_t t0 = HtpProfile::nowUs();
    const int err =
      timed ? nntr_hvx_mm_u8i4_gate_up_swiglu_timed(
                session, M, K, handle_gate_up, act_f32, act_len, out_ah,
                static_cast<int>(out_ah_bytes), act_scale_scratch_.data(),
                static_cast<int>(m_pad), act_zp_scratch_.data(),
                static_cast<int>(m_pad), stage_us, HTP_GU_N_STAGES)
            : nntr_hvx_mm_u8i4_gate_up_swiglu(
                session, M, K, handle_gate_up, act_f32, act_len, out_ah,
                static_cast<int>(out_ah_bytes), act_scale_scratch_.data(),
                static_cast<int>(m_pad), act_zp_scratch_.data(),
                static_cast<int>(m_pad));
    const uint64_t elapsed = HtpProfile::nowUs() - t0;
    if (err != AEE_SUCCESS) {
      throw std::runtime_error(
        std::string(timed ? "nntr_hvx_mm_u8i4_gate_up_swiglu_timed"
                          : "nntr_hvx_mm_u8i4_gate_up_swiglu") +
        " failed: err=" + std::to_string(err));
    }
    if (l2CheckEnabled())
      l2CheckFinite("gate_up_swiglu out_scale", act_scale_scratch_.data(),
                    m_pad, M, K, 2 * inter);
    profile.addInvokeGateUpSwiglu(M, K, 2 * inter, elapsed,
                                  timed ? stage_us : nullptr);
  }

  /** @brief Down matmul via the u8in path, fed act_ah/scale/zp that are
   *  ALREADY quantized (by invokeGateUpSwiglu, just above) -- unlike
   *  invokeLayerU8In, this does not call htp_quant_pack_u8_ah itself; there
   *  is nothing f32 left to quantize, the DSP produced the bytes directly. */
  void invokeLayerU8InRaw(remote_handle64 session, const uint32_t *handles,
                          int num_handles, unsigned int m_pad_in,
                          float *matCdata, unsigned int M, unsigned int N,
                          unsigned int K) {
    const size_t act_ah_bytes = static_cast<size_t>(m_pad_in) * K;
    const int out_len = static_cast<int>(M) * static_cast<int>(N);

    std::lock_guard<std::mutex> lock(invoke_mutex_);
    ensureCapacity(out_buf_, static_cast<size_t>(out_len) * sizeof(float));
    const uint8_t *act_ah = act_ah_buf_->data();
    float *out_cat = reinterpret_cast<float *>(out_buf_->data());

    HtpProfile &profile = HtpProfile::global();
    if (profile.level() == 0) {
      const int err = nntr_hvx_mm_u8i4_layer_u8in(
        session, M, K, handles, num_handles, act_ah,
        static_cast<int>(act_ah_bytes), act_scale_scratch_.data(),
        static_cast<int>(m_pad_in), act_zp_scratch_.data(),
        static_cast<int>(m_pad_in), out_cat, out_len);
      if (err != AEE_SUCCESS) {
        throw std::runtime_error("nntr_hvx_mm_u8i4_layer_u8in failed: err=" +
                                 std::to_string(err));
      }
      if (l2CheckEnabled())
        l2CheckFinite("down out", out_cat, static_cast<size_t>(out_len), M, K,
                      N);
      stagedMemcpy(matCdata, out_cat,
                   static_cast<size_t>(out_len) * sizeof(float));
      return;
    }

    uint32_t stage_us[HTP_N_STAGES] = {0};
    const bool timed = profile.level() >= 2;
    const uint64_t t0 = HtpProfile::nowUs();
    const int err =
      timed ? nntr_hvx_mm_u8i4_layer_u8in_timed(
                session, M, K, handles, num_handles, act_ah,
                static_cast<int>(act_ah_bytes), act_scale_scratch_.data(),
                static_cast<int>(m_pad_in), act_zp_scratch_.data(),
                static_cast<int>(m_pad_in), out_cat, out_len, stage_us,
                HTP_N_STAGES)
            : nntr_hvx_mm_u8i4_layer_u8in(
                session, M, K, handles, num_handles, act_ah,
                static_cast<int>(act_ah_bytes), act_scale_scratch_.data(),
                static_cast<int>(m_pad_in), act_zp_scratch_.data(),
                static_cast<int>(m_pad_in), out_cat, out_len);
    const uint64_t elapsed = HtpProfile::nowUs() - t0;
    if (err != AEE_SUCCESS) {
      throw std::runtime_error(
        std::string(timed ? "nntr_hvx_mm_u8i4_layer_u8in_timed"
                          : "nntr_hvx_mm_u8i4_layer_u8in") +
        " failed: err=" + std::to_string(err));
    }
    if (l2CheckEnabled())
      l2CheckFinite("down out", out_cat, static_cast<size_t>(out_len), M, K, N);
    stagedMemcpy(matCdata, out_cat,
                 static_cast<size_t>(out_len) * sizeof(float));
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

  // ION-backed activation/output scratch, reused across calls and grown on
  // demand (ensureCapacity) -- see invokeLayer's comment. Guarded by the
  // same mutex that serializes every call into the one HTP session.
  std::mutex invoke_mutex_;
  std::unique_ptr<HtpRpcBuffer> act_buf_;
  std::unique_ptr<HtpRpcBuffer> out_buf_;

  // invokeLayerU8In's scratch: the AH-packed activation (ION-backed, same
  // reasoning as act_buf_/out_buf_) and the small per-row scale/zp arrays
  // it produces alongside it (plain heap -- a few KB at most, not worth
  // ION's pin/map bookkeeping).
  std::unique_ptr<HtpRpcBuffer> act_ah_buf_;
  std::vector<float> act_scale_scratch_;
  std::vector<int32_t> act_zp_scratch_;
};

ComputeOps *get_htp_ops() {
  static HtpComputeOps instance;
  return &instance;
}

} // namespace nntrainer

#endif // ENABLE_HEXKL
