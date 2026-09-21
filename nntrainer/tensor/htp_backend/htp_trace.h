// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   htp_trace.h
 * @date   21 Sep 2026
 * @brief  Per-call HTP timeline recorder that writes a Chrome-trace JSON
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * NNTR_HTP_PROFILE sums every layer call into per-shape buckets and prints
 * them at exit; that says how much, never when. This records the same
 * measurements per call -- the host clock at the FastRPC call, the host
 * wall time, and the DSP's own stage_us[] -- and writes them as one
 * trace.json in Chrome Trace Event Format (tools/nntr_trace/viewer.html,
 * ui.perfetto.dev, chrome://tracing) when the process exits.
 *
 * Enable with NNTR_TRACE=/data/local/tmp/trace.json. Setting it also raises
 * NNTR_HTP_PROFILE to 2 so the timed entry points run and the DSP breakdown
 * is present; the summary print stays as it was.
 *
 * What the file contains, and what it does not (docs/backend_guide/
 * HTP_TRACE_PROFILER.md phase P2, "stage" level, without the DSP-side ring):
 * - host (pid 1): a `wait <call>` span per FastRPC call on the calling
 *   thread, the seam's marshal/return halves (transport = host - dsp,
 *   split evenly), staging memcpys, weight registration, and the
 *   prefill / decode-token phases NeuralNetwork::incremental_inference
 *   marks.
 * - HTP (pid 2): an entry span per call holding the DSP's stage totals laid
 *   out back to back on the lane that runs each stage (HVX, HMX, DMA).
 *   Positions inside a call are therefore sequential by construction; the
 *   durations, the call order and the host timeline are measured. Lane
 *   concurrency (HMX || HVX) needs the DSP ring buffer of P3 and is not in
 *   this file: the viewer's overlap and compression read 0 / 1.0x here,
 *   which is the truth of what this level records, not a measurement of
 *   the kernels.
 *
 * Cost when off: one static bool test per call. When on: one 200-byte record
 * appended under a mutex per call (a decode token is ~140 calls), written
 * once at exit.
 */

#ifndef __HTP_TRACE_H__
#define __HTP_TRACE_H__
#ifdef __cplusplus

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace nntrainer {

class HtpTrace {
public:
  /** @brief Which FastRPC entry a call went through; picks the stage layout. */
  enum Kind : uint8_t {
    KIND_FC = 0,  /**< mm_u8i4_layer(_u8in): HTP_T_* slots */
    KIND_FUSED,   /**< mm_u8i4_layer_fused: HTP_FU_T_* slots */
    KIND_GATE_UP, /**< mm_u8i4_gate_up_swiglu: HTP_GU_T_* slots */
    KIND_MOE,     /**< mm_u8i4_moe_layer: HTP_MOE_T_* slots */
    KIND_N
  };

  static constexpr unsigned kMaxStages = 24;

  static HtpTrace &global();

  /** @brief True when NNTR_TRACE names an output path. */
  bool enabled() const { return enabled_; }

  /** @brief Microseconds on the steady clock; the same clock HtpProfile uses.
   */
  static uint64_t nowUs();

  /**
   * @brief One FastRPC layer call.
   * @param t0_us     nowUs() right before the call
   * @param host_us   wall time of the call as the host saw it
   * @param stage_us  the DSP's slots, nullptr when the untimed entry ran
   * @param n_stages  slot count of @a stage_us (<= kMaxStages)
   */
  void call(Kind kind, unsigned M, unsigned K, unsigned N, unsigned n_handles,
            uint64_t t0_us, uint64_t host_us, const uint32_t *stage_us,
            unsigned n_stages, uint64_t bytes_in, uint64_t bytes_out);

  /** @brief A host memcpy into or out of the FastRPC buffer. */
  void staging(uint64_t t0_us, uint64_t us, uint64_t bytes);

  /** @brief One weight registration (convert + bake/register FastRPC). */
  void registration(uint64_t t0_us, uint64_t total_us, uint64_t convert_us,
                    uint64_t rpc_us, unsigned K, unsigned N);

  /**
   * @brief A model-level phase: prefill, or one decode token.
   * @param from,to  the token range incremental_inference ran
   */
  void phase(uint64_t t0_us, uint64_t us, unsigned from, unsigned to);

  /** @brief Copied into the file's metadata by HtpProfile. */
  void setMeta(int profile_level, int qos_mode);

  /** @brief Writes the file; idempotent. The destructor calls it too. */
  void write();

  ~HtpTrace();
  HtpTrace(const HtpTrace &) = delete;
  HtpTrace &operator=(const HtpTrace &) = delete;

private:
  HtpTrace();

  struct Call {
    uint64_t t0;
    uint64_t host_us;
    uint64_t bytes_in;
    uint64_t bytes_out;
    uint32_t stage[kMaxStages];
    unsigned M, K, N, n_handles;
    uint32_t tid;
    uint8_t kind;
    uint8_t n_stages;
    bool timed;
  };
  struct Span {
    uint64_t t0;
    uint64_t us;
    uint64_t a;
    uint64_t b;
    uint32_t tid;
    uint8_t what; /**< 0 staging, 1 registration, 2 phase */
  };

  static uint32_t currentTid();

  std::mutex mutex_;
  bool enabled_ = false;
  bool written_ = false;
  std::string path_;
  int profile_level_ = 0;
  int qos_mode_ = 0;
  uint64_t epoch_us_ = 0; /**< nowUs() at construction; ts are relative */
  std::vector<Call> calls_;
  std::vector<Span> spans_;
};

} // namespace nntrainer

#endif // __cplusplus
#endif // __HTP_TRACE_H__
