// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   compute_ops.h
 * @date   04 April 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  ComputeOps abstract interface for backend-agnostic dispatch
 *
 * Each Context (CPU/GPU/NPU) provides a concrete ComputeOps subclass.
 * Tensor operations call through this interface, enabling runtime
 * dispatch to the correct backend (ARM NEON, x86 AVX, OpenCL, CUDA,
 * QNN/HMX, ...) without #ifdef and — crucially — letting backend
 * subclasses carry their own state (cl_command_queue, npu_session,
 * kernel cache, ...) as member variables. That is the difference
 * between this and a function-pointer table: virtual dispatch lets
 * the impl reach back into per-backend resources without leaking a
 * `this` pointer through every call.
 *
 * Default method bodies throw std::runtime_error("not implemented").
 * Concrete subclasses override every op they want to support. For
 * accelerator-only ops (GPU batch/accel variants), pair the op with
 * a supports_*() predicate so callers can pick a CPU path on backends
 * that don't have an accelerated impl.
 */

#ifndef __COMPUTE_OPS_H__
#define __COMPUTE_OPS_H__
#ifdef __cplusplus

#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <vector>

#ifdef ENABLE_FP16
#include <tensor_dim.h>
#endif

namespace nntrainer {

/**
 * @class ComputeOps
 * @brief Abstract dispatch interface for tensor compute kernels.
 */
class ComputeOps {
public:
  virtual ~ComputeOps() = default;

  // ===========================================================================
  // FP32 BLAS
  // ===========================================================================
  virtual void sgemm_fp32(const unsigned int TStorageOrder, bool TransA,
                          bool TransB, const unsigned int M,
                          const unsigned int N, const unsigned int K,
                          const float alpha, const float *A,
                          const unsigned int lda, const float *B,
                          const unsigned int ldb, const float beta, float *C,
                          const unsigned int ldc);

  virtual void sgemv_fp32(const unsigned int TStorageOrder, bool TransA,
                          const unsigned int M, const unsigned int N,
                          const float alpha, const float *A,
                          const unsigned int lda, const float *X,
                          const unsigned int incX, const float beta, float *Y,
                          const unsigned int incY);

  virtual float sdot_fp32(const unsigned int N, const float *X,
                          const unsigned int incX, const float *Y,
                          const unsigned int incY);

  virtual void saxpy_fp32(const unsigned int N, const float alpha,
                          const float *X, const unsigned int incX, float *Y,
                          const unsigned int incY);

  virtual void scopy_fp32(const unsigned int N, const float *X,
                          const unsigned int incX, float *Y,
                          const unsigned int incY);

  virtual void sscal_fp32(const unsigned int N, const float alpha, float *X,
                          const unsigned int incX);

  virtual float snrm2_fp32(const unsigned int N, const float *X,
                           const unsigned int incX);

  virtual unsigned int isamax_fp32(const unsigned int N, const float *X,
                                   const unsigned int incX);

  // ===========================================================================
  // FP32 Element-wise
  // ===========================================================================
  virtual void ele_mul_fp32(const unsigned int N, const float *X,
                            const float *Y, float *Z, float alpha, float beta,
                            unsigned int i_stride, unsigned int o_stride);
  virtual void ele_add_fp32(const unsigned int N, const float *X,
                            const float *Y, float *Z, float alpha, float beta,
                            unsigned int i_stride, unsigned int o_stride);
  virtual void ele_sub_fp32(const unsigned int N, const float *X,
                            const float *Y, float *Z, float alpha, float beta,
                            unsigned int i_stride, unsigned int o_stride);
  virtual void ele_div_fp32(const unsigned int N, const float *X,
                            const float *Y, float *Z, float alpha, float beta,
                            unsigned int i_stride, unsigned int o_stride);

  // ===========================================================================
  // FP32 Activation / Special
  // ===========================================================================
  virtual void swiglu_fp32(const unsigned int N, float *X, float *Y, float *Z);
  virtual void swiglu_alpha_fp32(const unsigned int N, float *X, float *Y,
                                 float *Z, float alpha);
  virtual void tanh_gelu_fp32(const unsigned int N, const float *X, float *Y);
  virtual void gelu_v2_fp32(const unsigned int N, const float *X, float *Y);
  virtual void tanh_gelu_v2_fp32(const unsigned int N, const float *X,
                                 float *Y);
  virtual void tanh_gelu_mul_fp32(const unsigned int N, float *X, float *Y,
                                  float *Z);
  virtual void tanh_gelu_v2_mul_fp32(const unsigned int N, float *X, float *Y,
                                     float *Z);
  virtual float max_val_fp32(const unsigned int N, float *X);
  virtual void softmax_fp32(const unsigned int N, float *X, float *Y);
  virtual bool is_valid_fp32(const unsigned int N, const float *X);

  // ===========================================================================
  // FP32 Matrix ops
  // ===========================================================================
  virtual void transpose_matrix_fp32(const unsigned int M, const unsigned int N,
                                     const float *src, unsigned int ld_src,
                                     float *dst, unsigned int ld_dst);

  // ===========================================================================
  // FP32 Data conversion / Copy
  // ===========================================================================
  virtual void scopy_u8(const unsigned int N, const uint8_t *X,
                        const unsigned int incX, uint8_t *Y,
                        const unsigned int incY);
  virtual void scopy_s8(const unsigned int N, const int8_t *X,
                        const unsigned int incX, int8_t *Y,
                        const unsigned int incY);
  virtual void scopy_int4_to_float32(const unsigned int N, const uint8_t *X,
                                     const unsigned int incX, float *Y,
                                     const unsigned int incY);
  virtual void copy_s16_fp32(const unsigned int N, const int16_t *X, float *Y);
  virtual void copy_u16_fp32(const unsigned int N, const uint16_t *X, float *Y);
  virtual void copy_fp32_u32(const unsigned int N, const float *X, uint32_t *Y);
  virtual void copy_fp32_u16(const unsigned int N, const float *X, uint16_t *Y);
  virtual void copy_fp32_u8(const unsigned int N, const float *X, uint8_t *Y);
  virtual void copy_fp32_s16(const unsigned int N, const float *X, int16_t *Y);
  virtual void copy_fp32_s8(const unsigned int N, const float *X, int8_t *Y);

  // ===========================================================================
  // Quantized GEMM (GGUF format)
  // ===========================================================================
  virtual void gemm_q4_0_fp32(const unsigned int M, const unsigned int N,
                              const unsigned int K, const float *A,
                              const unsigned int lda, const void *B,
                              const unsigned int ldb, float *C,
                              const unsigned int ldc);
  virtual void gemm_q4_K_fp32(const unsigned int M, const unsigned int N,
                              const unsigned int K, const float *A,
                              const unsigned int lda, const void *B,
                              const unsigned int ldb, float *C,
                              const unsigned int ldc);
  virtual void gemm_q6_K_fp32(const unsigned int M, const unsigned int N,
                              const unsigned int K, const float *A,
                              const unsigned int lda, const void *B,
                              const unsigned int ldb, float *C,
                              const unsigned int ldc);

  // ===========================================================================
  // Quantized weight packing / quantization
  // ===========================================================================
  virtual void unpack_q4_0(const void *in_q4_0x, void *out_q4_0,
                           size_t data_size, const unsigned int M,
                           const unsigned int N);
  virtual void unpack_q4_0x8_transpose16(const void *src, uint16_t *d_out,
                                         uint16_t *qs_out, int N, int K);
  virtual size_t quantize_q4_0(const float *src, void *dst, int64_t nrow,
                               int64_t n_per_row, const float *quant_weights);
  virtual void dequantize_row_q4_0(const void *x, float *y, int64_t k);
  virtual void repack_q4_0(void *dst, void *src, size_t data_size,
                           const unsigned int M, const unsigned int N);

  // ===========================================================================
  // Clamp
  // ===========================================================================
  virtual void clamp_fp32(const float *input, float *output, size_t length,
                          float lower_bound, float upper_bound);

  // ===========================================================================
  // Data conversion (int8 → FP32)
  // ===========================================================================
  virtual void scopy_int8_to_fp32_u(const unsigned int N, const uint8_t *X,
                                    const unsigned int incX, float *Y,
                                    const unsigned int incY);
  virtual void scopy_int8_to_fp32_s(const unsigned int N, const int8_t *X,
                                    const unsigned int incX, float *Y,
                                    const unsigned int incY);

  // ===========================================================================
  // Accelerator-only (GPU/NPU) ops — query supports_* before calling.
  // CPU subclasses leave both the impl (default-throw) and predicate
  // (default false) untouched. Accelerator subclasses override both.
  // ===========================================================================
  virtual bool supports_gemm_q4_0_batch_fp32() const { return false; }
  virtual void gemm_q4_0_batch_fp32(std::vector<void *> matAdata,
                                    float *matBdata,
                                    std::vector<float *> matCdata,
                                    unsigned int M, std::vector<unsigned int> N,
                                    unsigned int K);

  virtual bool supports_gemm_q4_0_accel_fp32() const { return false; }
  virtual void gemm_q4_0_accel_fp32(void *matAdata, float *matBdata,
                                    float *matCdata, unsigned int M,
                                    unsigned int N, unsigned int K);

  // The M > 1 gate FloatTensor::dot() applies around both calls above
  // exists because, for CPU/GPU, a single-row GEMM is already fast
  // without going through the batch/accel path -- that reasoning does
  // not hold for HTP, where M == 1 (decode) is the FastRPC-amortization
  // shape these kernels exist for (see docs/htp_attention/
  // 40_moe_ffn_htp_task.md section2.2, 41_moe_ffn_e2e_and_perf_task.md
  // sectionB3). This predicate lets a backend opt in at M == 1 without
  // touching CPU/GPU's existing M > 1 behavior, which is intentional and
  // must not change.
  virtual bool accelerates_q4_0_at_m1() const { return false; }

  // QS4CX weights reach an accelerator without the Q4_0 detour: a model
  // quantized straight from FP32 into QS4CX carries the same int4 values
  // HexKL's registry wants, so the conversion at the seam is a bit
  // rearrangement plus a colsum rather than a second quantization (see
  // htp_qs4cx_from_packed). matAdata / matAscale are QS4CX_Tensor's
  // getData()/getScale() regions.
  virtual bool supports_gemm_qs4cx_accel_fp32() const { return false; }
  virtual void gemm_qs4cx_accel_fp32(void *matAdata, float *matAscale,
                                     float *matBdata, float *matCdata,
                                     unsigned int M, unsigned int N,
                                     unsigned int K);

  // Same grouping as gemm_q4_0_batch_fp32 (several weights sharing one
  // activation, e.g. decode's top-K experts' gate_up projections), for
  // QS4CX weights. Without this override, FloatTensor::dot's vector
  // overload had no fast path for QS4CX at all and fell through to a
  // per-weight loop of individual dotQs4cx() calls -- one FastRPC call per
  // expert instead of one per call, same shape of miss as the Q4_0 batch
  // override being written but the caller-side grouping shipping first
  // (htp_compute_ops.cpp's gemm_q4_0_batch_fp32 comment).
  virtual bool supports_gemm_qs4cx_batch_fp32() const { return false; }
  virtual void
  gemm_qs4cx_batch_fp32(std::vector<void *> matAdata,
                        std::vector<float *> matAscale, float *matBdata,
                        std::vector<float *> matCdata, unsigned int M,
                        std::vector<unsigned int> N, unsigned int K);

  // Fused MoE expert FFN (doc 43 §[L2]): ONE accelerator call computes
  // gate_up @ act -> SwiGLU -> down @, the SwiGLU intermediate never
  // leaving the accelerator. matAdata/matAscale/N carry exactly TWO QS4CX
  // weights, [gate_up (K x 2I), down (I x N_out)] -- N[0] == 2 * down.K is
  // the SwiGLU contract the impl must check. Callers gate on M > 1
  // themselves (decode's single token cannot amortize the fused call's
  // 64-row pad tax); without an override the caller keeps its two-dot +
  // host-activation fallback.
  virtual bool supports_gemm_qs4cx_fused_swiglu_fp32() const { return false; }
  virtual void gemm_qs4cx_fused_swiglu_fp32(std::vector<void *> matAdata,
                                            std::vector<float *> matAscale,
                                            float *matBdata, float *matCdata,
                                            unsigned int M,
                                            std::vector<unsigned int> N,
                                            unsigned int K);

  // A whole MoE FFN layer -- routing, every expert, and the scatter-add --
  // in one accelerator call (doc 46). Where gemm_qs4cx_fused_swiglu_fp32
  // above is one expert and the caller loops, this takes the routing table
  // and does the looping on the accelerator, so the token gather, the
  // routing multiply and the scatter-add stop crossing the boundary too.
  //
  // gate_up[e] is K x 2*inter and down[e] is inter x N_out. row_index holds
  // the token row each (expert, slot) works on, grouped by expert in expert
  // order; row_count says how many belong to each and must sum to
  // row_index's size; row_weight is the routing weight per entry. out is
  // M x N_out, zeroed by the implementation and accumulated into, because
  // experts share token rows.
  //
  // By const reference, unlike the vector-by-value overloads above: a
  // prefill layer's row_index is one entry per token per expert -- 1776 for
  // this model -- and copying that per layer is the kind of cost this call
  // exists to remove.
  //
  // weights_wh says the nibbles are already in the HMX weight-tile layout
  // (htp_wh_layout.h) with a per-output-channel column sum after the scales,
  // so the implementation registers them as they are instead of converting
  // and baking. It is a flag rather than a colsum pointer because the sums
  // sit immediately after the scales and the callee already knows N.
  virtual bool supports_gemm_qs4cx_moe_layer_fp32() const { return false; }
  virtual void
  gemm_qs4cx_moe_layer_fp32(const std::vector<void *> &gate_up_data,
                            const std::vector<float *> &gate_up_scale,
                            const std::vector<void *> &down_data,
                            const std::vector<float *> &down_scale,
                            const std::vector<unsigned int> &row_index,
                            const std::vector<unsigned int> &row_count,
                            const std::vector<float> &row_weight,
                            const float *act, float *out, unsigned int M,
                            unsigned int K, unsigned int inter,
                            unsigned int N_out, bool weights_wh);

  // [#85] Hands the accelerator the decode step's op list (the words of
  // htp_backend/htp_graph_desc.h, built by the model) so it can validate
  // the list once and run resident ops from one entry per token. Words,
  // not a struct: core gains no accelerator type. A backend without a
  // per-token entry returns false and the model runs as before.
  virtual bool set_decode_graph_desc(const std::vector<uint32_t> &words) {
    (void)words;
    return false;
  }

  // [#132 Part B] One Q4_0 weight of the decode list's FC, dense FFN or
  // lm_head ops, handed by the model at load in list order: K x N as the
  // FC reads it, canonical block_q4_0 rows when @a canonical (the tied
  // embedding table), else this ISA's repack. The backend keeps the
  // pointer until its graph init. Returns false when it takes none.
  virtual bool add_decode_graph_q4_0(const void *data, unsigned K, unsigned N,
                                     bool canonical) {
    (void)data;
    (void)K;
    (void)N;
    (void)canonical;
    return false;
  }

  // [#130] The per-token hook a CPU layer calls at one decode row before
  // running its own kernel: kind is the op kind of htp_graph_desc.h, pos
  // the absolute token position, in / out the row (in may be null when
  // the accelerator already holds the stretch's input), param the op's
  // f32 parameter (a norm's gamma, a conv op's w0 | w1 | w2, the RoPE
  // table for attention), state the conv state x_{t-2} | x_{t-1}. Returns
  // 0 when the op is not resident (the layer runs its CPU path), 1 when
  // out was written by the accelerator, 2 when the attention cache must
  // be seeded first: the layer then hands rows [0, pos) of its KV cache to
  // decode_kv_seed_fp32 and calls again. Plain scalars and pointers: core
  // gains no accelerator type.
  virtual int decode_op_fp32(unsigned kind, unsigned pos, const float *in,
                             unsigned in_len, float *out, unsigned out_len,
                             const float *param, unsigned param_len,
                             const float *state, unsigned state_len,
                             float eps) {
    (void)kind;
    (void)pos;
    (void)in;
    (void)in_len;
    (void)out;
    (void)out_len;
    (void)param;
    (void)param_len;
    (void)state;
    (void)state_len;
    (void)eps;
    return 0;
  }
  // [#132 Part B] True while the accelerator runs the whole decode row at
  // pos (every kind resident: the token is one stretch, and its first
  // hook has handed the row over): every CPU FC result of the row is then
  // discarded, so the layers skip their GEMVs. False everywhere else, so
  // the CPU path is unchanged.
  virtual bool decode_row_resident(unsigned pos) {
    (void)pos;
    return false;
  }
  // [#132 Part B E3] The two-session decode (NNTR_HTP_E2E=1). finish: the
  // model handed its Q4_0 weights and mapped everything else it maps at
  // load (after repack_weight): the backend may open its second session
  // and register them now. set_decode_logits(false): the caller
  // takes each decode token's id (take_decode_token_id) instead of its
  // logits, so only the id travels back; true (the default) keeps the
  // logits. take_decode_token_id: the id of the last decode token whose
  // logits did not come back, once; false when there is none.
  virtual bool finish_decode_graph_q4_0() { return false; }
  virtual void set_decode_logits(bool want) { (void)want; }
  // [#132 Part B E3] The ids the caller's greedy pick sets to -inf (its
  // bad words): an id taken with take_decode_token_id skips them too.
  virtual void set_decode_ban(const unsigned *ids, unsigned n) {
    (void)ids;
    (void)n;
  }
  virtual bool take_decode_token_id(unsigned *id) {
    (void)id;
    return false;
  }
  virtual bool decode_kv_seed_fp32(unsigned n_rows, const float *k_rows,
                                   const float *v_rows) {
    (void)n_rows;
    (void)k_rows;
    (void)v_rows;
    return false;
  }

  // Registers one K x N expert weight with the accelerator ahead of its
  // first use, so a model's load pays that cost rather than its first
  // prefill: for the 1408 weights of LFM2-8B-A1B it is 747 ms, 31% of the
  // measured prefill (doc 46 section 50.2 P6). The same call at forward time
  // then finds the weight already registered. A backend with nothing to
  // register returns false and the caller moves on.
  virtual bool register_qs4cx_weight(void *data, const float *scale,
                                     unsigned int K, unsigned int N,
                                     bool weights_wh) {
    (void)data;
    (void)scale;
    (void)K;
    (void)N;
    (void)weights_wh;
    return false;
  }

  // The Q4_0 twin, for a fully_connected layer under engine=htp: its
  // weight is otherwise converted and registered by the first dot() that
  // reaches gemm_q4_0_accel_fp32, which is inside the first prefill (doc
  // 50). Called from Transformer::repack_weight with the data pointer that
  // dot() will pass, so the forward-time lookup is a cache hit.
  virtual bool register_q4_0_weight(void *data, unsigned int K,
                                    unsigned int N) {
    (void)data;
    (void)K;
    (void)N;
    return false;
  }

  // A QS4CX_WH expert pair the loader never read (a virtual weight, doc
  // 52), and where its bytes are: gate_up [K, 2 * inter] at off_gu and down
  // [inter, N_out] at off_dn in the model file behind fd, each laid out as
  // QS4CX_WH_Tensor writes it -- [WH nibbles][N scales][N column sums].
  // The keys are what the caller passes as gate_up_data / down_data, with
  // a null scale, to gemm_qs4cx_moe_layer_fp32 once the expert is in.
  struct ExpertFileDesc {
    const void *key_gu;
    const void *key_dn;
    int fd;
    size_t off_gu, off_dn;
    unsigned int K, inter, N_out;
  };

  // How many expert slots the caller will ever hold at once (the LRU's
  // capacity), so the backend can size its memory to that instead of to a
  // fixed chunk. Advisory; call before the first register. Doc 52 section
  // 10.14: at NNTR_MOE_CACHE_EXPERTS=1 the pool is 116 MiB, and a 256 MiB
  // chunk would hide the saving.
  virtual void reserve_qs4cx_wh_expert_slots(size_t n) { (void)n; }

  // Reads one expert from the model file into a slot the backend owns and
  // registers it. Idempotent for a key already resident. at_load says
  // whether the profile counts it as load-time registration or as a cache
  // miss. A backend without a slot pool returns false.
  virtual bool register_qs4cx_wh_expert_file(const ExpertFileDesc &d,
                                             bool at_load) {
    (void)d;
    (void)at_load;
    return false;
  }

  // The same for several experts, none at load (doc 52 section 10.23): a
  // backend with a batched register makes one round trip for all of them.
  virtual bool
  register_qs4cx_wh_expert_files(const std::vector<ExpertFileDesc> &ds) {
    for (const ExpertFileDesc &d : ds)
      if (!register_qs4cx_wh_expert_file(d, /*at_load=*/false))
        return false;
    return true;
  }

  // The same, split so the file reads overlap other work (doc 52 sections
  // 10.10, 10.20): _begin queues one batch -- a slot per expert, read in the
  // background -- and returns; several batches may be in flight. _end waits
  // for the OLDEST batch, registers it, and returns the key_gu of each of
  // its experts now resident (empty when none is queued). The caller must
  // _end a batch before any ComputeOps call that touches its experts, only
  // between accelerator calls, and must have made room -- _begin takes
  // only free or new slots. False / empty from a backend without a slot
  // pool.
  virtual bool
  prefetch_qs4cx_wh_experts_begin(const std::vector<ExpertFileDesc> &ds) {
    (void)ds;
    return false;
  }
  virtual std::vector<const void *> prefetch_qs4cx_wh_experts_end() {
    return {};
  }

  // [plan 201 S1] The pool's policy for the per-token entry's miss path:
  // makes every key of need resident, calling evict for each key that
  // leaves (before any load) and load for each that must come in -- the
  // layer's ExpertLru::acquire.
  using ExpertPoolFn =
    std::function<void(const std::vector<const void *> &need,
                       const std::function<void(const void *)> &load,
                       const std::function<void(const void *)> &evict)>;

  // [plan 201 S1] One MoE layer's experts (all of them, expert order), in
  // layer order across calls, and the pool's policy: with NNTR_HTP_E2E=1
  // the backend serves S1's misses from these while a decode token runs.
  // A backend without the per-token entry ignores it.
  virtual void set_decode_moe_experts(const std::vector<ExpertFileDesc> &all,
                                      const ExpertPoolFn &pool) {
    (void)all;
    (void)pool;
  }

  // Undoes the above for one expert: both handles released, the slot back
  // in the pool for the next register_qs4cx_wh_expert_file. False when the
  // key is not resident.
  virtual bool release_qs4cx_wh_expert(const void *key_gu) {
    (void)key_gu;
    return false;
  }

  // The dense SwiGLU FFN as ONE accelerator call (doc 51): up and gate
  // [K x I] and down [I x N], all Q4_0x4 as loaded; act [M x K] f32 ->
  // out [M x N] f32 = (silu(act . gate) * (act . up)) . down. The HTP
  // answers it with its MoE layer kernel over column chunks of I, each
  // chunk a whole "expert" whose down output is summed into out.
  // register_q4_0_dense_ffn is the load-time twin, like the two above.
  virtual bool supports_gemm_q4_0_dense_ffn_fp32() const { return false; }
  virtual void gemm_q4_0_dense_ffn_fp32(void *up, void *gate, void *down,
                                        const float *act, float *out,
                                        unsigned int M, unsigned int K,
                                        unsigned int I, unsigned int N) {
    (void)up;
    (void)gate;
    (void)down;
    (void)act;
    (void)out;
    (void)M;
    (void)K;
    (void)I;
    (void)N;
    throw std::runtime_error(
      "ComputeOps::gemm_q4_0_dense_ffn_fp32 not implemented by this backend");
  }
  virtual bool register_q4_0_dense_ffn(void *up, void *gate, void *down,
                                       unsigned int K, unsigned int I,
                                       unsigned int N) {
    (void)up;
    (void)gate;
    (void)down;
    (void)K;
    (void)I;
    (void)N;
    return false;
  }

  // An LFM2 conv block as ONE accelerator call (doc 51 section 2):
  // in_proj [K x 3C] and out_proj [C x N] Q4_0x4 as loaded, conv_w [3 x C]
  // f32 (w0 for row t, w1 for t-1, w2 for t-2); act [M x K] f32 ->
  // out [M x N] f32 = ((b * conv1d(a * c)) . out_proj) with a | b | c the
  // column thirds of act . in_proj, and state [2 x C] f32 <- the conv
  // input's rows M-2 and M-1 (zero where M is shorter), the state the
  // CPU decode path continues from. register_q4_0_conv_block is the
  // load-time twin.
  virtual bool supports_gemm_q4_0_conv_block_fp32() const { return false; }
  virtual void gemm_q4_0_conv_block_fp32(void *in_proj, const float *conv_w,
                                         void *out_proj, const float *act,
                                         float *out, float *state,
                                         unsigned int M, unsigned int K,
                                         unsigned int C, unsigned int N) {
    (void)in_proj;
    (void)conv_w;
    (void)out_proj;
    (void)act;
    (void)out;
    (void)state;
    (void)M;
    (void)K;
    (void)C;
    (void)N;
    throw std::runtime_error(
      "ComputeOps::gemm_q4_0_conv_block_fp32 not implemented by this backend");
  }
  virtual bool register_q4_0_conv_block(void *in_proj, void *out_proj,
                                        unsigned int K, unsigned int C,
                                        unsigned int N) {
    (void)in_proj;
    (void)out_proj;
    (void)K;
    (void)C;
    (void)N;
    return false;
  }

  /**
   * @brief Fused causal attention over an fp16 KV cache (MHACoreLayer's
   *        compute_kcaches + softmax_triangle + compute_fp16vcache in one
   *        call, no materialized logits).
   *
   * out[q][(n*G+g)*hd + d] = softmax_k(q_row . k_row / sqrt(hd)) . v over
   * cache rows [max(0, pos+1-window), pos], pos = cache_from + q, with
   * G = n_head_q / n_head_kv query heads per KV head. Optional logit
   * softcap tanh(s/softcap)*softcap (0 = off) and per-head sink logits
   * (nullptr = none), both with MHACoreLayer's semantics.
   *
   * @param q          f32 [n_q][q_stride], post-RoPE, head h at column h*hd
   * @param k_cache    fp16 bit patterns [cache_to][kv_stride], post-RoPE
   * @param v_cache    fp16 bit patterns [cache_to][kv_stride]
   * @param window     sliding window; 0 means unlimited
   * @param sinks      n_head_q floats or nullptr
   * @param out        f32 [n_q][out_stride]
   * @return true on success; false means the accelerator could not run
   *         this call (shape or transport) and the caller should take the
   *         CPU path for it. Unlike the gemm ops above this never throws
   *         for a runtime failure: attention sits on the per-token hot
   *         path and a fallback is always available.
   */
  virtual bool supports_sdpa_fp16_kvcache() const { return false; }
  virtual bool sdpa_fp16_kvcache(
    const float *q, unsigned int q_stride, const uint16_t *k_cache,
    const uint16_t *v_cache, unsigned int kv_stride, unsigned int n_q,
    unsigned int cache_from, unsigned int cache_to, unsigned int n_head_q,
    unsigned int n_head_kv, unsigned int head_dim, unsigned int window,
    float softcap, const float *sinks, float *out, unsigned int out_stride);

  /**
   * @brief Memory the backend's accelerator reads in place.
   *
   * sdpa_fp16_kvcache's K/V cache reaches the accelerator without a copy
   * only when it lives in memory obtained here: on the HTP this is rpcmem,
   * a dma-buf FastRPC maps into the DSP and cache-maintains per call, where
   * a cache in ordinary memory is copied into a scratch buffer on every
   * call (the whole used range, every token). The default backend has no
   * such memory and returns nullptr; a caller then allocates as usual and
   * still gets correct results, just with the copy.
   *
   * @param bytes block size
   * @return a block at least 16-byte aligned, or nullptr when this backend
   *         has no shared memory or the allocation failed
   */
  virtual void *alloc_shared(size_t bytes) {
    (void)bytes;
    return nullptr;
  }

  /**
   * @brief Releases a block from alloc_shared(). nullptr is a no-op.
   */
  virtual void free_shared(void *block) { (void)block; }

  /**
   * @brief A quantized copy of an attention layer's KV cache that lives on
   *        the accelerator, appended row by row and attended over in place.
   *
   * The layer keeps its fp16 cache as the source of truth and mirrors the
   * rows it writes: kv_cache_q_register() once per (layer, batch) with the
   * cache's capacity, kv_cache_q_append() with the fp16 rows of each step
   * (or of any range that changed, e.g. after a cache load or rewind --
   * rows may be rewritten), then sdpa_q_kvcache() by handle with the same
   * contract as sdpa_fp16_kvcache() minus the cache pointers. The
   * accelerator quantizes at append time (per-token scales) and never
   * revisits a row; how it stores the cache is its own business. The
   * default backend has none: register returns -1 and the layer stays on
   * its fp16 path.
   *
   * @param kind      0 = int8 (A8W8), 1 = int4 (A8W4)
   * @param max_rows  cache capacity in rows
   * @return a handle >= 0, or -1 when this backend has no quantized cache
   *         or the registration failed
   */
  virtual bool supports_kv_cache_q() const { return false; }
  virtual int kv_cache_q_register(unsigned int kind, unsigned int max_rows,
                                  unsigned int n_head_kv,
                                  unsigned int head_dim) {
    (void)kind;
    (void)max_rows;
    (void)n_head_kv;
    (void)head_dim;
    return -1;
  }

  /**
   * @brief Writes cache rows [row0, row0 + n_rows) of the fp16 cache into
   *        the accelerator's copy.
   *
   * @param kv_stride  elements per cache row (n_head_kv * head_dim); rows
   *                   are dense fp16 bit patterns, K post-RoPE, V raw
   * @return false on a transport failure; the caller then drops the
   *         handle and takes the fp16 path
   */
  virtual bool kv_cache_q_append(int handle, unsigned int row0,
                                 unsigned int n_rows, unsigned int kv_stride,
                                 const uint16_t *k_rows,
                                 const uint16_t *v_rows) {
    (void)handle;
    (void)row0;
    (void)n_rows;
    (void)kv_stride;
    (void)k_rows;
    (void)v_rows;
    return false;
  }

  virtual void kv_cache_q_release(int handle) { (void)handle; }

  /**
   * @brief sdpa_fp16_kvcache() over a registered quantized cache, with the
   *        rows the cache is missing appended in the same round trip:
   *        rows [append_row0, append_row0 + append_rows) are written from
   *        k_rows / v_rows (dense, kv_stride elements per row) before the
   *        attention runs. append_rows == 0 attends over the rows already
   *        there. One accelerator call per layer per step is what keeps
   *        the transport cost at the fp16 path's.
   */
  virtual bool sdpa_q_kvcache(
    int handle, unsigned int append_row0, unsigned int append_rows,
    unsigned int kv_stride, const uint16_t *k_rows, const uint16_t *v_rows,
    const float *q, unsigned int q_stride, unsigned int n_q,
    unsigned int cache_from, unsigned int cache_to, unsigned int n_head_q,
    unsigned int n_head_kv, unsigned int head_dim, unsigned int window,
    float softcap, const float *sinks, float *out, unsigned int out_stride) {
    (void)append_row0;
    (void)append_rows;
    (void)kv_stride;
    (void)k_rows;
    (void)v_rows;
    (void)handle;
    (void)q;
    (void)q_stride;
    (void)n_q;
    (void)cache_from;
    (void)cache_to;
    (void)n_head_q;
    (void)n_head_kv;
    (void)head_dim;
    (void)window;
    (void)softcap;
    (void)sinks;
    (void)out;
    (void)out_stride;
    return false;
  }

  virtual bool supports_gemv_int4_batch_fp32() const { return false; }
  virtual void gemv_int4_batch_fp32(std::vector<void *> weights,
                                    std::vector<uint16_t *> scales,
                                    float *input, std::vector<float *> outputs,
                                    unsigned int K,
                                    std::vector<unsigned int> Ns,
                                    unsigned int group_size);

  virtual bool supports_gemm_int4_batch_fp32() const { return false; }
  virtual void gemm_int4_batch_fp32(float *input, std::vector<void *> weights,
                                    std::vector<uint16_t *> scales,
                                    std::vector<float *> matCdata,
                                    unsigned int M,
                                    std::vector<unsigned int> Ns,
                                    unsigned int K, unsigned int group_size);

  virtual bool supports_gemv_int4_accel_fp32() const { return false; }
  virtual void gemv_int4_accel_fp32(char *weight, uint16_t *scale, float *input,
                                    float *output, unsigned int K,
                                    unsigned int N, unsigned int group_size);

  virtual bool supports_sgemm_int4_accel_fp32() const { return false; }
  virtual void sgemm_int4_accel_fp32(float *input, char *weight,
                                     uint16_t *scale, float *output,
                                     unsigned int M, unsigned int N,
                                     unsigned int K, unsigned int group_size);

#ifdef ENABLE_FP16
  // ===========================================================================
  // FP16 BLAS
  // ===========================================================================
  virtual void sgemm_fp16(const unsigned int TStorageOrder, bool TransA,
                          bool TransB, const unsigned int M,
                          const unsigned int N, const unsigned int K,
                          const float alpha, const _FP16 *A,
                          const unsigned int lda, const _FP16 *B,
                          const unsigned int ldb, const float beta, _FP16 *C,
                          const unsigned int ldc);
  virtual void sgemv_fp16(const unsigned int TStorageOrder, bool TransA,
                          const unsigned int M, const unsigned int N,
                          const float alpha, const _FP16 *A,
                          const unsigned int lda, const _FP16 *X,
                          const unsigned int incX, const float beta, _FP16 *Y,
                          const unsigned int incY);
  virtual _FP16 sdot_fp16(const unsigned int N, const _FP16 *X,
                          const unsigned int incX, const _FP16 *Y,
                          const unsigned int incY);
  virtual void saxpy_fp16(const unsigned int N, const float alpha,
                          const _FP16 *X, const unsigned int incX, _FP16 *Y,
                          const unsigned int incY);
  virtual void scopy_fp16(const unsigned int N, const _FP16 *X,
                          const unsigned int incX, _FP16 *Y,
                          const unsigned int incY);
  virtual void scopy_fp32_to_fp16(const unsigned int N, const float *X,
                                  const unsigned int incX, _FP16 *Y,
                                  const unsigned int incY);
  virtual void scopy_fp16_to_fp32(const unsigned int N, const _FP16 *X,
                                  const unsigned int incX, float *Y,
                                  const unsigned int incY);
  virtual void sscal_fp16(const unsigned int N, const float alpha, _FP16 *X,
                          const unsigned int incX);
  virtual _FP16 snrm2_fp16(const unsigned int N, const _FP16 *X,
                           const unsigned int incX);
  virtual unsigned int isamax_fp16(const unsigned int N, const _FP16 *X,
                                   const unsigned int incX);

  // ===========================================================================
  // FP16 Element-wise
  // ===========================================================================
  virtual void ele_mul_fp16(const unsigned int N, const _FP16 *X,
                            const _FP16 *Y, _FP16 *Z, float alpha, float beta,
                            unsigned int i_stride, unsigned int o_stride);
  virtual void ele_add_fp16(const unsigned int N, const _FP16 *X,
                            const _FP16 *Y, _FP16 *Z, float alpha, float beta,
                            unsigned int i_stride, unsigned int o_stride);
  virtual void ele_sub_fp16(const unsigned int N, const _FP16 *X,
                            const _FP16 *Y, _FP16 *Z, float alpha, float beta,
                            unsigned int i_stride, unsigned int o_stride);
  virtual void ele_div_fp16(const unsigned int N, const _FP16 *X,
                            const _FP16 *Y, _FP16 *Z, float alpha, float beta,
                            unsigned int i_stride, unsigned int o_stride);

  // ===========================================================================
  // FP16 Activation / Special
  // ===========================================================================
  virtual void swiglu_fp16(const unsigned int N, _FP16 *X, _FP16 *Y, _FP16 *Z);
  virtual _FP16 max_val_fp16(const unsigned int N, _FP16 *X);
  virtual void softmax_fp16(const unsigned int N, _FP16 *X, _FP16 *Y);
  virtual bool is_valid_fp16(const unsigned int N, const _FP16 *X);
  virtual void inv_sqrt_inplace_fp16(const unsigned int N, _FP16 *X);

  // ===========================================================================
  // FP16 Matrix ops
  // ===========================================================================
  virtual void transpose_matrix_fp16(const unsigned int M, const unsigned int N,
                                     const _FP16 *src, unsigned int ld_src,
                                     _FP16 *dst, unsigned int ld_dst);

  // ===========================================================================
  // FP16 Data conversion
  // ===========================================================================
  virtual void scopy_int4_to_float16(const unsigned int N, const uint8_t *X,
                                     const unsigned int incX, _FP16 *Y,
                                     const unsigned int incY);
  virtual void scopy_int8_to_float16_u(const unsigned int N, const uint8_t *X,
                                       const unsigned int incX, _FP16 *Y,
                                       const unsigned int incY);
  virtual void scopy_int8_to_float16_s(const unsigned int N, const int8_t *X,
                                       const unsigned int incX, _FP16 *Y,
                                       const unsigned int incY);

  // ===========================================================================
  // Mixed precision BLAS
  // ===========================================================================
  virtual void shgemm(const unsigned int TStorageOrder, bool TransA,
                      bool TransB, const unsigned int M, const unsigned int N,
                      const unsigned int K, const float alpha, const float *A,
                      const unsigned int lda, const _FP16 *B,
                      const unsigned int ldb, const float beta, float *C,
                      const unsigned int ldc);
  virtual void shgemv(const unsigned int TStorageOrder, bool TransA,
                      const unsigned int M, const unsigned int N,
                      const float alpha, const float *A, const unsigned int lda,
                      const _FP16 *X, const unsigned int incX, const float beta,
                      float *Y, const unsigned int incY);
  virtual void hsgemm(const unsigned int TStorageOrder, bool TransA,
                      bool TransB, const unsigned int M, const unsigned int N,
                      const unsigned int K, const float alpha, const _FP16 *A,
                      const unsigned int lda, const float *B,
                      const unsigned int ldb, const float beta, float *C,
                      const unsigned int ldc);
  virtual void hsgemv(const unsigned int TStorageOrder, bool TransA,
                      const unsigned int M, const unsigned int N,
                      const float alpha, const _FP16 *A, const unsigned int lda,
                      const float *X, const unsigned int incX, const float beta,
                      float *Y, const unsigned int incY);

  // ===========================================================================
  // Quantized GEMM (FP16 variants)
  // ===========================================================================
  virtual void gemm_q4_0_fp16(const unsigned int M, const unsigned int N,
                              const unsigned int K, const _FP16 *A,
                              const unsigned int lda, const void *B,
                              const unsigned int ldb, _FP16 *C,
                              const unsigned int ldc);
  virtual void gemm_q6_K_fp16(const unsigned int M, const unsigned int N,
                              const unsigned int K, const _FP16 *A,
                              const unsigned int lda, const void *B,
                              const unsigned int ldb, _FP16 *C,
                              const unsigned int ldc);

  // ===========================================================================
  // Rotary embedding
  // ===========================================================================
  virtual void compute_rotary_embedding_value(unsigned int dim,
                                              unsigned int half_,
                                              unsigned int w, _FP16 *in,
                                              _FP16 *out, float *cos_,
                                              float *sin_);
#endif // ENABLE_FP16

protected:
  /**
   * @brief Helper used by default impls to throw a uniform "not
   *        implemented" runtime_error tagged with the op name.
   */
  [[noreturn]] static void throwNotImplemented(const char *op);
};

/**
 * @brief Global compute ops pointer.
 *
 * Set once during init_backend(). When a Context-specific ops table is
 * available (via ContextData), that takes precedence.
 */
extern ComputeOps *g_compute_ops;

/**
 * @brief Ensure the global compute ops is initialized.
 */
void ensureComputeOps();

/**
 * @brief Get the active compute ops with lazy initialization.
 */
inline ComputeOps *getComputeOps() {
#if defined(__GNUC__) || defined(__clang__)
  if (__builtin_expect(g_compute_ops == nullptr, 0))
#else
  if (g_compute_ops == nullptr)
#endif
    ensureComputeOps();
  return g_compute_ops;
}

/**
 * @brief Initialize the CPU compute backend.
 *
 * Sets up architecture-specific resources (e.g., GGML, OpenBLAS threads)
 * and assigns g_compute_ops to the matching concrete ComputeOps
 * subclass for the current CPU architecture.
 */
void init_backend();

/**
 * @brief Backend-specific compute ops getters.
 *
 * `get_cpu_ops()` returns a process-wide singleton of the unified
 * `CpuComputeOps` subclass. The same singleton works for ARM / x86 /
 * fallback because each arch's compute_backend.cpp provides its own
 * specialised body for `nntrainer::sgemm` etc.; the wrapper class is
 * arch-agnostic and only needs to be defined once.
 */
ComputeOps *get_cpu_ops();
#ifdef ENABLE_OPENCL
/** @brief OpenCL accelerator ComputeOps singleton. Defined when
 *  enable-opencl is on, in cl_operations/cl_compute_ops.cpp. */
ComputeOps *get_cl_ops();
#endif
#ifdef ENABLE_HEXKL
/** @brief HTP (Hexagon/HMX) accelerator ComputeOps singleton. Defined
 *  when enable-htp is on, in htp_backend/htp_compute_ops.cpp. */
ComputeOps *get_htp_ops();
#endif

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __COMPUTE_OPS_H__ */
