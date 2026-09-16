// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   custom_tie_word_embedding_layer.h
 * @date   21 May 2025
 * @brief  This is Tie_Word_Embedding Layer Class of Neural Network
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __CUSTOM_TIE_WORD_EMBEDDING_H__
#define __CUSTOM_TIE_WORD_EMBEDDING_H__
#ifdef __cplusplus

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#include <common_properties.h>
#include <layer_devel.h>
#include <layer_impl.h>

#include <vector>

namespace causallm {

/**
 * @class   TieWordEmbedding
 * @brief   TieWordEmbedding
 * @todo    Support setBatch for TieWordEmbedding
 */
WIN_EXPORT class TieWordEmbedding : public nntrainer::LayerImpl {
public:
  /**
   * @brief     Constructor of Embedding Layer
   */
  WIN_EXPORT TieWordEmbedding();

  /**
   * @brief     Destructor of Embedding Layer
   */
  WIN_EXPORT ~TieWordEmbedding() = default;

  /**
   *  @brief  Move constructor.
   *  @param[in] TieWordEmbedding &&
   */
  WIN_EXPORT TieWordEmbedding(TieWordEmbedding &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs TieWordEmbedding to be moved.
   */
  WIN_EXPORT TieWordEmbedding &operator=(TieWordEmbedding &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  WIN_EXPORT void finalize(nntrainer::InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  WIN_EXPORT void forwarding(nntrainer::RunLayerContext &context,
                             bool training) override;

  /**
￼   * @copydoc Layer::incremental_forwarding(RunLayerContext &context, unsigned
￼   * int from, unsigned int to, bool training)
￼   */
  WIN_EXPORT void incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  WIN_EXPORT void calcDerivative(nntrainer::RunLayerContext &context) override;

  /**
   * @copydoc Layer::calcGradient(RunLayerContext &context)
   */
  WIN_EXPORT void calcGradient(nntrainer::RunLayerContext &context) override;

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods
   * method)
   */
  WIN_EXPORT void
  exportTo(nntrainer::Exporter &exporter,
           const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Layer::getType()
   */
  WIN_EXPORT const std::string getType() const override {
    return TieWordEmbedding::type;
  };

  /**
   * @copydoc Layer::supportBackwarding()
   */
  WIN_EXPORT bool supportBackwarding() const override { return false; }

  WIN_EXPORT void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override;

  /**
   * @copydoc Layer::read()
   */
  WIN_EXPORT void read(std::ifstream &file, nntrainer::RunLayerContext &context,
                       bool opt_var, ml::train::ExecutionMode mode,
                       bool trainable,
                       nntrainer::TensorDim::DataType definedWeightDataType,
                       bool fsu = false, size_t start_offset = 0,
                       bool read_from_offset = false,
                       int file_fd = -1) override;

  /**
   * @copydoc Layer::read() (ReadSource/mmap variant)
   */
  WIN_EXPORT void read(nntrainer::ReadSource src,
                       nntrainer::RunLayerContext &context, bool opt_var,
                       ml::train::ExecutionMode mode, bool trainable,
                       nntrainer::TensorDim::DataType definedWeightDataType,
                       bool fsu, size_t start_offset = 0,
                       bool read_from_offset = false,
                       int file_fd = -1) override;

  /**
   * @copydoc Layer::save()
   */
  WIN_EXPORT void save(
    std::ofstream &file, nntrainer::RunLayerContext &run_context, bool opt_var,
    ml::train::ExecutionMode mode, bool trainable,
    nntrainer::TensorDim::DataType dtype = nntrainer::TensorDim::DataType::NONE,
    ml::train::ISA target_isa = ml::train::ISA::DEFAULT) const override;

  using Layer::setProperty;

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;

  inline static const std::string type = "tie_word_embeddings";

  /**
   * @brief Builds the lm_head's blocked twin now, at load, instead of on
   *        the first lm_head call.
   *
   * The first call is the tail of the first prefill, and the repack of a
   * 75 MB weight there read 64 ms on the prefill's clock (doc 47 section
   * 15). Called from Transformer::repack_weight, after every weight is
   * loaded -- the one place that is certain, since this node's own read()
   * reads nothing (the tied weight is the embedding's). A no-op for the
   * embedding-mode node and for a non-Q4_0 weight; the forward's own
   * build stays as the fallback for a caller that never repacks.
   */
  WIN_EXPORT void prepareLmhead(nntrainer::RunLayerContext &context);

private:
  std::tuple<nntrainer::props::InDim, nntrainer::props::OutDim,
             nntrainer::props::Unit, nntrainer::props::Scale>
    tieword_embedding_props;
  enum mode { embedding, lm_head };
  enum mode mode_;
  std::array<unsigned int, 4> weight_idx; /**< indices of the weights */
  bool skip_prefill = false;

  /**
   * @brief The lm_head's weight in the blocked Q4_0 layout, built once.
   *
   * A tied model stores ONE weight and quantize_stream writes it with
   * repack=false, because an embedding lookup has to address a token's row
   * directly. That leaves the lm_head -- the same bytes read as a matmul --
   * on the per-row vec_dot path while every other Q4_0 matmul in the model
   * gets the 4-row interleaved one. Measured in the same run: 2.9 GB/s here
   * against 22 GB/s for a repacked FC, 24.4% of a 54 s generation (doc 46
   * sections 44 and 45).
   *
   * So keep both. The canonical weight stays exactly as it is for the
   * lookup, and this is its blocked twin, the same size (a block_q4_0x4 is
   * four block_q4_0), built on the first lm_head call because that is the
   * first point at which the weight is certainly loaded.
   *
   * Empty means no repack: a dtype other than Q4_0, or a shape the
   * interleave does not divide. Both fall back to the row-wise path, which
   * stays correct.
   */
  std::vector<char> lmhead_blocked_;
  bool lmhead_blocked_tried_ = false;

  WIN_EXPORT void finalize_embedding(nntrainer::InitLayerContext &context);
  WIN_EXPORT void finalize_lmhead(nntrainer::InitLayerContext &context);
  WIN_EXPORT void
  incremental_forwarding_embedding(nntrainer::RunLayerContext &context,
                                   unsigned int from, unsigned int to,
                                   bool training);
  WIN_EXPORT void
  incremental_forwarding_lmhead(nntrainer::RunLayerContext &context,
                                unsigned int from, unsigned int to,
                                bool training);
  WIN_EXPORT void buildLmheadBlocked(const nntrainer::Tensor &weight,
                                     unsigned int vocab_size,
                                     unsigned int hidden_size);
};
} // namespace causallm

#endif /* __cplusplus */
#endif /* __CUSTOM_TIE_WORD_EMBEDDING_H__ */
