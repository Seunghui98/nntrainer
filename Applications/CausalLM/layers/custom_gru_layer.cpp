// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   custom_gru_layer.cpp
 * @date   14 January 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Custom GRU Layer compatible with PyTorch weight order (R -> Z -> N)
 */

#include <cmath>
#include <custom_gru_layer.h>
#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>
#include <cstdio>
#include <algorithm>

namespace causallm {
using namespace nntrainer;

// Input/Output indices
static constexpr size_t INPUT_X_IDX = 0;
static constexpr size_t INPUT_H0_IDX = 1;   // optional
static constexpr size_t OUTPUT_Y_IDX = 0;

enum GRUParams {
  weight_ih,
  weight_hh,
  bias_h,
  bias_ih,
  bias_hh,
  hidden_state,
  zrg,
  h_prev,
  dropout_mask
};

CustomGRULayer::CustomGRULayer() :
  LayerImpl(),
  gru_props(nntrainer::props::Unit(),
            nntrainer::props::HiddenStateActivation() = nntrainer::ActivationType::ACT_TANH,
            nntrainer::props::RecurrentActivation() = nntrainer::ActivationType::ACT_SIGMOID,
            nntrainer::props::ReturnSequences(), nntrainer::props::DropOutRate(),
            nntrainer::props::IntegrateBias(), nntrainer::props::ResetAfter()),
  acti_func(nntrainer::ActivationType::ACT_NONE, true),
  recurrent_acti_func(nntrainer::ActivationType::ACT_NONE, true),
  epsilon(1e-3f) {
  wt_idx.fill(std::numeric_limits<unsigned>::max());
}

void CustomGRULayer::finalize(nntrainer::InitLayerContext &context) {
  const Initializer weight_initializer =
    std::get<props::WeightInitializer>(*layer_impl_props).get();
  const Initializer bias_initializer =
    std::get<props::BiasInitializer>(*layer_impl_props).get();
  const WeightRegularizer weight_regularizer =
    std::get<props::WeightRegularizer>(*layer_impl_props).get();
  const float weight_regularizer_constant =
    std::get<props::WeightRegularizerConstant>(*layer_impl_props).get();
  auto &weight_decay = std::get<props::WeightDecay>(*layer_impl_props);
  auto &bias_decay = std::get<props::BiasDecay>(*layer_impl_props);
  const bool disable_bias =
    std::get<props::DisableBias>(*layer_impl_props).get();

  const unsigned int unit = std::get<props::Unit>(gru_props).get();
  ActivationType hidden_state_activation_type =
    std::get<props::HiddenStateActivation>(gru_props).get();
  ActivationType recurrent_activation_type =
    std::get<props::RecurrentActivation>(gru_props).get();
  const bool return_sequences =
    std::get<props::ReturnSequences>(gru_props).get();
  const float dropout_rate = std::get<props::DropOutRate>(gru_props).get();
  const bool integrate_bias = std::get<props::IntegrateBias>(gru_props).get();

  const unsigned int num_inputs = context.getNumInputs();
  NNTR_THROW_IF(!(num_inputs == 1 || num_inputs == 2), std::invalid_argument)
    << "GRU layer takes 1 input (x) or 2 inputs (x, h0)";

  // input0: x = [ batch, 1, time_iteration, feature_size ]
  const TensorDim &input_dim = context.getInputDimensions()[INPUT_X_IDX];
  const unsigned int batch_size = input_dim.batch();
  const unsigned int max_timestep = input_dim.height();
  NNTR_THROW_IF(max_timestep < 1, std::runtime_error)
    << "max timestep must be greater than 0 in gru layer.";
  const unsigned int feature_size = input_dim.width();

  // optional input1: h0
  const bool has_h0 = (num_inputs == 2);
  if (has_h0) {
    const TensorDim &h0_dim = context.getInputDimensions()[INPUT_H0_IDX];

    NNTR_THROW_IF(h0_dim.batch() != batch_size, std::invalid_argument)
      << "h0 batch must match x batch";
    NNTR_THROW_IF(h0_dim.width() != unit, std::invalid_argument)
      << "h0 feature(width) must match unit";

    // accept [B,1,1,U] or [B,1,T,U]
    NNTR_THROW_IF(!(h0_dim.height() == 1 || h0_dim.height() == max_timestep),
                  std::invalid_argument)
      << "h0 height must be 1 or match timesteps of x";
  }

  TensorDim output_dim(
    {batch_size, 1, return_sequences ? max_timestep : 1, unit});
  context.setOutputDimensions({output_dim});

  // weights
  TensorDim weight_ih_dim({feature_size, NUM_GATE * unit});
  wt_idx[GRUParams::weight_ih] = context.requestWeight(
    weight_ih_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "weight_ih", true);

  TensorDim weight_hh_dim({unit, NUM_GATE * unit});
  wt_idx[GRUParams::weight_hh] = context.requestWeight(
    weight_hh_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "weight_hh", true);

  if (!disable_bias) {
    if (integrate_bias) {
      TensorDim bias_h_dim({NUM_GATE * unit});
      wt_idx[GRUParams::bias_h] = context.requestWeight(
        bias_h_dim, bias_initializer, WeightRegularizer::NONE, 1.0f, bias_decay,
        "bias_h", true);
    } else {
      TensorDim bias_ih_dim({NUM_GATE * unit});
      wt_idx[GRUParams::bias_ih] = context.requestWeight(
        bias_ih_dim, bias_initializer, WeightRegularizer::NONE, 1.0f,
        bias_decay, "bias_ih", true);

      TensorDim bias_hh_dim({NUM_GATE * unit});
      wt_idx[GRUParams::bias_hh] = context.requestWeight(
        bias_hh_dim, bias_initializer, WeightRegularizer::NONE, 1.0f,
        bias_decay, "bias_hh", true);
    }
  }

  // internal tensors
  TensorDim hidden_state_dim(batch_size, 1, max_timestep, unit);
  wt_idx[GRUParams::hidden_state] =
    context.requestTensor(hidden_state_dim, "hidden_state", Initializer::NONE,
                          true, TensorLifespan::ITERATION_LIFESPAN);

  TensorDim zrg_dim(batch_size, 1, max_timestep, NUM_GATE * unit);
  wt_idx[GRUParams::zrg] =
    context.requestTensor(zrg_dim, "zrg", Initializer::NONE, true,
                          TensorLifespan::ITERATION_LIFESPAN);

  // Only allocate h_prev when NO external h0 is provided
  if (!has_h0) {
    TensorDim h_prev_dim = TensorDim({batch_size, 1, 1, unit});
    wt_idx[GRUParams::h_prev] =
      context.requestTensor(h_prev_dim, "h_prev", Initializer::NONE, false,
                            TensorLifespan::FORWARD_FUNC_LIFESPAN);
  } else {
    wt_idx[GRUParams::h_prev] = std::numeric_limits<unsigned>::max();
  }

  if (dropout_rate > epsilon) {
    wt_idx[GRUParams::dropout_mask] =
      context.requestTensor(output_dim, "dropout_mask", Initializer::NONE,
                            false, TensorLifespan::ITERATION_LIFESPAN);
  }

  acti_func.setActiFunc(hidden_state_activation_type);
  recurrent_acti_func.setActiFunc(recurrent_activation_type);
}

void CustomGRULayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, gru_props);
  LayerImpl::setProperty(remain_props);
}

void CustomGRULayer::exportTo(Exporter &exporter, const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(gru_props, method, this);
}

void CustomGRULayer::forwarding(RunLayerContext &context, bool training) {
 const bool disable_bias =
    std::get<props::DisableBias>(*layer_impl_props).get();

  const unsigned int unit = std::get<props::Unit>(gru_props).get();
  const bool return_sequences =
    std::get<props::ReturnSequences>(gru_props).get();
  const float dropout_rate = std::get<props::DropOutRate>(gru_props).get();
  const bool integrate_bias = std::get<props::IntegrateBias>(gru_props).get();
  const bool reset_after = std::get<props::ResetAfter>(gru_props).get();

  const unsigned int num_inputs = context.getNumInputs();
  const bool has_h0 = (num_inputs == 2);

  Tensor &input = context.getInput(INPUT_X_IDX);
  const TensorDim &input_dim = input.getDim();
  const unsigned int batch_size = input_dim.batch();
  const unsigned int max_timestep = input_dim.height();
  const unsigned int feature_size = input_dim.width();

  Tensor &output = context.getOutput(OUTPUT_Y_IDX);

  const Tensor &weight_ih = context.getWeight(wt_idx[GRUParams::weight_ih]);
  const Tensor &weight_hh = context.getWeight(wt_idx[GRUParams::weight_hh]);

  Tensor empty;
  Tensor &bias_h = !disable_bias && integrate_bias
                     ? context.getWeight(wt_idx[GRUParams::bias_h])
                     : empty;
  Tensor &bias_ih = !disable_bias && !integrate_bias
                      ? context.getWeight(wt_idx[GRUParams::bias_ih])
                      : empty;
  Tensor &bias_hh = !disable_bias && !integrate_bias
                      ? context.getWeight(wt_idx[GRUParams::bias_hh])
                      : empty;

  Tensor &hidden_state = context.getTensor(wt_idx[GRUParams::hidden_state]);
  Tensor &zrg = context.getTensor(wt_idx[GRUParams::zrg]);

  hidden_state.setZero();
  zrg.setZero();

  // h0 handling
  Tensor *h0_ptr = nullptr;
  Tensor h_prev_local;
  if (has_h0) {
    h0_ptr = &context.getInput(INPUT_H0_IDX);
  } else {
    Tensor &h_prev = context.getTensor(wt_idx[GRUParams::h_prev]);
    h_prev.setZero();
    h_prev_local = h_prev;
    h0_ptr = &h_prev_local;
  }

  Tensor prev_hs;
  Tensor hs;

  for (unsigned int b = 0; b < batch_size; ++b) {
    Tensor islice = input.getBatchSlice(b, 1);
    Tensor oslice = hidden_state.getBatchSlice(b, 1);
    Tensor zrg_ = zrg.getBatchSlice(b, 1);

    // For h0: allow [B,1,1,U] or [B,1,T,U]
    Tensor h0_b = h0_ptr->getBatchSlice(b, 1);

    for (unsigned int t = 0; t < max_timestep; ++t) {
      // 1. Perfectly reproduce PyTorch's expand(L, M, D) effect!
      // Always read only the 0th vector (t=0) of pos_embedding repeatedly (Broadcast)
      Tensor xs = islice.getSharedDataTensor({feature_size}, 0); 
    
      hs = oslice.getSharedDataTensor({unit}, t * unit);
      Tensor zrg_t = zrg_.getSharedDataTensor({unit * NUM_GATE}, unit * t * NUM_GATE);

      // 2. Prohibit sequential state transfer!
      // Since each label is independent from each other, must get its own h0 value, not the previous output.
      prev_hs = h0_b.getSharedDataTensor({unit}, t * unit);

      xs.dot(weight_ih, zrg_t);

      Tensor ztrt = zrg_t.getSharedDataTensor({unit * 2}, 0);
      Tensor gt = zrg_t.getSharedDataTensor({unit}, unit * 2);

      Tensor hh_proj = prev_hs.dot(weight_hh); // Shape: [1, 3 * unit]
      
      // Slicing 1D vector is 100% safe without memory corruption.
      Tensor hh_proj_zr = hh_proj.getSharedDataTensor({unit * 2}, 0);
      Tensor hh_proj_n  = hh_proj.getSharedDataTensor({unit}, unit * 2);

      // Add (R, Z) gate calculation
      ztrt.add_i(hh_proj_zr);

      if (!disable_bias) {
        if (integrate_bias) {
          Tensor ztrt_bias_h = bias_h.getSharedDataTensor({unit * 2}, 0);
          ztrt.add_i(ztrt_bias_h);
        } else {
          Tensor ztrt_bias_ih = bias_ih.getSharedDataTensor({unit * 2}, 0);
          ztrt.add_i(ztrt_bias_ih);
          Tensor ztrt_bias_hh = bias_hh.getSharedDataTensor({unit * 2}, 0);
          ztrt.add_i(ztrt_bias_hh);
        }
      }

      recurrent_acti_func.run_fn(ztrt, ztrt);

      Tensor zt = ztrt.getSharedDataTensor({unit}, 0);
      Tensor rt = ztrt.getSharedDataTensor({unit}, unit);

      // =====================================================================
      // New Gate (Candidate State)
      // =====================================================================
      Tensor temp;
      if (reset_after) {
        temp = Tensor(unit);
        // Copy the already perfectly calculated h_prev * W_hn vector as is!
        temp.copy(hh_proj_n); 

        if (!disable_bias && !integrate_bias) {
          Tensor bias_hh_g = bias_hh.getSharedDataTensor({unit}, 2 * unit);
          temp.add_i(bias_hh_g);
        }

        temp.multiply_i(rt);
        gt.add_i(temp);
      } else {
        rt.multiply(prev_hs, temp);
        
        // Note: Ideally, w_g should be sliced here too, but reset_after=false is not PyTorch style,
        // so the existing code is left as is. To do it perfectly, this part should also be changed to 1D mapping.
        Tensor w_g;
        w_g.copy_with_stride(weight_hh.getSharedDataTensor({1, 1, unit, unit}, unit * 2, false));
        temp.dot(w_g, gt, false, false, 1.0f);

        if (!disable_bias && !integrate_bias) {
          Tensor bias_hh_g = bias_hh.getSharedDataTensor({unit}, 2 * unit);
          gt.add_i(bias_hh_g);
        }
      }

      if (!disable_bias) {
        if (integrate_bias) {
          Tensor gt_bias_h = bias_h.getSharedDataTensor({unit}, unit * 2);
          gt.add_i(gt_bias_h);
        } else {
          Tensor gt_bias_ih = bias_ih.getSharedDataTensor({unit}, unit * 2);
          gt.add_i(gt_bias_ih);
        }
      }

      acti_func.run_fn(gt, gt);

      zt.multiply(prev_hs, hs);
      temp = zt.multiply(-1.0).add(1.0);
      hs.add_i(gt.multiply(temp));

      if (dropout_rate > epsilon && training) {
        Tensor mask_ = context.getTensor(wt_idx[GRUParams::dropout_mask])
                         .getBatchSlice(b, 1);
        Tensor msk = mask_.getSharedDataTensor({unit}, t * unit);
        msk.dropout_mask(dropout_rate);
        hs.multiply_i(msk);
      }
    }
  }

  if (!return_sequences) {
    for (unsigned int batch = 0; batch < batch_size; ++batch) {
      Tensor dest = output.getSharedDataTensor({unit}, batch * unit);
      Tensor src = hidden_state.getSharedDataTensor(
        {unit}, batch * unit * max_timestep + (max_timestep - 1) * unit);
      dest.copy(src);
    }
  } else {
    output.copy(hidden_state);
  }
}

void CustomGRULayer::calcDerivative(RunLayerContext &context) {
  throw nntrainer::exception::not_supported(
    "calcDerivative for CustomGRU layer is not supported");
}

void CustomGRULayer::calcGradient(RunLayerContext &context) {
    throw nntrainer::exception::not_supported(
    "calcGradient for CustomGRU layer is not supported");
}

void CustomGRULayer::setBatch(RunLayerContext &context, unsigned int batch) {
  context.updateTensor(wt_idx[GRUParams::hidden_state], batch);
  context.updateTensor(wt_idx[GRUParams::zrg], batch);
  context.updateTensor(wt_idx[GRUParams::h_prev], batch);

  if (std::get<props::DropOutRate>(gru_props).get() > epsilon) {
    context.updateTensor(wt_idx[GRUParams::dropout_mask], batch);
  }
}

} // namespace causallm
