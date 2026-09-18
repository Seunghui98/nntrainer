// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   QNNGraph.cpp
 * @date   10 Jan 2025
 * @brief  This is QNN Graph Layer Class of Neural Network
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include "QNNGraph.h"
#include "QnnTypes.h"
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <engine.h>
#include <fcntl.h>
#include <fstream>
#include <inttypes.h>
#include <memory>
#include <qnn_context.h>
#include <sys/mman.h>
#include <unistd.h>

#include <sys/resource.h>
#include <thread>

#include "QnnSampleAppUtils.hpp"
#include "Utils/DataUtil.hpp"
#include <common_properties.h>
#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>

std::chrono::duration<double> exec_seconds;

namespace nntrainer {

namespace {

inline double now_us() {
  return std::chrono::duration<double, std::micro>(
           std::chrono::steady_clock::now().time_since_epoch())
    .count();
}

/// Byte length of a QNN tensor from its dims and data type.
size_t qnnTensorBytes(Qnn_Tensor_t *t) {
  std::vector<size_t> dims;
  const uint32_t rank = QNN_TENSOR_GET_RANK(t);
  uint32_t *d = QNN_TENSOR_GET_DIMENSIONS(t);
  for (uint32_t r = 0; r < rank; ++r)
    dims.push_back(d[r]);
  auto [st, len] = datautil::calculateLength(dims, QNN_TENSOR_GET_DATA_TYPE(t));
  return st == datautil::StatusCode::SUCCESS ? len : 0;
}

bool iequals(const std::string &a, const std::string &b) {
  return a.size() == b.size() &&
         std::equal(a.begin(), a.end(), b.begin(),
                    [](unsigned char x, unsigned char y) {
                      return std::tolower(x) == std::tolower(y);
                    });
}

/// Bytes of one row (width x element) of a [B, C, H, W] nntrainer tensor.
size_t tensorRowBytes(const Tensor &t) {
  return (size_t)t.width() * t.getDim().getDataTypeSize();
}

} // namespace

std::shared_ptr<QNNVar> getQNNVar(RunLayerContext &context) {
  std::shared_ptr<QNNVar> qc_var =
    (std::static_pointer_cast<QNNBackendVar>(context.getContextData()))
      ->getVar();
  return qc_var;
}

QNNGraph::QNNGraph() :
  LayerImpl(), graph_props({}, {}, {}, props::FilePath(), {}, {}) {
  m_isContextCreated = false;
  m_inputDataType = iotensor::InputDataType::NATIVE;
  counter = 0;
  profile = std::getenv("NNTR_QNN_PROFILE") != nullptr;
  if (const char *d = std::getenv("NNTR_QNN_DUMP"))
    dump_dir = d;
}

QNNGraph::~QNNGraph() {
  if (profile && n_exec > 0) {
    std::cout << "[QNNGraph] " << bin_path << " total n=" << n_exec
              << " copy_in=" << t_copy_in / 1000.0
              << "ms exec=" << t_exec / 1000.0
              << "ms copy_out=" << t_copy_out / 1000.0 << "ms (avg exec "
              << t_exec / (double)n_exec << "us)" << std::endl;
  }

  if (m_context) {
    if (QNN_CONTEXT_NO_ERROR !=
        m_qnnFunctionPointers.qnnInterface.contextFree(m_context, nullptr)) {
      ml_loge("Failed to free Context");
    }
  }

  // Free the QNN context stored in QNNVar::ct_map using the stored bin_path.
  // Access QNNContext through Engine (same pattern as NeuralNetwork::load).
  if (!bin_path.empty()) {
    LOGD("[QNNGraph] ~QNNGraph: freeing context for bin_path=%s",
         bin_path.c_str());
    auto *ctx = Engine::Global().getRegisteredContext("qnn");
    if (ctx) {
      auto *qnn_ctx = static_cast<QNNContext *>(ctx);
      auto qnn_data = qnn_ctx->getQnnData();
      if (qnn_data && qnn_data->findContext(bin_path).has_value()) {
        LOGD("[QNNGraph] ~QNNGraph: calling freeContext for bin_path=%s",
             bin_path.c_str());
        qnn_data->freeContext(bin_path);
        LOGD("[QNNGraph] ~QNNGraph: freeContext completed for bin_path=%s",
             bin_path.c_str());
      } else {
        LOGD(
          "[QNNGraph] ~QNNGraph: context not found in ct_map for bin_path=%s",
          bin_path.c_str());
      }
    } else {
      LOGD("[QNNGraph] ~QNNGraph: qnn context not registered in Engine");
    }
  } else {
    LOGD("[QNNGraph] ~QNNGraph: bin_path is empty, skipping freeContext");
  }
}

void QNNGraph::finalize(InitLayerContext &context) {
  bin_path = std::get<props::FilePath>(graph_props).get();

  auto &dims = std::get<std::vector<props::TensorDimension>>(graph_props);
  t_dims.assign(dims.begin(), dims.end());

  t_dtype = std::get<std::vector<props::TensorDataType>>(graph_props);

  t_type = std::get<std::vector<props::TensorType>>(graph_props);

  NNTR_THROW_IF(t_dims.size() != t_dtype.size(), std::invalid_argument)
    << "Size of Dimension, DataTypes must be same!";
  NNTR_THROW_IF(t_dims.size() != t_type.size(), std::invalid_argument)
    << "Size of Dimension, Types must be same!";

  std::vector<TensorDim> out_dim;

  for (unsigned int i = 0; i < t_dims.size(); ++i) {
    t_dims[i].setFormat(context.getFormat());
    t_dims[i].setDataType(t_dtype[i]);

    std::string name = "w_" + std::to_string(i);

    switch (t_type[i]) {
    case nntrainer::TensorType_::OUT_TENSOR:
      out_dim.push_back(t_dims[i]);
      break;
    case nntrainer::TensorType_::IN_TENSOR:
      tensor_idx.push_back(
        context.requestTensor(t_dims[i], name, Initializer::NONE, true,
                              TensorLifespan::FORWARD_FUNC_LIFESPAN));
      break;
    default:
      break;
    }
  }

  /// @todo fc actaully supports multidimensions. EffDimFlag shouldn't be fixed
  /// like this.
  context.setEffDimFlagInputDimension(0, 0b1001);
  context.setDynDimFlagInputDimension(0, 0b1000);

  /** set output dimensions */
  context.setOutputDimensions(out_dim);
}

void QNNGraph::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, graph_props);

  LayerImpl::setProperty(remain_props);
}

StatusCode QNNGraph::freeContext(RunLayerContext &context) {
  std::shared_ptr<QNNVar> qc_var = getQNNVar(context);

  if (m_context) {
    if (QNN_CONTEXT_NO_ERROR !=
        m_qnnFunctionPointers.qnnInterface.contextFree(m_context, nullptr)) {
      ml_loge("Faile to free Context");
      return StatusCode::FAILURE;
    }
    m_isContextCreated = false;
  }
  m_context = nullptr;
  return StatusCode::SUCCESS;
}

StatusCode QNNGraph::makeContext(RunLayerContext &context) {

  std::shared_ptr<QNNVar> qc_var = getQNNVar(context);

  return qc_var->makeContext(bin_path);
}

void QNNGraph::read(std::ifstream &file, RunLayerContext &run_context,
                    bool opt_var, ml::train::ExecutionMode mode, bool trainable,
                    TensorDim::DataType defineWeightDataType, bool fsu,
                    size_t start_offset, bool read_from_offset, int file_fd) {}

// ---------------------------------------------------------------------------
// one-time setup
// ---------------------------------------------------------------------------

void QNNGraph::scanBuckets(RunLayerContext &context) {
  if (buckets_scanned)
    return;
  buckets_scanned = true;

  auto qc_var = getQNNVar(context);
  if (!qc_var->findContext(bin_path)) {
    ml_logw("Context is not created. Create Now");
    qc_var->makeContext(bin_path);
  }
  auto op = qc_var->findContext(bin_path);
  NNTR_THROW_IF(!op, std::invalid_argument)
    << "cannot create QNN context from " << bin_path;
  Qnn_Context_Graph_t &ci = *op;

  // "<layer name>_m<N>": nntrainer lowercases layer names, QNN keeps the
  // binary's case, so compare case-insensitively.
  const std::string prefix = context.getName() + "_m";
  for (auto &kv : ci.graph_map) {
    const std::string &g = kv.first;
    if (g.size() <= prefix.size() ||
        !iequals(g.substr(0, prefix.size()), prefix))
      continue;
    const std::string num = g.substr(prefix.size());
    if (num.empty() ||
        !std::all_of(num.begin(), num.end(),
                     [](unsigned char c) { return std::isdigit(c); }))
      continue;
    buckets.push_back((unsigned int)std::stoul(num));
  }
  std::sort(buckets.begin(), buckets.end());
  buckets.erase(std::unique(buckets.begin(), buckets.end()), buckets.end());

  if (!buckets.empty()) {
    std::string s;
    for (auto b : buckets)
      s += std::to_string(b) + " ";
    ml_logi("[QNNGraph:%s] bucketed graphs: %s", context.getName().c_str(),
            s.c_str());
  }
}

void QNNGraph::applyQuantParams(GraphExec &ge) {
  if (!qp_parsed) {
    for (auto &param :
         std::get<std::vector<props::InputQuantParam>>(graph_props)) {
      auto p = param.get();
      in_qp[p.first] = p.second;
    }
    for (auto &param :
         std::get<std::vector<props::OutputQuantParam>>(graph_props)) {
      auto p = param.get();
      out_qp[p.first] = p.second;
    }
    qp_parsed = true;
  }
  for (uint32_t i = 0; i < ge.graphInfo->numInputTensors; ++i) {
    auto key = ge.inputs[i].v1.name;
    NNTR_THROW_IF(in_qp.find(key) == in_qp.end(), std::invalid_argument)
      << "no input_quant_param for QNN input tensor " << key;
    auto value = in_qp[key];
    ge.inputs[i].v1.quantizeParams.scaleOffsetEncoding.scale = value.first;
    ge.inputs[i].v1.quantizeParams.scaleOffsetEncoding.offset = value.second;
  }
  for (uint32_t i = 0; i < ge.graphInfo->numOutputTensors; ++i) {
    auto key = ge.outputs[i].v1.name;
    NNTR_THROW_IF(out_qp.find(key) == out_qp.end(), std::invalid_argument)
      << "no output_quant_param for QNN output tensor " << key;
    auto value = out_qp[key];
    ge.outputs[i].v1.quantizeParams.scaleOffsetEncoding.scale = value.first;
    ge.outputs[i].v1.quantizeParams.scaleOffsetEncoding.offset = value.second;
  }
}

QNNGraph::GraphExec &QNNGraph::getExec(RunLayerContext &context,
                                       unsigned int rows) {
  auto it = exec_cache.find(rows);
  if (it != exec_cache.end())
    return it->second;

  auto qc_var = getQNNVar(context);
  if (!qc_var->findContext(bin_path)) {
    ml_logw("Context is not created. Create Now");
    qc_var->makeContext(bin_path);
  }
  auto op = qc_var->findContext(bin_path);
  NNTR_THROW_IF(!op, std::invalid_argument)
    << "cannot create QNN context from " << bin_path;
  Qnn_Context_Graph_t &ci = *op;

  const std::string gname =
    rows ? context.getName() + "_m" + std::to_string(rows) : context.getName();

  GraphExec ge;
  ge.rows = rows;
  ge.graphInfo = qc_var->graphRetrieve(bin_path, gname);
  NNTR_THROW_IF(!ge.graphInfo, std::invalid_argument)
    << "cannot retrieve graph " << gname << " from " << bin_path;

  NNTR_THROW_IF(context.getNumInputs() != ge.graphInfo->numInputTensors,
                std::invalid_argument)
    << "Number of NNtrainer's inputs " << context.getNumInputs()
    << " does not match with number of QNN's input tensors "
    << ge.graphInfo->numInputTensors << "!";
  NNTR_THROW_IF(context.getNumOutputs() != ge.graphInfo->numOutputTensors,
                std::invalid_argument)
    << "Number of NNtrainer's outputs " << context.getNumOutputs()
    << " does not match with number of QNN's output tensors "
    << ge.graphInfo->numOutputTensors << "!";

  qc_var->m_ioTensor.setupInputAndOutputTensors(&ge.inputs, &ge.outputs,
                                                *ge.graphInfo);
  NNTR_THROW_IF(!ge.inputs || !ge.outputs, std::runtime_error)
    << "failed to set up QNN IO tensors for " << gname;

  for (uint32_t i = 0; i < ge.graphInfo->numInputTensors; ++i)
    ge.in_bytes.push_back(qnnTensorBytes(&ge.inputs[i]));
  for (uint32_t i = 0; i < ge.graphInfo->numOutputTensors; ++i)
    ge.out_bytes.push_back(qnnTensorBytes(&ge.outputs[i]));

  applyQuantParams(ge);

  if (rows) {
    // Bucketed: dedicated rpcmem staging buffers, registered exactly once.
    // Their size must equal rows x (nntrainer row bytes): a mismatch means
    // the graph was built for another activation dtype (e.g. FP16 graph vs
    // FP32 model_tensor_type) or another hidden size.
    for (uint32_t i = 0; i < ge.graphInfo->numInputTensors; ++i) {
      const size_t want = (size_t)rows * tensorRowBytes(context.getInput(i));
      NNTR_THROW_IF(want != ge.in_bytes[i], std::invalid_argument)
        << "[" << gname << "] input " << i << " (" << ge.inputs[i].v1.name
        << ") QNN bytes " << ge.in_bytes[i] << " != nntrainer rows*row_bytes "
        << want << " (dtype/width mismatch between graph and model)";
      void *p = nullptr;
      qc_var->RpcMem->alloc(&p, ge.in_bytes[i], 4096);
      std::memset(p, 0, ge.in_bytes[i]);
      qc_var->RpcMem->registerQnnTensor(p, ge.inputs[i], ci.m_context);
      ge.in_ptrs.push_back(p);
    }
    for (uint32_t i = 0; i < ge.graphInfo->numOutputTensors; ++i) {
      const size_t want = (size_t)rows * tensorRowBytes(context.getOutput(i));
      NNTR_THROW_IF(want != ge.out_bytes[i], std::invalid_argument)
        << "[" << gname << "] output " << i << " (" << ge.outputs[i].v1.name
        << ") QNN bytes " << ge.out_bytes[i] << " != nntrainer rows*row_bytes "
        << want;
      void *p = nullptr;
      qc_var->RpcMem->alloc(&p, ge.out_bytes[i], 4096);
      std::memset(p, 0, ge.out_bytes[i]);
      qc_var->RpcMem->registerQnnTensor(p, ge.outputs[i], ci.m_context);
      ge.out_ptrs.push_back(p);
    }
  } else {
    ge.in_ptrs.assign(ge.graphInfo->numInputTensors, nullptr);
    ge.out_ptrs.assign(ge.graphInfo->numOutputTensors, nullptr);
  }

  ml_logi("[QNNGraph:%s] graph %s ready (%u in, %u out%s)",
          context.getName().c_str(), ge.graphInfo->graphName,
          ge.graphInfo->numInputTensors, ge.graphInfo->numOutputTensors,
          rows ? ", staged" : "");
  return exec_cache.emplace(rows, std::move(ge)).first->second;
}

// ---------------------------------------------------------------------------
// execution
// ---------------------------------------------------------------------------

void QNNGraph::execute(RunLayerContext &context, GraphExec &ge) {
  auto qc_var = getQNNVar(context);
  const double t0 = profile ? now_us() : 0.0;

  QnnGraph_Config_t **customGraphConfigs{nullptr};
  uint32_t configCount{0};
  auto backend_extensions = qc_var->m_backendExtensions;
  if (nullptr != backend_extensions && backend_extensions->interface()) {
    if (!backend_extensions->interface()->beforeExecute(
          ge.graphInfo->graphName, &customGraphConfigs, &configCount)) {
      QNN_ERROR("Extensions Failure in beforeExecute()");
    }
    if (customGraphConfigs) {
      std::vector<const QnnGraph_Config_t *> graphConfigsPointers(
        configCount + 1, nullptr);
      for (size_t idx = 0u; idx < configCount; idx++) {
        graphConfigsPointers[idx] = customGraphConfigs[idx];
      }
      if (QNN_SUCCESS !=
          qc_var->m_qnnFunctionPointers.qnnInterface.graphSetConfig(
            ge.graphInfo->graph, graphConfigsPointers.data())) {
        QNN_ERROR("Failure in setGraphConfigsBeforeExecute()");
      }
    }
  }

  Qnn_ErrorHandle_t executeStatus =
    qc_var->m_qnnFunctionPointers.qnnInterface.graphExecute(
      ge.graphInfo->graph, ge.inputs, ge.graphInfo->numInputTensors, ge.outputs,
      ge.graphInfo->numOutputTensors, qc_var->m_profileBackendHandle, nullptr);

  if (nullptr != backend_extensions && backend_extensions->interface()) {
    if (!backend_extensions->interface()->afterExecute()) {
      QNN_ERROR("Extensions Failure in afterExecute()");
    }
  }

  if (profile) {
    const double dt = now_us() - t0;
    t_exec += dt;
    exec_seconds += std::chrono::duration<double>(dt * 1e-6);
  }
  n_exec++;
  counter++;

  if (QNN_GRAPH_NO_ERROR != executeStatus) {
    ml_loge("Execution of Graph %s failed (%" PRIu64 ")",
            ge.graphInfo->graphName, (uint64_t)executeStatus);
    std::cout << "Execution of Graph " << ge.graphInfo->graphName << " failed!"
              << std::endl;
  }

  if (profile && (n_exec % 64) == 0) {
    std::cout << "[QNNGraph:" << context.getName() << "] n=" << n_exec
              << " avg copy_in=" << t_copy_in / (double)n_exec
              << "us exec=" << t_exec / (double)n_exec
              << "us copy_out=" << t_copy_out / (double)n_exec << "us"
              << std::endl;
  }
  if (!dump_dir.empty())
    dumpIO(ge);
}

void QNNGraph::dumpIO(GraphExec &ge) {
  const std::string g = ge.graphInfo->graphName;
  unsigned int &n = dump_count[g];
  if (n >= 4)
    return;
  auto write = [&](const std::string &tag, uint32_t i, const void *p,
                   size_t bytes) {
    if (!p)
      return;
    std::ofstream f(dump_dir + "/" + g + "." + tag + std::to_string(i) + "." +
                      std::to_string(n) + ".raw",
                    std::ios::binary);
    f.write(reinterpret_cast<const char *>(p), bytes);
  };
  for (uint32_t i = 0; i < ge.graphInfo->numInputTensors; ++i)
    write("in", i, ge.in_ptrs[i], ge.in_bytes[i]);
  for (uint32_t i = 0; i < ge.graphInfo->numOutputTensors; ++i)
    write("out", i, ge.out_ptrs[i], ge.out_bytes[i]);
  n++;
}

void QNNGraph::bindLegacy(RunLayerContext &context, GraphExec &ge) {
  auto qc_var = getQNNVar(context);
  auto op = qc_var->findContext(bin_path);
  Qnn_Context_Graph_t &ci = *op;

  // The layer's own tensors are rpcmem-backed (qnn allocator). They are
  // registered on first sight; a forward whose data pointers did not move
  // does no populate/register work at all.
  currentInputBuffers.clear();
  currentOutputBuffers.clear();
  for (size_t i = 0; i < context.getNumInputs(); ++i)
    updateBufferType(currentInputBuffers, context.getInput(i));
  for (size_t i = 0; i < ge.graphInfo->numOutputTensors; ++i)
    updateBufferType(currentOutputBuffers, context.getOutput(i));

  for (size_t i = 0; i < context.getNumInputs(); ++i) {
    void *p = context.getInput(i).getData<char>();
    if (p != ge.in_ptrs[i]) {
      populateTensor(qc_var, ci, currentInputBuffers[i], &(ge.inputs[i]));
      ge.in_ptrs[i] = p;
    }
  }
  for (size_t i = 0; i < context.getNumOutputs(); ++i) {
    void *p = context.getOutput(i).getData<char>();
    if (p != ge.out_ptrs[i]) {
      populateTensor(qc_var, ci, currentOutputBuffers[i], &(ge.outputs[i]));
      ge.out_ptrs[i] = p;
    }
  }
}

void QNNGraph::runBucketed(RunLayerContext &context, unsigned int from,
                           unsigned int to) {
  NNTR_THROW_IF(context.getInput(0).batch() != 1, std::invalid_argument)
    << "[QNNGraph] bucketed graphs support batch 1 only";
  NNTR_THROW_IF(to <= from, std::invalid_argument)
    << "[QNNGraph] empty row range " << from << ".." << to;

  unsigned int pos = from;
  while (pos < to) {
    const unsigned int need = to - pos;
    unsigned int bucket = buckets.back();
    for (auto b : buckets) {
      if (b >= need) {
        bucket = b;
        break;
      }
    }
    const unsigned int m = std::min(need, bucket);
    GraphExec &ge = getExec(context, bucket);

    double t0 = profile ? now_us() : 0.0;
    for (size_t i = 0; i < context.getNumInputs(); ++i) {
      Tensor &in = context.getInput(i);
      const size_t rb = tensorRowBytes(in);
      const char *src = in.getData<char>() + (size_t)pos * rb;
      char *dst = static_cast<char *>(ge.in_ptrs[i]);
      std::memcpy(dst, src, (size_t)m * rb);
      if (m < bucket)
        std::memset(dst + (size_t)m * rb, 0, (size_t)(bucket - m) * rb);
    }
    if (profile)
      t_copy_in += now_us() - t0;

    execute(context, ge);

    t0 = profile ? now_us() : 0.0;
    for (size_t i = 0; i < context.getNumOutputs(); ++i) {
      Tensor &out = context.getOutput(i);
      const size_t rb = tensorRowBytes(out);
      std::memcpy(out.getData<char>() + (size_t)pos * rb, ge.out_ptrs[i],
                  (size_t)m * rb);
    }
    if (profile)
      t_copy_out += now_us() - t0;

    pos += m;
  }
}

void QNNGraph::forwarding(RunLayerContext &context, bool training) {
  scanBuckets(context);
  if (!buckets.empty()) {
    runBucketed(context, 0, context.getInput(0).height());
    return;
  }
  GraphExec &ge = getExec(context, 0);
  bindLegacy(context, ge);
  execute(context, ge);
}

void QNNGraph::incremental_forwarding(RunLayerContext &context,
                                      unsigned int from, unsigned int to,
                                      bool training) {
  scanBuckets(context);
  if (buckets.empty()) {
    // legacy graphs run on the whole tensor, as before
    forwarding(context, training);
    return;
  }
  runBucketed(context, from, to);
}

void QNNGraph::updateTensorsByInputDimensions(
  RunLayerContext &context, std::vector<TensorDim> input_dimensions) {
  for (size_t i = 0; i < context.getNumInputs(); ++i) {
    TensorDim d = context.getInput(i).getDim();
    d.batch(input_dimensions[0].batch());
    d.height(input_dimensions[0].height());
    context.updateInput(i, d);
  }
  for (size_t i = 0; i < context.getNumOutputs(); ++i) {
    TensorDim d = context.getOutput(i).getDim();
    d.batch(input_dimensions[0].batch());
    d.height(input_dimensions[0].height());
    context.updateOutput(i, d);
  }
  // staging buffers are sized per bucket, not per sequence: nothing to redo
}

void QNNGraph::updateBufferType(std::vector<BufferTypePtr> &buffers,
                                Tensor &T) {
  Tdatatype type = T.getDataType();
  switch (type) {
  case Tdatatype::UINT4:
  case Tdatatype::UINT8:
    buffers.push_back(T.getData<uint8_t>());
    break;
  case Tdatatype::QINT8:
    // int8 activations (W8A8 path): same bytes, QNN sees them as the graph's
    // declared data type
    buffers.push_back(reinterpret_cast<uint8_t *>(T.getData<int8_t>()));
    break;
  case Tdatatype::UINT16:
    buffers.push_back(T.getData<uint16_t>());
    break;
#ifdef ENABLE_FP16
  case Tdatatype::FP16:
    buffers.push_back(reinterpret_cast<uint16_t *>(T.getData<_FP16>()));
    break;
#endif
  case Tdatatype::FP32:
    buffers.push_back(T.getData<float>());
    break;
  default:
    break;
  }
}

void QNNGraph::populateTensor(std::shared_ptr<QNNVar> qc_var,
                              Qnn_Context_Graph_t &context_i,
                              BufferTypePtr buffers, Qnn_Tensor_t *T) {
  switch (buffers.index()) {
  case 1: // uint8_t *
  {
    qc_var->m_ioTensor.populateInputTensor(std::get<uint8_t *>(buffers), T,
                                           m_inputDataType);
    qc_var->RpcMem->registerQnnTensor(std::get<uint8_t *>(buffers), *T,
                                      context_i.m_context);
  } break;
  case 2: // uint16_t*
  {
    qc_var->m_ioTensor.populateInputTensor(std::get<uint16_t *>(buffers), T,
                                           m_inputDataType);
    qc_var->RpcMem->registerQnnTensor(std::get<uint16_t *>(buffers), *T,
                                      context_i.m_context);
  } break;
  case 3: {
    qc_var->m_ioTensor.populateInputTensor(std::get<float *>(buffers), T,
                                           m_inputDataType);
    qc_var->RpcMem->registerQnnTensor(std::get<float *>(buffers), *T,
                                      context_i.m_context);
  } break;
  default:
    std::cout << "Unknown type: " << buffers.index() << std::endl;
    break;
  }
}

} // namespace nntrainer
