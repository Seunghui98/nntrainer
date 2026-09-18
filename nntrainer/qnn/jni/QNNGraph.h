// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   QNNGraph.h
 * @date   10 Jan 2025
 * @brief  This is QNN Graph Layer Class of Neural Network
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __NNTR_QNNGRAPH_H__
#define __NNTR_QNNGRAPH_H__

#include <iostream>
#include <layer_impl.h>
#include <map>
#include <qnn_context_var.h>
#include <qnn_properties.h>
#include <qnn_rpc_manager.h>

namespace nntrainer {

/**
 * @brief QNN Graph layer that wraps Qualcomm Neural Network graph execution.
 *
 * Two execution modes, chosen once from the context binary's graph names:
 *
 * - legacy: the binary carries a graph named exactly like this layer. The
 *   layer's own input/output tensors are registered (rpcmem) and executed in
 *   place, as before.
 *
 * - bucketed: the binary carries "<layer name>_m<N>" graphs (built by
 *   Applications/quick_ai/tools/npu/build_decoder_graphs.py). For a forward
 *   over T tokens the smallest bucket N >= T is picked (T > max bucket runs
 *   in chunks). Rows are copied through per-bucket rpcmem staging buffers
 *   that are allocated and registered once, so incremental decoding at an
 *   arbitrary row offset works (ION registration has no offset) and no
 *   memRegister/graphRetrieve/tensor setup happens on the forward path.
 *
 * Environment:
 *   NNTR_QNN_PROFILE=1     accumulate copy-in / execute / copy-out time per
 *                          layer, print every 64 executes and at teardown
 *   NNTR_QNN_DUMP=<dir>    dump the first 4 executes' raw IO per graph
 *                          (<graph>.in<i>.<n>.raw / .out<i>.<n>.raw) for
 *                          tools/npu/verify_htp.py
 */
class QNNGraph : public LayerImpl {
public:
  using BufferTypePtr =
    std::variant<std::monostate, uint8_t *, uint16_t *, float *>;

  QNNGraph();
  ~QNNGraph();

  inline static const std::string type = "qnn_graph";

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return QNNGraph::type; };

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(InitLayerContext &context) override;

  /**
   * @copydoc Layer::supportBackwarding()
   */
  bool supportBackwarding() const override { return false; }

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  void calcDerivative(RunLayerContext &context) override {};

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::incremental_forwarding(RunLayerContext &context, unsigned
   * int from, unsigned int to, bool training)
   */
  void incremental_forwarding(RunLayerContext &context, unsigned int from,
                              unsigned int to, bool training) override;

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  void setProperty(const std::vector<std::string> &values) override;

  StatusCode makeContext(RunLayerContext &context);

  StatusCode freeContext(RunLayerContext &context);

  void read(std::ifstream &file, RunLayerContext &run_context, bool opt_var,
            ml::train::ExecutionMode mode, bool trainable,
            TensorDim::DataType defineWeightDataType, bool fsu = false,
            size_t start_offset = 0, bool read_from_offset = false,
            int file_fd = -1) override;

  /**
   * @copydoc Layer::updateTensorsByInputDimensions
   *
   * Outputs follow the input's row count (like fully_connected) so the layer
   * can sit inside an incrementally decoded transformer block; the width is
   * the graph's own.
   */
  void updateTensorsByInputDimensions(
    RunLayerContext &context, std::vector<TensorDim> input_dimensions) override;

  void updateBufferType(std::vector<BufferTypePtr> &buffers, Tensor &T);

  void populateTensor(std::shared_ptr<QNNVar> qc_var,
                      Qnn_Context_Graph_t &context_i, BufferTypePtr buffer,
                      Qnn_Tensor_t *T);

private:
  /**
   * @brief Everything needed to execute one QNN graph, built on first use.
   */
  struct GraphExec {
    qnn_wrapper_api::GraphInfo_t *graphInfo = nullptr;
    Qnn_Tensor_t *inputs = nullptr;
    Qnn_Tensor_t *outputs = nullptr;
    std::vector<size_t> in_bytes;  /**< byte length of each QNN input */
    std::vector<size_t> out_bytes; /**< byte length of each QNN output */
    std::vector<void *> in_ptrs;   /**< bound buffers (staging or tensor) */
    std::vector<void *> out_ptrs;
    unsigned int rows = 0; /**< bucket size (0 in legacy mode) */
  };

  /**
   * @brief Resolve the graph for @p rows (bucket or legacy), set it up once
   * and cache it.
   */
  GraphExec &getExec(RunLayerContext &context, unsigned int rows);

  /**
   * @brief Execute @p ge with its currently bound buffers.
   */
  void execute(RunLayerContext &context, GraphExec &ge);

  /**
   * @brief Bind (and register once) the layer's own tensors -- legacy mode.
   */
  void bindLegacy(RunLayerContext &context, GraphExec &ge);

  /**
   * @brief Run rows [from, to) of the layer IO through bucket graphs.
   */
  void runBucketed(RunLayerContext &context, unsigned int from,
                   unsigned int to);

  /**
   * @brief Apply the static quant params from the layer properties to the
   * QNN IO tensors (once per GraphExec).
   */
  void applyQuantParams(GraphExec &ge);

  /**
   * @brief Scan the context binary once for "<name>_m<N>" graphs.
   */
  void scanBuckets(RunLayerContext &context);

  void dumpIO(GraphExec &ge);

  std::tuple<std::vector<props::TensorDimension>,
             std::vector<props::TensorDataType>, std::vector<props::TensorType>,
             props::FilePath, std::vector<props::InputQuantParam>,
             std::vector<props::OutputQuantParam>>
    graph_props;

  std::vector<unsigned int> weight_idx;
  std::vector<unsigned int> tensor_idx;

  Qnn_ContextHandle_t m_context = nullptr;
  std::string bin_path;
  bool m_isContextCreated;

  iotensor::InputDataType m_inputDataType;

  std::vector<props::TensorDataType> t_dtype;
  std::vector<TensorDim> t_dims;
  std::vector<props::TensorType> t_type;

  std::vector<BufferTypePtr> currentInputBuffers;
  std::vector<BufferTypePtr> currentOutputBuffers;

  sample_app::QnnFunctionPointers m_qnnFunctionPointers;

  int counter;

  /** cached execution state: key = bucket rows (0 = legacy graph) */
  std::map<unsigned int, GraphExec> exec_cache;
  std::vector<unsigned int> buckets; /**< sorted ascending */
  bool buckets_scanned = false;
  std::map<std::string, std::pair<float, int>> in_qp, out_qp;
  bool qp_parsed = false;

  /** NNTR_QNN_PROFILE accumulators (microseconds) */
  bool profile = false;
  double t_copy_in = 0, t_exec = 0, t_copy_out = 0;
  unsigned long n_exec = 0;
  std::string dump_dir;
  std::map<std::string, unsigned int> dump_count;
};

} // namespace nntrainer

#endif
