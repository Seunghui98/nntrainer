// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   main.cpp
 * @date   14 July 2026
 * @brief  YOLOv7ReIDtiny pose+ReID (320x320, nkpt=87) inference on nntrainer.
 *
 *         Presets (env YOLO_TENSOR_TYPE):
 *           w32a32   : FP32 weights + FP32 activations  (stage 1, reference)
 *           w8a32    : Q8_0 weights + FP32 activations, NHWC (stage 2)
 *         Weights are loaded from a safetensors produced by weight_converter.py
 *         (and, for w8a32, quantized by nntr_quantize --conv_dtype Q8_0).
 *
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "yolov7_pose_graph.h"
#include "rms_norm.h"
#include "rtmcc_gau.h"
#include <app_context.h>
#include <engine.h>
#include <layer.h>
#include <model.h>
#include <tensor_api.h>

extern "C" void openblas_set_num_threads(int);

using ml::train::createLayer;
using ml::train::Tensor;
using ModelHandle = std::unique_ptr<ml::train::Model>;

namespace {

using namespace yolov7_pose;

constexpr int INPUT_SIZE = 320;

std::vector<float> loadBin(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f)
    throw std::runtime_error("Cannot open: " + path);
  f.seekg(0, std::ios::end);
  size_t n = f.tellg() / sizeof(float);
  f.seekg(0);
  std::vector<float> v(n);
  f.read(reinterpret_cast<char *>(v.data()), n * sizeof(float));
  return v;
}

struct Keypoint {
  float x, y, score;
};

// Decode RTMPose SimCC: pose is [2*NKPT, SIMCC_BINS] (cls_x rows then cls_y).
std::vector<Keypoint> decodeSimcc(const float *pose) {
  std::vector<Keypoint> kpts(NKPT);
  for (int k = 0; k < NKPT; ++k) {
    const float *rx = pose + static_cast<size_t>(k) * SIMCC_BINS;
    const float *ry = pose + static_cast<size_t>(NKPT + k) * SIMCC_BINS;
    int bx = 0, by = 0;
    float mx = rx[0], my = ry[0];
    for (int i = 1; i < SIMCC_BINS; ++i) {
      if (rx[i] > mx) { mx = rx[i]; bx = i; }
      if (ry[i] > my) { my = ry[i]; by = i; }
    }
    float score = std::min(mx, my);
    if (score <= 0.0f) {
      kpts[k] = {-1.0f, -1.0f, score};
    } else {
      kpts[k] = {bx / 2.0f, by / 2.0f, score};
    }
  }
  return kpts;
}

void printPeakRSS() {
#ifdef __linux__
  std::ifstream f("/proc/self/status");
  std::string line;
  while (std::getline(f, line))
    if (line.rfind("VmHWM:", 0) == 0)
      std::cout << "  " << line << std::endl;
#endif
}

} // namespace

int main(int argc, char **argv) {
  openblas_set_num_threads(4);

  std::string res_dir = (argc > 1) ? argv[1] : ".";
  std::string input_path =
    (argc > 2) ? argv[2] : res_dir + "/input_320.bin";
  std::cout << "[Pose] RES_DIR=" << res_dir << " INPUT=" << input_path
            << std::endl;

  try {
    auto &app_ctx = nntrainer::AppContext::Global();
    (void)app_ctx;

    // Register custom layers used by the pose head.
    {
      auto &ct_engine = nntrainer::Engine::Global();
      auto ctx = static_cast<nntrainer::AppContext *>(
        ct_engine.getRegisteredContext("cpu"));
      auto tryReg = [&](auto fn) {
        try {
          ctx->registerFactory(fn);
        } catch (std::invalid_argument &e) {
          std::cerr << "register: " << e.what() << std::endl;
        }
      };
      tryReg(nntrainer::createLayer<quick_ai::RMSNormLayer>);
      tryReg(nntrainer::createLayer<quick_ai::RTMCCGauLayer>);
    }

    ModelHandle model =
      ml::train::createModel(ml::train::ModelType::NEURAL_NET);
    model->setProperty({nntrainer::withKey("batch_size", "1")});

    bool preset_nhwc = false;
    bool preset_q = false;
    std::string weights_default = "yolov7_pose.safetensors";
    if (const char *tt = std::getenv("YOLO_TENSOR_TYPE")) {
      std::string s = tt;
      if (s == "w8a32" || s == "W8A32") {
        model->setProperty(
          {nntrainer::withKey("model_tensor_type", "FP32-FP32")});
        preset_nhwc = true;
        preset_q = true;
        yolov7_pose::quantWeightDtype() = "Q8_0";
        weights_default = "yolov7_pose_q8_0.safetensors";
        std::cout << "[Pose] Preset=w8a32 (Q8_0 weights + FP32 act + NHWC)"
                  << std::endl;
      } else { // w32a32 default
        model->setProperty(
          {nntrainer::withKey("model_tensor_type", "FP32-FP32")});
        std::cout << "[Pose] Preset=w32a32 (FP32 weights + FP32 act + NCHW)"
                  << std::endl;
      }
    } else {
      model->setProperty(
        {nntrainer::withKey("model_tensor_type", "FP32-FP32")});
      std::cout << "[Pose] Preset=w32a32 (default)" << std::endl;
    }

    bool format_nhwc = false;
    // YOLO_FORCE_NCHW=1 keeps NCHW even for NHWC presets (x86 has no NHWC
    // quantized-conv kernel; used to check Q8_0 numerics on x86).
    if (std::getenv("YOLO_FORCE_NCHW"))
      preset_nhwc = false;
    if (preset_nhwc || std::getenv("YOLO_NHWC")) {
      model->setProperty({nntrainer::withKey("tensor_format", "NHWC")});
      format_nhwc = true;
      std::cout << "[Pose] tensor_format=NHWC" << std::endl;
    }

    auto x = Tensor(
      ml::train::TensorDim(1, 3, INPUT_SIZE, INPUT_SIZE,
                           ml::train::TensorDim::Format::NCHW,
                           ml::train::TensorDim::DataType::FP32),
      "input0");

    auto nodes = yolov7_pose::buildBackbone(x, preset_q);
    auto pose_feat = yolov7_pose::buildNeck("backbone.features", nodes, preset_q);
    auto reid_feat =
      yolov7_pose::buildNeck("backbone.features_feat", nodes, preset_q);
    auto pose_out = yolov7_pose::buildPoseHead(pose_feat, preset_q);
    auto reid_out = yolov7_pose::buildReidHead(reid_feat);

    std::vector<Tensor> graph_outputs = {pose_out, reid_out};
    if (int ret = model->compile(x, graph_outputs,
                                 ml::train::ExecutionMode::INFERENCE))
      throw std::runtime_error("compile failed: " + std::to_string(ret));

    std::string weights_path = res_dir + "/" + weights_default;
    if (const char *w = std::getenv("YOLO_WEIGHTS"))
      weights_path = (w[0] == '/') ? std::string(w) : res_dir + "/" + w;
    model->load(weights_path,
                ml::train::ModelFormat::MODEL_FORMAT_SAFETENSORS);
    std::cout << "[Pose] Weights loaded: " << weights_path << std::endl;

    auto input = loadBin(input_path);
    if (input.size() != static_cast<size_t>(3 * INPUT_SIZE * INPUT_SIZE))
      throw std::runtime_error("input size mismatch: " +
                               std::to_string(input.size()));

    if (format_nhwc) {
      const int C = 3, H = INPUT_SIZE, W = INPUT_SIZE;
      std::vector<float> nhwc(input.size());
      for (int c = 0; c < C; ++c)
        for (int h = 0; h < H; ++h)
          for (int w = 0; w < W; ++w)
            nhwc[(h * W + w) * C + c] = input[(c * H + h) * W + w];
      input.swap(nhwc);
    }

    std::vector<float *> in_ptr = {input.data()};
    int iters = std::getenv("YOLO_BENCH_ITERS")
                  ? std::max(1, std::atoi(std::getenv("YOLO_BENCH_ITERS")))
                  : 1;
    std::vector<float *> outs;
    double total_ms = 0.0;
    for (int it = 0; it < iters; ++it) {
      auto t0 = std::chrono::steady_clock::now();
      outs = model->inference(1, in_ptr, std::vector<float *>());
      auto t1 = std::chrono::steady_clock::now();
      total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    std::cout << "[Pose] Inference done, outputs=" << outs.size()
              << " avg=" << (total_ms / iters) << " ms" << std::endl;
    printPeakRSS();

    // outs[0] = pose [1, 2*NKPT, SIMCC_BINS], outs[1] = reid [1, EMBED_DIM]
    if (const char *dump = std::getenv("POSE_DUMP")) {
      std::string base = dump;
      std::ofstream pf(base + "_pose.bin", std::ios::binary);
      pf.write(reinterpret_cast<const char *>(outs[0]),
               sizeof(float) * 2 * NKPT * SIMCC_BINS);
      std::ofstream rf(base + "_reid.bin", std::ios::binary);
      rf.write(reinterpret_cast<const char *>(outs[1]),
               sizeof(float) * EMBED_DIM);
      std::cout << "[Pose] raw outputs dumped to " << base << "_{pose,reid}.bin"
                << std::endl;
    }

    auto kpts = decodeSimcc(outs[0]);

    float score_thr =
      std::getenv("POSE_THR") ? std::stof(std::getenv("POSE_THR")) : 0.5f;
    int visible = 0;
    for (auto &k : kpts)
      if (k.score >= score_thr && k.x >= 0)
        ++visible;
    std::cout << "[Pose] visible keypoints: " << visible << "/" << NKPT
              << " (thr=" << score_thr << ")" << std::endl;

    std::cout << "[";
    for (int k = 0; k < NKPT; ++k) {
      if (k)
        std::cout << ",";
      std::printf("\n  {\"i\": %d, \"x\": %.4g, \"y\": %.4g, \"s\": %.4g}", k,
                  kpts[k].x, kpts[k].y, kpts[k].score);
    }
    std::cout << "\n]" << std::endl;

    // reid embedding norm (sanity)
    float norm = 0.0f;
    for (int i = 0; i < EMBED_DIM; ++i)
      norm += outs[1][i] * outs[1][i];
    std::cout << "[ReID] embed_dim=" << EMBED_DIM << " l2norm=" << std::sqrt(norm)
              << std::endl;

    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}
