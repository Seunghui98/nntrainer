// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   profiler_lite.h
 * @date   20 July 2026
 * @brief  Env-gated, zero-dependency per-layer / per-op wall-clock profiler.
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NaN or Inf
 *
 * @details A minimal opt-in profiler for inference-time hot-spot analysis that
 * does NOT require building nntrainer with ENABLE_PROFILE. It is enabled only
 * when the environment variable NNTR_LAYER_PROFILE is set; when unset every
 * entry point is a single bool test, so it is safe to leave the instrumentation
 * compiled in.
 *
 * Two granularities are recorded and rendered hierarchically:
 *   - layer total : wall time of one layer's forward, keyed by layer name
 *                   (recorded by the graph forward loop).
 *   - op          : a named phase inside a layer (e.g. a conv's gather / matmul
 *                   / epilogue), rendered as a "- " sub-item under its layer.
 * Op scopes deep in the call stack attribute themselves to the layer currently
 * executing via a thread-local "current layer" that the layer sets on entry
 * (layer forward runs single-threaded at layer granularity; intra-layer
 * parallelism starts below the scope, so the scope times the whole phase).
 */

#ifndef __NNTR_PROFILER_LITE_H__
#define __NNTR_PROFILER_LITE_H__

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <ostream>
#include <string>
#include <vector>

namespace nntrainer {

/**
 * @brief Process-wide accumulator for the lite profiler.
 */
class LiteProf {
public:
  static LiteProf &get() {
    static LiteProf p;
    return p;
  }

  /** @brief Whether NNTR_LAYER_PROFILE is set (profiling active). */
  bool on() const { return enabled_; }

  /** @brief Accumulate one layer-forward wall time (ms), keyed by layer name. */
  void addLayer(const std::string &name, const std::string &type, double ms) {
    std::lock_guard<std::mutex> lk(mtx_);
    LayerAcc &a = fetch(name);
    a.ms += ms;
    a.n += 1;
    if (a.type.empty())
      a.type = type;
  }

  /** @brief Accumulate one op wall time (ms) under a layer. */
  void addOp(const std::string &layer, const std::string &op, double ms) {
    std::lock_guard<std::mutex> lk(mtx_);
    OpAcc &o = fetch(layer).ops[op];
    o.ms += ms;
    o.n += 1;
  }

  /** @brief Drop everything recorded so far (e.g. after an untimed warmup). */
  void reset() {
    std::lock_guard<std::mutex> lk(mtx_);
    layers_.clear();
    order_.clear();
  }

  /** @brief Render a report sorted by layer total time, descending. */
  void report(std::ostream &os) {
    std::lock_guard<std::mutex> lk(mtx_);
    double grand = 0.0;
    for (auto &kv : layers_)
      grand += kv.second.ms;

    std::vector<const std::string *> keys;
    keys.reserve(order_.size());
    for (auto &n : order_)
      keys.push_back(&n);
    std::sort(keys.begin(), keys.end(),
              [&](const std::string *a, const std::string *b) {
                return layers_.at(*a).ms > layers_.at(*b).ms;
              });

    os << "\n================[ NNTR layer profile ]================\n";
    os << layers_.size() << " layers, total " << grand << " ms\n";
    for (const std::string *k : keys) {
      const LayerAcc &a = layers_.at(*k);
      const double pct = grand > 0 ? 100.0 * a.ms / grand : 0.0;
      os << "  " << pad(*k, 44) << " " << fixed3(a.ms) << " ms  " << fixed1(pct)
         << "%";
      if (a.n > 1)
        os << "  (x" << a.n << ", " << fixed3(a.ms / a.n) << " avg)";
      if (!a.type.empty())
        os << "  [" << a.type << "]";
      os << "\n";
      // ops within the layer, largest first
      std::vector<const std::pair<const std::string, OpAcc> *> ops;
      for (auto &op : a.ops)
        ops.push_back(&op);
      std::sort(ops.begin(), ops.end(),
                [](const auto *x, const auto *y) {
                  return x->second.ms > y->second.ms;
                });
      for (auto *op : ops)
        os << "      - " << pad(op->first, 20) << " " << fixed3(op->second.ms)
           << " ms\n";
    }
    os << "=====================================================\n";
  }

  /** @brief Set the layer whose forward is currently running (this thread).
   *  A no-op when profiling is off, so it is free to call unconditionally. */
  static void setCurrent(const std::string &n) {
    if (get().on())
      current() = n;
  }
  /** @brief The current layer name (empty if none / profiling off). */
  static const std::string &getCurrent() { return current(); }

private:
  LiteProf() { enabled_ = std::getenv("NNTR_LAYER_PROFILE") != nullptr; }

  static std::string &current() {
    thread_local std::string c;
    return c;
  }

  struct OpAcc {
    double ms = 0;
    long n = 0;
  };
  struct LayerAcc {
    double ms = 0;
    long n = 0;
    std::string type;
    std::map<std::string, OpAcc> ops;
  };

  LayerAcc &fetch(const std::string &name) {
    auto it = layers_.find(name);
    if (it == layers_.end()) {
      order_.push_back(name);
      it = layers_.emplace(name, LayerAcc{}).first;
    }
    return it->second;
  }

  static std::string pad(const std::string &s, size_t w) {
    if (s.size() >= w)
      return s;
    return s + std::string(w - s.size(), ' ');
  }
  static std::string fixed3(double v) {
    char b[32];
    std::snprintf(b, sizeof(b), "%.3f", v);
    return b;
  }
  static std::string fixed1(double v) {
    char b[32];
    std::snprintf(b, sizeof(b), "%.1f", v);
    return b;
  }

  bool enabled_;
  std::mutex mtx_;
  std::map<std::string, LayerAcc> layers_;
  std::vector<std::string> order_;
};

/**
 * @brief RAII scope timer. Two forms:
 *   LiteScope s(layer_name, type, true);  // 3-arg: whole-layer total
 *   LiteScope s(layer_name, op_name);     // 2-arg: one op under a layer
 * Both are no-ops (a single bool copy) when profiling is off.
 */
struct LiteScope {
  bool on_;
  bool is_layer_;
  std::string layer_;
  std::string tag_; // op name, or (for a layer scope) the layer type
  std::chrono::high_resolution_clock::time_point t0_;

  LiteScope(const std::string &layer, const std::string &type, bool /*layer*/)
    : on_(LiteProf::get().on()), is_layer_(true), layer_(layer), tag_(type) {
    if (on_)
      t0_ = std::chrono::high_resolution_clock::now();
  }
  LiteScope(const std::string &layer, const std::string &op)
    : on_(LiteProf::get().on()), is_layer_(false), layer_(layer), tag_(op) {
    if (on_)
      t0_ = std::chrono::high_resolution_clock::now();
  }
  ~LiteScope() {
    if (!on_)
      return;
    const double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - t0_)
                        .count();
    if (is_layer_)
      LiteProf::get().addLayer(layer_, tag_, ms);
    else
      LiteProf::get().addOp(layer_, tag_, ms);
  }

  LiteScope(const LiteScope &) = delete;
  LiteScope &operator=(const LiteScope &) = delete;
};

} // namespace nntrainer

#endif /* __NNTR_PROFILER_LITE_H__ */
