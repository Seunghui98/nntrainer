// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   htp_trace.cpp
 * @date   21 Sep 2026
 * @brief  Per-call HTP timeline recorder that writes a Chrome-trace JSON
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "htp_trace.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <thread>

#if defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace nntrainer {

namespace {

/** @brief pids / tids in the file. DSP tids follow the QNN optrace convention
 *  (DMA 256, HVX 512.., HMX 768) so the viewer lines the two up. */
constexpr int PID_HOST = 1;
constexpr int PID_DSP = 2;
constexpr uint32_t TID_SEAM = 20;
constexpr uint32_t TID_REGISTER = 30;
constexpr uint32_t TID_DSP_MAIN = 1;
constexpr uint32_t TID_DMA = 256;
constexpr uint32_t TID_HVX = 512;
constexpr uint32_t TID_HMX = 768;

enum Lane : uint8_t { L_MAIN = 0, L_HMX, L_HVX, L_DMA };

struct LaneInfo {
  uint32_t tid;
  const char *cat;
  const char *engine; /**< nullptr when the lane is not a compute engine */
};
const LaneInfo kLane[] = {
  {TID_DSP_MAIN, "dsp.sync", nullptr},
  {TID_HMX, "dsp.hmx", "HMX"},
  {TID_HVX, "dsp.hvx", "HVX"},
  {TID_DMA, "dsp.dma", nullptr},
};

/** @brief One timed slot of a stage_us[] layout: where it is drawn and how
 *  it is classed for the viewer's cycles/element reference. `elems` names
 *  the element count the slot processes, in the call's shape terms. */
struct StageDesc {
  uint8_t slot;
  const char *name;
  uint8_t lane;
  const char *cls;
  uint8_t elems; /**< 0 none, 1 M*K, 2 M*N*nh, 3 ops 2*M*K*N*nh */
};

/** @brief Slots that are not times (counts, bytes, strides): kept as args. */
struct ArgDesc {
  uint8_t slot;
  const char *name;
};

struct Layout {
  const char *entry;  /**< FastRPC entry point name */
  const char *op;     /**< short op name for args.op / labels */
  uint8_t total_slot; /**< DSP_TOTAL */
  bool has_mm;        /**< a measured mm slot exists; else residual = mm */
  const StageDesc *stages;
  unsigned n_stages;
  const ArgDesc *args;
  unsigned n_args;
};

/* HTP_T_*: [dsp_total, quant, dequant, acc_read, acc_copy, drain, acc_stride]
 */
const StageDesc kFcStages[] = {
  {1, "quant f32->u8 AH", L_HVX, "quant", 1},
  {3, "acc_read", L_HMX, "matmul", 0},
  {4, "acc_copy", L_HVX, "elementwise", 2},
  {2, "dequant i32->f32", L_HVX, "dequant", 2},
  {5, "dma drain", L_DMA, "dma", 0},
};
const ArgDesc kFcArgs[] = {{6, "acc_stride"}};

/* HTP_FU_T_*: [total, quant, swiglu, dequant, acc_read, acc_copy, drain,
 * stride] */
const StageDesc kFusedStages[] = {
  {1, "quant f32->u8 AH", L_HVX, "quant", 1},
  {4, "acc_read", L_HMX, "matmul", 0},
  {5, "acc_copy", L_HVX, "elementwise", 2},
  {2, "swiglu", L_HVX, "elementwise", 0},
  {3, "dequant i32->f32", L_HVX, "dequant", 2},
  {6, "dma drain", L_DMA, "dma", 0},
};
const ArgDesc kFusedArgs[] = {{7, "acc_stride"}};

/* HTP_GU_T_*: [total, quant, swiglu, dequant, acc_read, drain, stride] */
const StageDesc kGateUpStages[] = {
  {1, "quant f32->u8 AH", L_HVX, "quant", 1},
  {4, "acc_read", L_HMX, "matmul", 0},
  {2, "swiglu", L_HVX, "elementwise", 0},
  {3, "dequant / requant u8", L_HVX, "dequant", 2},
  {5, "dma drain", L_DMA, "dma", 0},
};
const ArgDesc kGateUpArgs[] = {{6, "acc_stride"}};

/* HTP_MOE_T_*: [total, quant, swiglu, dequant, acc_read, drain, scatter,
   gather, requant, blocks, mm, dma_kb, dma_first, alloc, dma_first_kb,
   drain_dn, push, stage, acc_stride] */
const StageDesc kMoeStages[] = {
  {13, "alloc", L_MAIN, "sync", 0},
  {7, "gather rows", L_HVX, "elementwise", 1},
  {1, "quant f32->u8 AH", L_HVX, "quant", 1},
  {16, "dma push", L_DMA, "dma", 0},
  {10, "micro-mm (measured)", L_HMX, "matmul", 3},
  {5, "dma drain (gate_up)", L_DMA, "dma", 0},
  {4, "acc_read", L_HMX, "matmul", 0},
  {8, "requant", L_HVX, "quant", 0},
  {2, "swiglu", L_HVX, "elementwise", 0},
  {15, "dma drain (down)", L_DMA, "dma", 0},
  {3, "dequant i32->f32", L_HVX, "dequant", 2},
  {6, "scatter-add", L_HVX, "elementwise", 2},
  {17, "stage copies", L_MAIN, "sync", 0},
};
const ArgDesc kMoeArgs[] = {{9, "blocks"},
                            {11, "dma_kb"},
                            {12, "dma_first_us"},
                            {14, "dma_first_kb"},
                            {18, "acc_stride"}};

const Layout kLayouts[HtpTrace::KIND_N] = {
  {"mm_u8i4_layer", "fc", 0, false, kFcStages, 5, kFcArgs, 1},
  {"mm_u8i4_layer_fused", "fused_ffn", 0, false, kFusedStages, 6, kFusedArgs,
   1},
  {"mm_u8i4_gate_up_swiglu", "gate_up_swiglu", 0, false, kGateUpStages, 5,
   kGateUpArgs, 1},
  {"mm_u8i4_moe_layer", "moe_ffn", 0, true, kMoeStages, 13, kMoeArgs, 5},
};

const char *kTokenNames[] = {"prefill", "decode"};

} // namespace

HtpTrace &HtpTrace::global() {
  static HtpTrace instance;
  return instance;
}

HtpTrace::HtpTrace() {
  const char *env = std::getenv("NNTR_TRACE");
  if (env == nullptr || *env == '\0')
    return;
  // "path[,level=...]": the level suffix is accepted for forward
  // compatibility with the ring-buffer levels and ignored here.
  const char *comma = std::strchr(env, ',');
  path_ = comma ? std::string(env, comma - env) : std::string(env);
  enabled_ = !path_.empty();
  epoch_us_ = nowUs();
  calls_.reserve(1 << 14);
}

HtpTrace::~HtpTrace() { write(); }

uint64_t HtpTrace::nowUs() {
  return static_cast<uint64_t>(
    std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::steady_clock::now().time_since_epoch())
      .count());
}

uint32_t HtpTrace::currentTid() {
#if defined(__linux__)
  return static_cast<uint32_t>(::syscall(SYS_gettid));
#else
  return static_cast<uint32_t>(
    std::hash<std::thread::id>()(std::this_thread::get_id()) & 0x7fffffffu);
#endif
}

void HtpTrace::setMeta(int profile_level, int qos_mode) {
  std::lock_guard<std::mutex> lock(mutex_);
  profile_level_ = profile_level;
  qos_mode_ = qos_mode;
}

void HtpTrace::call(Kind kind, unsigned M, unsigned K, unsigned N,
                    unsigned n_handles, uint64_t t0_us, uint64_t host_us,
                    const uint32_t *stage_us, unsigned n_stages,
                    uint64_t bytes_in, uint64_t bytes_out) {
  if (!enabled_)
    return;
  Call c{};
  c.t0 = t0_us;
  c.host_us = host_us;
  c.bytes_in = bytes_in;
  c.bytes_out = bytes_out;
  c.M = M;
  c.K = K;
  c.N = N;
  c.n_handles = n_handles;
  c.tid = currentTid();
  c.kind = static_cast<uint8_t>(kind);
  c.timed = stage_us != nullptr;
  if (c.timed) {
    if (n_stages > kMaxStages)
      n_stages = kMaxStages;
    c.n_stages = static_cast<uint8_t>(n_stages);
    std::memcpy(c.stage, stage_us, n_stages * sizeof(uint32_t));
  }
  std::lock_guard<std::mutex> lock(mutex_);
  calls_.push_back(c);
}

void HtpTrace::staging(uint64_t t0_us, uint64_t us, uint64_t bytes) {
  if (!enabled_)
    return;
  std::lock_guard<std::mutex> lock(mutex_);
  spans_.push_back(Span{t0_us, us, bytes, 0, currentTid(), 0});
}

void HtpTrace::registration(uint64_t t0_us, uint64_t total_us,
                            uint64_t convert_us, uint64_t rpc_us, unsigned K,
                            unsigned N) {
  if (!enabled_)
    return;
  std::lock_guard<std::mutex> lock(mutex_);
  spans_.push_back(
    Span{t0_us, total_us, convert_us | (static_cast<uint64_t>(K) << 32),
         rpc_us | (static_cast<uint64_t>(N) << 32), currentTid(), 1});
}

void HtpTrace::phase(uint64_t t0_us, uint64_t us, unsigned from, unsigned to) {
  if (!enabled_)
    return;
  std::lock_guard<std::mutex> lock(mutex_);
  spans_.push_back(Span{t0_us, us, from, to, currentTid(), 2});
}

namespace {

struct Writer {
  FILE *f;
  bool first = true;
  uint64_t epoch;

  void sep() {
    if (!first)
      std::fputs(",\n", f);
    first = false;
  }
  static double rel(uint64_t t, uint64_t epoch) {
    return t >= epoch ? static_cast<double>(t - epoch) : 0.0;
  }
  void meta(int pid, int tid, const char *what, const char *name, int sort) {
    sep();
    if (tid < 0)
      std::fprintf(
        f,
        "{\"ph\":\"M\",\"pid\":%d,\"name\":\"%s\",\"args\":{\"name\":\"%s\"}}",
        pid, what, name);
    else
      std::fprintf(f,
                   "{\"ph\":\"M\",\"pid\":%d,\"tid\":%d,\"name\":\"%s\","
                   "\"args\":{\"name\":\"%s\"}}",
                   pid, tid, what, name);
    sep();
    if (tid < 0)
      std::fprintf(f,
                   "{\"ph\":\"M\",\"pid\":%d,\"name\":\"process_sort_index\","
                   "\"args\":{\"sort_index\":%d}}",
                   pid, sort);
    else
      std::fprintf(f,
                   "{\"ph\":\"M\",\"pid\":%d,\"tid\":%d,\"name\":\"thread_sort_"
                   "index\",\"args\":{\"sort_index\":%d}}",
                   pid, tid, sort);
  }
  /** @brief Opens an X event; the caller appends args with arg*() then end().
   */
  void begin(int pid, uint32_t tid, const char *name, const char *cat,
             uint64_t t0, uint64_t us) {
    sep();
    std::fprintf(f,
                 "{\"ph\":\"X\",\"pid\":%d,\"tid\":%u,\"name\":\"%s\",\"cat\":"
                 "\"%s\",\"ts\":%.3f,\"dur\":%.3f,\"args\":{",
                 pid, tid, name, cat, rel(t0, epoch), static_cast<double>(us));
    argFirst = true;
  }
  bool argFirst = true;
  void argSep() {
    if (!argFirst)
      std::fputc(',', f);
    argFirst = false;
  }
  void argS(const char *k, const char *v) {
    argSep();
    std::fprintf(f, "\"%s\":\"%s\"", k, v);
  }
  void argU(const char *k, uint64_t v) {
    argSep();
    std::fprintf(f, "\"%s\":%llu", k, static_cast<unsigned long long>(v));
  }
  void argD(const char *k, double v) {
    argSep();
    std::fprintf(f, "\"%s\":%.3f", k, v);
  }
  void end() { std::fputs("}}", f); }
};

uint64_t elemsOf(uint8_t which, unsigned M, unsigned K, unsigned N,
                 unsigned nh) {
  switch (which) {
  case 1:
    return static_cast<uint64_t>(M) * K;
  case 2:
    return static_cast<uint64_t>(M) * N * nh;
  case 3:
    return 2ull * M * K * N * nh;
  default:
    return 0;
  }
}

} // namespace

void HtpTrace::write() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!enabled_ || written_)
    return;
  written_ = true;
  FILE *f = std::fopen(path_.c_str(), "w");
  if (f == nullptr) {
    std::fprintf(stderr, "[HTP-TRACE] cannot open %s for writing\n",
                 path_.c_str());
    return;
  }
  Writer w{f, true, epoch_us_};
  std::fputs("{\"displayTimeUnit\":\"ms\",\"traceEvents\":[\n", f);

  // --- names ---
  w.meta(PID_HOST, -1, "process_name", "host (CPU)", 0);
  w.meta(PID_DSP, -1, "process_name", "HTP (cDSP, stage totals)", 1);
  std::vector<uint32_t> host_tids;
  auto seen = [&](uint32_t tid) {
    for (uint32_t t : host_tids)
      if (t == tid)
        return true;
    host_tids.push_back(tid);
    return false;
  };
  for (const Call &c : calls_)
    if (!seen(c.tid)) {
      char name[48];
      std::snprintf(name, sizeof(name), "host tid %u", c.tid);
      w.meta(PID_HOST, static_cast<int>(c.tid), "thread_name", name,
             host_tids.size() == 1 ? 0 : 1);
    }
  for (const Span &s : spans_)
    if (!seen(s.tid)) {
      char name[48];
      std::snprintf(name, sizeof(name), "host tid %u", s.tid);
      w.meta(PID_HOST, static_cast<int>(s.tid), "thread_name", name, 1);
    }
  w.meta(PID_HOST, TID_SEAM, "thread_name", "fastrpc seam", 5);
  w.meta(PID_HOST, TID_REGISTER, "thread_name", "weight register", 6);
  w.meta(PID_DSP, TID_DSP_MAIN, "thread_name", "dsp main (caller)", 0);
  w.meta(PID_DSP, TID_HMX, "thread_name", "HMX", 1);
  w.meta(PID_DSP, TID_HVX, "thread_name", "HVX (pool total)", 2);
  w.meta(PID_DSP, TID_DMA, "thread_name", "DMA", 10);

  // --- phases, staging, registration ---
  unsigned token = 0;
  for (const Span &s : spans_) {
    if (s.what == 0) {
      w.begin(PID_HOST, s.tid, "staging memcpy", "host.cpu", s.t0, s.us);
      w.argS("engine", "CPU");
      w.argU("bytes", s.a);
      w.end();
    } else if (s.what == 1) {
      const uint64_t convert = s.a & 0xffffffffu, rpc = s.b & 0xffffffffu;
      const unsigned K = static_cast<unsigned>(s.a >> 32),
                     N = static_cast<unsigned>(s.b >> 32);
      w.begin(PID_HOST, TID_REGISTER, "weight register", "host.load", s.t0,
              s.us);
      w.argS("engine", "CPU");
      w.argU("K", K);
      w.argU("N", N);
      w.argU("convert_us", convert);
      w.argU("rpc_us", rpc);
      w.argU("host_tid", s.tid);
      w.end();
    } else {
      const bool prefill = s.b > s.a + 1;
      char name[48];
      if (prefill)
        std::snprintf(name, sizeof(name), "prefill (%u tokens)",
                      static_cast<unsigned>(s.b - s.a));
      else
        std::snprintf(name, sizeof(name), "decode token %u", ++token);
      w.begin(PID_HOST, s.tid, name, "host.phase", s.t0, s.us);
      w.argS("phase", kTokenNames[prefill ? 0 : 1]);
      if (!prefill)
        w.argU("token", token);
      w.argU("from", s.a);
      w.argU("to", s.b);
      w.end();
    }
  }

  // --- calls ---
  for (const Call &c : calls_) {
    const Layout &L =
      kLayouts[c.kind < static_cast<uint8_t>(KIND_N) ? c.kind : 0];
    const uint64_t dsp =
      (c.timed && L.total_slot < c.n_stages) ? c.stage[L.total_slot] : 0;
    const uint64_t transport = c.host_us > dsp ? c.host_us - dsp : 0;
    const uint64_t half = transport / 2;
    char name[96];
    std::snprintf(name, sizeof(name), "wait %s", L.entry);

    w.begin(PID_HOST, c.tid, name, "host.wait", c.t0, c.host_us);
    w.argS("op", L.op);
    w.argS("engine", "HTP");
    w.argU("M", c.M);
    w.argU("K", c.K);
    w.argU("N", c.N);
    w.argU("n_handles", c.n_handles);
    w.argU("bytes_in", c.bytes_in);
    w.argU("bytes_out", c.bytes_out);
    w.argU("host_us", c.host_us);
    w.argU("dsp_us", dsp);
    w.argU("transport_us", transport);
    if (!c.timed)
      w.argS("note", "untimed entry: no DSP breakdown");
    w.end();

    std::snprintf(name, sizeof(name), "marshal %s", L.entry);
    w.begin(PID_HOST, TID_SEAM, name, "host.rpc", c.t0, half);
    w.argS("op", L.op);
    w.argU("bytes", c.bytes_in);
    w.end();
    std::snprintf(name, sizeof(name), "return %s", L.entry);
    w.begin(PID_HOST, TID_SEAM, name, "host.rpc", c.t0 + half + dsp,
            c.host_us - half - dsp);
    w.argS("op", L.op);
    w.argU("bytes", c.bytes_out);
    w.end();

    if (!c.timed || dsp == 0)
      continue;
    const uint64_t d0 = c.t0 + half;
    std::snprintf(name, sizeof(name), "nntr_hvx_%s", L.entry);
    w.begin(PID_DSP, TID_DSP_MAIN, name, "dsp.entry", d0, dsp);
    w.argS("op", L.op);
    w.argU("M", c.M);
    w.argU("K", c.K);
    w.argU("N", c.N);
    w.argU("n_handles", c.n_handles);
    w.argU("dsp_total_us", dsp);
    for (unsigned i = 0; i < L.n_args; ++i)
      if (L.args[i].slot < c.n_stages)
        w.argU(L.args[i].name, c.stage[L.args[i].slot]);
    w.argS("layout", "sequential: DSP stage totals laid out back to back (no "
                     "lane overlap at this level)");
    w.end();

    // stages back to back on their lanes
    uint64_t t = d0, sum = 0;
    for (unsigned i = 0; i < L.n_stages; ++i) {
      const StageDesc &s = L.stages[i];
      if (s.slot >= c.n_stages)
        continue;
      const uint64_t us = c.stage[s.slot];
      if (us == 0)
        continue;
      const LaneInfo &lane = kLane[s.lane];
      w.begin(PID_DSP, lane.tid, s.name, lane.cat, t, us);
      w.argS("op", L.op);
      w.argS("stage", s.name);
      w.argS("class", s.cls);
      if (lane.engine)
        w.argS("engine", lane.engine);
      const uint64_t e = elemsOf(s.elems, c.M, c.K, c.N, c.n_handles);
      if (e) {
        if (s.elems == 3) {
          w.argU("ops", e);
          w.argU("elems", e);
        } else
          w.argU("elems", e);
      }
      w.argU("M", c.M);
      w.argU("K", c.K);
      w.argU("N", c.N);
      w.argU("n_handles", c.n_handles);
      w.end();
      t += us;
      sum += us;
    }
    if (dsp > sum) {
      const uint64_t rest = dsp - sum;
      if (!L.has_mm) {
        // No slot times the HMX issue loop on this entry; the DSP total minus
        // every named stage is the matmul plus whatever the probes do not
        // name (HtpProfile prints the same number as "mm").
        w.begin(PID_DSP, TID_HMX, "micro-mm (residual)", "dsp.hmx", t, rest);
        w.argS("op", L.op);
        w.argS("stage", "mm_residual");
        w.argS("class", "matmul");
        w.argS("engine", "HMX");
        const uint64_t ops = elemsOf(3, c.M, c.K, c.N, c.n_handles);
        w.argU("ops", ops);
        w.argU("elems", ops);
        w.argU("M", c.M);
        w.argU("K", c.K);
        w.argU("N", c.N);
        w.argU("n_handles", c.n_handles);
        w.end();
      } else {
        w.begin(PID_DSP, TID_DSP_MAIN, "unaccounted", "dsp.sync", t, rest);
        w.argS("op", L.op);
        w.argS("stage", "rest");
        w.end();
      }
    }
  }

  std::fprintf(
    f,
    "\n],\"metadata\":{\"tool\":\"htp_trace (host per-call timing; DSP stage "
    "totals laid out sequentially)\","
    "\"level\":\"stage-seq\",\"profile_level\":%d,\"qos_mode\":%d,\"htp_"
    "enabled\":true,"
    "\"dsp_clock_mhz\":1200,\"dsp_clock_mhz_note\":\"assumed for "
    "cycles/element; no PMU cycles at this level\","
    "\"hvx_threads\":6,\"budgets\":{\"VTCM "
    "(KB)\":8192},\"dropped\":{\"host\":0,\"dsp\":0},"
    "\"clock_sync\":{\"method\":\"host clock only; DSP spans placed inside "
    "their host call\",\"offset_us\":0,\"rtt_us\":0,\"violations\":0},"
    "\"calls\":%llu,\"untimed_calls\":%llu}}\n",
    profile_level_, qos_mode_, static_cast<unsigned long long>(calls_.size()),
    static_cast<unsigned long long>([&] {
      uint64_t n = 0;
      for (const Call &c : calls_)
        if (!c.timed)
          ++n;
      return n;
    }()));
  std::fclose(f);
  std::fprintf(stderr, "[HTP-TRACE] wrote %s: %llu calls, %llu spans\n",
               path_.c_str(), static_cast<unsigned long long>(calls_.size()),
               static_cast<unsigned long long>(spans_.size()));
}

} // namespace nntrainer
