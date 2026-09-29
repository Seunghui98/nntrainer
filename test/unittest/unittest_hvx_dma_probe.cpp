// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   unittest_hvx_dma_probe.cpp
 * @date   21 Sep 2026
 * @brief  Device probe: arena DMA rate by shape / workers / vote, and DDR
 *         bandwidth with the CPU and the DSP reading at once
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * MoeChunkReplay (#87) replays the MoE layer call's own M=1 descriptor
 * list through dma_replay, next to one real traced call, then the 16 #100
 * cells (DMA_REPLAY_X lines, docs/plans/100-dma-chunk-list.md section 3.2)
 * with a content check that fails on a stale window. TwoReaderDdr and
 * PrefetchOverlap (#90, docs/plans/90-two-reader-ddr-probe.md) measure the
 * CPU and the DSP reading DDR alone and at once, and what a CPU touch of
 * the next layer's weights during the MoE call costs the DSP and stages.
 *
 * Runs on an Android device only. Requires libnntr_hvx_skel.so on
 * ADSP_LIBRARY_PATH; run once with the vote-on skel and once with the
 * vote-off one (test/htp/build.sh, HEX_EXTRA_CFLAGS=-DNNTR_HVX_NO_BUS_VOTE).
 * Reports rather than asserts on bandwidth -- the numbers are the answer
 * (docs/plans/77-first-handoff.md section 3.4 / 3.5, LEDGER items 3, 4).
 * What it does assert: the mapping is live (checksum) and the descriptor
 * count matches the host-side plan.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sched.h>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#include <AEEStdErr.h>
#include <remote.h>

#include "htp_backend/hmx/hexkl_dma_trace.h"
#include "nntr_dma_probe_plan.h"
#include "nntr_hvx.h"
#include "nntr_moe_dma_plan.h"

namespace {

std::string hex(int err) {
  std::ostringstream os;
  os << "0x" << std::hex << std::setw(8) << std::setfill('0')
     << static_cast<unsigned>(err);
  return os.str();
}

/** @brief nntr_dma_pattern: the byte at every 64-aligned offset is 0xA5,
 *  so the DSP's 64-stride coverage checksum over any descriptor payload is
 *  0xA5 per sample whatever the plan put there; 32 bytes on sits a tag of
 *  the 4 KiB page (#100), which the replay's res[12] sums. */
inline uint8_t pattern(size_t i) {
  return nntr_dma_pattern(static_cast<uint32_t>(i));
}

struct Shape {
  const char *name;
  uint32_t row_size, nrows, src_stride;
};

const Shape kShapes[] = {
  {"i", 4096u, 256u, 4096u},
  {"i1", 1048576u, 1u, 1048576u},
  {"ii", 16384u, 64u, 57344u},
  {"iii", 8192u, 64u, 57344u},
};

/** @brief Session + two rpcmem chunks attached as arenas, the mapping path
 *  the model uses (HtpComputeOps::place, kArenaChunkMax = 256 MiB). */
class HvxDmaProbe : public ::testing::Test {
protected:
  using RpcAlloc = void *(*)(int, uint32_t, int);
  using RpcFree = void (*)(void *);
  using RpcToFd = int (*)(void *);
  using FastrpcMmap = int (*)(int, int, void *, int, size_t, int);
  using FastrpcMunmap = int (*)(int, int, void *, size_t);

  void SetUp() override {
    remote_rpc_control_unsigned_module unsigned_pd = {CDSP_DOMAIN_ID, 1};
    int err = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE,
                                     &unsigned_pd, sizeof(unsigned_pd));
    ASSERT_EQ(err, AEE_SUCCESS) << "enabling unsigned PD failed: " << hex(err);
    const std::string uri = std::string(nntr_hvx_URI) + "&_dom=cdsp";
    err = nntr_hvx_open(uri.c_str(), &handle_);
    ASSERT_EQ(err, AEE_SUCCESS)
      << "nntr_hvx_open failed: " << hex(err)
      << " -- is libnntr_hvx_skel.so on ADSP_LIBRARY_PATH?";

    auto init = (void (*)(void))dlsym(RTLD_DEFAULT, "rpcmem_init");
    alloc_ = (RpcAlloc)dlsym(RTLD_DEFAULT, "rpcmem_alloc");
    free_ = (RpcFree)dlsym(RTLD_DEFAULT, "rpcmem_free");
    to_fd_ = (RpcToFd)dlsym(RTLD_DEFAULT, "rpcmem_to_fd");
    mmap_ = (FastrpcMmap)dlsym(RTLD_DEFAULT, "fastrpc_mmap");
    munmap_ = (FastrpcMunmap)dlsym(RTLD_DEFAULT, "fastrpc_munmap");
    if (!alloc_ || !free_ || !to_fd_ || !mmap_) {
      GTEST_SKIP() << "rpcmem/fastrpc_mmap not available";
    }
    if (init) {
      init();
    }
    // 2 x 256 MiB, falling back to 2 x 128 MiB (plan section 5); the
    // bytes= field of every line says which one was in use.
    for (uint32_t bytes : {256u << 20, 128u << 20}) {
      if (TryAttach(bytes)) {
        break;
      }
    }
    ASSERT_EQ(chunks_.size(), 2u) << "could not attach two arena chunks";
  }

  bool TryAttach(uint32_t bytes) {
    Detach();
    for (int k = 0; k < 2; ++k) {
      Chunk c;
      c.bytes = bytes;
      c.buf = alloc_(25 /*RPCMEM_HEAP_ID_SYSTEM*/, 1, static_cast<int>(bytes));
      if (c.buf == nullptr) {
        std::cout << "DMA_PROBE_NOTE rpcmem_alloc(" << bytes << ") failed\n";
        Detach();
        return false;
      }
      auto *p = static_cast<uint8_t *>(c.buf);
      for (size_t i = 0; i < bytes; ++i) {
        p[i] = pattern(i);
      }
      c.fd = to_fd_(c.buf);
      const int rc = mmap_(CDSP_DOMAIN_ID, c.fd, c.buf, 0, bytes,
                           static_cast<int>(FASTRPC_MAP_FD));
      if (rc != 0) {
        std::cout << "DMA_PROBE_NOTE fastrpc_mmap rc=" << hex(rc) << "\n";
        free_(c.buf);
        Detach();
        return false;
      }
      c.mapped = true;
      const int err = nntr_hvx_arena_attach(handle_, c.fd, bytes, &c.arena);
      if (err != AEE_SUCCESS) {
        std::cout << "DMA_PROBE_NOTE arena_attach err=" << hex(err) << "\n";
        chunks_.push_back(c); // so Detach releases the mapping
        Detach();
        return false;
      }
      c.attached = true;
      chunks_.push_back(c);
    }
    return true;
  }

  void Detach() {
    for (auto &c : chunks_) {
      if (c.attached) {
        nntr_hvx_arena_detach(handle_, c.arena);
      }
      if (c.mapped && munmap_) {
        munmap_(CDSP_DOMAIN_ID, c.fd, c.buf, c.bytes);
      }
      if (c.buf) {
        free_(c.buf);
      }
    }
    chunks_.clear();
  }

  void TearDown() override {
    if (handle_) {
      Detach();
      nntr_hvx_close(handle_);
    }
  }

  struct Probe {
    double gbs = 0;
    uint64_t us = 0, bytes = 0;
    bool checksum_ok = false;
    uint32_t workers_used = 0, vote = 0, n_desc = 0;
    int err = AEE_SUCCESS;
  };

  /** @brief Passes so one call moves at least @a min_bytes. */
  uint32_t PassesFor(const Shape &s, uint32_t workers, uint64_t min_bytes,
                     uint32_t *n_desc_out = nullptr) const {
    static nntr_dma_probe_desc plan[NNTR_DMA_PROBE_MAX_DESC];
    const uint32_t vtcm_guess = ((8u << 20) / workers) & ~127u;
    const uint32_t n =
      nntr_dma_probe_plan(chunks_[0].bytes, s.row_size, s.nrows, s.src_stride,
                          workers, vtcm_guess, plan, NNTR_DMA_PROBE_MAX_DESC);
    if (n_desc_out) {
      *n_desc_out = n;
    }
    if (n == 0) {
      return 1;
    }
    const uint64_t per_pass = static_cast<uint64_t>(n) * s.row_size * s.nrows;
    return static_cast<uint32_t>((min_bytes + per_pass - 1) / per_pass);
  }

  Probe Run(uint32_t chunk, const Shape &s, uint32_t workers, uint32_t passes) {
    Probe r;
    std::vector<uint32_t> res(10, 0);
    r.err =
      nntr_hvx_dma_probe(handle_, chunks_[chunk].arena, 0, chunks_[chunk].bytes,
                         s.row_size, s.nrows, s.src_stride, workers, passes,
                         res.data(), static_cast<int>(res.size()));
    if (r.err != AEE_SUCCESS) {
      return r;
    }
    r.us = res[0];
    r.bytes = (static_cast<uint64_t>(res[2]) << 32) | res[1];
    r.workers_used = res[4];
    r.vote = res[5];
    r.n_desc = res[6];
    const uint32_t sum_bytes = std::min(res[7], res[8]);
    const uint32_t want = 0xA5u * ((sum_bytes + 63u) / 64u);
    r.checksum_ok = (res[3] == want);
    r.gbs = r.us ? static_cast<double>(r.bytes) / r.us / 1e3 : 0.0;
    return r;
  }

  /** @brief One grep-able line per cell: 3 runs, the best kept. */
  Probe Cell(const Shape &s, uint32_t workers) {
    uint32_t n_plan = 0;
    const uint32_t passes = PassesFor(s, workers, 512ull << 20, &n_plan);
    Probe best;
    for (int rep = 0; rep < 3; ++rep) {
      Probe r = Run(rep & 1, s, workers, passes);
      if (r.err != AEE_SUCCESS) {
        best = r;
        break;
      }
      if (r.gbs > best.gbs) {
        best = r;
      }
    }
    if (best.err != AEE_SUCCESS) {
      std::cout << "DMA_PROBE shape=" << s.name << " workers=" << workers
                << " skipped err=" << hex(best.err) << "\n";
      return best;
    }
    std::cout << std::fixed << std::setprecision(1)
              << "DMA_PROBE shape=" << s.name << " workers=" << workers
              << " vote=" << best.vote << " gbs=" << best.gbs
              << " us=" << best.us << " bytes=" << best.bytes
              << " checksum_ok=" << (best.checksum_ok ? "y" : "n")
              << " workers_used=" << best.workers_used
              << " n_desc=" << best.n_desc << " passes=" << passes << "\n";
    EXPECT_TRUE(best.checksum_ok) << s.name << " workers=" << workers;
    EXPECT_EQ(best.n_desc, n_plan) << s.name << " workers=" << workers;
    return best;
  }

  struct Chunk {
    void *buf = nullptr;
    uint32_t bytes = 0;
    int fd = -1;
    uint32_t arena = 0;
    bool mapped = false, attached = false;
  };

  remote_handle64 handle_ = 0;
  RpcAlloc alloc_ = nullptr;
  RpcFree free_ = nullptr;
  RpcToFd to_fd_ = nullptr;
  FastrpcMmap mmap_ = nullptr;
  FastrpcMunmap munmap_ = nullptr;
  std::vector<Chunk> chunks_;
};

/** @brief XOR-reads [p, p + n) once; the result is returned so the loads
 *  cannot be dropped. NEON 4 x 16 B per step on arm64, uint64 elsewhere. */
uint64_t stream_xor(const uint8_t *p, size_t n) {
#if defined(__ARM_NEON)
  uint8x16_t a0 = vdupq_n_u8(0), a1 = a0, a2 = a0, a3 = a0;
  size_t i = 0;
  for (; i + 64 <= n; i += 64) {
    a0 = veorq_u8(a0, vld1q_u8(p + i));
    a1 = veorq_u8(a1, vld1q_u8(p + i + 16));
    a2 = veorq_u8(a2, vld1q_u8(p + i + 32));
    a3 = veorq_u8(a3, vld1q_u8(p + i + 48));
  }
  uint8x16_t a = veorq_u8(veorq_u8(a0, a1), veorq_u8(a2, a3));
  uint64_t r = vgetq_lane_u64(vreinterpretq_u64_u8(a), 0) ^
               vgetq_lane_u64(vreinterpretq_u64_u8(a), 1);
  for (; i < n; ++i) {
    r ^= p[i];
  }
  return r;
#else
  uint64_t r = 0;
  size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    uint64_t v;
    std::memcpy(&v, p + i, 8);
    r ^= v;
  }
  for (; i < n; ++i) {
    r ^= p[i];
  }
  return r;
#endif
}

} // namespace

/**
 * @brief Section 3.4: 4 shapes x 4 worker counts, this skel's vote state.
 */
TEST_F(HvxDmaProbe, DmaProbeShapes) {
  for (const Shape &s : kShapes) {
    for (uint32_t w = 1; w <= 4; ++w) {
      Cell(s, w);
    }
  }
}

namespace {

/** @brief [#90] The LFM2 expert shape and one M=1 call's weight bytes
 *  (4 experts x (gate_up + down), 22 020 096 B), and the stage slots read
 *  (test/htp/nntr_hvx_mm_u8i4.c's MOE_N_STAGES layout). */
constexpr uint32_t kK = 2048, kI = 1792, kN = 2048, kE = 4;
constexpr int kMoeStages = 31, kStDsp = 0, kStMm = 10, kStPath = 29,
              kStFeed = 30;
constexpr uint32_t kSetsPerChunkMax = 8;

/** @brief moe_set_opts' word (unittest_hvx_mm_u8i4.cpp MoeGemvOpts):
 *  GEMV path, lead / rows1 / feed authoritative, lead in 64 KB units. */
uint32_t GemvOpts(uint32_t lead_kb, bool rows1, bool feed) {
  return 1u | 0x80u | 0x40u | 0x20u | (((lead_kb / 64u) & 0xFFu) << 8) |
         (rows1 ? 0x10000u : 0u) | (feed ? 0x20000u : 0u);
}

double Median(std::vector<double> v) {
  if (v.empty()) {
    return 0.0;
  }
  std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
  return v[v.size() / 2];
}

/** @brief Expert sets over the fixture's two arena chunks, each set four
 *  consecutive regions (gate_up at 0, down at nntr_moe_dma_down_off), as
 *  MoeChunkReplay registers them: the pattern bytes are nibbles to the
 *  GEMV, and only the time is read. 8 sets a chunk (32 regions, 168 MiB)
 *  at 256 MiB, 5 at 128 MiB. Released on destruction. */
struct MoeSets {
  remote_handle64 h = 0;
  std::vector<uint32_t> gu, dn;
  uint32_t n_sets = 0;
  std::vector<float> act = std::vector<float>(kK), out = std::vector<float>(kN);
  std::vector<uint32_t> stage = std::vector<uint32_t>(kMoeStages);
  const std::vector<uint32_t> row_count = std::vector<uint32_t>(kE, 1u),
                              row_index = std::vector<uint32_t>(kE, 0u);
  const std::vector<float> row_weight = std::vector<float>(kE, 0.25f);

  static uint64_t CallBytes() {
    return static_cast<uint64_t>(kE) *
           (nntr_moe_dma_gu_bytes(kK, kI) + nntr_moe_dma_dn_bytes(kI, kN));
  }

  int Register(remote_handle64 handle, const uint32_t arena[2],
               uint32_t chunk_bytes) {
    h = handle;
    const uint32_t region = nntr_moe_dma_region_bytes(kK, kI, kN);
    const uint32_t dn_off = nntr_moe_dma_down_off(kK, kI);
    const uint32_t per_chunk =
      std::min(kSetsPerChunkMax, (chunk_bytes / region - 1u) / kE);
    std::vector<float> ws_gu(2 * kI, 0.01f), bias_gu(2 * kI, 0.f);
    std::vector<int32_t> cs_gu(2 * kI, 0);
    std::vector<float> ws_dn(kN, 0.01f), bias_dn(kN, 0.f);
    std::vector<int32_t> cs_dn(kN, 0);
    for (uint32_t i = 0; i < kK; ++i) {
      act[i] = static_cast<float>((i * 7919u) % 1000u) / 500.f - 1.f;
    }
    for (uint32_t c = 0; c < 2; ++c) {
      for (uint32_t r = 0; r < per_chunk * kE; ++r) {
        uint32_t hg = 0, hd = 0;
        int err = nntr_hvx_weight_register_u8i4_arena(
          h, kK, 2 * kI, arena[c], r * region, ws_gu.data(), (int)(2 * kI),
          cs_gu.data(), (int)(2 * kI), bias_gu.data(), (int)(2 * kI), &hg);
        if (err != AEE_SUCCESS) {
          return err;
        }
        gu.push_back(hg);
        err = nntr_hvx_weight_register_u8i4_arena(
          h, kI, kN, arena[c], r * region + dn_off, ws_dn.data(), (int)kN,
          cs_dn.data(), (int)kN, bias_dn.data(), (int)kN, &hd);
        if (err != AEE_SUCCESS) {
          return err;
        }
        dn.push_back(hd);
      }
    }
    n_sets = static_cast<uint32_t>(gu.size()) / kE;
    return AEE_SUCCESS;
  }

  /** @brief One M=1 MoE call on set @a set mod n_sets; fills stage. */
  int Call(uint32_t set) {
    const uint32_t *g = gu.data() + (set % n_sets) * kE,
                   *d = dn.data() + (set % n_sets) * kE;
    std::fill(stage.begin(), stage.end(), 0u);
    return nntr_hvx_mm_u8i4_moe_layer_timed(
      h, 1, kK, kI, kN, g, (int)kE, d, (int)kE, row_index.data(), (int)kE,
      row_count.data(), (int)kE, row_weight.data(), (int)kE, act.data(),
      (int)kK, out.data(), (int)kN, stage.data(), kMoeStages);
  }

  ~MoeSets() {
    for (size_t i = 0; i < gu.size(); ++i) {
      nntr_hvx_weight_release_u8i4(h, gu[i]);
    }
    for (size_t i = 0; i < dn.size(); ++i) {
      nntr_hvx_weight_release_u8i4(h, dn[i]);
    }
  }
};

/** @brief "a,b,c" of @a v. */
std::string Join(const std::vector<int> &v) {
  std::ostringstream os;
  for (size_t i = 0; i < v.size(); ++i) {
    os << (i ? "," : "") << v[i];
  }
  return os.str();
}

/** @brief CPU_CACHE lines from sysfs for cpu0 and cpu7 (the SLC is not
 *  listed there; plan 90 section 3.1 infers it from PrefetchOverlap). */
void PrintCpuCache() {
  for (int cpu : {0, 7}) {
    for (int idx = 0; idx < 8; ++idx) {
      const std::string d = "/sys/devices/system/cpu/cpu" +
                            std::to_string(cpu) + "/cache/index" +
                            std::to_string(idx) + "/";
      std::string level, size, shared, type;
      std::ifstream(d + "level") >> level;
      if (level.empty()) {
        break;
      }
      std::ifstream(d + "size") >> size;
      std::ifstream(d + "shared_cpu_list") >> shared;
      std::ifstream(d + "type") >> type;
      std::cout << "CPU_CACHE cpu=" << cpu << " index=" << idx
                << " level=" << level << " type=" << type << " size=" << size
                << " shared_cpu_list=" << shared << "\n";
    }
  }
}

/** @brief t threads streaming [base, base + t x slice) in 4 MiB steps until
 *  Finish; the bytes are the steps completed inside the window. Each
 *  thread records the CPU it ended on (unpinned, as the app's pool). */
struct CpuStreamer {
  using clock = std::chrono::steady_clock;
  static constexpr size_t kStep = 4u << 20;
  std::atomic<int> go{0}, stop{0};
  std::atomic<uint64_t> bytes{0}, xr{0};
  std::vector<std::thread> th;
  std::vector<int> cpus;
  clock::time_point t0;
  void Start(const uint8_t *base, unsigned threads, size_t slice) {
    cpus.assign(threads, -1);
    for (unsigned t = 0; t < threads; ++t) {
      th.emplace_back([this, t, p = base + t * slice, slice]() {
        uint64_t x = 0, n = 0;
        while (!go.load(std::memory_order_acquire)) {
        }
        while (!stop.load(std::memory_order_relaxed)) {
          for (size_t o = 0; o < slice && !stop.load(std::memory_order_relaxed);
               o += kStep) {
            x ^= stream_xor(p + o, kStep);
            n += kStep;
          }
        }
        cpus[t] = sched_getcpu();
        bytes.fetch_add(n);
        xr.fetch_xor(x);
      });
    }
    t0 = clock::now();
    go.store(1, std::memory_order_release);
  }
  /** @brief Stops the threads; prints DDR_CPU; returns GB/s. */
  double Finish(unsigned threads, const char *tag) {
    stop.store(1);
    for (auto &t : th) {
      t.join();
    }
    const double s = std::chrono::duration<double>(clock::now() - t0).count();
    const uint64_t b = bytes.load();
    const uint64_t us = static_cast<uint64_t>(s * 1e6);
    const double gbs = b / s / 1e9;
    const bool valid = nntr_ddr_rate_valid(b, us, NNTR_DDR_CEILING_GBS);
    std::cout << std::fixed << std::setprecision(2)
              << "DDR_CPU threads=" << threads << " phase=" << tag
              << " bytes=" << b << " s=" << s << " gbs=" << gbs
              << " valid=" << (valid ? "y" : "n INVALID")
              << " cpus=" << Join(cpus) << " xor=" << std::hex << xr.load()
              << std::dec << "\n";
    EXPECT_TRUE(valid) << "DDR_CPU threads=" << threads << " " << tag;
    return gbs;
  }
};

/** @brief One DSP reader's stream: bytes over the DSP-side microseconds. */
struct DspStream {
  uint64_t bytes = 0, us = 0;
  bool ok = false;
  double gbs() const { return us ? static_cast<double>(bytes) / us / 1e3 : 0; }
};

} // namespace

/**
 * @brief [#90, plan section 3.1] DDR read rates of the CPU and the DSP,
 *        alone and at once.
 *
 * CPU: a 512 MiB stream at 1, 2, 4, 8 unpinned threads, 1.5 s each. DSP,
 * two readers: "ring" is the MoE feed's own f2 list through dma_replay with
 * fresh = 1 (32 regions x 5.25 MiB = 168 MiB of distinct DDR, 500-call
 * chunks until 1.2 s, every chunk's res[12] tag held against the
 * simulator); "hvx" is the real M=1 MoE call with the feed off (D192, the
 * GEMV's direct arena read) over 16 expert sets, rate = bytes / mm_us.
 * Then both at once: ring with 1/2/4/8 CPU threads, hvx with 2/8. Any DDR
 * rate or aggregate above NNTR_DDR_CEILING_GBS prints INVALID and fails, as
 * does a ring stream with a bad tag or off the f2 20-call reference by
 * more than 20 % (alone; under contention only the upper side).
 */
TEST_F(HvxDmaProbe, TwoReaderDdr) {
  const size_t kCpuBytes = 512ull << 20;
  const double kCpuSeconds = 1.5;
  PrintCpuCache();
  std::vector<uint8_t> cpu_buf(kCpuBytes);
  for (size_t i = 0; i < kCpuBytes; ++i) {
    cpu_buf[i] = pattern(i);
  }

  // 1. CPU alone.
  const unsigned kThreads[] = {1, 2, 4, 8};
  double cpu_alone[9] = {0};
  for (unsigned t : kThreads) {
    CpuStreamer run;
    run.Start(cpu_buf.data(), t, kCpuBytes / t);
    std::this_thread::sleep_for(std::chrono::duration<double>(kCpuSeconds));
    cpu_alone[t] = run.Finish(t, "alone");
  }

  // 2. The ring reader and its in-run reference.
  const Chunk &ch = chunks_[0];
  const uint32_t region = nntr_moe_dma_region_bytes(kK, kI, kN);
  static nntr_moe_dma_item items[NNTR_MOE_DMA_PLAN_MAX];
  static uint8_t samples[(8u << 20) / 64u + 1u];
  nntr_two_reader_spec spec;
  const uint32_t n_items =
    nntr_two_reader_cell(items, NNTR_MOE_DMA_PLAN_MAX, &spec);
  ASSERT_NE(n_items, 0u);
  std::vector<uint32_t> sched;
  for (uint32_t k = 0; k < n_items; ++k) {
    const nntr_moe_dma_item &it = items[k];
    const uint32_t w[8] = {nntr_moe_dma_word0(&it),
                           it.expert,
                           it.src_off,
                           it.dst_off,
                           it.row_size,
                           it.nrows,
                           it.src_stride,
                           0u};
    sched.insert(sched.end(), w, w + 8);
  }
  auto replay = [&](uint32_t fresh, uint32_t calls,
                    std::vector<uint32_t> &res) {
    res.assign(13, 0);
    return nntr_hvx_dma_replay(handle_, ch.arena, region, sched.data(),
                               (int)sched.size(), 1, 0, 0, fresh, 0, calls,
                               res.data(), (int)res.size());
  };
  std::vector<uint32_t> res;
  ASSERT_EQ(replay(0, 20, res), AEE_SUCCESS) << "a skel without #100?";
  const uint32_t n_regions = res[7];
  const bool ref_ok =
    res[12] ==
    nntr_moe_dma_tag_sum(items, n_items, 20, 0, n_regions, region, samples);
  const double ref_gbs = res[0] ? res[2] * 20.0 / res[0] / 1e3 : 0.0;
  std::cout << std::fixed << std::setprecision(2)
            << "DDR_DSP_REF name=f2 fresh=0 calls=20 us=" << res[0]
            << " bytes_per_call=" << res[2] << " gbs=" << ref_gbs
            << " checksum_ok=" << (ref_ok ? "y" : "n") << "\n";
  EXPECT_TRUE(ref_ok) << "DDR_DSP_REF";
  const uint32_t chunk_want =
    nntr_moe_dma_tag_sum(items, n_items, spec.calls_per_chunk, spec.fresh,
                         n_regions, region, samples);
  auto ring = [&]() {
    DspStream d;
    d.ok = true;
    std::vector<uint32_t> r;
    for (uint32_t c = 0; c < spec.chunks_max && d.us < spec.min_us; ++c) {
      const int err = replay(spec.fresh, spec.calls_per_chunk, r);
      if (err != AEE_SUCCESS) {
        ADD_FAILURE() << "ring replay err=" << hex(err);
        d.ok = false;
        break;
      }
      d.ok = d.ok && r[12] == chunk_want;
      d.us += r[0];
      d.bytes += static_cast<uint64_t>(r[2]) * r[1];
    }
    return d;
  };

  // 3. The HVX-direct reader: the D192 GEMV, feed off.
  MoeSets sets;
  {
    const uint32_t arenas[2] = {chunks_[0].arena, chunks_[1].arena};
    ASSERT_EQ(sets.Register(handle_, arenas, ch.bytes), AEE_SUCCESS);
  }
  uint32_t applied = 0;
  const uint32_t d192 = GemvOpts(192, true, false);
  ASSERT_EQ(nntr_hvx_moe_set_opts(handle_, d192, &applied), AEE_SUCCESS);
  ASSERT_EQ(applied, d192) << "the skel does not know #117's feed bit";
  uint32_t hvx_call = 0;
  for (uint32_t s = 0; s < sets.n_sets; ++s) { // page state
    ASSERT_EQ(sets.Call(s), AEE_SUCCESS);
  }
  uint64_t hvx_dsp_us = 0, hvx_window_us = 0;
  auto hvx = [&]() {
    DspStream d;
    d.ok = true;
    hvx_dsp_us = 0;
    const auto t0 = std::chrono::steady_clock::now();
    // at most 10000 calls (~10 s): a skel that reports mm = 0 ends here
    for (uint32_t n = 0; d.us < spec.min_us && n < 10000u; ++n) {
      const int err = sets.Call(hvx_call++);
      if (err != AEE_SUCCESS || sets.stage[kStPath] != 1u ||
          sets.stage[kStFeed] != 0u) {
        ADD_FAILURE() << "hvx reader err=" << hex(err)
                      << " path=" << sets.stage[kStPath]
                      << " feed=" << sets.stage[kStFeed];
        d.ok = false;
        break;
      }
      d.us += sets.stage[kStMm];
      hvx_dsp_us += sets.stage[kStDsp];
      d.bytes += MoeSets::CallBytes();
    }
    hvx_window_us =
      static_cast<uint64_t>(std::chrono::duration<double, std::micro>(
                              std::chrono::steady_clock::now() - t0)
                              .count());
    return d;
  };

  auto print_dsp = [&](const char *reader, const DspStream &d, bool valid) {
    std::cout << std::fixed << std::setprecision(2)
              << "DDR_DSP reader=" << reader << " bytes=" << d.bytes
              << " us=" << d.us << " gbs=" << d.gbs()
              << " checksum_ok=" << (d.ok ? "y" : "n");
    if (std::strcmp(reader, "ring") == 0) {
      std::cout << " ref_gbs=" << ref_gbs << " regions=" << n_regions
                << " footprint_mib="
                << (nntr_two_reader_footprint_bytes(n_regions, region) >> 20);
    } else {
      std::cout << " dsp_us=" << hvx_dsp_us << " window_us=" << hvx_window_us
                << " duty="
                << (hvx_window_us ? 1.0 * hvx_dsp_us / hvx_window_us : 0.0)
                << " sets=" << sets.n_sets << " footprint_mib="
                << (sets.n_sets * kE * static_cast<uint64_t>(region) >> 20);
    }
    std::cout << " chunk_bytes=" << ch.bytes
              << " valid=" << (valid ? "y" : "n INVALID") << "\n";
    EXPECT_TRUE(valid) << "DDR_DSP reader=" << reader;
  };
  const DspStream ring_alone = ring();
  print_dsp("ring", ring_alone,
            nntr_two_reader_verdict(ring_alone.gbs(), ref_gbs, ring_alone.ok));
  const DspStream hvx_alone = hvx();
  print_dsp("hvx", hvx_alone,
            hvx_alone.ok && nntr_ddr_rate_valid(hvx_alone.bytes, hvx_alone.us,
                                                NNTR_DDR_CEILING_GBS));

  // 4. Both at once: the CPU threads released just before the (blocking)
  //    DSP calls, stopped as soon as they return.
  struct Pair {
    const char *reader;
    unsigned t;
  };
  const Pair pairs[] = {{"ring", 1}, {"ring", 2}, {"ring", 4},
                        {"ring", 8}, {"hvx", 2},  {"hvx", 8}};
  for (const Pair &p : pairs) {
    const bool is_ring = std::strcmp(p.reader, "ring") == 0;
    CpuStreamer run;
    run.Start(cpu_buf.data(), p.t, kCpuBytes / p.t);
    const DspStream d = is_ring ? ring() : hvx();
    const double cpu_with = run.Finish(p.t, is_ring ? "with_ring" : "with_hvx");
    const DspStream &alone = is_ring ? ring_alone : hvx_alone;
    const double aggregate = cpu_with + d.gbs();
    const bool valid =
      d.ok && aggregate <= NNTR_DDR_CEILING_GBS &&
      nntr_ddr_rate_valid(d.bytes, d.us, NNTR_DDR_CEILING_GBS) &&
      (!is_ring || d.gbs() <= ref_gbs * (1.0 + NNTR_TWO_READER_REF_TOL));
    std::cout << std::fixed << std::setprecision(2)
              << "DDR_TWO_READER cpu_alone=" << cpu_alone[p.t]
              << " dsp_alone=" << alone.gbs() << " cpu_with=" << cpu_with
              << " dsp_with=" << d.gbs() << " aggregate=" << aggregate
              << " cpu_threads=" << p.t << " chunk_bytes=" << ch.bytes
              << " reader=" << p.reader
              << " cpu_loss_pct=" << 100.0 * (1.0 - cpu_with / cpu_alone[p.t])
              << " dsp_loss_pct=" << 100.0 * (1.0 - d.gbs() / alone.gbs())
              << " checksum_ok=" << (d.ok ? "y" : "n")
              << " valid=" << (valid ? "y" : "n INVALID") << "\n";
    EXPECT_TRUE(valid) << "DDR_TWO_READER " << p.reader << " t=" << p.t;
  }
  EXPECT_EQ(nntr_hvx_moe_set_opts(handle_, 0u, &applied), AEE_SUCCESS);
}

namespace {

/** @brief Workers that spin between jobs (a persistent pool like the
 *  app's): Go(n, fn) runs fn(j) on workers j < n, Busy() says whether one
 *  is still running, and each worker keeps the CPU it last ran a job on
 *  and when it finished. */
class SpinPool {
public:
  static constexpr unsigned kMax = 8;
  using clock = std::chrono::steady_clock;
  explicit SpinPool(unsigned n) : n_(n) {
    for (unsigned j = 0; j < n_; ++j) {
      th_.emplace_back([this, j]() { Loop(j); });
    }
  }
  ~SpinPool() {
    quit_.store(1);
    for (unsigned j = 0; j < n_; ++j) {
      go_[j].fetch_add(1, std::memory_order_release);
    }
    for (auto &t : th_) {
      t.join();
    }
  }
  void Go(unsigned n, std::function<void(unsigned)> fn) {
    fn_ = std::move(fn);
    left_.store(n, std::memory_order_relaxed);
    for (unsigned j = 0; j < n; ++j) {
      go_[j].fetch_add(1, std::memory_order_release);
    }
  }
  bool Busy() const { return left_.load(std::memory_order_acquire) != 0; }
  void Wait() const {
    while (Busy()) {
    }
  }
  int Cpu(unsigned j) const { return cpu_[j]; }
  clock::time_point End(unsigned j) const { return end_[j]; }

private:
  void Loop(unsigned j) {
    uint32_t seen = 0;
    for (;;) {
      uint32_t g;
      while ((g = go_[j].load(std::memory_order_acquire)) == seen) {
      }
      seen = g;
      if (quit_.load()) {
        return;
      }
      fn_(j);
      end_[j] = clock::now();
      cpu_[j] = sched_getcpu();
      left_.fetch_sub(1, std::memory_order_release);
    }
  }
  unsigned n_;
  std::vector<std::thread> th_;
  std::atomic<uint32_t> go_[kMax] = {};
  std::atomic<unsigned> left_{0};
  std::atomic<int> quit_{0};
  std::function<void(unsigned)> fn_;
  int cpu_[kMax] = {};
  clock::time_point end_[kMax];
};

} // namespace

/**
 * @brief [#90, plan section 3.2] What a CPU prefetch of the next layer's
 *        FC weights during the MoE call costs the DSP, and what it stages.
 *
 * 8 readers (7 spinning workers + the main thread, NNTR_NUM_THREADS=8) and
 * a ring of S MiB buffers, >= 192 MiB in all, so a buffer is cold when it
 * comes round. Per S = 4, 10, 20, 32: COLD (one real M=1 MoE call with the
 * app's default feed = 1, no touch; then the 8 readers re-read B), HOT at
 * T = 1, 2, 7 (T workers touch B, no call; then the re-read) and OVERLAP
 * at T = 1, 2, 7 (the call with T workers touching B, released as it is
 * issued; then the re-read). 64 iterations a cell, medians. The re-read is
 * a stream, not the Q4_0 GEMV, so saved_us is a byte saving. Cache re-read
 * times are not DDR rates and carry no bound.
 */
TEST_F(HvxDmaProbe, PrefetchOverlap) {
  using clock = std::chrono::steady_clock;
  const unsigned kReaders = 8, kWorkers = kReaders - 1;
  const int kIters = 64;
  const size_t kRingBytes = 200ull << 20; // max over S of ceil(192 / S) x S
  MoeSets sets;
  {
    const uint32_t arenas[2] = {chunks_[0].arena, chunks_[1].arena};
    ASSERT_EQ(sets.Register(handle_, arenas, chunks_[0].bytes), AEE_SUCCESS);
  }
  uint32_t applied = 0;
  const uint32_t feed = GemvOpts(192, true, true);
  ASSERT_EQ(nntr_hvx_moe_set_opts(handle_, feed, &applied), AEE_SUCCESS);
  ASSERT_EQ(applied, feed) << "the skel does not know #117's feed bit";
  uint32_t call_i = 0;
  auto moe_call = [&](double *dsp_us, double *mm_us) {
    const int err = sets.Call(call_i++);
    ASSERT_EQ(err, AEE_SUCCESS) << hex(err);
    ASSERT_EQ(sets.stage[kStPath], 1u) << "not the M=1 GEMV path";
    ASSERT_EQ(sets.stage[kStFeed], 1u) << "not the VTCM feed";
    *dsp_us = sets.stage[kStDsp];
    *mm_us = sets.stage[kStMm];
  };
  for (uint32_t s = 0; s < sets.n_sets; ++s) { // page state
    double a, b;
    ASSERT_NO_FATAL_FAILURE(moe_call(&a, &b));
  }

  std::vector<uint8_t> ring(kRingBytes, 0x5a); // written, so faulted in
  std::atomic<uint64_t> sink{0};
  SpinPool pool(kWorkers);
  std::cout << "PREFETCH_CONFIG readers=" << kReaders << " workers=" << kWorkers
            << " ring_mib=" << (kRingBytes >> 20) << " sets=" << sets.n_sets
            << " opts=0x" << std::hex << feed << std::dec
            << " chunk_bytes=" << chunks_[0].bytes << "\n";

  size_t g = 0; // the ring position runs on across cells
  for (uint32_t s_mib : {4u, 10u, 20u, 32u}) {
    const size_t s_bytes = static_cast<size_t>(s_mib) << 20;
    const size_t n_bufs = ((192ull << 20) + s_bytes - 1) / s_bytes;
    const size_t slice = s_bytes / kReaders;
    double cold_us = 0, mm_alone = 0, dsp_alone = 0;
    double hot_us[kReaders] = {0};
    for (int kind = 0; kind < 3; ++kind) { // COLD, HOT, OVERLAP
      for (unsigned t : {1u, 2u, 7u}) {
        if (kind == 0 && t != 1u) {
          continue;
        }
        std::vector<double> dsp, mm, touch, reread, moved;
        int late = 0;
        for (int it = 0; it < kIters; ++it) {
          const uint8_t *buf = ring.data() + ((g++) % n_bufs) * s_bytes;
          auto touch_fn = [&sink, buf, t, slice](unsigned j) {
            uint64_t x = 0;
            for (unsigned k = j; k < kReaders; k += t) {
              x ^= stream_xor(buf + k * slice, slice);
            }
            sink.fetch_xor(x, std::memory_order_relaxed);
          };
          // phase 1: the window
          const auto t_go = clock::now();
          if (kind != 0) {
            pool.Go(t, touch_fn);
          }
          if (kind != 1) {
            double d, m;
            ASSERT_NO_FATAL_FAILURE(moe_call(&d, &m));
            dsp.push_back(d);
            mm.push_back(m);
            late += kind == 2 && pool.Busy();
          }
          if (kind != 0) {
            pool.Wait();
            double us = 0;
            for (unsigned j = 0; j < t; ++j) {
              us = std::max(us, std::chrono::duration<double, std::micro>(
                                  pool.End(j) - t_go)
                                  .count());
            }
            touch.push_back(us);
          }
          int touch_cpu[kReaders];
          for (unsigned j = 0; j < t && kind != 0; ++j) {
            touch_cpu[j] = pool.Cpu(j);
          }
          // phase 2: the 8 readers re-read the buffer (the consumer)
          const auto t0 = clock::now();
          pool.Go(kWorkers, [&sink, buf, slice](unsigned j) {
            sink.fetch_xor(stream_xor(buf + j * slice, slice),
                           std::memory_order_relaxed);
          });
          sink.fetch_xor(stream_xor(buf + kWorkers * slice, slice),
                         std::memory_order_relaxed);
          pool.Wait();
          reread.push_back(
            std::chrono::duration<double, std::micro>(clock::now() - t0)
              .count());
          if (kind != 0) {
            int n = 0;
            for (unsigned k = 0; k < kReaders; ++k) {
              const int cpu = k < kWorkers ? pool.Cpu(k) : sched_getcpu();
              n += cpu != touch_cpu[k % t];
            }
            moved.push_back(n);
          }
        }
        const double r_us = Median(reread);
        std::cout << std::fixed << std::setprecision(1);
        if (kind == 0) {
          cold_us = r_us;
          mm_alone = Median(mm);
          dsp_alone = Median(dsp);
          std::cout << "PREFETCH_COLD size_mib=" << s_mib << " bufs=" << n_bufs
                    << " dsp_us=" << dsp_alone << " mm_us=" << mm_alone
                    << " cold_us=" << cold_us
                    << " cold_gbs=" << s_bytes / cold_us / 1e3 << "\n";
          continue;
        }
        const double touch_us = Median(touch);
        if (kind == 1) {
          hot_us[t] = r_us;
          std::cout << "PREFETCH_HOT size_mib=" << s_mib
                    << " touch_threads=" << t << " touch_us=" << touch_us
                    << " hot_us=" << r_us << " moved=" << Median(moved) << "\n";
          continue;
        }
        const double dsp_us = Median(dsp), mm_us = Median(mm);
        const double staged =
          cold_us > hot_us[t] ? (cold_us - r_us) / (cold_us - hot_us[t]) : 0.0;
        const double net =
          22.0 * nntr_prefetch_net_us(cold_us, r_us, dsp_us, dsp_alone) / 1e3;
        std::cout << "PREFETCH_OVERLAP size_mib=" << s_mib
                  << " touch_threads=" << t << " dsp_us=" << dsp_us
                  << " mm_us=" << mm_us << " mm_alone_us=" << mm_alone
                  << " dmm_pct=" << 100.0 * (mm_us / mm_alone - 1.0)
                  << " touch_us=" << touch_us
                  << " touch_gbs=" << s_bytes / touch_us / 1e3
                  << " touch_late=" << late << "/" << kIters
                  << " cold_us=" << cold_us << " hot_us=" << hot_us[t]
                  << " reread_us=" << r_us << std::setprecision(2)
                  << " staged=" << staged << " staged_mib=" << staged * s_mib
                  << std::setprecision(1) << " saved_us=" << cold_us - r_us
                  << std::setprecision(3) << " net_ms_per_token=" << net
                  << std::setprecision(1) << " moved=" << Median(moved) << "\n";
      }
    }
  }
  std::cout << "PREFETCH_SINK " << std::hex << sink.load() << std::dec << "\n";
  EXPECT_EQ(nntr_hvx_moe_set_opts(handle_, 0u, &applied), AEE_SUCCESS);
}

/**
 * @brief [#87, plan section 3.3] The MoE layer call's own M=1 descriptor
 *        list, replayed with nothing between the pushes.
 *
 * Step 0 runs one real mm_u8i4_moe_layer_timed at M=1 over four expert
 * regions of the pattern arena (registered as arena weights: the HMX does
 * not care what the nibbles are, so no bake is needed for a timing) and
 * reads its trace back -- the DMA_REPLAY_TRACE lines are the in-situ
 * timeline the replays are held against, and its issue times pace the
 * pace=1 cells. Reports rather than asserts on the numbers; asserts the
 * mapping is live (checksum) and the trace has the planner's shape.
 */
TEST_F(HvxDmaProbe, MoeChunkReplay) {
  const uint32_t K = 2048, I = 1792, N = 2048, E = 4, ACC_TILES = 32;
  const uint32_t region = nntr_moe_dma_region_bytes(K, I, N);
  const uint32_t gu_bytes = nntr_moe_dma_gu_bytes(K, I);
  const uint32_t dn_bytes = nntr_moe_dma_dn_bytes(I, N);
  const uint32_t dn_off = nntr_moe_dma_down_off(K, I);
  const Chunk &ch = chunks_[0];
  ASSERT_GE(ch.bytes, (E + 1) * region) << "chunk too small for 4 regions";

  // VTCM layout for the replay: gate_up, down, activation, copy scratch.
  const uint32_t v_gu = 0, v_dn = gu_bytes, v_act = v_dn + dn_bytes,
                 v_copy = v_act + (K / 32u) * 2048u;
  static nntr_moe_dma_item plan[NNTR_MOE_DMA_PLAN_MAX];
  const uint32_t n_plan =
    nntr_moe_dma_plan_m1(K, I, N, E, ACC_TILES, v_gu, v_dn, v_act, v_copy, plan,
                         NNTR_MOE_DMA_PLAN_MAX);
  ASSERT_NE(n_plan, 0u);
  uint32_t plan_push = 0, plan_wait = 0;
  for (uint32_t k = 0; k < n_plan; ++k) {
    (plan[k].op == NNTR_MOE_DMA_OP_PUSH ? plan_push : plan_wait)++;
  }

  // --- step 0: one real timed call over the same regions -----------------
  std::vector<uint32_t> h_gu(E), h_dn(E);
  {
    std::vector<float> ws_gu(2 * I, 0.01f), bias_gu(2 * I, 0.f);
    std::vector<int32_t> cs_gu(2 * I, 0);
    std::vector<float> ws_dn(N, 0.01f), bias_dn(N, 0.f);
    std::vector<int32_t> cs_dn(N, 0);
    for (uint32_t e = 0; e < E; ++e) {
      ASSERT_EQ(nntr_hvx_weight_register_u8i4_arena(
                  handle_, K, 2 * I, ch.arena, e * region, ws_gu.data(),
                  (int)(2 * I), cs_gu.data(), (int)(2 * I), bias_gu.data(),
                  (int)(2 * I), &h_gu[e]),
                AEE_SUCCESS);
      ASSERT_EQ(nntr_hvx_weight_register_u8i4_arena(
                  handle_, I, N, ch.arena, e * region + dn_off, ws_dn.data(),
                  (int)N, cs_dn.data(), (int)N, bias_dn.data(), (int)N,
                  &h_dn[e]),
                AEE_SUCCESS);
    }
  }
  // One row routed to all four experts: decode's shape.
  const std::vector<uint32_t> row_count(E, 1u), row_index(E, 0u);
  const std::vector<float> row_weight(E, 0.25f);
  std::vector<float> act(K), out(N, 0.f);
  for (uint32_t i = 0; i < K; ++i) {
    act[i] = static_cast<float>((i * 7919u) % 1000u) / 500.f - 1.f;
  }
  // test/htp/nntr_hvx_mm_u8i4.c's MOE_N_STAGES (mirrored as
  // HTP_MOE_N_STAGES in htp_compute_ops.cpp): 19 before #87 + 10 (#87's
  // DMA slots) + 1 (#80's MOE_T_PATH) + 1 (#117's MOE_T_M1_FEED, after
  // them; the slots read below did not move).
  const int kMoeStages = 31;
  std::vector<uint32_t> stage(kMoeStages, 0);
  std::vector<uint32_t> trace;
  uint32_t n_words = 0;
  bool traced = false;
  {
    // Two calls: the first warms the page state, the second is traced.
    int err = AEE_SUCCESS;
    for (int rep = 0; rep < 2 && err == AEE_SUCCESS; ++rep) {
      err = nntr_hvx_mm_u8i4_moe_layer_timed(
        handle_, 1, K, I, N, h_gu.data(), (int)E, h_dn.data(), (int)E,
        row_index.data(), (int)E, row_count.data(), (int)E, row_weight.data(),
        (int)E, act.data(), (int)K, out.data(), (int)N, stage.data(),
        kMoeStages);
    }
    if (err != AEE_SUCCESS) {
      std::cout << "DMA_REPLAY_NOTE moe_layer_timed err=" << hex(err)
                << " -- pace=1 cells fall back to pace=0\n";
    } else {
      trace.resize(HEXKL_DMA_TRACE_MAX_WORDS);
      err = nntr_hvx_moe_dma_trace_read(handle_, trace.data(),
                                        (int)trace.size(), &n_words);
      traced = err == AEE_SUCCESS && n_words >= HEXKL_DMA_TRACE_HDR_WORDS &&
               trace[0] == plan_push && trace[1] == plan_wait;
      std::cout << "DMA_REPLAY_TRACE dsp_us=" << stage[0]
                << " desc=" << stage[19] << " waits=" << stage[20]
                << " blocked=" << stage[21] << " wait_us=" << stage[22]
                << " wait_act_us=" << stage[23] << " busy_us=" << stage[24]
                << ".." << stage[25] << " depth_max=" << stage[26]
                << " first_ready_us=" << stage[27]
                << " last_issue_us=" << stage[28] << " trace_words=" << n_words
                << " plan_shape_ok=" << (traced ? "y" : "n") << "\n";
    }
  }
  const uint32_t pw = HEXKL_DMA_TRACE_PUSH_WORDS,
                 ww = HEXKL_DMA_TRACE_WAIT_WORDS;
  auto us = [](uint32_t ticks) { return ticks / 19.2; };
  if (traced) {
    static const char *const kKind[] = {"act", "gate", "up", "down", "copy"};
    static const char *const kSite[] = {"act", "gu", "dn", "copy_in",
                                        "copy_out"};
    const uint32_t *p = trace.data() + HEXKL_DMA_TRACE_HDR_WORDS;
    for (uint32_t k = 0; k < trace[0]; ++k, p += pw) {
      std::cout << std::fixed << std::setprecision(1)
                << "DMA_REPLAY_TRACE push k=" << k << " t=" << us(p[0])
                << " kind=" << (p[1] < 5 ? kKind[p[1]] : "?") << " e=" << p[2]
                << " c=" << p[3] << " bytes=" << p[4] << " row=" << p[5]
                << " nrows=" << p[6] << " stride=" << p[7] << " depth=" << p[9]
                << " done=" << us(p[10]) << ".." << us(p[11]) << "\n";
    }
    for (uint32_t k = 0; k < trace[1]; ++k, p += ww) {
      std::cout << std::fixed << std::setprecision(1)
                << "DMA_REPLAY_TRACE wait k=" << k
                << " site=" << (p[3] < 5 ? kSite[p[3]] : "?")
                << " t=" << us(p[0]) << ".." << us(p[1])
                << " blocked=" << (p[4] ? "y" : "n") << "\n";
    }
  }

  // --- the schedule: the plan, with the traced call's issue times ---------
  // Pushes land strided, the kernel's VTCM layout (hexkl_mm_u8i4_moe.c
  // push2d: dst_stride = src_stride); the replay default is packed, which
  // overlaps the gate/up chunks and fails the footprint sum below (#99).
  // weight_bytes is the in-situ `weight DMA:` basis; bytes_per_call also
  // counts the activation and copy pushes.
  std::vector<uint32_t> sched;
  uint32_t weight_bytes = 0;
  sched.reserve(n_plan * 8);
  {
    const uint32_t *pp = trace.data() + HEXKL_DMA_TRACE_HDR_WORDS;
    const uint32_t *wp = pp + plan_push * pw;
    uint32_t ip = 0, iw = 0;
    for (uint32_t k = 0; k < n_plan; ++k) {
      const nntr_moe_dma_item &it = plan[k];
      uint32_t t_rel = 0;
      if (traced) {
        t_rel =
          it.op == NNTR_MOE_DMA_OP_PUSH ? pp[(ip++) * pw] : wp[(iw++) * ww];
      }
      const bool push = it.op == NNTR_MOE_DMA_OP_PUSH;
      if (push && it.kind != NNTR_MOE_DMA_KIND_ACT &&
          it.kind != NNTR_MOE_DMA_KIND_COPY) {
        weight_bytes += it.row_size * it.nrows;
      }
      const uint32_t words[8] = {nntr_moe_dma_word0(&it) |
                                   (push ? NNTR_MOE_DMA_DST_STRIDED << 16 : 0u),
                                 it.expert,
                                 it.src_off,
                                 it.dst_off,
                                 it.row_size,
                                 it.nrows,
                                 it.src_stride,
                                 t_rel};
      sched.insert(sched.end(), words, words + 8);
    }
  }

  const uint32_t want_sum = 0xA5u * (gu_bytes / 64u);
  auto cell = [&](uint32_t workers, uint32_t load, uint32_t pace,
                  uint32_t fresh, uint32_t gap_us) {
    const uint32_t calls = 20;
    std::vector<uint32_t> res(12, 0);
    if (pace && !traced) {
      pace = 0;
    }
    const int err = nntr_hvx_dma_replay(
      handle_, ch.arena, region, sched.data(), (int)sched.size(), workers, load,
      pace, fresh, gap_us, calls, res.data(), (int)res.size());
    if (err != AEE_SUCCESS) {
      std::cout << "DMA_REPLAY workers=" << workers << " load=" << load
                << " pace=" << pace << " fresh=" << fresh
                << " gap_us=" << gap_us << " skipped err=" << hex(err) << "\n";
      return;
    }
    const double us_per_call = static_cast<double>(res[0]) / calls;
    const double gbs = us_per_call > 0 ? res[2] / us_per_call / 1e3 : 0.0;
    std::cout << std::fixed << std::setprecision(1)
              << "DMA_REPLAY workers=" << workers << " load=" << load
              << " pace=" << pace << " fresh=" << fresh << " gap_us=" << gap_us
              << " calls=" << calls << " us_per_call=" << us_per_call
              << " bytes_per_call=" << res[2]
              << " weight_bytes=" << weight_bytes << " gbs=" << gbs
              << " wait_us=" << res[3] / (double)calls << " blocked=" << res[4]
              << "/" << plan_wait * calls << " depth_max=" << res[5]
              << " busy_us=" << res[10] / (double)calls << ".."
              << res[11] / (double)calls << " regions=" << res[7]
              << " workers_used=" << res[8] << " load_units=" << res[9]
              << " checksum_ok=" << (res[6] == want_sum ? "y" : "n") << "\n";
    EXPECT_EQ(res[6], want_sum) << "workers=" << workers << " load=" << load;
  };
  for (uint32_t pace = 0; pace <= 1; ++pace) {
    for (uint32_t w : {1u, 2u, 4u}) {
      cell(w, 0, pace, 0, 0);
    }
  }
  cell(1, 1, 1, 0, 0);
  cell(1, 2, 1, 0, 0);
  cell(1, 0, 1, 1, 0);
  cell(1, 0, 1, 0, 600);
  cell(1, 0, 1, 1, 600);

  // --- [#100] the cells of plan 100 section 3.2, one run, workers 1 ------
  // res[12] is the tag sum over every push's VTCM window; the host
  // simulates the same list (nntr_moe_dma_tag_sum, checked against a byte
  // copy by replay_cells_host_check) with the skel's region count, so a
  // transfer that did not land -- or, for fresh = 1, landed a call late --
  // fails the line. The rates are reported, not asserted. The expectation
  // takes list order as landing order; only the traced* cells (the packed
  // gate/up chunks overlap, #99) and iii_chain (a slot rewritten 8 pushes
  // on) have overlapping destinations in flight at once, every other cell
  // waits or drains before it rewrites a destination.
  {
    static nntr_moe_dma_item items[NNTR_MOE_DMA_PLAN_MAX];
    static uint8_t samples[(8u << 20) / 64u + 1u];
    const uint32_t calls = 20;
    for (uint32_t id = 0; id < NNTR_MOE_DMA_N_CELLS; ++id) {
      const char *name = "?";
      uint32_t fresh = 0, load = 0, n_push = 0, modes = 0;
      const uint32_t n =
        nntr_moe_dma_cell(id, plan, n_plan, K, I, N, E, items,
                          NNTR_MOE_DMA_PLAN_MAX, &name, &fresh, &load);
      ASSERT_NE(n, 0u) << "cell " << id;
      std::vector<uint32_t> s;
      for (uint32_t k = 0; k < n; ++k) {
        const nntr_moe_dma_item &it = items[k];
        const uint32_t words[8] = {nntr_moe_dma_word0(&it),
                                   it.expert,
                                   it.src_off,
                                   it.dst_off,
                                   it.row_size,
                                   it.nrows,
                                   it.src_stride,
                                   0u};
        s.insert(s.end(), words, words + 8);
        if (it.op == NNTR_MOE_DMA_OP_PUSH) {
          ++n_push;
          // flags 0 is the replay's default: packed
          modes |= (it.flags & NNTR_MOE_DMA_DST_STRIDED) ? 2u : 1u;
        }
      }
      std::vector<uint32_t> res(13, 0);
      const int err = nntr_hvx_dma_replay(handle_, ch.arena, region, s.data(),
                                          (int)s.size(), 1, load, 0, fresh, 0,
                                          calls, res.data(), (int)res.size());
      if (err != AEE_SUCCESS) {
        std::cout << "DMA_REPLAY_X name=" << name << " skipped err=" << hex(err)
                  << "\n";
        ADD_FAILURE() << name << ": " << hex(err) << " -- a skel without #100?";
        continue;
      }
      const uint32_t want =
        nntr_moe_dma_tag_sum(items, n, calls, fresh, res[7], region, samples);
      const double us_per_call = static_cast<double>(res[0]) / calls;
      const double gbs = us_per_call > 0 ? res[2] / us_per_call / 1e3 : 0.0;
      static const char *const kMode[] = {"?", "packed", "strided", "mixed"};
      std::cout << std::fixed << std::setprecision(1)
                << "DMA_REPLAY_X name=" << name << " dst=" << kMode[modes]
                << " load=" << load << " fresh=" << fresh << " desc=" << n_push
                << " bytes_per_call=" << res[2]
                << " us_per_call=" << us_per_call << " gbs=" << gbs
                << " wait_us=" << res[3] / (double)calls
                << " busy_us=" << res[10] / (double)calls << ".."
                << res[11] / (double)calls << " depth_max=" << res[5]
                << " regions=" << res[7] << " load_units=" << res[9]
                << " tag=" << res[12] << "/" << want << " checksum_ok="
                << (res[12] == want ? "y" : "n")
                // a skel without #100 leaves res[12] at 0 on the unflagged
                // cells; a live window never sums to 0 (host check)
                << (res[12] == 0u ? " stale_skel_or_nothing_landed" : "")
                << "\n";
      EXPECT_EQ(res[12], want) << name;
    }
  }

  for (uint32_t e = 0; e < E; ++e) {
    nntr_hvx_weight_release_u8i4(handle_, h_gu[e]);
    nntr_hvx_weight_release_u8i4(handle_, h_dn[e]);
  }
}

int main(int argc, char **argv) {
  int result = -1;
  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error during InitGoogleTest" << std::endl;
    return 0;
  }
  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error during RUN_ALL_TESTS()" << std::endl;
  }
  return result;
}
