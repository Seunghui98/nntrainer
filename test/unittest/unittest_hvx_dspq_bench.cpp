// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   unittest_hvx_dspq_bench.cpp
 * @date   28 Sep 2026
 * @brief  [#141] Device microbench: one FastRPC invoke vs one dspqueue
 *         request/response round trip with a 12 KiB payload, in one session
 *         at the app's QoS (plan docs/plans/141-dspqueue-bench.md)
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * Device only. Every dspqueue_* symbol is resolved with dlsym, so the binary
 * has no dspqueue import and still loads (and prints the FastRPC rows) on a
 * runtime without it; a missing symbol is a loud FAIL, never a skip. Each
 * row is 100 warm-up + 1000 timed round trips; bad counts round trips whose
 * response had the wrong sequence number, status or echoed payload.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

#include "dspqueue.h"
#include "htp_rpc_bench.h"
#include "htp_rpcmem.h"
#include "nntr_hvx.h"

namespace {

using DspqApi = nntrainer::HtpDspqApi;
using nntrainer::HtpRpcBuffer;
using nntrainer::HtpRpcMemApi;
using Clock = std::chrono::steady_clock;

constexpr int kWarm = 100;
constexpr int kTimed = 1000;
constexpr uint32_t kPay = 12288;             // one activation row, 12 KiB
constexpr uint32_t kReadTimeoutUs = 5000000; // arm=B: 5 s, never a hang
constexpr uint32_t kOpEcho = 1, kOpQuit = 2; // nntr_hvx_dspq_bench.c

/** @brief One row's statistics over the timed samples (µs). */
struct Row {
  double median = 0, p90 = 0, min = 0, max = 0;
  int bad = 0;
  int err = 0; /**< first transport error, which ends the row */
};

void print_row(const char *name, std::vector<double> &us, Row &r) {
  if (!us.empty()) {
    std::sort(us.begin(), us.end());
    r.median = us[us.size() / 2];
    r.p90 = us[us.size() * 9 / 10];
    r.min = us.front();
    r.max = us.back();
  }
  std::printf("DSPQ_BENCH row=%s n=%zu median_us=%.1f p90_us=%.1f "
              "min_us=%.1f max_us=%.1f bad=%d%s%s\n",
              name, us.size(), r.median, r.p90, r.min, r.max, r.bad,
              r.err ? " err=" : "", r.err ? hex(r.err).c_str() : "");
  std::fflush(stdout);
  EXPECT_EQ(r.err, 0) << name;
  EXPECT_EQ(r.bad, 0) << name;
}

/** @brief F rows: nntr_hvx_add_f32 on n floats, as-is buffers. */
Row run_f(remote_handle64 h, const char *name, float *a, float *b, float *c,
          int n) {
  for (int i = 0; i < n; ++i) {
    a[i] = static_cast<float>(i);
    b[i] = 0.5f;
  }
  Row r;
  std::vector<double> us;
  for (int it = 0; it < kWarm + kTimed; ++it) {
    const auto t0 = Clock::now();
    const int err = nntr_hvx_add_f32(h, a, n, b, n, c, n);
    const auto t1 = Clock::now();
    if (err != AEE_SUCCESS) {
      r.err = err;
      break;
    }
    if (c[n - 1] != a[n - 1] + 0.5f) {
      ++r.bad;
    }
    if (it >= kWarm) {
      us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
  }
  print_row(name, us, r);
  return r;
}

/** @brief Words of the payload the echo is checked on: first, middle, last. */
constexpr uint32_t kProbe[3] = {0, kPay / 8, kPay / 4 - 1};

/**
 * @brief Q rows: one request/response packet per round trip.
 * @param arm_spin spin on read_noblock instead of a blocking read
 * @param pay 12 KiB in -> out echo through two buffer references
 * @param sent incremented per packet the DSP answered (dsp_stats check)
 */
Row run_q(const DspqApi &dq, dspqueue_t q, const char *name, bool arm_spin,
          bool pay, HtpRpcBuffer &in, HtpRpcBuffer &out, uint32_t *sent) {
  auto *iw = reinterpret_cast<uint32_t *>(in.data());
  auto *ow = reinterpret_cast<const uint32_t *>(out.data());
  Row r;
  std::vector<double> us;
  for (int it = 0; it < kWarm + kTimed; ++it) {
    const uint32_t seq = static_cast<uint32_t>(it);
    const uint32_t msg[2] = {kOpEcho, seq};
    struct dspqueue_buffer bufs[2] = {};
    if (pay) {
      for (int k = 0; k < 3; ++k) {
        iw[kProbe[k]] = seq * 2654435761u + k;
      }
      bufs[0].fd = in.fd();
      bufs[0].size = kPay;
      bufs[0].flags = DSPQUEUE_BUFFER_FLAG_REF |
                      DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER |
                      DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT;
      bufs[0].ptr = in.data();
      bufs[1].fd = out.fd();
      bufs[1].size = kPay;
      bufs[1].flags = DSPQUEUE_BUFFER_FLAG_REF;
      bufs[1].ptr = out.data();
    }
    const uint32_t nb = pay ? 2 : 0;
    uint32_t flags = 0, rnb = 0, len = 0, resp[2] = {~0u, ~0u};
    struct dspqueue_buffer rbufs[2] = {};

    const auto t0 = Clock::now();
    int err =
      dq.write(q, 0, nb, bufs, sizeof(msg),
               reinterpret_cast<const uint8_t *>(msg), DSPQUEUE_TIMEOUT_NONE);
    if (err == AEE_SUCCESS && !arm_spin) {
      err = dq.read(q, &flags, 2, &rnb, rbufs, sizeof(resp), &len,
                    reinterpret_cast<uint8_t *>(resp), kReadTimeoutUs);
    } else if (err == AEE_SUCCESS) {
      for (uint32_t spins = 1;; ++spins) {
        err = dq.read_noblock(q, &flags, 2, &rnb, rbufs, sizeof(resp), &len,
                              reinterpret_cast<uint8_t *>(resp));
        if (err != AEE_EWOULDBLOCK) {
          break;
        }
        if ((spins & 1023u) == 0 &&
            Clock::now() - t0 > std::chrono::microseconds(kReadTimeoutUs)) {
          err = AEE_EEXPIRED;
          break;
        }
      }
    }
    const auto t1 = Clock::now();
    if (err != AEE_SUCCESS) {
      r.err = err;
      break;
    }
    ++*sent;
    bool ok =
      len == sizeof(resp) && resp[0] == seq && resp[1] == 0 && rnb == nb;
    for (int k = 0; pay && k < 3; ++k) {
      ok = ok && ow[kProbe[k]] == iw[kProbe[k]];
    }
    if (!ok) {
      ++r.bad;
    }
    if (it >= kWarm) {
      us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
  }
  print_row(name, us, r);
  return r;
}

/** @brief Session as HtpFc::SetUp opens it, at the app's poll window. */
class DspqBench : public ::testing::Test {
protected:
  void SetUp() override {
    int err = htp_enable_unsigned_pd();
    ASSERT_EQ(err, AEE_SUCCESS) << "enabling unsigned PD failed: " << hex(err);
    const std::string uri = std::string(nntr_hvx_URI) + "&_dom=cdsp";
    err = nntr_hvx_open(uri.c_str(), &handle_);
    ASSERT_EQ(err, AEE_SUCCESS)
      << "nntr_hvx_open failed: " << hex(err)
      << " -- is libnntr_hvx_skel.so on ADSP_LIBRARY_PATH?";
    const char *env = std::getenv("NNTR_HTP_POLL_US");
    poll_us_ = env ? static_cast<uint32_t>(std::strtoul(env, nullptr, 10))
                   : 5000; // htp_backend.cpp's default
    qos_mode_ = htp_set_latency_qos(handle_, poll_us_);
  }

  void TearDown() override {
    if (handle_) {
      nntr_hvx_close(handle_);
    }
  }

  remote_handle64 handle_ = 0;
  uint32_t poll_us_ = 0;
  int qos_mode_ = 0;
};

/** @brief One Q phase: DSP thread in mode, four rows, optional F12p. */
struct Phase {
  const DspqApi &dq;
  remote_handle64 h;
  HtpRpcBuffer &in, &out;

  /** @return false when the phase could not start (message printed) */
  bool run(uint32_t mode, const char *const names[4], Row rows[4],
           const std::function<void()> &parked) {
    dspqueue_t q = nullptr;
    int err = dq.create(CDSP_DOMAIN_ID, 0, 0, 0, nullptr, nullptr, nullptr, &q);
    if (err != AEE_SUCCESS) {
      std::printf("DSPQ_BENCH unsupported side=arm err=%s\n", hex(err).c_str());
      ADD_FAILURE() << "dspqueue_create failed: " << hex(err);
      return false;
    }
    uint64_t id = 0;
    err = dq.export_(q, &id);
    if (err == AEE_SUCCESS) {
      err = nntr_hvx_dspq_bench_start(h, id, mode);
    }
    if (err != AEE_SUCCESS) {
      if (err == AEE_EUNSUPPORTED) {
        std::printf("DSPQ_BENCH unsupported side=dsp\n");
      }
      ADD_FAILURE() << "dspq_bench_start(mode " << mode
                    << ") failed: " << hex(err)
                    << " (0x8000040e = stale skel, rule 3)";
      dq.close(q);
      return false;
    }
    uint32_t sent = 0;
    for (int i = 0; i < 4; ++i) {
      rows[i] = run_q(dq, q, names[i], /*arm_spin=*/i % 2 == 1,
                      /*pay=*/i < 2, in, out, &sent);
      if (rows[i].err != 0) {
        break; // an abandoned request would shift every later row's seq
      }
    }
    if (parked) {
      parked();
    }
    const uint32_t quit[2] = {kOpQuit, 0};
    dq.write(q, 0, 0, nullptr, sizeof(quit),
             reinterpret_cast<const uint8_t *>(quit), DSPQUEUE_TIMEOUT_NONE);
    uint32_t res[4] = {};
    err = nntr_hvx_dspq_bench_stop(h, res, 4);
    std::printf("DSPQ_BENCH dsp_stats mode=%u served=%u empty=%u bad=%u "
                "arm_answered=%u\n",
                res[3], res[0], res[1], res[2], sent);
    EXPECT_EQ(err, AEE_SUCCESS) << "dspq_bench_stop: " << hex(err);
    EXPECT_EQ(res[2], 0u) << "DSP-side bad packets";
    EXPECT_EQ(res[0], sent) << "DSP served != ARM answered: not the same PD?";
    err = dq.close(q);
    EXPECT_EQ(err, AEE_SUCCESS) << "dspqueue_close: " << hex(err);
    return true;
  }
};

} // namespace

TEST_F(DspqBench, RoundTrip) {
  // F rows: the bare invoke on plain heap, then the 12 KiB-class invoke on
  // ION, the reference the decision is taken against.
  std::vector<float> a0(32), b0(32), c0(32);
  run_f(handle_, "F0", a0.data(), b0.data(), c0.data(), 32);
  HtpRpcBuffer fa(kPay), fb(kPay), fc(kPay);
  auto *pa = reinterpret_cast<float *>(fa.data());
  auto *pb = reinterpret_cast<float *>(fb.data());
  auto *pc = reinterpret_cast<float *>(fc.data());
  const int n12 = static_cast<int>(kPay / sizeof(float));
  const Row f12 = run_f(handle_, "F12", pa, pb, pc, n12);

  // Queue buffers, mapped once by fd as the arenas are
  // (htp_compute_ops.cpp).
  HtpRpcBuffer in(kPay), out(kPay);
  const HtpRpcMemApi &mem = HtpRpcMemApi::get();
  const bool ion = in.isIon() && out.isIon();
  bool mapped = ion && mem.mmap != nullptr && in.fd() >= 0 && out.fd() >= 0;
  if (mapped) {
    mapped = mem.mmap(CDSP_DOMAIN_ID, in.fd(), in.data(), 0, kPay,
                      FASTRPC_MAP_FD) == 0;
    if (mem.mmap(CDSP_DOMAIN_ID, out.fd(), out.data(), 0, kPay,
                 FASTRPC_MAP_FD) != 0) {
      mapped = false;
    }
  }

  const DspqApi dq;
  uint64_t sig = ~0ull;
  if (dq.missing == nullptr) {
    dspqueue_t probe = nullptr;
    if (dq.create(CDSP_DOMAIN_ID, 0, 0, 0, nullptr, nullptr, nullptr, &probe) ==
        AEE_SUCCESS) {
      if (dq.get_stat(probe, DSPQUEUE_STAT_SIGNALING_PERF, &sig) != 0) {
        sig = ~0ull;
      }
      dq.close(probe);
    }
  }
  std::printf("DSPQ_BENCH env qos_mode=%d poll_us=%u ion=%c mapped=%c "
              "signaling_perf=%lld\n",
              qos_mode_, poll_us_, ion ? 'y' : 'n', mapped ? 'y' : 'n',
              sig == ~0ull ? -1ll : static_cast<long long>(sig));
  EXPECT_TRUE(mapped) << "queue buffers are not ION + fastrpc_mmap'd";
  if (dq.missing != nullptr) {
    std::printf("DSPQ_BENCH unsupported side=arm sym=%s\n", dq.missing);
    FAIL() << "device libcdsprpc.so has no " << dq.missing;
  }

  Phase ph{dq, handle_, in, out};
  const char *const qb[4] = {"QBB12", "QBS12", "QBB0", "QBS0"};
  const char *const qs[4] = {"QSB12", "QSS12", "QSB0", "QSS0"};
  Row rb[4], rs[4];
  const bool okb = ph.run(0, qb, rb, nullptr);
  const bool oks = ph.run(1, qs, rs, [&] {
    run_f(handle_, "F12p", pa, pb, pc, n12); // S thread parked, queue idle
  });

  if (mapped) {
    mem.munmap(CDSP_DOMAIN_ID, in.fd(), in.data(), kPay);
    mem.munmap(CDSP_DOMAIN_ID, out.fd(), out.data(), kPay);
  }
  ASSERT_TRUE(okb && oks) << "a dspqueue phase did not start";

  // Plan 141 section 3: delta = F12 - QSS12 median; >= 50 adopt, < 25 drop.
  const double delta = f12.median - rs[1].median;
  const bool valid = f12.err == 0 && f12.bad == 0 && rs[1].err == 0 &&
                     rs[1].bad == 0 && rs[1].median > 0;
  const char *rule = !valid        ? "invalid"
                     : delta >= 50 ? "adopt"
                     : delta < 25  ? "drop"
                                   : "user";
  std::printf("DSPQ_BENCH verdict t_f_us=%.1f t_q_us=%.1f qbb12_us=%.1f "
              "cache_share_us=%.1f delta_us=%.1f s95_ms=%.2f s51_ms=%.2f "
              "s3_ms=%.3f s1_ms=%.3f rule=%s\n",
              f12.median, rs[1].median, rb[0].median,
              rs[1].median - rs[3].median, delta, 95 * delta / 1000,
              51 * delta / 1000, 3 * delta / 1000, delta / 1000, rule);
}

/** @brief googletest_main is gtest-all.cc only; each binary brings main. */
int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
