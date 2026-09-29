// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   unittest_hvx_two_sessions.cpp
 * @date   29 Sep 2026
 * @brief  [#178] Device probe: a second cDSP session (its own PD) beside the
 *         loaded app's -- address space, hop cost, DDR and VTCM sharing,
 *         teardown (plan docs/plans/178-second-dsp-session.md section 4)
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * Device only; reports S2_FIELD key=value lines, gates nothing but the
 * transport (a call that should work and fails is a FAIL, a number is
 * data). The five tests run in order and share one state: S1 is opened as
 * the app opens it and holds the arena ladder (3840 MiB expected) from Q1
 * to the end; S2 is reserved and opened in Q1 and closed in Q5. A test
 * whose precondition an earlier stop rule removed is SKIPPED, with the
 * reason. Never FASTRPC_SESSION_CLOSE (it closes S1 too); a hung S2 open is
 * killed with FASTRPC_REMOTE_PROCESS_KILL on S2's effective domain only.
 *
 * S2's heap probe beside its held mapping ladder runs the PD's address
 * space dry, which took S2's shell down in the 2026-09-29 23:20 sitting
 * (crash in __wrap_malloc; every later S2 call AEE_ENOSUCH 0x27): it is
 * Q5's last S2 cell, Q1 probes S2's heap with nothing mapped and a
 * 1024 MiB cap, and every S2 test starts with a liveness check
 * (S2_STOP rule=s2_dead).
 *
 * Stop rules (plan section 1): s2_reserve_rc = 0x73 (AEE_ENOSESSION) -> no
 * second session, Q2-Q4 skip; s2_mmap_mib < 512 -> the 4 GiB is per HLOS
 * process (printed as S2_STOP, the remaining cells still run: the heap may
 * still hold the weights); a Q4 FC set above 13 ms/token -> S2_STOP.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sched.h>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <AEEStdErr.h>
#include <remote.h>

#include "../htp/host/q4_gemv_cases.h"
#include "../htp/nntr_moe_dma_plan.h"
#include "htp_rpcmem.h"
#include "nntr_hvx.h"

namespace {

using nntrainer::HtpDspqApi;
using nntrainer::HtpRpcMemApi;
using Clock = std::chrono::steady_clock;

std::string hex(int err) {
  std::ostringstream os;
  os << "0x" << std::hex << std::setw(8) << std::setfill('0')
     << static_cast<unsigned>(err);
  return os.str();
}

template <typename T> void field(const char *k, const T &v) {
  std::cout << "S2_FIELD " << k << "=" << v << std::endl;
}

double us_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
}

constexpr size_t kStep = size_t(256) << 20; /**< ladder step, 256 MiB */
constexpr uint32_t FC_INTRIN = 1u, FC_FEED_VTCM = 1u << 16,
                   FC_FEED_L2 = 1u << 17; /**< fc_q4m1_f32's variant word */
constexpr uint32_t kMbxBytes = 64u * 1024u;

/** @brief One rpcmem buffer mapped into a session (and maybe attached). */
struct Mapped {
  void *buf = nullptr;
  int fd = -1;
  size_t bytes = 0;
  int domain = 0;
  uint32_t arena = ~0u;
};

/** @brief A canonical Q4_0 weight registered as Q4M1 on one session. */
struct FcWeight {
  uint32_t K = 0, N = 0, h = ~0u;
  std::vector<uint8_t> canon;
};

/** @brief One small HMX call (mm_u8i4_from_f32, M=1 K=N=256, fixed
 *  inputs); @return rc, the int32 accumulators in @a acc. Run on S1 at the
 *  start and after S2's close, the two must be equal (HMX + VTCM intact);
 *  on a lite S2 it must be AEE_EUNSUPPORTED. */
int hmx_smoke(remote_handle64 h, std::vector<int32_t> *acc) {
  // m_pad: HEXKL_HMX_INT8_BLOCK_N_ROW (64); a wrong length is the skel's
  // AEE_EBADPARM, which reads like a stale skel (the 23:20 sitting)
  const uint32_t M = 1, K = 256, N = 256, m_pad = 64;
  std::vector<float> x(K), d(N, 0.01f), bias(N, 0.0f), scale(m_pad), out(N);
  std::vector<int8_t> w(static_cast<size_t>(K) * N);
  std::vector<int32_t> colsum(N, 0), zp(m_pad);
  std::vector<uint8_t> ah(static_cast<size_t>(m_pad) * K);
  for (uint32_t i = 0; i < K; ++i) {
    x[i] = static_cast<float>((i * 37u) % 101u) / 50.0f - 1.0f;
  }
  for (size_t i = 0; i < w.size(); ++i) {
    w[i] = static_cast<int8_t>(static_cast<int>((i * 7u) % 16u) - 8);
  }
  acc->assign(static_cast<size_t>(m_pad) * N, 0);
  return nntr_hvx_mm_u8i4_from_f32(
    h, M, K, N, x.data(), static_cast<int>(K), w.data(),
    static_cast<int>(w.size()), d.data(), static_cast<int>(N), colsum.data(),
    static_cast<int>(N), bias.data(), static_cast<int>(N), ah.data(),
    static_cast<int>(ah.size()), scale.data(), static_cast<int>(m_pad),
    zp.data(), static_cast<int>(m_pad), acc->data(),
    static_cast<int>(acc->size()), out.data(), static_cast<int>(N));
}

/** @brief State the five tests share, in test order. */
struct Shared {
  remote_handle64 h1 = 0, h2 = 0;
  uint32_t effdom2 = 0;
  bool s2_open = false;
  std::vector<Mapped> s1_maps;
  uint32_t s1_info[7] = {};
  std::vector<int32_t> s1_acc; /**< hmx_smoke on S1 before S2 existed */
  FcWeight fc7168;             /**< on S2, from Q2 */
  uint32_t fc_feed = 0; /**< S2's best weight feed (VTCM or L2), from Q2 */
} g;

/** @brief rpcmem + fastrpc_mmap on @a domain, then arena_attach on @a h
 *  when @a attach. @return the fastrpc_mmap rc (or -1: no rpcmem). */
int map_chunk(remote_handle64 h, int domain, size_t bytes, bool attach,
              Mapped *m) {
  const HtpRpcMemApi &api = HtpRpcMemApi::get();
  if (!api.alloc || !api.to_fd || !api.mmap) {
    return -1;
  }
  m->buf =
    api.alloc(nntrainer::HTP_RPC_HEAP_ID_SYSTEM,
              nntrainer::HTP_RPC_FLAGS_UNCACHED, static_cast<int>(bytes));
  if (!m->buf) {
    return -2;
  }
  m->fd = api.to_fd(m->buf);
  m->bytes = bytes;
  m->domain = domain;
  int rc =
    api.mmap(domain, m->fd, m->buf, 0, bytes, static_cast<int>(FASTRPC_MAP_FD));
  if (rc == 0 && attach) {
    rc =
      nntr_hvx_arena_attach(h, m->fd, static_cast<uint32_t>(bytes), &m->arena);
    if (rc != AEE_SUCCESS) {
      api.munmap(domain, m->fd, m->buf, bytes);
    }
  }
  if (rc != 0) {
    api.free_(m->buf);
    m->buf = nullptr;
  }
  return rc;
}

void unmap_chunk(remote_handle64 h, Mapped *m) {
  const HtpRpcMemApi &api = HtpRpcMemApi::get();
  if (!m->buf) {
    return;
  }
  if (m->arena != ~0u) {
    nntr_hvx_arena_detach(h, m->arena);
  }
  if (api.munmap) {
    api.munmap(m->domain, m->fd, m->buf, m->bytes);
  }
  api.free_(m->buf);
  m->buf = nullptr;
}

/** @brief 256 MiB steps up to 4 GiB; @return MiB held, the stop reason and
 *  rc in @a why / @a rc. */
size_t ladder(remote_handle64 h, int domain, bool attach,
              std::vector<Mapped> *out, std::string *why, int *rc) {
  size_t total = 0;
  *why = "limit_4gib";
  *rc = 0;
  for (int i = 0; i < 16; ++i) {
    Mapped m;
    *rc = map_chunk(h, domain, kStep, attach, &m);
    if (*rc != 0) {
      *why = *rc == -2 ? "rpcmem_alloc" : "fastrpc_mmap_or_attach";
      break;
    }
    out->push_back(m);
    total += kStep;
  }
  return total >> 20;
}

/** @brief 1 MiB heap probe: allocates until refused or @a cap_mib, frees.
 *  A probe that runs the PD's address space dry can take the PD's own
 *  shell down with it (S2 crashed in __wrap_malloc right after one, the
 *  23:20 sitting), so every S2 probe that can reach the end runs last. */
uint32_t heap_mib(remote_handle64 h, uint32_t cap_mib = 4096) {
  uint32_t n = 0;
  uint64 sum = 0;
  const int rc = nntr_hvx_mem_probe_dsp_heap(h, 1, cap_mib, &n, 1, &sum, 1);
  return rc == AEE_SUCCESS ? n : 0u;
}

bool session_info(remote_handle64 h, uint32_t info[7], const char *who) {
  const int rc = nntr_hvx_session_info(h, info, 7);
  std::cout << "S2_FIELD session_info who=" << who << " rc=" << hex(rc)
            << " hmx_locked=" << info[0] << " vtcm_size=" << info[1]
            << " vtcm_avail_kib=" << info[2] << " vtcm_max_page_kib=" << info[3]
            << " open_path=" << info[5] << " hvx_units=" << info[6]
            << std::endl;
  EXPECT_EQ(rc, AEE_SUCCESS)
    << who << ": session_info (0x8000040e = stale skel, rule 3)";
  return rc == AEE_SUCCESS;
}

/** @brief S2 still answers (session_info); a dead PD (AEE_ENOSUCH 0x27
 *  from the framework) ends the S2 cells with an S2_STOP line. */
bool s2_alive(const char *where) {
  if (!g.s2_open) {
    return false;
  }
  uint32_t info[7] = {};
  const int rc = nntr_hvx_session_info(g.h2, info, 7);
  if (rc == AEE_SUCCESS) {
    return true;
  }
  std::cout << "S2_STOP rule=s2_dead where=" << where << " rc=" << hex(rc)
            << std::endl;
  g.s2_open = false;
  nntr_hvx_close(g.h2);
  g.h2 = 0;
  return false;
}

int fc_register(remote_handle64 h, uint32_t K, uint32_t N, FcWeight *w) {
  w->K = K;
  w->N = N;
  w->canon.resize(static_cast<size_t>(N) * (K / 32u) * 18u);
  make_weights(w->canon.data(), K, N);
  std::vector<uint8_t> m1(q4m1_bytes(K, N));
  q4m1_from_q4_0(w->canon.data(), K, N, m1.data());
  return nntr_hvx_q4m1_register(h, K, N, m1.data(), static_cast<int>(m1.size()),
                                &w->h);
}

/** @brief Output words that differ from the spec's, bit for bit. */
int fc_bad(const FcWeight &w, const std::vector<float> &x,
           const std::vector<float> &y) {
  std::vector<int8_t> q(w.K);
  std::vector<uint16_t> d(w.K / 32u);
  std::vector<float> ref(w.N);
  q8_0_quant_cpu_det(x.data(), w.K, q.data(), d.data());
  q4_gemv_cpu_det(w.canon.data(), q.data(), d.data(), w.K, w.N, ref.data());
  int bad = 0;
  for (uint32_t i = 0; i < w.N; ++i) {
    bad += std::memcmp(&y[i], &ref[i], sizeof(float)) != 0;
  }
  return bad;
}

/** @brief One fc_q4m1_f32 call on @a h; @return rc, DSP us in @a us. */
int fc_call(remote_handle64 h, const FcWeight &w, uint32_t variant,
            uint32_t lanes, uint32_t reps, const std::vector<float> &x,
            std::vector<float> *y, double *us) {
  std::vector<uint32_t> st(8, 0);
  y->assign(w.N, 0.0f);
  const int rc = nntr_hvx_fc_q4m1_f32(h, w.h, variant, lanes, reps, x.data(),
                                      static_cast<int>(w.K), y->data(),
                                      static_cast<int>(w.N), st.data(), 8);
  *us = st[0];
  return rc;
}

double fc_bytes(const FcWeight &w) {
  return static_cast<double>(w.N) * w.K * 18.0 / 32.0;
}

const char *feed_name(uint32_t f) {
  return f == FC_FEED_VTCM ? "vtcm" : f == FC_FEED_L2 ? "l2" : "direct";
}

/** @brief Runs @a fn on a thread pinned to @a cpu (-1: not pinned). */
template <typename F> std::thread pinned(int cpu, F fn) {
  return std::thread([cpu, fn] {
    if (cpu >= 0) {
      cpu_set_t set;
      CPU_ZERO(&set);
      CPU_SET(cpu, &set);
      sched_setaffinity(0, sizeof(set), &set);
    }
    fn();
  });
}

class TwoSessions : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    remote_rpc_control_unsigned_module unsigned_pd = {CDSP_DOMAIN_ID, 1};
    int err = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE,
                                     &unsigned_pd, sizeof(unsigned_pd));
    field("s1_unsigned_rc", hex(err));
    const std::string uri = std::string(nntr_hvx_URI) + "&_dom=cdsp";
    err = nntr_hvx_open(uri.c_str(), &g.h1);
    field("s1_open_rc", hex(err));
    if (err != AEE_SUCCESS) {
      g.h1 = 0;
    }
  }

  static void TearDownTestSuite() {
    if (g.s2_open) {
      nntr_hvx_close(g.h2);
      g.s2_open = false;
    }
    for (Mapped &m : g.s1_maps) {
      unmap_chunk(g.h1, &m);
    }
    g.s1_maps.clear();
    if (g.h1) {
      nntr_hvx_close(g.h1);
      g.h1 = 0;
    }
  }

  void SetUp() override {
    ASSERT_NE(g.h1, 0u) << "S1 did not open -- is libnntr_hvx_skel.so on "
                           "ADSP_LIBRARY_PATH?";
  }
};

/** @brief Q1: S1 at the loaded app's mapping, then a reserved S2: does it
 *  open, and what can it map and allocate beside S1's 3840 MiB? */
TEST_F(TwoSessions, Q1_SecondSession) {
  std::string why;
  int rc = 0;
  const size_t s1_mib =
    ladder(g.h1, CDSP_DOMAIN_ID, true, &g.s1_maps, &why, &rc);
  field("s1_mmap_mib", s1_mib);
  field("s1_mmap_stopped_by", why + ":" + hex(rc));
  field("s1_heap_mib", heap_mib(g.h1));
  session_info(g.h1, g.s1_info, "s1_start");
  rc = hmx_smoke(g.h1, &g.s1_acc);
  field("s1_hmx_smoke_rc", hex(rc));
  EXPECT_EQ(rc, AEE_SUCCESS) << "S1's HMX before S2";

  remote_rpc_reserve_new_session_t rs;
  std::memset(&rs, 0, sizeof(rs));
  char dom[] = "cdsp", sname[] = "nntr_s2";
  rs.domain_name = dom;
  rs.domain_name_len = std::strlen(dom);
  rs.session_name = sname;
  rs.session_name_len = std::strlen(sname);
  rc = remote_session_control(FASTRPC_RESERVE_NEW_SESSION, &rs, sizeof(rs));
  field("s2_reserve_rc", hex(rc));
  field("s2_session_id", rs.session_id);
  field("s2_effdom", rs.effective_domain_id);
  if (rc != AEE_SUCCESS) {
    std::cout << "S2_STOP rule="
              << (rc == 0x73 ? "no_second_session_0x73" : "reserve_failed")
              << " (Q2-Q4 skip; Q5 runs the S1 check)" << std::endl;
    return;
  }
  remote_rpc_effective_domain_id_t ed;
  std::memset(&ed, 0, sizeof(ed));
  ed.domain_name = dom;
  ed.domain_name_len = rs.domain_name_len;
  ed.session_id = rs.session_id;
  rc = remote_session_control(FASTRPC_GET_EFFECTIVE_DOMAIN_ID, &ed, sizeof(ed));
  field("s2_get_effdom_rc", hex(rc));
  field("s2_effdom_get", ed.effective_domain_id);
  ASSERT_EQ(rc, AEE_SUCCESS);
  g.effdom2 = ed.effective_domain_id;

  std::string mod(nntr_hvx_URI);
  std::vector<char> uri2(mod.size() + 64, '\0');
  remote_rpc_get_uri_t gu;
  std::memset(&gu, 0, sizeof(gu));
  gu.domain_name = dom;
  gu.domain_name_len = rs.domain_name_len;
  gu.session_id = rs.session_id;
  gu.module_uri = &mod[0];
  gu.module_uri_len = mod.size();
  gu.uri = uri2.data();
  gu.uri_len = uri2.size();
  rc = remote_session_control(FASTRPC_GET_URI, &gu, sizeof(gu));
  field("s2_get_uri_rc", hex(rc));
  field("s2_uri", std::string(uri2.data()));
  ASSERT_EQ(rc, AEE_SUCCESS);

  remote_rpc_control_unsigned_module up = {static_cast<int>(g.effdom2), 1};
  rc = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &up, sizeof(up));
  field("s2_unsigned_rc", hex(rc));
  ASSERT_EQ(rc, AEE_SUCCESS);

  // The open on a thread: S2's hw_init may wait on VTCM / HMX instead of
  // failing; after 10 s it is killed on S2's domain (never S1's).
  std::atomic<int> open_rc{AEE_EFAILED};
  std::atomic<bool> open_done{false};
  remote_handle64 h2 = 0;
  const auto t0 = Clock::now();
  std::thread opener([&] {
    open_rc = nntr_hvx_open(uri2.data(), &h2);
    open_done = true;
  });
  while (!open_done && us_since(t0) < 10e6) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  const double open_us = us_since(t0);
  if (!open_done) {
    remote_rpc_process_clean_params kill = {static_cast<int>(g.effdom2)};
    const int krc =
      remote_session_control(FASTRPC_REMOTE_PROCESS_KILL, &kill, sizeof(kill));
    std::cout << "S2_STOP rule=s2_open_hang kill_rc=" << hex(krc)
              << " (read the FARF of S2's hw_init)" << std::endl;
  }
  opener.join();
  field("s2_open_rc", hex(open_rc));
  field("s2_open_us", static_cast<uint64_t>(open_us));
  if (open_us > 2e6) {
    std::cout << "S2_STOP rule=s2_open_over_2s" << std::endl;
  }
  ASSERT_EQ(open_rc.load(), AEE_SUCCESS) << "S2 open";
  g.h2 = h2;
  g.s2_open = true;
  uint32_t info[7] = {};
  session_info(g.h2, info, "s2_open");
  {
    std::vector<int32_t> acc;
    field("s2_hmx_call_rc", hex(hmx_smoke(g.h2, &acc))); // lite: 0x80000414
  }
  session_info(g.h1, info, "s1_after_s2_open");

  // S2's own mapping ladder with S1's held (released at once; the heap
  // beside a held ladder runs the space dry and is Q5's last cell), then
  // its heap up to 1024 MiB with nothing mapped (the design needs 495).
  std::vector<Mapped> s2_maps;
  const size_t s2_mib =
    ladder(g.h2, static_cast<int>(g.effdom2), true, &s2_maps, &why, &rc);
  field("s2_mmap_mib", s2_mib);
  field("s2_mmap_stopped_by", why + ":" + hex(rc));
  for (Mapped &m : s2_maps) {
    unmap_chunk(g.h2, &m);
  }
  const uint32_t nomap = heap_mib(g.h2, 1024);
  field("s2_heap_mib_nomap",
        nomap == 1024 ? ">=1024 (cap)" : std::to_string(nomap));
  if (!s2_alive("q1_after_ladder")) {
    return;
  }
  if (s2_mib < 512) {
    std::cout << "S2_STOP rule=s2_mmap_lt_512 (the 4 GiB is per HLOS "
                 "process; heap cells still decide the weights)"
              << std::endl;
  }

  // The FC set (66 weights in 5 shapes, HvxFcQ4.Rate's table) plus the
  // lm_head in 16384-row slices, all held at once on S2's heap. The bytes
  // are a pattern: only the space is measured here.
  struct Shape {
    uint32_t K, N, count;
  };
  const Shape set[7] = {{2048u, 6144u, 18u}, {2048u, 2048u, 30u},
                        {2048u, 512u, 12u},  {2048u, 7168u, 4u},
                        {7168u, 2048u, 2u},  {2048u, 16384u, 7u},
                        {2048u, 13568u, 1u}};
  std::vector<uint32_t> held;
  size_t bytes = 0;
  int reg_rc = AEE_SUCCESS;
  for (const Shape &sh : set) {
    std::vector<uint8_t> w(q4m1_bytes(sh.K, sh.N), 0x5Au);
    for (uint32_t c = 0; c < sh.count && reg_rc == AEE_SUCCESS; ++c) {
      uint32_t h = ~0u;
      reg_rc = nntr_hvx_q4m1_register(g.h2, sh.K, sh.N, w.data(),
                                      static_cast<int>(w.size()), &h);
      if (reg_rc == AEE_SUCCESS) {
        held.push_back(h);
        bytes += w.size();
      }
    }
  }
  field("s2_q4m1_n", held.size());
  field("s2_q4m1_mib", bytes >> 20);
  field("s2_q4m1_stop_rc", hex(reg_rc));
  field("s2_q4m1_all_fit", held.size() == 74u ? "yes" : "no");
  for (uint32_t h : held) {
    nntr_hvx_q4m1_release(g.h2, h);
  }
  s2_alive("q1_end");
}

/** @brief One ARM-brokered hop row: echo on q1, then q2, alternating; each
 *  round trip one sample. DSP threads spin (dspq_bench mode 1). */
void hop_row(const HtpDspqApi &dq, dspqueue_t q[2], bool arm_spin, uint32_t pay,
             nntrainer::HtpRpcBuffer *in, nntrainer::HtpRpcBuffer *out,
             const char *name) {
  std::vector<double> us;
  int err = AEE_SUCCESS, bad = 0;
  for (int it = 0; it < 1100 && err == AEE_SUCCESS; ++it) {
    for (int s = 0; s < 2 && err == AEE_SUCCESS; ++s) {
      const uint32_t seq = static_cast<uint32_t>(it * 2 + s);
      const uint32_t msg[2] = {1u /*echo*/, seq};
      struct dspqueue_buffer bufs[2] = {};
      if (pay) {
        reinterpret_cast<uint32_t *>(in->data())[0] = seq;
        bufs[0].fd = in->fd();
        bufs[0].size = pay;
        bufs[0].ptr = in->data();
        bufs[0].flags = DSPQUEUE_BUFFER_FLAG_REF |
                        DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER |
                        DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT;
        bufs[1].fd = out->fd();
        bufs[1].size = pay;
        bufs[1].ptr = out->data();
        bufs[1].flags = DSPQUEUE_BUFFER_FLAG_REF;
      }
      uint32_t flags = 0, rnb = 0, len = 0, resp[2] = {~0u, ~0u};
      struct dspqueue_buffer rbufs[2] = {};
      const auto t0 = Clock::now();
      err =
        dq.write(q[s], 0, pay ? 2 : 0, bufs, sizeof(msg),
                 reinterpret_cast<const uint8_t *>(msg), DSPQUEUE_TIMEOUT_NONE);
      if (err == AEE_SUCCESS && !arm_spin) {
        err = dq.read(q[s], &flags, 2, &rnb, rbufs, sizeof(resp), &len,
                      reinterpret_cast<uint8_t *>(resp), 5000000);
      } else if (err == AEE_SUCCESS) {
        do {
          err = dq.read_noblock(q[s], &flags, 2, &rnb, rbufs, sizeof(resp),
                                &len, reinterpret_cast<uint8_t *>(resp));
        } while (err == AEE_EWOULDBLOCK && us_since(t0) < 5e6);
      }
      const double dt = us_since(t0);
      bad += err == AEE_SUCCESS &&
             (resp[0] != seq || resp[1] != 0 ||
              (pay && reinterpret_cast<uint32_t *>(out->data())[0] != seq));
      if (it >= 100) {
        us.push_back(dt);
      }
    }
  }
  std::sort(us.begin(), us.end());
  const double med = us.empty() ? 0 : us[us.size() / 2];
  std::cout << std::fixed << std::setprecision(1) << "S2_FIELD " << name << "="
            << med << " n=" << us.size()
            << " p90=" << (us.empty() ? 0 : us[us.size() * 9 / 10])
            << " bad=" << bad << " err=" << hex(err) << std::defaultfloat
            << std::endl;
  EXPECT_EQ(err, AEE_SUCCESS) << name;
  EXPECT_EQ(bad, 0) << name;
}

/** @brief mailbox_run on both sessions from two ARM threads; role 1 first. */
void mbox_pair(int fd, uint32_t n, uint32_t payload, uint32_t spin_us,
               uint32_t r0[5], uint32_t r1[5], int *rc0, int *rc1) {
  std::thread t1([&] {
    *rc1 =
      nntr_hvx_mailbox_run(g.h2, fd, kMbxBytes, 1u, n, payload, spin_us, r1, 5);
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  *rc0 =
    nntr_hvx_mailbox_run(g.h1, fd, kMbxBytes, 0u, n, payload, spin_us, r0, 5);
  t1.join();
}

/** @brief S2's FC at K=7168 on its best feed (VTCM at 6 lanes if it fits,
 *  else L2), reps calls per sample; median of 3 samples, us per call. */
double s2_fc_us(uint32_t reps, int *bad) {
  std::vector<float> x(g.fc7168.K), y;
  make_row(x.data(), g.fc7168.K, 0, 1);
  std::vector<double> s;
  for (int i = 0; i < 3; ++i) {
    double us = 0;
    const int rc =
      fc_call(g.h2, g.fc7168, FC_INTRIN | g.fc_feed, 6u, reps, x, &y, &us);
    EXPECT_EQ(rc, AEE_SUCCESS) << "S2 fc_q4m1_f32";
    s.push_back(us / reps);
  }
  *bad = fc_bad(g.fc7168, x, y);
  std::sort(s.begin(), s.end());
  return s[1];
}

/** @brief Q2: what one DSP-to-DSP hop costs, through the ARM (two
 *  dspqueues) and through a shared page both PDs poll. */
TEST_F(TwoSessions, Q2_HopCost) {
  if (!s2_alive("start")) {
    GTEST_SKIP() << "S2 not open or dead (an S2_STOP line says which)";
  }
  // S2's FC weight for Q2's thread-cost cell and Q3; its feed: VTCM if it
  // fits 6 lanes, else the L2 feed.
  ASSERT_EQ(fc_register(g.h2, 7168u, 2048u, &g.fc7168), AEE_SUCCESS);
  {
    std::vector<float> x(7168u), y;
    make_row(x.data(), 7168u, 0, 1);
    double us = 0;
    g.fc_feed = FC_FEED_VTCM;
    if (fc_call(g.h2, g.fc7168, FC_INTRIN | FC_FEED_VTCM, 6u, 1u, x, &y, &us) !=
        AEE_SUCCESS) {
      g.fc_feed = FC_FEED_L2;
    }
    field("s2_fc_feed", feed_name(g.fc_feed));
  }

  const HtpDspqApi &dq = HtpDspqApi::get();
  const HtpRpcMemApi &mem = HtpRpcMemApi::get();
  ASSERT_EQ(dq.missing, nullptr) << "no " << dq.missing;
  nntrainer::HtpRpcBuffer in(12288), out(12288);
  ASSERT_TRUE(in.isIon() && out.isIon() && mem.mmap);
  const int doms[2] = {CDSP_DOMAIN_ID, static_cast<int>(g.effdom2)};
  const remote_handle64 hs[2] = {g.h1, g.h2};
  for (int d : doms) {
    ASSERT_EQ(mem.mmap(d, in.fd(), in.data(), 0, 12288, FASTRPC_MAP_FD), 0);
    ASSERT_EQ(mem.mmap(d, out.fd(), out.data(), 0, 12288, FASTRPC_MAP_FD), 0);
  }
  dspqueue_t q[2] = {nullptr, nullptr};
  for (int s = 0; s < 2; ++s) {
    int err = dq.create(doms[s], 0, 0, 0, nullptr, nullptr, nullptr, &q[s]);
    field(s ? "s2_dspq_create_rc" : "s1_dspq_create_rc", hex(err));
    ASSERT_EQ(err, AEE_SUCCESS);
    uint64_t id = 0;
    ASSERT_EQ(dq.export_(q[s], &id), AEE_SUCCESS);
    err = nntr_hvx_dspq_bench_start(hs[s], id, 1u);
    ASSERT_EQ(err, AEE_SUCCESS)
      << "dspq_bench_start s" << s + 1 << " " << hex(err);
  }
  hop_row(dq, q, true, 0, &in, &out, "hop_arm_spin_us_0b");
  hop_row(dq, q, true, 12288, &in, &out, "hop_arm_spin_us_12k");
  hop_row(dq, q, false, 0, &in, &out, "hop_arm_block_us_0b");
  hop_row(dq, q, false, 12288, &in, &out, "hop_arm_block_us_12k");
  for (int s = 0; s < 2; ++s) {
    const uint32_t quit[2] = {2u, 0u};
    dq.write(q[s], 0, 0, nullptr, sizeof(quit),
             reinterpret_cast<const uint8_t *>(quit), DSPQUEUE_TIMEOUT_NONE);
    uint32_t res[4] = {};
    EXPECT_EQ(nntr_hvx_dspq_bench_stop(hs[s], res, 4), AEE_SUCCESS);
    dq.close(q[s]);
  }
  for (int d : doms) {
    mem.munmap(d, in.fd(), in.data(), 12288);
    mem.munmap(d, out.fd(), out.data(), 12288);
  }

  // The mailbox: one uncached page mapped in both sessions.
  nntrainer::HtpRpcBuffer page(kMbxBytes, nntrainer::HTP_RPC_FLAGS_UNCACHED);
  ASSERT_TRUE(page.isIon());
  for (int d : doms) {
    ASSERT_EQ(mem.mmap(d, page.fd(), page.data(), 0, kMbxBytes, FASTRPC_MAP_FD),
              0);
  }
  const uint32_t n = 10000u;
  for (uint32_t payload : {0u, 8192u}) {
    std::memset(page.data(), 0, kMbxBytes);
    uint32_t r0[5] = {}, r1[5] = {};
    int rc0 = 0, rc1 = 0;
    mbox_pair(page.fd(), n, payload, 1000u, r0, r1, &rc0, &rc1);
    std::cout << std::fixed << std::setprecision(2) << "S2_FIELD hop_mbox_us_"
              << (payload ? "8k" : "0b") << "=" << r0[0] / (2.0 * n)
              << " done=" << r0[1] << "/" << r1[1] << " timeouts=" << r0[2]
              << "/" << r1[2] << " bad=" << r0[4] << "/" << r1[4]
              << " checksum=" << r0[3] << "/" << r1[3] << " rc=" << hex(rc0)
              << "/" << hex(rc1) << std::defaultfloat << std::endl;
    EXPECT_EQ(rc0, AEE_SUCCESS);
    EXPECT_EQ(rc1, AEE_SUCCESS);
    EXPECT_EQ(r0[2] + r1[2], 0u) << "mailbox timeouts";
    EXPECT_EQ(r0[4] + r1[4], 0u) << "mailbox stale payload words";
  }

  // What S1's spinning mailbox thread takes from S2's FC: the FC alone,
  // then with S1 role 0 spinning on a pong that never comes (1 s spin).
  int bad = 0;
  const double parked = s2_fc_us(20u, &bad);
  std::memset(page.data(), 0, kMbxBytes);
  uint32_t r0[5] = {};
  int rc0 = 0;
  std::thread spinner([&] {
    rc0 = nntr_hvx_mailbox_run(g.h1, page.fd(), kMbxBytes, 0u, 1u, 0u, 1000000u,
                               r0, 5);
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  int bad2 = 0;
  const double spinning = s2_fc_us(20u, &bad2);
  spinner.join();
  std::cout << std::fixed << std::setprecision(1)
            << "S2_FIELD hop_mbox_thread_cost_pct="
            << 100.0 * (spinning / parked - 1.0) << " fc_parked_us=" << parked
            << " fc_spinning_us=" << spinning
            << " feed=" << feed_name(g.fc_feed) << " bad=" << bad << "/" << bad2
            << " spinner_timeouts=" << r0[2] << " rc=" << hex(rc0)
            << std::defaultfloat << std::endl;
  EXPECT_EQ(bad + bad2, 0);
  for (int d : doms) {
    mem.munmap(d, page.fd(), page.data(), kMbxBytes);
  }
}

/** @brief Q3: S1's MoE-shaped DMA (#158 cell: bypass, fresh, 1 queue) and
 *  S2's FC weight DMA alone, concurrent and sequential. */
TEST_F(TwoSessions, Q3_DdrShare) {
  if (!s2_alive("q3_start") || g.fc7168.h == ~0u || g.s1_maps.empty()) {
    GTEST_SKIP() << "needs S2, Q2's FC weight and S1's arena";
  }
  static nntr_moe_dma_item items[NNTR_MOE_DMA_PLAN_MAX];
  nntr_two_reader_spec spec;
  const uint32_t n_items =
    nntr_dma_settings_cell(1u, items, NNTR_MOE_DMA_PLAN_MAX, &spec);
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
  const uint32_t region = nntr_moe_dma_region_bytes(2048u, 1792u, 2048u);
  const uint32_t arena = g.s1_maps[0].arena;
  const uint32_t calls = 40u, reps = 40u;
  std::vector<float> x(g.fc7168.K), y;
  make_row(x.data(), g.fc7168.K, 0, 1);

  struct Side {
    int rc = 0;
    double dsp_us = 0, bytes = 0;
    Clock::time_point t0, t1;
  };
  auto s1_run = [&](Side *s) {
    std::vector<uint32_t> res(13, 0);
    s->t0 = Clock::now();
    s->rc = nntr_hvx_dma_replay(
      g.h1, arena, region, sched.data(), static_cast<int>(sched.size()), 1u, 0u,
      0u, spec.fresh, 0u, calls, res.data(), static_cast<int>(res.size()));
    s->t1 = Clock::now();
    s->dsp_us = res[0];
    s->bytes = static_cast<double>(res[2]) * res[1];
  };
  auto s2_run = [&](Side *s) {
    std::vector<float> yy;
    s->t0 = Clock::now();
    s->rc = fc_call(g.h2, g.fc7168, FC_INTRIN | g.fc_feed, 6u, reps, x, &yy,
                    &s->dsp_us);
    s->t1 = Clock::now();
    s->bytes = fc_bytes(g.fc7168) * reps;
  };
  auto gbs = [](double bytes, double us) {
    return us > 0 ? bytes / us / 1e3 : 0;
  };
  auto window = [](const Side &a, const Side &b) {
    const auto t0 = std::min(a.t0, b.t0);
    const auto t1 = std::max(a.t1, b.t1);
    return std::chrono::duration<double, std::micro>(t1 - t0).count();
  };

  Side a, b;
  s1_run(&a); // warm (page tables, clocks)
  s1_run(&a);
  s2_run(&b);
  field("ddr_s1_alone_gbs", gbs(a.bytes, a.dsp_us));
  field("ddr_s2_alone_gbs", gbs(b.bytes, b.dsp_us));
  field("ddr_s2_feed", feed_name(g.fc_feed));
  EXPECT_EQ(a.rc, AEE_SUCCESS) << "dma_replay " << hex(a.rc);
  EXPECT_EQ(b.rc, AEE_SUCCESS) << "fc_q4m1_f32 " << hex(b.rc);

  for (int pin = 0; pin < 2; ++pin) {
    std::atomic<int> ready{0};
    Side ca, cb;
    auto go = [&](Side *s, bool one) {
      ready.fetch_add(1);
      while (ready.load() < 2) {
      }
      one ? s1_run(s) : s2_run(s);
    };
    std::thread t1 = pinned(pin ? 0 : -1, [&] { go(&ca, true); });
    std::thread t2 = pinned(pin ? 1 : -1, [&] { go(&cb, false); });
    t1.join();
    t2.join();
    const double skew = std::abs(
      std::chrono::duration<double, std::micro>(ca.t0 - cb.t0).count());
    std::cout << std::fixed << std::setprecision(2)
              << "S2_FIELD ddr_concurrent_aggregate_gbs"
              << (pin ? "_pinned_cpu0_cpu1" : "") << "="
              << gbs(ca.bytes + cb.bytes, window(ca, cb))
              << " s1_gbs=" << gbs(ca.bytes, ca.dsp_us)
              << " s2_gbs=" << gbs(cb.bytes, cb.dsp_us)
              << " start_skew_us=" << skew << " rc=" << hex(ca.rc) << "/"
              << hex(cb.rc) << std::defaultfloat << std::endl;
  }
  Side sa, sb;
  s1_run(&sa);
  s2_run(&sb);
  std::cout << std::fixed << std::setprecision(2)
            << "S2_FIELD ddr_sequential_gbs="
            << gbs(sa.bytes + sb.bytes, window(sa, sb)) << " dsp_only_gbs="
            << gbs(sa.bytes + sb.bytes, sa.dsp_us + sb.dsp_us)
            << std::defaultfloat << std::endl;
}

/** @brief Q4: S2's VTCM beside S1's feed, and the exact FC's rate from
 *  VTCM, the L2 scratch and DDR directly -> the FC set's ms per token. */
TEST_F(TwoSessions, Q4_VtcmShare) {
  if (!s2_alive("start")) {
    GTEST_SKIP() << "S2 not open or dead (an S2_STOP line says which)";
  }
  uint32_t info[7] = {};
  session_info(g.h2, info, "s2_q4");
  field("s2_vtcm_avail_kib", info[2]);
  field("s2_vtcm_max_page_kib", info[3]);
  field("s2_vtcm_size_kib", info[1] >> 10);

  FcWeight w2048;
  ASSERT_EQ(fc_register(g.h2, 2048u, 6144u, &w2048), AEE_SUCCESS);
  if (g.fc7168.h == ~0u) {
    ASSERT_EQ(fc_register(g.h2, 7168u, 2048u, &g.fc7168), AEE_SUCCESS);
  }
  const FcWeight *ws[2] = {&g.fc7168, &w2048};
  const uint32_t feeds[3] = {FC_FEED_VTCM, FC_FEED_L2, 0u};
  double best[2][3] = {}; // [shape][feed] -> best GB/s
  int bad_total = 0;
  for (int si = 0; si < 2; ++si) {
    const FcWeight &w = *ws[si];
    std::vector<float> x(w.K), y;
    make_row(x.data(), w.K, 0, 1);
    for (int fi = 0; fi < 3; ++fi) {
      for (uint32_t lanes = 1; lanes <= 6; ++lanes) {
        double us = 0;
        const uint32_t v = FC_INTRIN | feeds[fi];
        int rc = fc_call(g.h2, w, v, lanes, 1u, x, &y, &us);
        if (rc == AEE_SUCCESS) {
          rc = fc_call(g.h2, w, v, lanes, 3u, x, &y, &us);
        }
        if (rc != AEE_SUCCESS) {
          std::cout << "S2_FIELD s2_fc_rate K=" << w.K << " N=" << w.N
                    << " feed=" << feed_name(feeds[fi]) << " lanes=" << lanes
                    << " skipped rc=" << hex(rc) << std::endl;
          continue;
        }
        const int bad = fc_bad(w, x, y);
        bad_total += bad;
        const double per = us / 3.0, rate = fc_bytes(w) / (per * 1e3);
        best[si][fi] = std::max(best[si][fi], rate);
        std::cout << std::fixed << std::setprecision(2)
                  << "S2_FIELD s2_fc_rate K=" << w.K << " N=" << w.N
                  << " feed=" << feed_name(feeds[fi]) << " lanes=" << lanes
                  << " us_per_call=" << per << " gbs=" << rate << " bad=" << bad
                  << std::defaultfloat << std::endl;
      }
    }
  }
  // The FC set + lm_head bytes by K (HvxFcQ4.Rate's per-token table).
  const double q4 = 18.0 / 32.0;
  const double b7168 = 2.0 * 7168 * 2048 * q4;
  const double b2048 =
    (18.0 * 6144 + 30.0 * 2048 + 12.0 * 512 + 4.0 * 7168 + 128000.0) * 2048 *
    q4;
  double best_ms = 0;
  for (int fi = 0; fi < 3; ++fi) {
    const double us7168 = fc_bytes(g.fc7168) / (best[0][fi] * 1e3);
    field(fi == 0   ? "s2_fc_rate_vtcm_us"
          : fi == 1 ? "s2_fc_rate_l2_us"
                    : "s2_fc_rate_direct_us",
          best[0][fi] > 0 ? us7168 : 0.0);
    if (best[0][fi] <= 0 || best[1][fi] <= 0) {
      continue;
    }
    const double ms = b7168 / best[0][fi] / 1e6 + b2048 / best[1][fi] / 1e6;
    std::cout << std::fixed << std::setprecision(3)
              << "S2_FIELD s2_fc_ms_per_token feed=" << feed_name(feeds[fi])
              << " ms=" << ms << " gbs_k7168=" << best[0][fi]
              << " gbs_k2048=" << best[1][fi]
              << " set_mb=" << (b7168 + b2048) / 1e6 << " cpu_reference_ms=7.4"
              << std::defaultfloat << std::endl;
    best_ms = best_ms == 0 ? ms : std::min(best_ms, ms);
  }
  field("s2_fc_ms_per_token_best", best_ms);
  if (best_ms == 0 || best_ms > 13.0) {
    std::cout << "S2_STOP rule=fc_set_over_13ms (the E2E path cannot beat A;"
                 " recommendation (c) of #132)"
              << std::endl;
  }
  field("s2_fc_bad_total", bad_total);
  EXPECT_EQ(bad_total, 0);
  nntr_hvx_q4m1_release(g.h2, w2048.h);
}

/** @brief Q5: S2 closes first; S1 keeps its HMX lock and VTCM. */
TEST_F(TwoSessions, Q5_Teardown) {
  if (s2_alive("q5_start")) {
    // The one cell that may run S2's address space dry (and its shell with
    // it): S2's heap with its whole mapping ladder held, beside S1's.
    std::vector<Mapped> s2_maps;
    std::string why;
    int rc = 0;
    const size_t mib =
      ladder(g.h2, static_cast<int>(g.effdom2), true, &s2_maps, &why, &rc);
    field("s2_mmap_mib_q5", mib);
    field("s2_heap_mib", heap_mib(g.h2)); // with the ladder held
    for (Mapped &m : s2_maps) {
      unmap_chunk(g.h2, &m);
    }
    s2_alive("q5_after_heap_probe");
  }
  if (g.s2_open) {
    const int rc = nntr_hvx_close(g.h2);
    field("s2_close_rc", hex(rc));
    g.s2_open = false;
    g.h2 = 0;
  }
  uint32_t info[7] = {};
  ASSERT_TRUE(session_info(g.h1, info, "s1_end"));
  std::vector<int32_t> acc;
  const int rc = hmx_smoke(g.h1, &acc);
  const bool same = rc == AEE_SUCCESS && acc == g.s1_acc;
  field("s1_hmx_smoke_end_rc", hex(rc));
  field("s1_keeps_hmx", info[0] == 1u && same ? "yes" : "no");
  field("s1_keeps_vtcm", info[1] == g.s1_info[1] && same ? "yes" : "no");
  field("vtcm_avail_kib_start_end",
        std::to_string(g.s1_info[2]) + "/" + std::to_string(info[2]));
  EXPECT_EQ(info[0], 1u) << "S1 lost the HMX lock";
  EXPECT_EQ(info[1], g.s1_info[1]) << "S1's VTCM changed";
  EXPECT_TRUE(same) << "S1's HMX call after S2 differs from before";
}

} // namespace

/** @brief googletest_main is gtest-all.cc only; each binary brings main. */
int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
