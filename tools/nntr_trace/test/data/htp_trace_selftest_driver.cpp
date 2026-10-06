#include "htp_trace.h"
#include <cstdlib>
#include <thread>
using nntrainer::HtpTrace;
int main() {
  setenv("NNTR_TRACE", "selftest_trace.json", 1);
  HtpTrace &t = HtpTrace::global();
  t.setMeta(2, 2);
  uint64_t now = HtpTrace::nowUs();
  // registration
  t.registration(now, 1500, 700, 90, 2048, 2048);
  now += 1600;
  // prefill phase with one MoE call and one FC call
  uint64_t p0 = now;
  uint32_t moe[19] = {14916, 274, 3300, 905, 2927, 171, 53, 126, 1060, 1035, 8606, 164786, 0, 107, 1024, 60, 57, 383, 32};
  t.staging(now, 40, 1 << 20); now += 45;
  t.call(HtpTrace::KIND_MOE, 444, 2048, 2048, 4, now, 17296, moe, 19, 444 * 2048 * 4, 444 * 2048 * 4); now += 17300;
  uint32_t conv[19] = {4000, 120, 3300, 300, 400, 80, 0, 0, 0, 64, 2500, 0, 0, 30, 0, 0, 20, 60, 32};
  t.call(HtpTrace::KIND_CONV, 444, 2048, 2048, 2, now, 4700, conv, 19, 444 * 2048 * 4, 444 * 2048 * 4); now += 4750;
  uint32_t fc[7] = {3202, 357, 794, 501, 0, 58, 32};
  t.call(HtpTrace::KIND_FC, 444, 2048, 6144, 1, now, 3984, fc, 7, 444 * 2048 * 4, 444 * 6144 * 4); now += 4000;
  t.phase(p0, now - p0, 0, 444);
  // three decode tokens
  for (int tok = 0; tok < 3; ++tok) {
    uint64_t t0 = now;
    for (int l = 0; l < 4; ++l) {
      uint32_t d[19] = {1354, 15, 120, 13, 259, 106, 1, 139, 42, 4, 748, 21504, 0, 0, 1024, 8, 3, 4, 32};
      t.call(HtpTrace::KIND_MOE, 1, 2048, 2048, 4, now, 2169, d, 19, 8192, 8192); now += 2200;
      t.call(HtpTrace::KIND_GATE_UP, 1, 2048, 12288, 1, now, 900, nullptr, 0, 8192, 12288); now += 950;
    }
    t.phase(t0, now - t0, 444 + tok, 445 + tok);
    now += 30;
  }
  t.write();
  return 0;
}
