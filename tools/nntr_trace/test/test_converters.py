# SPDX-License-Identifier: Apache-2.0
"""The two device-side inputs open through summarize.py.

- htp_profile_to_trace.py on a real NNTR_HTP_PROFILE=2 print (LFM2.5-8B-A1B
  on S25U, test/data/htp_profile_sample.log): one representative call per
  shape, the stage totals on their lanes.
- The output HtpTrace (htp_trace.cpp on the HTP branch) wrote from its x86
  self-test driver (test/data/htp_trace_selftest.json): per-call spans,
  phases, staging, registration.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(HERE, "data")
sys.path.insert(0, ROOT)
import summarize  # noqa: E402


def metrics_of(path):
  with open(path) as f:
    model = summarize.parse_events(json.load(f))
  return summarize.metrics_json(model, summarize.resolve_range(model, "all")), model


class HtpProfileSummary(unittest.TestCase):
  def test_summary_print_converts(self):
    out = os.path.join(tempfile.mkdtemp(), "profile.json")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "htp_profile_to_trace.py"),
                        os.path.join(DATA, "htp_profile_sample.log"), "-o", out],
                       capture_output=True, text=True)
    self.assertEqual(r.returncode, 0, r.stderr)
    m, model = metrics_of(out)
    names = [k["name"] for k in m["kernels"]]
    self.assertIn("K2048_N2048_prefill: micro-mm", names)
    mm = next(k for k in m["kernels"] if k["name"] == "K2048_N2048_prefill: micro-mm")
    self.assertAlmostEqual(mm["total_us"], 8606.3, places=1)  # straight from the log
    waits = [e for e in model["all"] if e["cat"] == "host.wait"]
    self.assertEqual(len(waits), 3)
    decode = next(e for e in waits if e["args"]["op"] == "K2048_N2048_decode")
    self.assertEqual(decode["args"]["calls"], 11264)
    self.assertAlmostEqual(decode["dur"], 2168.9, places=1)
    self.assertEqual(m["buckets"]["load"], 1554.9 * 1000)


class HtpTraceSelftest(unittest.TestCase):
  def test_recorder_output(self):
    m, model = metrics_of(os.path.join(DATA, "htp_trace_selftest.json"))
    self.assertEqual([t["name"] for t in m["tokens"]],
                     ["prefill (444 tokens)", "decode token 1", "decode token 2", "decode token 3"])
    self.assertEqual([t["calls"] for t in m["tokens"]], [2, 8, 8, 8])
    self.assertEqual(m["transport"]["n"], 26)
    self.assertEqual(m["warnings"]["items"], [])
    moe = next(k for k in m["kernels"] if k["name"] == "moe_ffn: micro-mm (measured)")
    self.assertEqual(moe["calls"], 13)
    fc = next(k for k in m["kernels"] if k["name"] == "fc: micro-mm (residual)")
    self.assertAlmostEqual(fc["total_us"], 3202 - 357 - 794 - 501 - 58, places=3)
    untimed = [e for e in model["all"] if e["cat"] == "host.wait" and "note" in e["args"]]
    self.assertEqual(len(untimed), 12)  # gate_up calls passed stage_us=nullptr


if __name__ == "__main__":
  unittest.main()
