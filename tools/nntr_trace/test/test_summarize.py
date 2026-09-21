# SPDX-License-Identifier: Apache-2.0
"""summarize.py must reproduce the viewer's metrics JSON to 1e-6 relative.

check.js dumps window.__nntr.metricsJSON() for the as-built sample into
fixtures/viewer_metrics.json (whole-trace range); this compares every number
against summarize.py on the same trace, so the two implementations cannot
drift apart (VIEWER_PLAN.md W12).
"""
import json
import os
import subprocess
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIX = os.path.join(HERE, "fixtures")
sys.path.insert(0, ROOT)
import summarize  # noqa: E402


def walk(a, b, path, diffs, tol=1e-6):
  if isinstance(a, dict):
    if set(a) != set(b):
      diffs.append("%s: keys %s vs %s" % (path, sorted(set(a) ^ set(b)), ""))
      return
    for k in a:
      walk(a[k], b[k], path + "." + k, diffs)
  elif isinstance(a, list):
    if len(a) != len(b):
      diffs.append("%s: len %d vs %d" % (path, len(a), len(b)))
      return
    for i, (x, y) in enumerate(zip(a, b)):
      walk(x, y, "%s[%d]" % (path, i), diffs)
  elif isinstance(a, bool) or a is None or isinstance(a, str):
    if a != b:
      diffs.append("%s: %r vs %r" % (path, a, b))
  else:
    if abs(a - b) > tol * max(1.0, abs(a), abs(b)):
      diffs.append("%s: %r vs %r" % (path, a, b))


class SummarizeMatchesViewer(unittest.TestCase):
  def setUp(self):
    self.vm = os.path.join(FIX, "viewer_metrics.json")
    self.trace = os.path.join(FIX, "trace_asbuilt.json")
    if not (os.path.exists(self.vm) and os.path.exists(self.trace)):
      self.skipTest("run test/run.sh first (check.js writes the fixture)")

  def test_matches_viewer(self):
    with open(self.vm) as f:
      viewer = json.load(f)
    with open(self.trace) as f:
      model = summarize.parse_events(json.load(f))
    ours = summarize.metrics_json(model, summarize.resolve_range(model, "all"))
    diffs = []
    walk(viewer, ours, "$", diffs)
    self.assertEqual(diffs[:10], [], "viewer vs summarize.py differ (%d): %s" % (len(diffs), diffs[:10]))

  def test_cli_and_gates(self):
    out = subprocess.run([sys.executable, os.path.join(ROOT, "summarize.py"), self.trace,
                          "--fail-if", "compression<9"], capture_output=True, text=True)
    self.assertEqual(out.returncode, 1, out.stderr)
    self.assertIn("GATE FAILED: compression", out.stderr)
    out = subprocess.run([sys.executable, os.path.join(ROOT, "summarize.py"), self.trace,
                          "--range", "phase:0", "--fail-if", "idle_ratio>0.5"],
                         capture_output=True, text=True)
    self.assertEqual(out.returncode, 0, out.stderr)
    self.assertIn('"prefill', out.stdout)


if __name__ == "__main__":
  unittest.main()
