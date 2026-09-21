# SPDX-License-Identifier: Apache-2.0
"""Placeholder until summarize.py lands (VIEWER_PLAN.md W12)."""
import json
import os
import unittest

FIX = os.path.join(os.path.dirname(__file__), "fixtures")


class ViewerMetricsFixture(unittest.TestCase):
  def test_viewer_dumped_metrics(self):
    p = os.path.join(FIX, "viewer_metrics.json")
    if not os.path.exists(p):
      self.skipTest("run check.js first")
    with open(p) as f:
      m = json.load(f)
    self.assertEqual(m["schema"], "nntr_trace.metrics.v1")
    self.assertGreater(m["wall_us"], 0)


if __name__ == "__main__":
  unittest.main()
