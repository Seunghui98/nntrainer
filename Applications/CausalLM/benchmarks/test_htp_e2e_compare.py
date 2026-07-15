import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
import htp_e2e_compare as cmp


class TestBuildRunCmd(unittest.TestCase):
    def test_no_decode_npu_env_var(self):
        # Decode routes to NPU by default now (no runtime toggle) —
        # build_run_cmd never injects NNTR_HTP_DECODE_NPU.
        cmd = cmp.build_run_cmd("/data/local/tmp/qwen3-0.6b")
        self.assertNotIn("NNTR_HTP_DECODE_NPU", cmd)
        self.assertIn("HTP_E2E_DUMP_IDS=1", cmd)


class TestParsers(unittest.TestCase):
    STDOUT = (
        "some log noise\n"
        "TOKEN_IDS: 10,20,30\n"
        "GENERATED_TEXT_BEGIN\n"
        "Hello, world.\nSecond line.\n"
        "GENERATED_TEXT_END\n"
        "more noise\n"
    )

    def test_parse_token_ids(self):
        self.assertEqual(cmp.parse_token_ids(self.STDOUT), [10, 20, 30])

    def test_parse_generated_text(self):
        self.assertEqual(cmp.parse_generated_text(self.STDOUT),
                         "Hello, world.\nSecond line.")

    def test_parse_generated_text_absent(self):
        self.assertEqual(cmp.parse_generated_text("no markers here"), "")


class TestEnvParameterization(unittest.TestCase):
    def test_install_dir_in_run_cmd(self):
        cmd = cmp.build_run_cmd(cmp.MODEL_DIR)
        self.assertIn(cmp.INSTALL_DIR, cmd)
        self.assertIn(f"cd {cmp.INSTALL_DIR}", cmd)
        self.assertIn(f"LD_LIBRARY_PATH={cmp.INSTALL_DIR}", cmd)
        self.assertIn(f"ADSP_LIBRARY_PATH={cmp.INSTALL_DIR}", cmd)

    def test_defaults_are_golden_device(self):
        # With no env overrides set, DEVICE defaults to the golden S25 Ultra.
        self.assertEqual(
            os.environ.get("HTP_E2E_DEVICE", "R3CY205ZMND"), cmp.DEVICE)


if __name__ == "__main__":
    unittest.main()
