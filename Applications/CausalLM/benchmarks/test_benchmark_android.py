import io
import json
import sys
import os
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(__file__))
import benchmark_android as bm


class TestParseStrListArg(unittest.TestCase):
    def test_single(self):
        self.assertEqual(bm.parse_str_list_arg("htp"), ["htp"])

    def test_multiple(self):
        self.assertEqual(bm.parse_str_list_arg("htp,cpu"), ["htp", "cpu"])

    def test_whitespace(self):
        self.assertEqual(bm.parse_str_list_arg("htp, cpu"), ["htp", "cpu"])

    def test_empty_segment_ignored(self):
        self.assertEqual(bm.parse_str_list_arg("htp,,cpu"), ["htp", "cpu"])


class TestConfigModifierComputeEngine(unittest.TestCase):
    def _make_mock_config(self):
        return {
            "init_seq_len": 16,
            "num_to_generate": 32,
            "batch_size": 1,
            "compute_engine": "cpu",
            "sample_input": "hello",
        }

    def _run_enter(self, compute_engine):
        """Run ConfigModifier.__enter__ with mocked adb/tokenizer calls."""
        cfg = self._make_mock_config()
        adb_cat_result = MagicMock(returncode=0, stdout=json.dumps(cfg))
        adb_push_result = MagicMock(returncode=0)

        with patch("subprocess.run", side_effect=[
            adb_cat_result,   # adb shell cat (backup read)
            MagicMock(),      # adb shell cp (backup)
            adb_push_result,  # adb push modified config (tokenizer_file absent: no pulls)
        ]), patch("tempfile.NamedTemporaryFile") as mock_tmp, \
            patch("os.remove"), patch("shutil.rmtree"), \
            patch("tempfile.mkdtemp", return_value="/tmp/fake_tok"):

            mock_tmp.return_value.__enter__.return_value.name = "/tmp/fake.json"
            modifier = bm.backup_and_modify_config(
                "/data/local/tmp/nntrainer/models/qwen3-0.6b",
                32, 0, 1, compute_engine=compute_engine
            )
            written_config = {}

            def fake_json_dump(obj, f, **kwargs):
                written_config.update(obj)

            with patch("json.dump", side_effect=fake_json_dump):
                modifier.__enter__()

        return written_config

    def test_compute_engine_set_when_specified(self):
        cfg = self._run_enter("htp")
        self.assertEqual(cfg.get("compute_engine"), "htp")

    def test_compute_engine_unchanged_when_none(self):
        cfg = self._run_enter(None)
        # None means: don't touch compute_engine; original "cpu" stays
        self.assertEqual(cfg.get("compute_engine"), "cpu")

    def test_compute_engine_not_injected_when_none_and_key_absent(self):
        """When compute_engine=None and device config has no compute_engine key,
        the key must not be injected into the written config."""
        cfg_no_key = {
            "init_seq_len": 16,
            "num_to_generate": 32,
            "batch_size": 1,
            "sample_input": "hello",
            # no "compute_engine" key
        }
        adb_cat_result = MagicMock(returncode=0, stdout=json.dumps(cfg_no_key))
        adb_push_result = MagicMock(returncode=0)

        with patch("subprocess.run", side_effect=[
            adb_cat_result,
            MagicMock(),
            adb_push_result,
        ]), patch("tempfile.NamedTemporaryFile") as mock_tmp, \
            patch("os.remove"), patch("shutil.rmtree"), \
            patch("tempfile.mkdtemp", return_value="/tmp/fake_tok"):

            mock_tmp.return_value.__enter__.return_value.name = "/tmp/fake.json"
            modifier = bm.backup_and_modify_config(
                "/data/local/tmp/nntrainer/models/qwen3-0.6b",
                32, 0, 1, compute_engine=None
            )
            written_config = {}

            def fake_json_dump(obj, f, **kwargs):
                written_config.update(obj)

            with patch("json.dump", side_effect=fake_json_dump):
                modifier.__enter__()

        self.assertNotIn("compute_engine", written_config)


class TestMainArgparse(unittest.TestCase):
    def test_compute_engine_default_none(self):
        with patch("sys.argv", [
            "benchmark_android.py",
            "-m", "/data/local/tmp/nntrainer/models/qwen3-0.6b",
        ]):
            parser = bm._build_parser()
            args = parser.parse_args([
                "-m", "/data/local/tmp/nntrainer/models/qwen3-0.6b"
            ])
            self.assertIsNone(args.compute_engine)

    def test_compute_engine_parsed(self):
        parser = bm._build_parser()
        args = parser.parse_args([
            "-m", "/data/local/tmp/nntrainer/models/qwen3-0.6b",
            "--compute-engine", "htp,cpu",
        ])
        self.assertEqual(args.compute_engine, "htp,cpu")

    def test_extra_env_parsed(self):
        parser = bm._build_parser()
        args = parser.parse_args([
            "-m", "/data/local/tmp/nntrainer/models/qwen3-0.6b",
            "--env", "NNTR_HTP_QINT8_ATTENTION_ENABLE=0",
            "--env", "NNTR_HTP_QINT8_TRACE=1",
        ])
        self.assertEqual(
            args.env,
            ["NNTR_HTP_QINT8_ATTENTION_ENABLE=0", "NNTR_HTP_QINT8_TRACE=1"],
        )


class TestEnvAssignments(unittest.TestCase):
    def test_parse_env_assignments(self):
        parsed = bm.parse_env_assignments([
            "NNTR_HTP_QINT8_ATTENTION_ENABLE=0",
            "NNTR_HTP_QINT8_TRACE=1",
        ])
        self.assertEqual(
            parsed,
            {
                "NNTR_HTP_QINT8_ATTENTION_ENABLE": "0",
                "NNTR_HTP_QINT8_TRACE": "1",
            },
        )

    def test_parse_env_assignments_rejects_invalid_key(self):
        with self.assertRaises(ValueError):
            bm.parse_env_assignments(["BAD-KEY=1"])

    def test_build_remote_command_includes_env(self):
        cmd = bm.build_remote_command(
            "/data/local/tmp/nntrainer/models/qwen3-0.6b",
            omp_threads=4,
            extra_env={
                "NNTR_HTP_QINT8_ATTENTION_ENABLE": "0",
                "NNTR_HTP_QINT8_TRACE": "1",
            },
        )
        self.assertIn("OMP_NUM_THREADS=4", cmd)
        self.assertIn("NNTR_HTP_QINT8_ATTENTION_ENABLE=0", cmd)
        self.assertIn("NNTR_HTP_QINT8_TRACE=1", cmd)
        self.assertIn("./run_causallm.sh '/data/local/tmp/nntrainer/models/qwen3-0.6b'", cmd)


class TestOutputResultsTable(unittest.TestCase):
    def _make_results(self):
        return [
            {"compute_engine": "htp", "n_threads": 4, "n_prompt": 32,
             "n_gen": 0, "prefill_mean": 2.5, "prefill_std": 0.1,
             "gen_mean": 0.0, "gen_std": 0.0},
            {"compute_engine": "cpu", "n_threads": 4, "n_prompt": 32,
             "n_gen": 0, "prefill_mean": 8.2, "prefill_std": 0.2,
             "gen_mean": 0.0, "gen_std": 0.0},
        ]

    def test_engine_column_present(self):
        f = io.StringIO()
        with redirect_stdout(f):
            bm.output_results_table(
                self._make_results(), "qwen3-0.6b", "1.2 GiB",
                "CausalLM", "FP16-FP32", "S25U"
            )
        output = f.getvalue()
        self.assertIn("Engine", output)
        self.assertIn("htp", output)
        self.assertIn("cpu", output)


class TestEnginesProductIntegration(unittest.TestCase):
    def test_no_compute_engine_produces_none_engine(self):
        """Omitting --compute-engine yields engines=[None], preserving old behavior."""
        parser = bm._build_parser()
        args = parser.parse_args([
            "-m", "/data/local/tmp/nntrainer/models/qwen3-0.6b",
            "-p", "32,64",
            "-n", "0",
            "-t", "4",
        ])
        engines = bm.parse_str_list_arg(args.compute_engine) if args.compute_engine else [None]
        self.assertEqual(engines, [None])

    def test_compute_engine_expands_configs(self):
        """--compute-engine htp,cpu doubles the config count vs omitting it."""
        from itertools import product as iterproduct
        parser = bm._build_parser()

        args_no_engine = parser.parse_args([
            "-m", "/data/local/tmp/nntrainer/models/qwen3-0.6b",
            "-p", "32,64", "-n", "0", "-t", "4",
        ])
        engines_no = [None]
        configs_no = list(iterproduct(
            engines_no,
            bm.parse_list_arg(args_no_engine.n_prompt),
            bm.parse_list_arg(args_no_engine.n_gen),
            bm.parse_list_arg(args_no_engine.n_threads),
        ))

        args_with_engine = parser.parse_args([
            "-m", "/data/local/tmp/nntrainer/models/qwen3-0.6b",
            "-p", "32,64", "-n", "0", "-t", "4",
            "--compute-engine", "htp,cpu",
        ])
        engines_with = bm.parse_str_list_arg(args_with_engine.compute_engine)
        configs_with = list(iterproduct(
            engines_with,
            bm.parse_list_arg(args_with_engine.n_prompt),
            bm.parse_list_arg(args_with_engine.n_gen),
            bm.parse_list_arg(args_with_engine.n_threads),
        ))

        self.assertEqual(len(configs_with), 2 * len(configs_no))
        self.assertEqual(configs_no[0][0], None)   # engine is first element
        self.assertEqual(configs_with[0][0], "htp")
        self.assertEqual(configs_with[2][0], "cpu")


if __name__ == "__main__":
    unittest.main()
