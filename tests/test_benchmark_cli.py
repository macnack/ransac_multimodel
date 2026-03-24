import json
import os
import subprocess
import sys
import tempfile
import unittest

from benchmarks.benchmark_numpy_vs_torch import parse_sample_ids


class TestBenchmarkCLI(unittest.TestCase):
    def test_parse_sample_ids_csv(self):
        self.assertEqual(parse_sample_ids("98,122,128"), [98, 122, 128])

    def test_parse_sample_ids_range(self):
        self.assertEqual(parse_sample_ids("3-5"), [3, 4, 5])

    def test_parse_sample_ids_invalid(self):
        with self.assertRaises(ValueError):
            parse_sample_ids("5-3")

    def test_synthetic_cli_smoke(self):
        with tempfile.TemporaryDirectory() as td:
            out_path = os.path.join(td, "latest.json")
            cmd = [
                sys.executable,
                "-m",
                "benchmarks.benchmark_numpy_vs_torch",
                "--synthetic",
                "--synthetic-n",
                "1024",
                "--modes",
                "residual",
                "--repeats",
                "1",
                "--warmup",
                "0",
                "--device",
                "cpu",
                "--quiet",
                "--output",
                out_path,
            ]
            proc = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(__file__)), capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertTrue(os.path.exists(out_path))

            with open(out_path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            self.assertIn("cases", payload)
            self.assertGreaterEqual(len(payload["cases"]), 1)
            self.assertIn("residual", payload["cases"][0])


if __name__ == "__main__":
    unittest.main()
