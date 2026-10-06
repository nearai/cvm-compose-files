#!/usr/bin/env python3
# How to run: python3 -m unittest scripts.test_gpu13_glm53_tp2 (needs ruby; CI runs it in ruby:3.3-alpine)
"""gpu13's GLM service in prod/small-models.yaml: two TP2 replicas, validated by validate_ds4f_migration.rb."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SMALL = Path("prod/small-models.yaml")
VALIDATOR = Path("scripts/validate_ds4f_migration.rb")
A = "model-sg-glm53-w4afp8-tp2-r1a"
B = "model-sg-glm53-w4afp8-tp2-r1b"


class Gpu13Tp2Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        shutil.copytree(ROOT / "prod", self.root / "prod")
        (self.root / "scripts").mkdir()
        shutil.copyfile(ROOT / VALIDATOR, self.root / VALIDATOR)
        self.valid = (self.root / SMALL).read_text()

    def run_validator(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(["ruby", str(self.root / VALIDATOR)], capture_output=True, text=True, cwd=self.root, check=False)

    def assert_fails(self, mutated: str, message: str) -> None:
        self.assertNotEqual(mutated, self.valid)
        (self.root / SMALL).write_text(mutated)
        try:
            result = self.run_validator()
            output = result.stdout + result.stderr
            self.assertNotEqual(result.returncode, 0, output)
            self.assertIn(message, output)
        finally:
            (self.root / SMALL).write_text(self.valid)

    def block(self, name: str) -> tuple[int, int]:
        start = self.valid.index(f"  {name}:\n")
        return start, self.valid.index("\n  # ---", start)

    def test_committed_file_passes(self) -> None:
        result = self.run_validator()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_tp4_service_is_gone_and_pool_has_both_replicas(self) -> None:
        services = self.valid[self.valid.index("\nservices:\n"):]
        self.assertNotIn("\n  model-sg-glm53-fp8-tp4:\n", services)
        self.assertIn(f"VLLM_BACKEND_URLS=http://{A}:8000,http://{B}:8000", self.valid)

    def test_rejects_drift(self) -> None:
        start_a, end_a = self.block(A)
        start_b, end_b = self.block(B)
        b_block = self.valid[start_b:end_b]
        cases = (
            (self.valid.replace("      --prefill-decode-interval 2\n", "      --prefill-decode-interval 2\n      --disable-overlap-schedule\n", 1), "must not disable the overlap scheduler"),
            (self.valid.replace("      --speculative-num-draft-tokens 5\n", "      --speculative-num-draft-tokens 5\n      --speculative-adaptive\n", 1), "must use fixed EAGLE"),
            (self.valid.replace("--mem-fraction-static 0.86", "--mem-fraction-static 0.80", 1), "runtime flag changed"),
            (self.valid.replace("--max-running-requests 16", "--max-running-requests 24", 1), "runtime flag changed"),
            (self.valid.replace("--max-queued-requests 4", "--max-queued-requests 8", 1), "runtime flag changed"),
            (self.valid.replace("--cuda-graph-max-bs-decode 16", "--cuda-graph-max-bs-decode 32", 1), "runtime flag changed"),
            (self.valid.replace("--max-mamba-cache-size 330", "--max-mamba-cache-size 165", 1), "runtime flag changed"),
            (self.valid.replace("--tp-size 2", "--tp-size 4", 1), "runtime flag changed"),
            (self.valid.replace("${GLM53_R1A_HICACHE_RAM_BUDGET:-325GiB}", "${GLM53_HICACHE_RAM_BUDGET:-80%}", 1), "HiCache host-memory contract changed"),
            (self.valid.replace("    - SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=1\n", "    - SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=1\n    - SGLANG_CHUNKED_PREFILL_ADMISSION_RESERVE=4096\n", 1), "must not set SGLANG_CHUNKED_PREFILL_ADMISSION_RESERVE"),
            (self.valid[:start_b] + b_block.replace("127.0.0.1:29511", "127.0.0.1:29510") + self.valid[end_b:], "needs a unique --dist-init-addr"),
            (self.valid[:start_b] + b_block.replace('device_ids: ["6","7"]', 'device_ids: ["4","5"]') + self.valid[end_b:], "must use GPUs 6,7"),
            (self.valid[:start_b] + b_block.replace('nearai.otel.instance: "1b"', 'nearai.otel.instance: "x"') + self.valid[end_b:], "instance must be 1b"),
            (self.valid.replace("pdi2-gpu13", "pdi1-overlap-off-gpu13"), "config_variant"),
            (self.valid.replace(f"http://{B}:8000", "", 1).replace(f"{A}:8000,", f"{A}:8000", 1), "must pool both TP2 replicas"),
        )
        for index, (mutated, message) in enumerate(cases):
            with self.subTest(index=index, message=message):
                self.assert_fails(mutated, message)


if __name__ == "__main__":
    unittest.main()
