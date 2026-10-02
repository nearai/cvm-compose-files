#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: python3 -m unittest scripts.test_glm53_w4afp8_tp2x4 (needs ruby for the validator cases)
"""The generated 4x TP2 base-tier canary file, its generator and its validator contract."""

import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts import prepare_glm53_w4afp8_tp2x4 as generator

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / generator.TARGET
VALIDATOR = Path("scripts/validate_glm53_prod_config.rb")
DCGM_VALIDATOR = Path("scripts/validate_glm53_dcgm_metrics.rb")
CANONICAL = Path("prod/GLM-5.3-Flash-SGL-TP4.yaml")
NAMES = [f"model-sg-glm53-w4afp8-tp2-r{replica}" for replica in (1, 2, 3, 4)]


def replace_nth(text: str, needle: str, index: int, replacement: str) -> str:
    positions = [match.start() for match in re.finditer(re.escape(needle), text)]
    target = positions[index]
    return text[:target] + replacement + text[target + len(needle):]


class GeneratedFileTest(unittest.TestCase):
    def setUp(self) -> None:
        self.source = (ROOT / generator.SOURCE).read_text()
        self.target = TARGET.read_text()

    def test_committed_file_matches_generator(self) -> None:
        self.assertEqual(self.target, generator.generate(self.source))

    def test_check_mode_passes(self) -> None:
        result = subprocess.run(
            ["python3", str(ROOT / "scripts/prepare_glm53_w4afp8_tp2x4.py"), "--check"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_header_marks_the_file_generated_and_names_the_rollback(self) -> None:
        header = self.target[: self.target.index("x-logging-conf:")]
        self.assertIn("generated from\n# prod/GLM-5.3-Flash-SGL-TP4-W4AFP8.yaml", header)
        self.assertIn("Do not hand-edit this file.", header)
        self.assertIn("ROLLBACK: prod/GLM-5.3-Flash-SGL-TP4-W4AFP8.yaml", header)
        self.assertIn(generator.IMAGE, header)

    def test_scrape_deployment_and_topology_preserve_service_identity(self) -> None:
        self.assertNotIn("tp4-r", self.target)
        self.assertEqual(self.target.count('                      deployment: "glm53-flash-sgl-tp4"\n'), 6)
        self.assertEqual(self.target.count('                      deployment: "glm53-flash-sgl-tp4"\n                      topology: "tp2x4"\n'), 6)
        self.assertNotIn('nearai.otel.deployment: "glm53-flash-sgl-tp4"', self.target)
        self.assertNotIn('deployment:glm53-flash-sgl-tp4', self.target)
        self.assertEqual(self.target.count("deployment:glm53-flash-sgl-tp2x4"), 12)

    def test_four_replicas_one_per_nvlink_pair(self) -> None:
        for name, devices in zip(NAMES, ('["0","1"]', '["2","3"]', '["4","5"]', '["6","7"]')):
            with self.subTest(replica=name):
                _, _, block = generator.section(self.target, f"  {name}:\n", "    labels:\n", name)
                self.assertIn(f"    container_name: {name}\n", block)
                self.assertIn(f"device_ids: {devices}", block)
                self.assertIn(f"- job_name: sglang-{name}\n", self.target)
                self.assertIn(f"set $$backend http://{name}:8000;", self.target)

    def test_routing_and_downloader_blocks_are_byte_identical_to_the_source(self) -> None:
        for start, end in (("  model-downloader:\n", "\n  # Dormant during normal deploys."), ("  hf-cleanup:\n", "\n  nginx:\n")):
            with self.subTest(block=start.strip()):
                # Only the deployment label may differ: it splits the canary host on dashboards.
                self.assertEqual(
                    generator.section(self.target, start, end, start)[2].replace(generator.DEPLOYMENT, generator.SOURCE_DEPLOYMENT),
                    generator.section(self.source, start, end, start)[2],
                )
        self.assertEqual(self.target[self.target.index("  nginx_conf:\n"):], self.source[self.source.index("  nginx_conf:\n"):])
        registrar = generator.section(self.target, "  registrar_script:\n", "  nginx_conf:\n", "registrar")[2]
        source_registrar = generator.section(self.source, "  registrar_script:\n", "  nginx_conf:\n", "registrar")[2]
        self.assertEqual(registrar.replace("(4x TP2/EP2)", "(2x TP4/EP4)"), source_registrar)

    def test_generator_refuses_its_own_output_and_a_drifted_source(self) -> None:
        with self.assertRaises(generator.GenerationError):
            generator.generate(self.target)
        for before in (
            "\n      --chunked-prefill-size 4096\n",
            "\n    - SGLANG_CHUNKED_PREFILL_ADMISSION_RESERVE=4096\n",
            '\n      - "8009:8009"\n',
            "\n        for replica in (1, 2):\n",
        ):
            with self.subTest(removed=before.strip()[:40]):
                self.assertEqual(self.source.count(before), 1)
                with self.assertRaises(generator.GenerationError):
                    generator.generate(self.source.replace(before, "\n"))
        with self.assertRaises(generator.GenerationError):
            generator.generate(self.source.replace("\n    - SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=1\n",
                                                   "\n    - SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=1\n    - SGLANG_HICACHE_POOLED_TRANSFERS=1\n"))


class ValidatorContractTest(unittest.TestCase):
    """Each mutation of the committed file must fail the production validator with its reason."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name in (CANONICAL, generator.SOURCE, generator.TARGET, VALIDATOR, DCGM_VALIDATOR):
            (self.root / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, self.root / name)
        self.target = self.root / generator.TARGET
        self.valid = self.target.read_text()

    def run_ruby(self, script: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(["ruby", str(self.root / script)], capture_output=True, text=True, check=False)

    def assert_fails(self, mutated: str, message: str, script: Path = VALIDATOR) -> None:
        self.assertNotEqual(mutated, self.valid)
        self.target.write_text(mutated)
        try:
            result = self.run_ruby(script)
            output = result.stdout + result.stderr
            self.assertEqual(result.returncode, 1, output)
            self.assertIn(message, output)
            self.assertNotIn("Traceback", output)
            self.assertNotRegex(output, r"\.rb:\d+:in")
        finally:
            self.target.write_text(self.valid)

    def replace_once(self, before: str, after: str) -> str:
        self.assertEqual(self.valid.count(before), 1, before)
        return self.valid.replace(before, after)

    def test_committed_files_pass(self) -> None:
        for script in (VALIDATOR, DCGM_VALIDATOR):
            with self.subTest(script=str(script)):
                result = self.run_ruby(script)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("GLM-5.3-Flash-SGL-TP2x4-W4AFP8.yaml", result.stdout)

    def test_rejects_engine_argv_drift(self) -> None:
        argv = "argv must be the lab-qualified TP2 argv exactly"
        cases = (
            ("\n      --tp-size 2\n", "\n      --tp-size 4\n"),
            ("\n      --ep-size 2\n", "\n      --ep-size 4\n"),
            ("\n      --chunked-prefill-size 8192\n", "\n      --chunked-prefill-size 4096\n"),
            ("\n      --hicache-write-policy write_through_selective\n", "\n      --hicache-write-policy write_through\n"),
            ("\n      --enable-hierarchical-cache\n", "\n"),
            ("\n      --max-running-requests 32\n", "\n      --max-running-requests 15\n"),
        )
        for before, after in cases:
            with self.subTest(mutation=after.strip()[-50:] or before.strip()):
                self.assert_fails(self.replace_once(before, after), argv)

    def test_rejects_engine_image_and_environment_drift(self) -> None:
        environment = "environment must be the W4AFP8 base engine environment plus"
        cases = (
            (f"\n  image: {generator.IMAGE}\n", f"\n  image: {generator.SOURCE_IMAGE}\n", "image must be"),
            ("    - SGLANG_CHUNKED_PREFILL_ADMISSION_RESERVE=4096\n", "", "must set SGLANG_CHUNKED_PREFILL_ADMISSION_RESERVE=4096"),
            ("${GLM53_HICACHE_RAM_BUDGET:-325GiB}", "${GLM53_HICACHE_RAM_BUDGET:-406GiB}", environment),
            ("    - SGLANG_DSA_INDEXER_QSPLIT=1\n", "", environment),
            ("${GLM53_HICACHE_CUDA_HOST_MEMORY:-1}", "${GLM53_HICACHE_CUDA_HOST_MEMORY:-0}", environment),
            ('device_ids: ["6","7"]', 'device_ids: ["4","5"]', "must use GPU device_ids 6,7"),
            (
                f"    container_name: {NAMES[3]}\n",
                f'    container_name: {NAMES[3]}\n    restart: "no"\n',
                "replicas must use identical runtime configuration",
            ),
        )
        for before, after, message in cases:
            with self.subTest(mutation=message):
                self.assert_fails(self.replace_once(before, after), message)

    def test_rejects_fan_out_drift(self) -> None:
        backends = ",".join(f"http://{name}:8000" for name in NAMES)
        cases = (
            (f"VLLM_BACKEND_URLS={backends}", f"VLLM_BACKEND_URLS={backends.rsplit(',', 1)[0]}", "must pool all four TP2 replicas"),
            ("for replica in (1, 2, 3, 4):", "for replica in (1, 2):", "glm53-perception-check must check replicas 1-4"),
            ('      - "8011:8011"\n', "", "glm53-soak-relay must publish"),
            (f"set $$backend http://{NAMES[3]}:8000;", f"set $$backend http://{NAMES[2]}:8000;", "glm53-soak-relay must map"),
            ("proxy_next_upstream off;", "proxy_next_upstream error;", "glm53-soak-relay config must match the W4AFP8 base file"),
        )
        for before, after, message in cases:
            with self.subTest(mutation=message):
                self.assert_fails(self.replace_once(before, after), message)

    def test_rejects_routing_drift_from_the_base_file(self) -> None:
        outside = "must match the W4AFP8 base file outside the engines"
        cases = (
            ('register_model "z-ai/glm-5.3-flash" "glm-5-3-flash.completions.near.ai"',
             'register_model "z-ai/glm-5.3-flash" "glm-5-3-flash-long.completions.near.ai"'),
            ("server_name glm-5-3-flash.completions.near.ai", "server_name glm-5-3-flash-long.completions.near.ai"),
            ("      - VLLM_BACKEND_CONVERSATION_AFFINITY=1\n", "      - VLLM_BACKEND_CONVERSATION_AFFINITY=1\n      - EXTRA=1\n"),
        )
        for before, after in cases:
            with self.subTest(mutation=after[:60]):
                self.assert_fails(self.replace_once(before, after), outside)

    def test_rejects_untruthful_telemetry(self) -> None:
        cases = (
            (f'nearai.otel.config_variant: "{generator.VARIANT}"', 'nearai.otel.config_variant: "incorrect-variant"', 3,
             "nearai.otel.config_variant must be"),
            (f"config_variant:{generator.VARIANT}", "config_variant:incorrect-variant", 2, "log metadata must carry exactly config_variant:"),
            ('      nearai.otel.engine_image: "47aff7910900"\n', '      nearai.otel.engine_image: "8bce6a7cc872"\n', 1, "nearai.otel.engine_image must be"),
            ('      nearai.otel.instance: "4"\n', '      nearai.otel.instance: "2"\n', 0, "nearai.otel.instance must be"),
            ('                      instance: "3"\n', '                      instance: "1"\n', 0, "scrape label instance"),
            ('                      engine_image: "47aff7910900"\n', '                      engine_image: "8bce6a7cc872"\n', 3, "scrape label engine_image"),
            ('nearai.otel.deployment: "glm53-flash-sgl-tp2x4"', 'nearai.otel.deployment: "glm53-flash-sgl-tp4"', 2,
             "must not carry the glm53-flash-sgl-tp4 deployment label"),
            ('"deployment:glm53-flash-sgl-tp2x4"', '"deployment:glm53-flash-sgl-tp4"', 1,
             "must not carry the glm53-flash-sgl-tp4 deployment label"),
            ('                      deployment: "glm53-flash-sgl-tp4"\n',
             '                      deployment: "glm53-flash-sgl-tp2x4"\n', 0, "scrape label deployment"),
            ('                      deployment: "glm53-flash-sgl-tp4"\n',
             '                      deployment: "glm53-flash-sgl-tp2x4"\n', 4, "dcgm-dcgm-glm53 scrape label deployment"),
            ('                      deployment: "glm53-flash-sgl-tp4"\n',
             '                      deployment: "glm53-flash-sgl-tp2x4"\n', 5, "inference-proxy-proxy-glm53 scrape label deployment"),
            ('                      topology: "tp2x4"\n', '', 0, "scrape label topology"),
            ('                      topology: "tp2x4"\n', '', 4, "dcgm-dcgm-glm53 scrape label topology"),
            ('                      topology: "tp2x4"\n', '', 5, "inference-proxy-proxy-glm53 scrape label topology"),
        )
        for needle, replacement, index, message in cases:
            with self.subTest(mutation=message):
                self.assert_fails(replace_nth(self.valid, needle, index, replacement), message)

    def test_rejects_a_surviving_tp4_reference(self) -> None:
        mutated = self.replace_once("    # Local snapshot path; nothing is fetched at engine start.\n",
                                    "    # model-sg-glm53-w4afp8-tp4-r1\n    # Local snapshot path; nothing is fetched at engine start.\n")
        self.assert_fails(mutated, "must not reference any TP4 engine")

    def test_dcgm_validator_covers_the_file(self) -> None:
        needle = "image: nvcr.io/nvidia/k8s/dcgm-exporter@sha256:613ab03c11d442fd960ff515f547e9921537454a712d08160bc8f677f89f1c35"
        mutated = self.replace_once(needle, "image: nvcr.io/nvidia/k8s/dcgm-exporter@sha256:" + "0" * 64)
        self.assert_fails(mutated, "GLM-5.3-Flash-SGL-TP2x4-W4AFP8.yaml", DCGM_VALIDATOR)


if __name__ == "__main__":
    _ = unittest.main()
