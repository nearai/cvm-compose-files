#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: uv run -m scripts.test_glm53_w4afp8_canary
import hashlib
import shlex
import subprocess
import unittest
from pathlib import Path
from typing import Final

from scripts import prepare_glm53_w4afp8_canary as canary

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "prod/GLM-5.3-Flash-SGL-TP4-W4AFP8-Canary.yaml"
GENERATOR = ROOT / "scripts/prepare_glm53_w4afp8_canary.py"
PROMOTED_IMAGE: Final = "docker.io/nearaidev/sglang@sha256:8bce6a7cc872a80faded3bd1ef0a64873a1d7abae34c94e5358775ca21f133cc"
LONG_CONTEXT_SHA256: Final = "b020c87160fd3ea0512017519fc0e1207f2ab7f580e02defa68be6048a377001"


class CommittedCanaryTest(unittest.TestCase):
    def test_generated_canary_exists(self) -> None:
        # Given the repository's production compose tree
        # When the W4AFP8 canary artifacts are inspected
        # Then a generated candidate artifact must be present.
        self.assertTrue(CANDIDATE.is_file(), CANDIDATE)

    def test_committed_canary_matches_generator(self) -> None:
        # Given the canonical compose file
        canonical = (ROOT / canary.COMPOSE).read_text()
        # When the candidate is regenerated in memory
        expected = canary.generate(canonical)
        # Then the committed generated file must be byte-identical.
        self.assertEqual(CANDIDATE.read_text(), expected)

    def test_candidate_has_exact_promoted_image(self) -> None:
        # Given the generated gpu02 long-context compose file
        text = CANDIDATE.read_text()
        _, _, service = canary.section(
            text,
            f"  {canary.CANDIDATE_SERVICE}:\n",
            "\n  model-sg-glm53-fp8-tp4-r2:\n",
            "candidate service",
        )
        # When the candidate r1 image override is inspected
        image_lines = [line.strip() for line in service.splitlines() if line.strip().startswith("image:")]
        # Then it is exactly the immutable signed combined-image digest, never a mutable tag.
        self.assertEqual(image_lines, [f"image: {PROMOTED_IMAGE}"])

    def test_candidate_has_exact_treatment_contract(self) -> None:
        # Given the generated gpu02 long-context compose file
        text = CANDIDATE.read_text()
        _, _, service = canary.section(
            text,
            f"  {canary.CANDIDATE_SERVICE}:\n",
            "\n  model-sg-glm53-fp8-tp4-r2:\n",
            "candidate service",
        )
        _, _, common = canary.section(
            text,
            "x-sg-glm53-flash-common: &sg-glm53-flash-common\n",
            "\nx-dcgm-common: &dcgm-common\n",
            "shared engine anchor",
        )
        # When the r1 treatment is inspected
        required = (
            f"--model-path /root/.cache/huggingface/hub/models--graphistry--GLM-5.3-Flash-W4AFP8/snapshots/{canary.CHECKPOINT_REVISION}",
            "--served-model-name z-ai/glm-5.3-flash",
            "--chunked-prefill-size 4096",
            "--max-prefill-tokens 32768",
            "--prefill-decode-interval 1",
            "--tp-size 4",
            "--ep-size 4",
            "--kv-cache-dtype bfloat16",
            "--speculative-algorithm EAGLE",
            "--speculative-num-steps 5",
            "--speculative-eagle-topk 1",
            "--speculative-num-draft-tokens 6",
            "--speculative-adaptive",
        )
        # Then every treatment setting remains exact.
        for item in required:
            with self.subTest(item=item):
                self.assertIn(item, service)
        self.assertEqual(service.count(canary.CHECKPOINT_REVISION), 1)
        self.assertNotIn("--revision", service)
        self.assertNotIn("--moe-runner-backend", service)
        self.assertNotIn("SGLANG_CHUNKED_PREFILL_ADMISSION_RESERVE", common)
        self.assertNotIn("SGLANG_ADMISSION_RESERVE_MAX_FRACTION", common)

    def test_candidate_command_renders_exact_argument_vector(self) -> None:
        # Given the folded Compose command generated for candidate r1
        text = CANDIDATE.read_text()
        _, _, service = canary.section(
            text,
            f"  {canary.CANDIDATE_SERVICE}:\n",
            "\n  model-sg-glm53-fp8-tp4-r2:\n",
            "candidate service",
        )
        _, _, command = canary.section(service, "    command: >\n", "    depends_on:\n", "candidate command")
        rendered = " ".join(line.strip() for line in command.splitlines()[1:])
        # When the folded scalar is tokenized as the runtime command
        arguments = shlex.split(rendered)
        required_flags = ("--model-path", "--served-model-name", "--tp-size", "--ep-size", "--kv-cache-dtype")
        # Then the executable and every treatment option are exact argv entries.
        self.assertEqual(arguments[:2], ["sglang", "serve"])
        for flag in required_flags:
            with self.subTest(flag=flag):
                self.assertIn(flag, arguments)
        self.assertFalse(any(argument.startswith(" ") for argument in arguments), arguments)

    def test_candidate_has_no_runtime_loader_patch_bootstrap(self) -> None:
        # Given the generated candidate built from the promoted combined image
        text = CANDIDATE.read_text()
        forbidden = (
            "BLOCKED",
            "exit 78",
            "git apply",
            "glm53_w4afp8_patch",
            "modules-to-not-convert.diff",
            "diff --git a/python/sglang/srt/layers/quantization/w4afp8.py",
        )
        # When the generated artifact is scanned for the obsolete runtime patch path
        # Then no blocker, patch application, patch mount, or embedded diff remains.
        for marker in forbidden:
            with self.subTest(marker=marker):
                self.assertNotIn(marker, text)

    def test_candidate_wiring_is_isolated_to_long_context_r1(self) -> None:
        # Given the generated candidate and deployed long-context source
        text = CANDIDATE.read_text()
        long_context = (ROOT / canary.COMPOSE).read_text()
        # When service, proxy, verification, and telemetry references are counted
        # Then every r1 route names the candidate and r2 stays the FP8 HiCache arm.
        self.assertIn("  model-sg-glm53-fp8-tp4-r2:\n", text)
        self.assertEqual(text.count(canary.CANDIDATE_SERVICE), 9)
        self.assertNotIn(canary.CONTROL_SERVICE, text)
        self.assertIn(canary.CONTROL_SERVICE, long_context)
        expected_perception_loop = (
            f'for replica, service in ((1, "{canary.CANDIDATE_SERVICE}"), '
            '(2, "model-sg-glm53-fp8-tp4-r2")):'
        )
        self.assertIn(
            expected_perception_loop,
            text,
        )

    def test_long_context_r2_definition_is_unchanged_except_safety_profile(self) -> None:
        # Given the deployed long-context source and generated candidate
        source = (ROOT / canary.COMPOSE).read_text()
        candidate = CANDIDATE.read_text()
        _, _, source_r2 = canary.section(
            source,
            "  model-sg-glm53-fp8-tp4-r2:\n",
            "\n  # Explicit operator-only semantic check;",
            "source r2 service",
        )
        _, _, candidate_r2 = canary.section(
            candidate,
            "  model-sg-glm53-fp8-tp4-r2:\n",
            "\n  # Explicit operator-only semantic check;",
            "candidate r2 service",
        )
        profile = f'    profiles: ["{canary.CANARY_PROFILE}"]\n'
        # When the alternate file's default-apply guard is ignored
        # Then gpu02 r2 is byte-identical to the deployed HiCache arm.
        self.assertEqual(candidate_r2.replace(profile, "", 1), source_r2)

    def test_canonical_long_context_source_is_unchanged(self) -> None:
        # Given the deployed long-context source is the generator input
        source = (ROOT / canary.COMPOSE).read_bytes()
        # When its immutable baseline digest is calculated
        digest = hashlib.sha256(source).hexdigest()
        # Then promotion work has not changed the canonical production file.
        self.assertEqual(digest, LONG_CONTEXT_SHA256)

    def test_customer_routing_requires_explicit_canary_services(self) -> None:
        # Given an operator accidentally applies the canary file without a service list
        text = CANDIDATE.read_text()
        profiled_services = (
            "model-downloader",
            canary.CANDIDATE_SERVICE,
            "model-proxy-registrar",
            "proxy-glm53",
            "model-sg-glm53-fp8-tp4-r2",
            "dcgm-glm53",
            "otelcol-contrib",
            "nginx",
        )
        # When Compose evaluates the default profile
        # Then candidate, download, customer-routing, registration, and changed telemetry stay disabled.
        for service in profiled_services:
            with self.subTest(service=service):
                marker = f'  {service}:\n    profiles: ["{canary.CANARY_PROFILE}"]\n'
                self.assertIn(marker, text)
        self.assertIn(
            f'  {canary.CANDIDATE_SERVICE}:\n    profiles: ["{canary.CANARY_PROFILE}"]\n    restart: unless-stopped\n',
            text,
        )

    def test_check_mode_passes(self) -> None:
        # Given the committed generated canary
        # When the generator runs in drift-detection mode
        result = subprocess.run(
            ["python3", str(GENERATOR), "--check"],
            capture_output=True,
            check=False,
            text=True,
        )
        # Then it reports no drift.
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    _ = unittest.main()
