#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: uv run -m scripts.test_glm53_w4afp8_canary
import hashlib
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts import prepare_glm53_w4afp8_canary as canary

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "prod/GLM-5.3-Flash-SGL-TP4-W4AFP8-Canary.yaml"
GENERATOR = ROOT / "scripts/prepare_glm53_w4afp8_canary.py"


class CommittedCanaryTest(unittest.TestCase):
    def test_generated_canary_exists(self) -> None:
        # Given the repository's production compose tree
        # When the W4AFP8 canary artifacts are inspected
        # Then a generated candidate artifact must be present.
        self.assertTrue(CANDIDATE.is_file(), CANDIDATE)

    def test_committed_canary_matches_generator(self) -> None:
        # Given the canonical compose file and reviewed loader patch
        canonical = (ROOT / canary.COMPOSE).read_text()
        patch = (ROOT / canary.PATCH).read_text()
        # When the candidate is regenerated in memory
        expected = canary.generate(canonical, patch)
        # Then the committed generated file must be byte-identical.
        self.assertEqual(CANDIDATE.read_text(), expected)

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
            "--chunked-prefill-size 16384",
            "--max-prefill-tokens 32768",
            "--prefill-decode-interval 1",
            canary.BASE_SOURCE_SHA256,
            canary.PATCH_SHA256,
            canary.PATCHED_SOURCE_SHA256,
        )
        # Then every replicated setting and fail-closed checksum must be present.
        for item in required:
            with self.subTest(item=item):
                self.assertIn(item, service)
        self.assertNotIn("--revision", service)
        self.assertNotIn("--moe-runner-backend", service)
        self.assertNotIn("SGLANG_CHUNKED_PREFILL_ADMISSION_RESERVE", common)
        self.assertNotIn("SGLANG_ADMISSION_RESERVE_MAX_FRACTION", common)

    def test_candidate_refuses_to_start_when_pool_clamp_image_is_not_pinned(self) -> None:
        # Given the generated canary still inherits the base engine image
        text = CANDIDATE.read_text()
        _, _, service = canary.section(
            text,
            f"  {canary.CANDIDATE_SERVICE}:\n",
            "\n  model-sg-glm53-fp8-tp4-r2:\n",
            "candidate service",
        )
        # When an operator explicitly targets the otherwise opt-in candidate service
        blocker = "exit 78"
        # Then startup must stop before SGLang can execute.
        self.assertIn(
            "BLOCKED: this file still inherits the base engine image",
            text,
        )
        self.assertIn(blocker, service)
        self.assertLess(service.index(blocker), service.index("exec sglang serve"))

    def test_loader_patch_bootstrap_is_restart_safe_and_fail_closed(self) -> None:
        # Given the exact bootstrap template and a disposable source/patch pair
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.txt"
            patch = root / "change.diff"
            git = root / "git"
            sha256sum = root / "sha256sum"
            _ = source.write_text("before\n")
            _ = patch.write_text("""--- a/source.txt
+++ b/source.txt
@@ -1 +1 @@
-before
+after
""")
            _ = git.write_text("""#!/bin/sh
set -eu
test "$1" = "apply"
if [ "${2:-}" = "--check" ]; then
  test "$(cat source.txt)" = "before"
else
  printf 'after\\n' > source.txt
fi
""")
            _ = git.chmod(0o755)
            _ = sha256sum.write_text("""#!/usr/bin/env python3
import hashlib, pathlib, sys
if '-c' in sys.argv or '--check' in sys.argv:
    expected, path = sys.stdin.read().strip().split(None, 1)
    actual = hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
    print(f'{path}: {"OK" if actual == expected else "FAILED"}')
    raise SystemExit(0 if actual == expected else 1)
for path in sys.argv[1:]:
    digest = hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
    print(f'{digest}  {path}')
""")
            _ = sha256sum.chmod(0o755)

            def digest(path: Path) -> str:
                return hashlib.sha256(path.read_bytes()).hexdigest()

            base_sha256 = digest(source)
            patch_sha256 = digest(patch)
            patched_sha256 = hashlib.sha256(b"after\n").hexdigest()
            script = "\n".join(
                line.strip()
                for line in canary.patch_bootstrap(
                    patch_target=patch.name,
                    source_target=source.name,
                    patch_sha256=patch_sha256,
                    base_sha256=base_sha256,
                    patched_sha256=patched_sha256,
                )
            ).replace("$$", "$")
            environment = {**os.environ, "PATH": f"{root}{os.pathsep}{os.environ['PATH']}"}
            # When the same writable layer starts from stock and then restarts
            first = subprocess.run(
                ["bash", "-c", script],
                cwd=root,
                env=environment,
                capture_output=True,
                check=False,
                text=True,
            )
            second = subprocess.run(
                ["bash", "-c", script],
                cwd=root,
                env=environment,
                capture_output=True,
                check=False,
                text=True,
            )
            _ = source.write_text("unexpected\n")
            drifted = subprocess.run(
                ["bash", "-c", script],
                cwd=root,
                env=environment,
                capture_output=True,
                check=False,
                text=True,
            )
        # Then stock applies, restart succeeds without reapplying, and drift fails closed.
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
        self.assertIn("already applied", second.stdout)
        self.assertNotEqual(drifted.returncode, 0)
        self.assertIn("Unexpected SHA256", drifted.stderr)

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
            f'  {canary.CANDIDATE_SERVICE}:\n    profiles: ["{canary.CANARY_PROFILE}"]\n    restart: "no"\n',
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
