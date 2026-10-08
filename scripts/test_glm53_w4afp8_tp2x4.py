#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: python3 -m unittest scripts.test_glm53_w4afp8_tp2x4 (needs ruby for the validator cases)
"""The generated v7 fleet base-tier file (4x TP2), its generator and its validator contract."""

import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts import prepare_glm53_w4afp8_tp2x4 as generator

from scripts.test_glm53_v7_fleet import FleetFilesTest, FleetRunbookTest, FleetValidatorContractTest, Gpu13FleetTest  # noqa: F401

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

    def test_no_tp4_engine_or_deployment_reference_survives(self) -> None:
        self.assertNotIn("tp4-r", self.target)
        self.assertNotIn('glm53-flash-sgl-tp4"', self.target)
        # 12 inherited log tags plus the ghost aggregator's.
        self.assertEqual(self.target.count("deployment:glm53-flash-sgl-tp2x4"), 13)

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

    def test_candidate_argv_is_the_control_argv_with_exactly_the_v7_fleet_edits(self) -> None:
        control = [line.strip() for line in generator.section(self.target, "x-sg-glm53-flash-common:", "\nx-sg-glm53-flash-candidate", "c")[2].split("command: >\n")[1].split("  volumes:")[0].splitlines() if line.strip()]
        candidate = [line.strip() for line in generator.section(self.target, "x-sg-glm53-flash-candidate:", "\nx-dcgm-common", "c")[2].split("command: >\n")[1].splitlines() if line.strip()]
        removed = sorted(set(control) - set(candidate))
        added = sorted(set(candidate) - set(control))
        self.assertEqual(
            removed,
            sorted(["--mem-fraction-static 0.80", "--max-running-requests 32", "--cuda-graph-max-bs-decode 32", "--speculative-num-steps 5",
                    "--speculative-num-draft-tokens 6", "--speculative-adaptive", "--max-mamba-cache-size 165", "--prefill-decode-interval 1",
                    "--dsa-prefill-backend tilelang", "--dsa-decode-backend tilelang", "--kv-cache-dtype bfloat16",
                    "--enable-hierarchical-cache", "--hicache-write-policy write_through_selective", "--hicache-io-backend direct",
                    "--hicache-mem-layout page_first_direct", "--max-queued-requests 8"]),
        )
        self.assertEqual(
            added,
            sorted(["--mem-fraction-static 0.86", "--max-running-requests 64", "--cuda-graph-max-bs-decode 64", "--speculative-num-steps 4",
                    "--speculative-num-draft-tokens 5", "--max-mamba-cache-size 380", "--prefill-decode-interval 2",
                    "--dsa-prefill-backend flashmla_kv", "--dsa-decode-backend flashmla_kv", "--kv-cache-dtype fp8_e4m3",
                    "--disable-overlap-schedule", "--max-queued-requests 32"]),
        )
        # -1 adaptive, -4 HiCache flags, +1 overlap-off.
        self.assertEqual(len(control) - 4, len(candidate))
        self.assertEqual(candidate[-2:], ["--mamba-ssm-dtype bfloat16", "--disable-overlap-schedule"])

    def test_all_replicas_run_the_candidate_and_differ_only_in_identity(self) -> None:
        def block(name: str) -> str:
            # r2-r4 repeat the anchor environment with their own ghost replica name (asserted below);
            # outside that list every replica is identical.
            text = generator.section(self.target, f"  {name}:\n", "    labels:\n", name)[2]
            if "    environment:\n" in text:
                start = text.index("    environment:\n")
                end = text.index("    depends_on:\n", start)
                environment = text[start:end]
                replica = f"r{NAMES.index(name) + 1}"
                self.assertEqual(environment.count(f"\n      - SGLANG_GHOST_CACHE_REPLICA={replica}\n"), 1, name)
                text = text[:start] + text[end:]
            return text
        first_block = block(NAMES[0]).replace(NAMES[0], "NAME").replace('["0","1"]', "DEV")
        self.assertIn("    <<: *sg-glm53-flash-candidate\n", first_block)
        for name, devices in zip(NAMES[1:], ('["2","3"]', '["4","5"]', '["6","7"]')):
            self.assertEqual(block(name).replace(name, "NAME").replace(devices, "DEV"), first_block)
        self.assertEqual(self.target.count("    <<: *sg-glm53-flash-common\n"), 0)
        self.assertEqual(self.target.count(generator.CANDIDATE_VARIANT), 12)
        self.assertEqual(self.target.count(generator.VARIANT), 0)

    def test_candidate_flag_values_are_pinned_literally(self) -> None:
        # Independent of the generator's edit list: a shared typo in the generator and validator must still fail here.
        _, _, anchor = generator.section(self.target, "x-sg-glm53-flash-candidate:", "\nx-dcgm-common", "candidate")
        for flag in ("--mem-fraction-static 0.86", "--max-running-requests 64", "--cuda-graph-max-bs-decode 64", "--speculative-num-steps 4",
                     "--speculative-eagle-topk 1", "--speculative-num-draft-tokens 5", "--max-mamba-cache-size 380", "--max-queued-requests 32",
                     "--chunked-prefill-size 8192", "--context-length 1048576", "--kv-cache-dtype fp8_e4m3", "--dsa-prefill-backend flashmla_kv",
                     "--dsa-decode-backend flashmla_kv", "--disable-overlap-schedule", "--mamba-ssm-dtype bfloat16"):
            self.assertEqual(anchor.count(f"      {flag}\n"), 1, flag)
        self.assertNotIn("--speculative-adaptive", anchor)
        # HiCache OFF exactly as gpu03 r3 runs it: no HiCache flag in the argv.
        self.assertFalse([t for t in anchor.split() if t.startswith("--hicache")])
        self.assertNotIn("--enable-hierarchical-cache", anchor)
        self.assertEqual(anchor.count("      --prefill-decode-interval 2\n"), 1)

    def test_candidate_derivation_refuses_a_drifted_control_argv(self) -> None:
        control = ["sglang serve", "--mem-fraction-static 0.80"]
        with self.assertRaises(generator.GenerationError):
            generator.candidate_arguments(control)

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


class AnchorUsageTest(unittest.TestCase):
    """The runbook for this file is docs/glm53-v7-fleet-rollout.md (tested in scripts/test_glm53_v7_fleet.py)."""

    def test_only_candidate_replicas_use_the_candidate_anchor(self) -> None:
        target = TARGET.read_text()
        self.assertEqual(target.count("<<: *sg-glm53-flash-candidate"), len(generator.CANDIDATE_REPLICAS))
        self.assertEqual(target.count("    <<: *sg-glm53-flash-common"), len(generator.REPLICAS) - len(generator.CANDIDATE_REPLICAS))


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

    def replace_in_anchor(self, anchor: str, before: str, after: str) -> str:
        """Replace `before` inside one engine anchor (the control or the candidate), once."""
        start = self.valid.index(f"{anchor}: &")
        end = self.valid.index("\nx-", start + 1)
        block = self.valid[start:end]
        self.assertEqual(block.count(before), 1, before)
        return self.valid[:start] + block.replace(before, after) + self.valid[end:]

    def test_committed_files_pass(self) -> None:
        for script in (VALIDATOR, DCGM_VALIDATOR):
            with self.subTest(script=str(script)):
                result = self.run_ruby(script)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("GLM-5.3-Flash-SGL-TP2x4-W4AFP8.yaml", result.stdout)

    def test_rejects_engine_argv_drift(self) -> None:
        argv = "argv must be the lab-qualified TP2 control argv exactly"
        control = "x-sg-glm53-flash-common"
        cases = (
            ("\n      --tp-size 2\n", "\n      --tp-size 4\n"),
            ("\n      --ep-size 2\n", "\n      --ep-size 4\n"),
            ("\n      --chunked-prefill-size 8192\n", "\n      --chunked-prefill-size 4096\n"),
            ("\n      --hicache-write-policy write_through_selective\n", "\n      --hicache-write-policy write_through\n"),
            ("\n      --enable-hierarchical-cache\n", "\n"),
            ("\n      --max-running-requests 32\n", "\n      --max-running-requests 15\n"),
            # A candidate flag leaking into the control replicas.
            ("\n      --mem-fraction-static 0.80\n", "\n      --mem-fraction-static 0.86\n"),
            ("\n      --speculative-adaptive\n", "\n"),
            ("\n      --max-mamba-cache-size 165\n", "\n      --max-mamba-cache-size 330\n"),
        )
        for before, after in cases:
            with self.subTest(mutation=after.strip()[-50:] or before.strip()):
                self.assert_fails(self.replace_in_anchor(control, before, after), argv)

    def test_rejects_candidate_argv_drift(self) -> None:
        argv = "argv must be the v7 fleet argv exactly"
        candidate = "x-sg-glm53-flash-candidate"
        cases = (
            ("\n      --mem-fraction-static 0.86\n", "\n      --mem-fraction-static 0.80\n"),
            ("\n      --mem-fraction-static 0.86\n", "\n      --mem-fraction-static 0.88\n"),
            ("\n      --max-running-requests 64\n", "\n      --max-running-requests 48\n"),
            ("\n      --prefill-decode-interval 2\n", "\n      --prefill-decode-interval 1\n"),
            ("\n      --speculative-num-steps 4\n", "\n      --speculative-num-steps 5\n"),
            ("\n      --speculative-num-draft-tokens 5\n", "\n      --speculative-num-draft-tokens 6\n"),
            ("\n      --speculative-eagle-topk 1\n", "\n      --speculative-eagle-topk 1\n      --speculative-adaptive\n"),
            ("\n      --chunked-prefill-size 8192\n", "\n      --chunked-prefill-size 16384\n"),
            # HiCache must stay OFF, FP8 KV needs its pairing, and the overlap scheduler stays off.
            ("\n      --mamba-ssm-dtype bfloat16\n", "\n      --mamba-ssm-dtype bfloat16\n      --enable-hierarchical-cache\n"),
            ("\n      --kv-cache-dtype fp8_e4m3\n", "\n      --kv-cache-dtype bfloat16\n"),
            ("\n      --dsa-decode-backend flashmla_kv\n", "\n      --dsa-decode-backend tilelang\n"),
            ("\n      --dsa-prefill-backend flashmla_kv\n", "\n      --dsa-prefill-backend tilelang\n"),
            ("\n      --disable-overlap-schedule\n", "\n"),
            ("\n      --max-queued-requests 32\n", "\n      --max-queued-requests 16\n"),
            ("\n      --context-length 1048576\n", "\n      --context-length 524288\n"),
        )
        for before, after in cases:
            with self.subTest(mutation=after.strip()[-50:]):
                self.assert_fails(self.replace_in_anchor(candidate, before, after), argv)

    def test_rejects_candidate_capacity_invariant_violations(self) -> None:
        candidate = "x-sg-glm53-flash-candidate"
        # Too few mamba slots for the running cap (5 per request) and a stale decode graph size.
        self.assert_fails(
            self.replace_in_anchor(candidate, "\n      --max-mamba-cache-size 380\n", "\n      --max-mamba-cache-size 165\n"),
            "cannot hold 64 running requests",
        )
        self.assert_fails(
            self.replace_in_anchor(candidate, "\n      --cuda-graph-max-bs-decode 64\n", "\n      --cuda-graph-max-bs-decode 32\n"),
            "--cuda-graph-max-bs-decode must equal --max-running-requests (64)",
        )

    def test_rejects_a_missing_capacity_flag(self) -> None:
        self.assert_fails(
            self.replace_in_anchor("x-sg-glm53-flash-candidate", "\n      --max-mamba-cache-size 380\n", "\n"),
            "must set --max-mamba-cache-size",
        )

    def test_rejects_candidate_args_with_the_control_variant_label(self) -> None:
        mutated = replace_nth(self.valid, f'nearai.otel.config_variant: "{generator.CANDIDATE_VARIANT}"', 0,
                              f'nearai.otel.config_variant: "{generator.VARIANT}"')
        self.assert_fails(mutated, "nearai.otel.config_variant must be")

    def test_rejects_candidate_environment_and_roles_drift(self) -> None:
        # The candidate inherits the control environment unchanged: an override on one candidate replica fails.
        # r3 carries its own copy of the environment (it needs its own ghost replica name); the
        # needle with six spaces only matches r2-r4's lists, so occurrence 1 is r3's.
        self.assert_fails(
            replace_nth(self.valid, "      - SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=1\n", 1,
                        "      - SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0\n"),
            "environment must be the W4AFP8 base engine environment plus",
        )
        # Moving a replica back onto the previous (common) anchor changes its argv, which is flagged.
        self.assert_fails(
            self.replace_once(
                f"  {NAMES[1]}:\n    <<: *sg-glm53-flash-candidate\n", f"  {NAMES[1]}:\n    <<: *sg-glm53-flash-common\n"
            ),
            "argv must be the v7 fleet argv exactly",
        )

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
        # r2-r4 repeat the anchor environment with their own ghost replica name, so environment
        # needles appear four times; the first occurrence is always the shared anchor (r1).
        for before, after, message in cases:
            with self.subTest(mutation=message):
                self.assert_fails(replace_nth(self.valid, before, 0, after), message)

    def test_rejects_missing_or_inconsistent_observability(self) -> None:
        required = "(opt-in observability is required on every replica)"
        cases = (
            # SGLANG_GHOST_CACHE dropped from one replica only: r1 (the anchor) or r3 (its own list).
            ("\n    - SGLANG_GHOST_CACHE=1\n", "\n", 0, f"{NAMES[0]} must set SGLANG_GHOST_CACHE=1 {required}"),
            ("\n      - SGLANG_GHOST_CACHE=1\n", "\n", 1, f"{NAMES[2]} must set SGLANG_GHOST_CACHE=1 {required}"),
            ("\n      - SGLANG_KV_TIER_METRICS=1\n", "\n", 2, f"{NAMES[3]} must set SGLANG_KV_TIER_METRICS=1 {required}"),
            ("SGLANG_GHOST_CACHE_REPLICA=r3\n", "SGLANG_GHOST_CACHE_REPLICA=r2\n", 0, "must use distinct SGLANG_GHOST_CACHE_REPLICA names"),
            ("SGLANG_GHOST_CACHE_SOCKET=/ghost/aggregator.sock\n", "SGLANG_GHOST_CACHE_SOCKET=/tmp/aggregator.sock\n", 3,
             "replicas must share one SGLANG_GHOST_CACHE_SOCKET"),
            ("SGLANG_GHOST_CACHE_KEY_FILE=/ghost/key\n", "SGLANG_GHOST_CACHE_KEY_FILE=/ghost/key2\n", 1,
             "replicas must share one SGLANG_GHOST_CACHE_KEY_FILE"),
            ("\n    - ghost:/ghost\n", "\n", 0, "must mount ghost:/ghost"),
            ('"--socket", "/ghost/aggregator.sock"', '"--socket", "/ghost/other.sock"', 0, "glm53-ghost-aggregator must run python3 -m"),
            ('"--sample", "16"', '"--sample", "1"', 0, "glm53-ghost-aggregator must run python3 -m"),
            (f"    image: {generator.IMAGE}\n    container_name: glm53-ghost-aggregator\n",
             f"    image: {generator.SOURCE_IMAGE}\n    container_name: glm53-ghost-aggregator\n", 0,
             "glm53-ghost-aggregator image must be the engines' image"),
            ("    runtime: runc\n    init: true\n", "    init: true\n", 0, "glm53-ghost-aggregator must run under runc"),
            ("      type: tmpfs\n", "      type: none\n", 0, "must declare the ghost volume as tmpfs"),
            ("['glm53-ghost-aggregator:9464']", "['glm53-ghost-aggregator:9465']", 0, "must scrape glm53-ghost-aggregator:9464"),
            ("              - job_name: ghost-aggregator-glm53-ghost-aggregator\n", "              - job_name: ghost-aggregator\n", 0,
             "missing ghost-aggregator-glm53-ghost-aggregator scrape job"),
            (f'nearai.otel.config_variant: "{generator.CANDIDATE_VARIANT}"', f'nearai.otel.config_variant: "{generator.CANDIDATE_VARIANT.replace("-obs-v1-v7", "-v7")}"', 0,
             "nearai.otel.config_variant must be"),
        )
        for before, after, index, message in cases:
            with self.subTest(mutation=message, occurrence=index):
                self.assert_fails(replace_nth(self.valid, before, index, after), message)

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
        candidate_variant = generator.CANDIDATE_VARIANT
        cases = (
            # Every replica must carry the candidate variant everywhere (log tag, metric label, scrape job).
            (f'nearai.otel.config_variant: "{candidate_variant}"', f'nearai.otel.config_variant: "{generator.VARIANT}"', 1,
             "nearai.otel.config_variant must be"),
            (f"config_variant:{candidate_variant}", f"config_variant:{generator.VARIANT}", 0, "log metadata must carry exactly config_variant:"),
            (f'                      config_variant: "{candidate_variant}"', f'                      config_variant: "{generator.VARIANT}"', 1,
             "scrape label config_variant must be"),
            (f'nearai.otel.config_variant: "{candidate_variant}"', 'nearai.otel.config_variant: "incorrect-variant"', 3,
             "nearai.otel.config_variant must be"),
            (f"config_variant:{candidate_variant}", "config_variant:incorrect-variant", 3, "log metadata must carry exactly config_variant:"),
            (f'      nearai.otel.engine_image: "{generator.ENGINE_IMAGE_LABEL}"\n', '      nearai.otel.engine_image: "8bce6a7cc872"\n', 1, "nearai.otel.engine_image must be"),
            (f'      nearai.otel.engine_image: "{generator.ENGINE_IMAGE_LABEL}"\n', f'      nearai.otel.engine_image: "{generator.V6_IMAGE.split(":")[-1][:12]}"\n', 2, "nearai.otel.engine_image must be"),
            ('"precision:int4-weights-fp8-activations-fp8-kv"', '"precision:int4-weights-fp8-activations-bf16-kv"', 1, "log metadata must carry precision:int4-weights-fp8-activations-fp8-kv"),
            ('                      precision: "int4-weights-fp8-activations-fp8-kv"\n', '                      precision: "int4-weights-fp8-activations-bf16-kv"\n', 2, "scrape label precision must be"),
            ('      nearai.otel.instance: "4"\n', '      nearai.otel.instance: "2"\n', 0, "nearai.otel.instance must be"),
            ('                      instance: "3"\n', '                      instance: "1"\n', 0, "scrape label instance"),
            (f'                      engine_image: "{generator.ENGINE_IMAGE_LABEL}"\n', '                      engine_image: "8bce6a7cc872"\n', 3, "scrape label engine_image"),
            ('nearai.otel.deployment: "glm53-flash-sgl-tp2x4"', 'nearai.otel.deployment: "glm53-flash-sgl-tp4"', 2,
             "must not carry the glm53-flash-sgl-tp4 deployment label"),
            ('"deployment:glm53-flash-sgl-tp2x4"', '"deployment:glm53-flash-sgl-tp4"', 1,
             "must not carry the glm53-flash-sgl-tp4 deployment label"),
        )
        for needle, replacement, index, message in cases:
            with self.subTest(mutation=message):
                self.assert_fails(replace_nth(self.valid, needle, index, replacement), message)

    def test_rejects_a_surviving_tp4_reference(self) -> None:
        mutated = replace_nth(self.valid, "    # Local snapshot path; nothing is fetched at engine start.\n", 0,
                              "    # model-sg-glm53-w4afp8-tp4-r1\n    # Local snapshot path; nothing is fetched at engine start.\n")
        self.assert_fails(mutated, "must not reference any TP4 engine")

    def test_dcgm_validator_covers_the_file(self) -> None:
        needle = "image: nvcr.io/nvidia/k8s/dcgm-exporter@sha256:613ab03c11d442fd960ff515f547e9921537454a712d08160bc8f677f89f1c35"
        mutated = self.replace_once(needle, "image: nvcr.io/nvidia/k8s/dcgm-exporter@sha256:" + "0" * 64)
        self.assert_fails(mutated, "GLM-5.3-Flash-SGL-TP2x4-W4AFP8.yaml", DCGM_VALIDATOR)


if __name__ == "__main__":
    _ = unittest.main()
