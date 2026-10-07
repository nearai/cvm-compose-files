#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: python3 -m unittest scripts.test_glm53_w4afp8_long_context (needs ruby for the validator cases)
"""The generated W4AFP8 + HiCache long-context file, its generator and its validator contract."""

import json
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts import glm53_v8_bundle as v8
from scripts import prepare_glm53_w4afp8_long_context as generator
# The v8 canary contract (release gate, env-map printer, render checks) is shared with the base file's tests.
from scripts.test_glm53_v8_bundle import SlotRenderChecks, V8ReleaseGateTest, _block, telemetry as telemetry_of  # noqa: F401
# CI runs this module in the ruby image; importing the gpu13 TP2 contract tests here runs them there too
# without a separate workflow step.
from scripts.test_gpu13_glm53_tp2 import Gpu13Tp2Test  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / generator.TARGET
VALIDATOR = Path("scripts/validate_glm53_prod_config.rb")
DCGM_VALIDATOR = Path("scripts/validate_glm53_dcgm_metrics.rb")
CANONICAL = Path("prod/GLM-5.3-Flash-SGL-TP4.yaml")
HICACHE = Path("prod/GLM-5.3-Flash-SGL-TP4-HiCache.yaml")
RELEASED_IMAGE = Path("docker/sglang-glm53-hicache/RELEASED_IMAGE")
PREVIOUS_R1_V2_IMAGE = "docker.io/nearaidev/sglang@sha256:8ff1a487b98a52fe08b781715bebd7c8c445d4fe068f312f03f527d5a3c77e84"
RELEASED_V3_IMAGE = "docker.io/nearaidev/sglang@sha256:47aff791090003a37f893e998c44794c410d3f7bdfc7fdd2dfab5eb5592b30bb"
RELEASED_V3_LABEL = "47aff7910900"
# glm53-hicache-w4afp8-v6 (#340, workflow run 37505271073): what every replica pins now.
RELEASED_V6_IMAGE = "docker.io/nearaidev/sglang@sha256:9c6ddd4319c4ab00e351d8650459e68b8830e36ffcc029d67fa5e19d0ac3ed17"
RELEASED_V6_LABEL = "9c6ddd4319c4"
V1_IMAGE = "docker.io/nearaidev/sglang@sha256:fde25985aea3ebabf1eb581ae21d53be8540e32933eef942ee8b962a1bfbea20"
UNKNOWN_IMAGE = "docker.io/nearaidev/sglang@sha256:" + "0" * 64
R1_VARIANT = "fc91d24-long-context-w4afp8-c8192-qsplit-offloop-v3-hicache-cuda-host-pooled-v1-admission-reserve-disabled-pool-clamp-pdi1-h200-tp4-ep4-eagle-adaptive-5-1-6-strict-budget8192-obs-v1"
# r2 also carries the host650g marker of the HiCache host-tier canary (its 650 GiB budget).
R2_VARIANT = "fc91d24-long-context-w4afp8-c8192-qsplit-offloop-v3-hicache-cuda-host-pooled-v1-host650g-admission-reserve-disabled-pool-clamp-pdi2-h200-tp4-ep4-eagle-adaptive-5-1-6-strict-budget8192-obs-v1"
R1_WITHOUT_OFFLOOP_VARIANT = R1_VARIANT.replace("-offloop-v3", "")
R2_WITHOUT_OFFLOOP_VARIANT = R2_VARIANT.replace("-offloop-v3", "")


PROXY_POOL = "VLLM_BACKEND_URLS=${GLM53_BACKEND_URLS:-http://model-sg-glm53-w4afp8-tp4-r1:8000,http://model-sg-glm53-w4afp8-tp4-r2:8000}"
TP2_VARIANT = generator.TP2_VARIANT
R2A = "model-sg-glm53-w4afp8-tp2-r2a"
R2B = "model-sg-glm53-w4afp8-tp2-r2b"
R1A = "model-sg-glm53-w4afp8-tp2-r1a"
R1B = "model-sg-glm53-w4afp8-tp2-r1b"


def replace_nth(text: str, needle: str, index: int, replacement: str) -> str:
    positions = []
    start = text.find(needle)
    while start != -1:
        positions.append(start)
        start = text.find(needle, start + 1)
    target = positions[index]
    return text[:target] + replacement + text[target + len(needle):]


ALL_TP2 = tuple(f"model-sg-glm53-w4afp8-tp2-r{suffix}" for suffix in ("2a", "2b", "1a", "1b"))


def slotify(name: str, text: str) -> str:
    """Spell a flag line the way the v8 slot (r2a) renders it: its value behind the slot's variable."""
    if name != "model-sg-glm53-w4afp8-tp2-r2a":
        return text
    for argument, (flag, variable, default) in generator.V8_ARGUMENT_VARIABLES.items():
        text = text.replace(argument, f"{flag} {v8.expression(generator.V8_PREFIX, variable, default)}")
    return text


def tp2_bounds(text: str, name: str) -> tuple[int, int]:
    """Start/end of one TP2 service block: up to the next TP2 section comment or the file's next service."""
    start = text.index(f"  {name}:\n")
    ends = [text.find(marker, start + 10) for marker in ("\n  # ---", "\n  # Explicit operator-only")]
    return start, min(end for end in ends if end != -1)


def rendered_engine_sections() -> tuple[str, str, str]:
    rendered = generator.generate((ROOT / generator.SOURCE).read_text())
    return rendered, generator.section(rendered, "x-sg-glm53-flash-common: &sg-glm53-flash-common\n", "\nx-dcgm-common: &dcgm-common\n", "shared engine anchor")[2], generator.section(rendered, "  model-sg-glm53-w4afp8-tp4-r2:\n", "\n  # --- GLM-5.3-Flash 2xTP2 memory-optimized canary replica 2a", "replica 2 service")[2]


class GeneratedFileTest(unittest.TestCase):
    def test_both_replicas_use_the_released_v3_image_identity(self) -> None:
        self.assertEqual(generator.REPLICA_IMAGE, {1: RELEASED_V6_IMAGE, 2: RELEASED_V6_IMAGE})
        self.assertEqual(generator.REPLICA_IMAGE_LABEL, {1: RELEASED_V6_LABEL, 2: RELEASED_V6_LABEL})

    def test_committed_file_matches_generator(self) -> None:
        # Given the committed long-context source, the regenerated target is byte-identical.
        self.assertEqual(TARGET.read_text(), generator.generate((ROOT / generator.SOURCE).read_text()))

    def test_check_mode_passes(self) -> None:
        result = subprocess.run(
            ["python3", str(ROOT / "scripts/prepare_glm53_w4afp8_long_context.py"), "--check"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_long_domain_routing_blocks_are_byte_identical_to_the_source(self) -> None:
        # The registrar script and the nginx config (the :8001 discovery stub included) are
        # the long-domain contract; the generator must never touch them.
        source = (ROOT / generator.SOURCE).read_text()
        target = TARGET.read_text()
        for start, end in (("  registrar_script:\n", "  nginx_conf:\n"), ("  nginx_conf:\n", "\x00")):
            with self.subTest(block=start.strip()):
                source_block = source[source.index(start):] if end == "\x00" else generator.section(source, start, end, start)[2]
                target_block = target[target.index(start):] if end == "\x00" else generator.section(target, start, end, start)[2]
                self.assertEqual(target_block, source_block)

    def test_generator_refuses_its_own_output_and_a_drifted_source(self) -> None:
        source = (ROOT / generator.SOURCE).read_text()
        with self.assertRaises(generator.GenerationError):
            generator.generate(TARGET.read_text())
        for before in ("\n      --moe-runner-backend deep_gemm\n", "    image: " + generator.SOURCE_HICACHE_IMAGE + "\n"):
            with self.subTest(removed=before.strip()[:40]):
                self.assertEqual(source.count(before), 1)
                with self.assertRaises(generator.GenerationError):
                    generator.generate(source.replace(before, "\n" if before.startswith("\n") else ""))

    def test_released_v3_image_is_pinned_on_both_replicas(self) -> None:
        _, anchor, r2 = rendered_engine_sections()
        self.assertIn(f"\n  image: {RELEASED_V6_IMAGE}\n", anchor)
        self.assertNotIn(PREVIOUS_R1_V2_IMAGE, anchor)
        self.assertIn(f"\n    image: {RELEASED_V6_IMAGE}\n", r2)
        self.assertNotIn(PREVIOUS_R1_V2_IMAGE, r2)

    def test_offloop_v3_marker_is_truthful_on_both_replicas(self) -> None:
        rendered, _, _ = rendered_engine_sections()
        self.assertEqual(rendered.count(R1_VARIANT), 3)
        self.assertEqual(rendered.count(R2_VARIANT), 3)
        self.assertNotIn(R1_WITHOUT_OFFLOOP_VARIANT, rendered)
        self.assertNotIn(R2_WITHOUT_OFFLOOP_VARIANT, rendered)

    def test_runtime_parameters_remain_the_no_flag_8192_split_arm(self) -> None:
        # Given the r1-only image promotion, runtime activation remains unconditional:
        # no dynamic-tokenizer flag, chunk 8192, QSPLIT=1 and pdi 1/2 stay observable.
        _, anchor, r2 = rendered_engine_sections()
        runtime = anchor + r2
        self.assertNotIn("DYNAMIC_BATCH_TOKENIZER", runtime.upper())
        self.assertNotIn("dynamic-batch-tokenizer", runtime.lower())
        self.assertIn("\n      --chunked-prefill-size 8192\n", anchor)
        self.assertIn("\n        --chunked-prefill-size 8192\n", r2)
        self.assertIn("\n      --prefill-decode-interval 1\n", anchor)
        self.assertIn("\n        --prefill-decode-interval 2\n", r2)
        self.assertEqual(anchor.count("\n    - SGLANG_DSA_INDEXER_QSPLIT=1\n"), 1)
        self.assertEqual(r2.count("\n      - SGLANG_DSA_INDEXER_QSPLIT=1\n"), 1)

    def test_tp2_canary_leaves_the_tp4_replicas_and_the_default_pool_untouched_for_gpu23(self) -> None:
        # gpu23 deploys r1 + r2 from this same file with a scoped services list and never sets
        # GLM53_BACKEND_URLS. Without the canary additions the file must be the same except for the
        # proxy line's override wrapper, whose default is the exact previous value.
        source = (ROOT / generator.SOURCE).read_text()
        with_canary = generator.generate(source)
        original_add = generator.add_tp2_canary
        try:
            generator.add_tp2_canary = lambda text: text
            without = generator.generate(source)
        finally:
            generator.add_tp2_canary = original_add
        r1 = "  model-sg-glm53-w4afp8-tp4-r1:\n"
        r2 = "  model-sg-glm53-w4afp8-tp4-r2:\n"
        self.assertEqual(
            generator.section(with_canary, r1, r2, "r1")[2], generator.section(without, r1, r2, "r1")[2]
        )
        self.assertEqual(
            generator.section(with_canary, r2, "\n  # --- GLM-5.3-Flash 2xTP2", "r2")[2],
            generator.section(without, r2, "\n  # Explicit operator-only", "r2")[2],
        )
        self.assertIn(f"- VLLM_BACKEND_URLS={generator.BACKEND_URLS_R1_R2}\n", without)
        self.assertIn("${GLM53_BACKEND_URLS:-" + generator.BACKEND_URLS_R1_R2 + "}", with_canary)

    def test_tp2_pair_is_memory_optimized_and_half_of_r2_host_ram(self) -> None:
        rendered = generator.generate((ROOT / generator.SOURCE).read_text())
        for suffix, spec in generator.TP2_REPLICAS.items():
            begin, finish = tp2_bounds(rendered, f"{generator.TP2_SERVICE_PREFIX}{suffix}")
            service = re.sub(r"\$\{GLM53_V8_[A-Z0-9_]+:-([^}]*)\}", r"\1", rendered[begin:finish])  # the slot's defaults are today's values
            with self.subTest(replica=suffix):
                for flag in ("--tp-size 2", "--ep-size 2", "--mem-fraction-static 0.86", "--max-mamba-cache-size 330", "--max-running-requests 12", "--cuda-graph-max-bs-decode 12",
                             "--max-queued-requests 4", "--speculative-num-steps 4", "--speculative-eagle-topk 1", "--speculative-num-draft-tokens 5",
                             "--mamba-ssm-dtype bfloat16", "--chunked-prefill-size 8192", "--hicache-write-policy write_through",
                             f"--dist-init-addr {spec['dist_init']}"):
                    self.assertIn(f"\n        {flag}\n", service)
                for forbidden in ("--speculative-adaptive", "--disable-overlap-schedule", "ADMISSION_RESERVE"):
                    self.assertNotIn(forbidden, service)
                self.assertIn(f"${{{spec['budget_var']}:-325GiB}}", service)
                self.assertEqual(2 * 325, 650)  # r2 default is 650 GiB

    def test_observability_is_on_every_replica_with_one_aggregator(self) -> None:
        # Every GLM engine in the file (TP4 r1/r2 and the gpu02 TP2 pair r2a/r2b) carries the full
        # observability environment under its own replica name; there is one sidecar and one job.
        rendered = generator.generate((ROOT / generator.SOURCE).read_text())
        blocks = {
            "r1": (generator.section(rendered, "x-sg-glm53-flash-common: &sg-glm53-flash-common\n", "\nx-dcgm-common: &dcgm-common\n", "anchor")[2], "    "),
            "r2": (generator.section(rendered, "  model-sg-glm53-w4afp8-tp4-r2:\n", "\n  # ", "r2")[2], "      "),
        }
        for suffix in generator.TP2_REPLICAS:
            name = f"{generator.TP2_SERVICE_PREFIX}{suffix}"
            blocks[f"r{suffix}"] = (generator.section(rendered, f"  {name}:\n", "\n  # ", name)[2], "      ")
        self.assertEqual(sorted(blocks), ["r1", "r1a", "r1b", "r2", "r2a", "r2b"])
        for replica, (block, indent) in blocks.items():
            with self.subTest(replica=replica):
                for entry in ("SGLANG_GHOST_CACHE=1", "SGLANG_GHOST_CACHE_SAMPLE=16", "SGLANG_GHOST_CACHE_KEY_FILE=/ghost/key",
                              "SGLANG_GHOST_CACHE_SOCKET=/ghost/aggregator.sock", f"SGLANG_GHOST_CACHE_REPLICA={replica}",
                              "SGLANG_KV_TIER_METRICS=1"):
                    self.assertEqual(block.count(f"\n{indent}- {entry}\n"), 1, entry)
        self.assertIn("\n    - ghost:/ghost\n", blocks["r1"][0])
        self.assertEqual(rendered.count("\n  glm53-ghost-aggregator:\n"), 1)
        self.assertEqual(rendered.count("- job_name: ghost-aggregator-glm53-ghost-aggregator\n"), 1)
        self.assertIn(f"    image: {generator.IMAGE}\n    container_name: glm53-ghost-aggregator\n", rendered)
        self.assertTrue(generator.TP2_VARIANT.endswith("-obs-v1"))


class LongSlotRenderTest(SlotRenderChecks, unittest.TestCase):
    """gpu02's r2a is the v8 canary slot of this file; gpu23, r2b, r1a, r1b and the TP4 replicas must render as before."""

    kind = "long"
    generator = generator
    target = TARGET
    slot = R2A
    engines = ALL_TP2
    sibling = R2B
    prefix = generator.V8_PREFIX
    other_kind = "base"
    canary_flags = {
        "--kv-cache-dtype": "fp8_e4m3",
        "--dsa-prefill-backend": "flashmla_kv",
        "--dsa-decode-backend": "flashmla_kv",
        "--max-running-requests": str(generator.V8_LONG_MAX_RUNNING),
        "--cuda-graph-max-bs-decode": str(generator.V8_LONG_MAX_RUNNING),
        "--max-queued-requests": str(generator.V8_LONG_MAX_QUEUED),
    }
    disable_slot = {"V8_SLOT": "none"}
    default_variant = generator.TP2_VARIANT
    extra_variable_names = {"MAX_QUEUED"}

    def adapt_to(self, name: str, argv: list[str]) -> list[str]:
        suffix = name.rsplit("r", 1)[1]
        out = list(argv)
        out[out.index("--dist-init-addr") + 1] = generator.TP2_REPLICAS[suffix]["dist_init"]
        return out

    def test_the_slot_is_r2a_and_everything_but_the_bundle_stays_the_tp2_argv(self) -> None:
        self.assertEqual(generator.V8_SLOT, "2a")
        self.assertEqual(generator.V8_PREFIX, "GLM53_V8_R2A_")
        argv = self.argv(self.slot, self.canary_env())
        for flag, value in (("--tp-size", "2"), ("--mem-fraction-static", "0.86"), ("--max-mamba-cache-size", "330"),
                            ("--speculative-num-steps", "4"), ("--speculative-num-draft-tokens", "5"), ("--chunked-prefill-size", "8192"),
                            ("--prefill-decode-interval", "2"), ("--hicache-write-policy", "write_through"), ("--dist-init-addr", "127.0.0.1:29512")):
            self.assertEqual(argv[argv.index(flag) + 1], value, flag)
        # 5 mamba slots per running request must fit the unchanged 330.
        self.assertGreaterEqual(330, 5 * generator.V8_LONG_MAX_RUNNING)

    def test_the_long_caps_are_single_generator_constants_and_the_file_keeps_12_4(self) -> None:
        self.assertEqual((generator.V8_LONG_MAX_RUNNING, generator.V8_LONG_MAX_QUEUED), (16, 6))
        self.assertEqual(self.argv(self.slot, {})[self.argv(self.slot, {}).index("--max-running-requests") + 1], "12")
        self.assertEqual(self.argv(self.slot, {})[self.argv(self.slot, {}).index("--max-queued-requests") + 1], "4")
        self.assertNotIn(f"-{generator.V8_LONG_MAX_RUNNING}q{generator.V8_LONG_MAX_QUEUED}", self.text)  # the caps live in the env map only

    def test_tp4_replicas_ignore_the_canary_env_map(self) -> None:
        for name in ("model-sg-glm53-w4afp8-tp4-r1", "model-sg-glm53-w4afp8-tp4-r2"):
            with self.subTest(replica=name):
                self.assertNotIn("GLM53_V8_", _block(self.text, name))
                self.assertEqual(self.argv(name, self.canary_env()), self.argv(name, {}))
                self.assertEqual(telemetry_of(self.text, name, self.canary_env()), telemetry_of(self.text, name, {}))


class ValidatorContractTest(unittest.TestCase):
    """Each mutation of the committed file must fail the production validator with its reason."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name in (CANONICAL, HICACHE, RELEASED_IMAGE, generator.SOURCE, generator.TARGET, VALIDATOR, DCGM_VALIDATOR):
            (self.root / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, self.root / name)
        self.target = self.root / generator.TARGET
        self.valid = self.target.read_text()

    def selected_candidate(self) -> str:
        return self.valid

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

    def replace_r2(self, before: str, after: str) -> str:
        """Mutate the TP4 r2 occurrence of a needle that the TP2 pair (rendered after r2) also carries."""
        self.assertEqual(self.valid.count(before), 5, before)
        return replace_nth(self.valid, before, 0, after)

    def test_committed_files_pass(self) -> None:
        for script in (VALIDATOR, DCGM_VALIDATOR):
            with self.subTest(script=str(script)):
                result = self.run_ruby(script)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("GLM-5.3-Flash-SGL-TP4-W4AFP8-LongContext.yaml", result.stdout)

    def test_selected_candidate_passes_the_real_validator(self) -> None:
        candidate = self.selected_candidate()
        self.target.write_text(candidate)
        result = self.run_ruby(VALIDATOR)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_rejects_image_and_split_capability_drift(self) -> None:
        candidate = self.selected_candidate()
        cases = (
            (candidate.replace(f"    image: {RELEASED_V6_IMAGE}\n", f"    image: {PREVIOUS_R1_V2_IMAGE}\n", 1), "model-sg-glm53-w4afp8-tp4-r2 image must be"),
            (candidate.replace(f"  image: {RELEASED_V6_IMAGE}\n", f"  image: {PREVIOUS_R1_V2_IMAGE}\n", 1), "model-sg-glm53-w4afp8-tp4-r1 image must be"),
            (candidate.replace(f"  image: {RELEASED_V6_IMAGE}\n", f"  image: {V1_IMAGE}\n", 1), "does not run an approved split-capable image"),
            (candidate.replace(f"  image: {RELEASED_V6_IMAGE}\n", f"  image: {UNKNOWN_IMAGE}\n", 1), "does not run an approved split-capable image"),
        )
        self.valid = candidate
        for mutated, message in cases:
            with self.subTest(message=message):
                self.assert_fails(mutated, message)

    def test_rejects_untruthful_offloop_marker_in_each_consumer(self) -> None:
        candidate = self.selected_candidate()
        for variant in (R1_VARIANT, R2_VARIANT):
            for index in range(3):
                with self.subTest(variant=variant, consumer=index):
                    mutated = replace_nth(candidate, variant, index, variant.replace("-offloop-v3", ""))
                    self.valid = candidate
                    self.assert_fails(mutated, "config_variant")

    def test_rejects_engine_argv_drift(self) -> None:
        argv = "argv must be campaign-2 arm L2 exactly"
        cases = (
            ("\n      --chunked-prefill-size 8192\n", "\n      --chunked-prefill-size 4096\n", argv),
            ("        --dist-init-addr 127.0.0.1:29511\n", "        --dist-init-addr 127.0.0.1:29510\n", "--dist-init-addr 127.0.0.1:29511"),
            ("\n        --hicache-io-backend direct\n", "\n        --hicache-io-backend kernel\n", argv),
            (
                "\n      --served-model-name z-ai/glm-5.3-flash\n",
                "\n      --served-model-name z-ai/glm-5.3-flash\n      --revision 84c6a6aa9497188e15a635ba793b0f95a79b1033\n",
                argv,
            ),
            ("\n      --max-queued-requests 8\n", "\n      --max-queued-requests 32\n", argv),
            # r1 is the pdi 1 control and r2 the pdi 2 canary; neither may take the other's value.
            ("\n      --prefill-decode-interval 1\n", "\n      --prefill-decode-interval 2\n", "--prefill-decode-interval 1"),
            ("\n        --prefill-decode-interval 2\n", "\n        --prefill-decode-interval 1\n", "--prefill-decode-interval 2"),
            # Both replicas run 8192 in this arm, so neither may drift off it.
            ("\n      --chunked-prefill-size 8192\n", "\n      --chunked-prefill-size 4096\n", "model-sg-glm53-w4afp8-tp4-r1 argv must be"),
            ("\n        --chunked-prefill-size 8192\n", "\n        --chunked-prefill-size 4096\n", "model-sg-glm53-w4afp8-tp4-r2 argv must be"),
        )
        for before, after, message in cases:
            with self.subTest(mutation=after.strip()[:60]):
                mutated = self.replace_r2(before, after) if before.startswith("\n        --") else self.replace_once(before, after)
                self.assert_fails(mutated, message)

    def test_rejects_c16384_without_the_indexer_split(self) -> None:
        """Raising r2 to the 16384 chunk without the split must be rejected by the pairing gate.

        This arm runs 8192, so the forbidden pairing has to be constructed: raise the chunk AND
        drop the split. Without the split that chunk left 0.04-0.65 GB free per GPU on a
        concurrent long burst, the condition that preceded the gpu02 crash. The assertion names
        the pairing error specifically, so the test cannot pass merely because some unrelated
        equality check fired first.
        """
        mutated = self.replace_r2("\n        --chunked-prefill-size 8192\n", "\n        --chunked-prefill-size 16384\n")
        mutated = mutated.replace("      - SGLANG_DSA_INDEXER_QSPLIT=1\n", "", 1)
        self.assert_fails(mutated, "without SGLANG_DSA_INDEXER_QSPLIT=1")

    def test_rejects_dropping_the_split_from_r2(self) -> None:
        """r2 carries the split in this arm; removing it is the whole variable under test."""
        self.assert_fails(self.replace_r2("      - SGLANG_DSA_INDEXER_QSPLIT=1\n", ""), "SGLANG_DSA_INDEXER_QSPLIT")

    def test_rejects_the_split_on_an_image_without_the_patch(self) -> None:
        """The split flag is inert and misleading on v1, which does not carry the patch.

        Both replicas now run v3, so this mutation has to name the v1 digest explicitly -- using
        generator.IMAGE would be a no-op and the test would pass without exercising anything.
        """
        v1 = "docker.io/nearaidev/sglang@sha256:fde25985aea3ebabf1eb581ae21d53be8540e32933eef942ee8b962a1bfbea20"
        self.assert_fails(replace_nth(self.valid, f"  image: {generator.IMAGE}\n", 0, f"  image: {v1}\n"), "image must be")

    def test_rejects_engine_image_and_environment_drift(self) -> None:
        # r2 now carries its own environment block (it needs SGLANG_DSA_INDEXER_QSPLIT=1 and a
        # YAML merge key replaces a list rather than extending it), so several of these needles
        # legitimately appear twice: once in the shared anchor and once in r2. Each case states
        # which occurrence it mutates instead of relying on the needle being unique.
        cases = (
            (f"\n  image: {generator.IMAGE}\n", "\n  image: docker.io/nearaidev/sglang@sha256:" + "0" * 64 + "\n", "image must be", 0),
            ("${GLM53_HICACHE_RAM_BUDGET:-406GiB}", "${GLM53_HICACHE_RAM_BUDGET:-80%}", "SGLANG_HICACHE_RAM_BUDGET=${GLM53_HICACHE_RAM_BUDGET:-80%}", 0),
            # r2 is the host-tier canary with its own budget variable; reverting it to the r1
            # budget, or dropping the canary default, must both fail.
            ("${GLM53_R2_HICACHE_RAM_BUDGET:-650GiB}", "${GLM53_HICACHE_RAM_BUDGET:-406GiB}", "SGLANG_HICACHE_RAM_BUDGET=${GLM53_HICACHE_RAM_BUDGET:-406GiB}", 0),
            ("${GLM53_R2_HICACHE_RAM_BUDGET:-650GiB}", "${GLM53_R2_HICACHE_RAM_BUDGET:-80%}", "SGLANG_HICACHE_RAM_BUDGET=${GLM53_R2_HICACHE_RAM_BUDGET:-80%}", 0),
            ("${GLM53_HICACHE_CUDA_HOST_MEMORY:-1}", "${GLM53_HICACHE_CUDA_HOST_MEMORY:-0}", "environment must be the long-context control environment", 0),
            ("${GLM53_HICACHE_CUDA_HOST_MEMORY:-1}", "${GLM53_HICACHE_CUDA_HOST_MEMORY:-0}", "environment must be the long-context control environment", 1),
            (
                "    - SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=1\n",
                "    - SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=1\n    - SGLANG_CHUNKED_PREFILL_ADMISSION_RESERVE=4096\n",
                "must not set admission-reserve environment",
                0,
            ),
            ('device_ids: ["4","5","6","7"]', 'device_ids: ["0","1","2","3"]', "must use GPU device_ids 4,5,6,7", 0),
            (
                "    container_name: model-sg-glm53-w4afp8-tp4-r2\n",
                '    container_name: model-sg-glm53-w4afp8-tp4-r2\n    restart: "no"\n',
                "replicas must share one runtime configuration",
                0,
            ),
        )
        for before, after, message, index in cases:
            with self.subTest(mutation=after.strip()[:50], occurrence=index):
                self.assert_fails(replace_nth(self.valid, before, index, after), message)

    def test_rejects_routing_drift_from_the_long_context_file(self) -> None:
        outside = "must match the long-context file outside the two engines"
        cases = (
            ("LONG_TIER_ONLY=${LONG_TIER_ONLY:-true}", "LONG_TIER_ONLY=${LONG_TIER_ONLY:-false}", outside),
            ('"id":"z-ai/glm-5.3-flash-long"', '"id":"z-ai/glm-5.3-flash"', outside),
            ("set $$backend http://model-sg-glm53-w4afp8-tp4-r2:8000;", "set $$backend http://model-sg-glm53-w4afp8-tp4-r1:8000;", outside),
            (
                PROXY_POOL,
                "VLLM_BACKEND_URLS=${GLM53_BACKEND_URLS:-http://model-sg-glm53-w4afp8-tp4-r1:8000}",
                "must pool both W4AFP8 replicas",
            ),
            # gpu23 never sets GLM53_BACKEND_URLS: dropping the override, or pooling the TP2 pair
            # by default, would change what gpu23 deploys.
            (PROXY_POOL, "VLLM_BACKEND_URLS=http://model-sg-glm53-w4afp8-tp4-r1:8000,http://model-sg-glm53-w4afp8-tp4-r2:8000", "must pool both W4AFP8 replicas"),
            (
                PROXY_POOL,
                "VLLM_BACKEND_URLS=${GLM53_BACKEND_URLS:-http://model-sg-glm53-w4afp8-tp4-r1:8000,http://model-sg-glm53-w4afp8-tp2-r2a:8000,http://model-sg-glm53-w4afp8-tp2-r2b:8000}",
                "must pool both W4AFP8 replicas",
            ),
        )
        for before, after, message in cases:
            with self.subTest(mutation=after[:60]):
                self.assert_fails(self.replace_once(before, after), message)

    def test_rejects_untruthful_telemetry(self) -> None:
        cases = (
            (f'nearai.otel.config_variant: "{generator.VARIANTS[1]}"', 'nearai.otel.config_variant: "incorrect-variant"', 0,
             "nearai.otel.config_variant must be"),
            (f'nearai.otel.config_variant: "{generator.VARIANTS[2]}"', f'nearai.otel.config_variant: "{generator.VARIANTS[1]}"', 0,
             "nearai.otel.config_variant must be"),
            (f"config_variant:{generator.VARIANTS[1]}", "config_variant:incorrect-variant", 0, "log metadata must carry exactly config_variant:"),
            (f'      nearai.otel.engine_image: "{generator.ENGINE_IMAGE_LABEL}"\n', '      nearai.otel.engine_image: "e9d29a1cb1cd"\n', 0, "nearai.otel.engine_image must be"),
            (f'      nearai.otel.engine_image: "{generator.R2_ENGINE_IMAGE_LABEL}"\n', '      nearai.otel.engine_image: "e9d29a1cb1cd"\n', 1, "nearai.otel.engine_image must be"),
            ('"precision:int4-weights-fp8-activations-bf16-kv"', '"precision:fp8-weights-bf16-kv"', 0, "log metadata must carry precision:"),
            (f'                      engine_image: "{generator.ENGINE_IMAGE_LABEL}"\n', '                      engine_image: "e9d29a1cb1cd"\n', 0, "scrape label engine_image"),
            (f'                      engine_image: "{generator.R2_ENGINE_IMAGE_LABEL}"\n', '                      engine_image: "e9d29a1cb1cd"\n', 1, "scrape label engine_image"),
            ('      nearai.otel.model_path: "graphistry/GLM-5.3-Flash-W4AFP8"\n', '      nearai.otel.model_path: "zai-org/GLM-5.3-Flash"\n', 6, "dcgm-glm53 nearai.otel.model_path"),
        )
        for needle, replacement, index, message in cases:
            with self.subTest(mutation=message):
                self.assert_fails(replace_nth(self.valid, needle, index, replacement), message)

    def test_tp2_canary_rejects_flag_and_memory_drift(self) -> None:
        argv = "argv must be the memory-optimized TP2 argv exactly"
        for name in ALL_TP2:
            start, end = tp2_bounds(self.valid, name)
            block = self.valid[start:end]
            cases = (
                ("--mem-fraction-static 0.86", "--mem-fraction-static 0.80", argv),
                ("--max-mamba-cache-size 330", "--max-mamba-cache-size 165", argv),
                ("--max-running-requests 12", "--max-running-requests 32", argv),
                ("--max-running-requests 12", "--max-running-requests 24", argv),
                ("--max-running-requests 12", "--max-running-requests 16", argv),
                ("--max-queued-requests 4", "--max-queued-requests 8", argv),
                ("--max-queued-requests 4", "--max-queued-requests 16", argv),
                ("--cuda-graph-max-bs-decode 12", "--cuda-graph-max-bs-decode 32", argv),
                ("--cuda-graph-max-bs-decode 12", "--cuda-graph-max-bs-decode 16", argv),
                ("--speculative-num-steps 4", "--speculative-num-steps 5", argv),
                ("--speculative-num-draft-tokens 5", "--speculative-num-draft-tokens 6", argv),
                ("--tp-size 2", "--tp-size 4", argv),
                ("--hicache-write-policy write_through", "--hicache-write-policy write_through_selective", argv),
                ("--chunked-prefill-size 8192", "--chunked-prefill-size 32768", "must not enable 32K prefill chunks"),
                ("--mamba-ssm-dtype bfloat16\n", "--mamba-ssm-dtype bfloat16\n        --speculative-adaptive\n", "must not set --speculative-adaptive"),
                ("--mamba-ssm-dtype bfloat16\n", "--mamba-ssm-dtype bfloat16\n        --disable-overlap-schedule\n", "must not set --disable-overlap-schedule"),
            )
            for before, after, message in cases:
                before = slotify(name, before)
                with self.subTest(replica=name, mutation=after.strip()[:50]):
                    self.assertEqual(block.count(before), 1, before)
                    self.assert_fails(self.valid[:start] + block.replace(before, after, 1) + self.valid[end:], message)

    def test_tp2_canary_rejects_environment_pinning_and_telemetry_drift(self) -> None:
        for name, devices, wrong_devices, budget, instance in (
            (R2A, '["4","5"]', '["4","6"]', "${GLM53_R2A_HICACHE_RAM_BUDGET:-325GiB}", "2a"),
            (R2B, '["6","7"]', '["0","1"]', "${GLM53_R2B_HICACHE_RAM_BUDGET:-325GiB}", "2b"),
            (R1A, '["0","1"]', '["0","2"]', "${GLM53_R1A_HICACHE_RAM_BUDGET:-325GiB}", "1a"),
            (R1B, '["2","3"]', '["4","5"]', "${GLM53_R1B_HICACHE_RAM_BUDGET:-325GiB}", "1b"),
        ):
            start, end = tp2_bounds(self.valid, name)
            block = self.valid[start:end]
            cases = (
                (f"device_ids: {devices}", f"device_ids: {wrong_devices}", "must use GPU device_ids"),
                ("    environment:\n", "    environment:\n      - SGLANG_CHUNKED_PREFILL_ADMISSION_RESERVE=4096\n", "must not set admission-reserve environment"),
                (budget, budget.replace("325GiB", "650GiB"), "environment must be the long-context environment plus per-replica"),
                ("      - SGLANG_DSA_INDEXER_QSPLIT=1\n", "", "environment must be the long-context environment plus per-replica"),
                (f'nearai.otel.instance: "{instance}"', 'nearai.otel.instance: "9"', "nearai.otel.instance must be"),
                ("nearai.otel.gpu_pair:", "nearai.otel.gpu_pairx:", "nearai.otel.gpu_pair must be"),
                (f'nearai.otel.config_variant: "{TP2_VARIANT}', 'nearai.otel.config_variant: "x', "nearai.otel.config_variant must be"),
                ('nearai.otel.deployment: "glm53-flash-sgl-tp4"', 'nearai.otel.deployment: "other"', "nearai.otel.deployment"),
                (f"image: {generator.V8_IMAGE_EXPRESSION if name == R2A else RELEASED_V6_IMAGE}\n", f"image: {PREVIOUS_R1_V2_IMAGE}\n", "image must be"),
                ("    container_name: " + name + "\n", "    container_name: " + name + '\n    restart: "no"\n', "must share one runtime configuration outside command, environment and image"),
            )
            for before, after, message in cases:
                with self.subTest(replica=name, message=message):
                    self.assertGreaterEqual(block.count(before), 1, before)
                    self.assert_fails(self.valid[:start] + block.replace(before, after, 1) + self.valid[end:], message)

    def test_rejects_v8_slot_drift(self) -> None:
        p = generator.V8_PREFIX
        extra = f"${{{p}EXTRA_ARGS:-}}"
        env_prefix = f"${{{p}ENV_PREFIX:-}}"
        image = generator.V8_IMAGE_EXPRESSION
        pinned = "is not a v8 slot variable with its pinned default"
        r2b_command_end = "        --max-mamba-cache-size 330\n        --mamba-ssm-dtype bfloat16\n"
        r2b_start, r2b_end = tp2_bounds(self.valid, R2B)
        r2b_block = self.valid[r2b_start:r2b_end]
        self.assertEqual(r2b_block.count(r2b_command_end), 1)
        cases = (
            # A default that is not today's value / empty: would turn the bundle on for gpu23 and every other replica.
            (self.valid.replace(extra, f"${{{p}EXTRA_ARGS:---disable-overlap-schedule}}"), pinned),
            (self.valid.replace(env_prefix, f"${{{p}ENV_PREFIX:-env SGLANG_PREPROCESS_WORKERS=4}}"), pinned),
            (self.valid.replace(f"${{{p}VARIANT_SUFFIX:-}}", f"${{{p}VARIANT_SUFFIX:--v8bundle}}"), pinned),
            (self.valid.replace(f"${{{p}MAX_RUNNING:-12}}", f"${{{p}MAX_RUNNING:-16}}"), pinned),
            (self.valid.replace(f"${{{p}MAX_QUEUED:-4}}", f"${{{p}MAX_QUEUED:-6}}"), pinned),
            (self.valid.replace(f"${{{p}KV_DTYPE:-bfloat16}}", f"${{{p}KV_DTYPE:-fp8_e4m3}}"), pinned),
            (self.valid.replace(f"${{{p}DSA_BACKEND:-tilelang}}", f"${{{p}DSA_BACKEND:-flashmla_kv}}"), pinned),
            (self.valid.replace(f"${{{p}PRECISION:-int4-weights-fp8-activations-bf16-kv}}", f"${{{p}PRECISION:-int4-weights-fp8-activations-fp8-kv}}"), pinned),
            (self.valid.replace(extra, f"${{{p}EXTRA_ARGS}}"), pinned),
            (self.valid.replace(extra, f"${p}EXTRA_ARGS"), "references GLM53_V8_ outside a ${NAME:-default} expression"),
            # The image: a placeholder digest as the default, a different default, or a bare reference.
            (self.valid.replace(image, f"${{{p}IMAGE:-docker.io/nearaidev/sglang@{v8.IMAGE_DIGEST_PLACEHOLDER}}}"), "contains the v8 placeholder"),
            (self.valid.replace(image, f"${{{p}IMAGE:-{PREVIOUS_R1_V2_IMAGE}}}"), pinned),
            (self.valid.replace(image, f"${{{p}IMAGE}}"), pinned),
            (self.valid.replace(env_prefix, "", 1), "argv must be the memory-optimized TP2 argv exactly"),
            # The flag, the environment, the dtype or the backend as literals anywhere.
            (self.valid.replace(extra, "--disable-overlap-schedule"), "must not hardcode --disable-overlap-schedule"),
            (self.valid.replace(extra, "env NEAR_SELF_PROFILE=1"), "must not hardcode NEAR_SELF_PROFILE"),
            (self.valid[:r2b_start] + r2b_block.replace("    environment:\n", "    environment:\n      - NEAR_SELF_PROFILE=1\n", 1) + self.valid[r2b_end:],
             "must not hardcode NEAR_SELF_PROFILE"),
            (self.valid.replace(env_prefix, "env SGLANG_PREPROCESS_WORKERS=4"), "must not hardcode SGLANG_PREPROCESS_"),
            (self.valid.replace(f"${{{p}KV_DTYPE:-bfloat16}}", "fp8_e4m3"), "must not hardcode fp8_e4m3"),
            (self.valid.replace(f"${{{p}DSA_BACKEND:-tilelang}}", "flashmla_kv", 1), "must not hardcode flashmla_kv"),
            # The slot's variables on the wrong replica (the sibling r2b, a gpu23-era r1a, a TP4 replica) or a shared anchor.
            (self.valid[:r2b_start] + r2b_block.replace(r2b_command_end, r2b_command_end + f"        {extra}\n") + self.valid[r2b_end:],
             f"{R2B} must not reference GLM53_V8_ variables; only {R2A} does"),
            (replace_nth(self.valid, 'nearai.otel.engine_image: "9c6ddd4319c4"', 1, f'nearai.otel.engine_image: "{generator.V8_IMAGE_LABEL_EXPRESSION}"'),
             "must not reference GLM53_V8_ variables; only"),
            (self.replace_once("      - VLLM_BACKEND_CONVERSATION_AFFINITY=1\n", f"      - VLLM_BACKEND_CONVERSATION_AFFINITY=1\n      - X={extra}\n"),
             "proxy-glm53 must not reference GLM53_V8_ variables"),
            (self.valid.replace("\n  image: " + RELEASED_V6_IMAGE + "\n", "\n  image: " + RELEASED_V6_IMAGE + f"\n  x-leak: {extra}\n", 1),
             "x-sg-glm53-flash-common must not reference GLM53_V8_ variables"),
            # Another file's slot prefix, a second variable for the graph batch.
            (self.valid.replace(extra, "${GLM53_V8_R4_EXTRA_ARGS:-}"), pinned),
            (self.valid.replace(f"--cuda-graph-max-bs-decode ${{{p}MAX_RUNNING:-12}}", f"--cuda-graph-max-bs-decode ${{{p}GRAPH_BS:-12}}"), pinned),
            # A telemetry place that lost the suffix or the precision/label expression (counts are pinned).
            (replace_nth(self.valid, generator.V8_VARIANT_EXPRESSION, 2, ""), "must appear exactly 3 time(s)"),
            (replace_nth(self.valid, generator.V8_PRECISION_EXPRESSION, 1, "int4-weights-fp8-activations-bf16-kv"), "must appear exactly 2 time(s)"),
            (replace_nth(self.valid, generator.V8_IMAGE_LABEL_EXPRESSION, 2, "9c6ddd4319c4"), "must appear exactly 3 time(s)"),
        )
        for mutated, message in cases:
            with self.subTest(expect=message, mutation=hash(mutated) % 10000):
                self.assertNotEqual(mutated, self.valid)
                self.assert_fails(mutated, message)

    def test_tp2_canary_rejects_duplicate_dist_init_port_and_gpu_overlap(self) -> None:
        self.assert_fails(self.replace_once("        --dist-init-addr 127.0.0.1:29513\n", "        --dist-init-addr 127.0.0.1:29512\n"), "is used by more than one engine")
        self.assert_fails(self.replace_once("        --dist-init-addr 127.0.0.1:29512\n", "        --dist-init-addr 127.0.0.1:29510\n"), "is used by more than one engine")
        self.assert_fails(self.replace_once('device_ids: ["6","7"]', 'device_ids: ["4","5"]'), "must not share GPUs")

    def test_tp2_rollout_rejects_cross_half_gpus_ports_and_overlap(self) -> None:
        # r1a/r1b replace TP4 r1 (GPUs 0-3) and must stay inside it; r2a/r2b stay inside r2 (4-7).
        cases = (
            ('device_ids: ["0","1"]', 'device_ids: ["4","5"]', "must stay within model-sg-glm53-w4afp8-tp4-r1's GPUs"),
            ('device_ids: ["2","3"]', 'device_ids: ["6","7"]', "must stay within model-sg-glm53-w4afp8-tp4-r1's GPUs"),
            ('device_ids: ["4","5"]', 'device_ids: ["0","1"]', "must stay within model-sg-glm53-w4afp8-tp4-r2's GPUs"),
            ('device_ids: ["2","3"]', 'device_ids: ["0","1"]', "must not share GPUs"),
            ('device_ids: ["0","1"]', 'device_ids: ["0","4"]', "must stay within model-sg-glm53-w4afp8-tp4-r1's GPUs"),
            ("        --dist-init-addr 127.0.0.1:29514\n", "        --dist-init-addr 127.0.0.1:29512\n", "is used by more than one engine"),
            ("        --dist-init-addr 127.0.0.1:29515\n", "        --dist-init-addr 127.0.0.1:29511\n", "is used by more than one engine"),
            ("        --dist-init-addr 127.0.0.1:29515\n", "        --dist-init-addr 127.0.0.1:29514\n", "is used by more than one engine"),
        )
        for before, after, message in cases:
            with self.subTest(mutation=after.strip()[:60], message=message):
                self.assert_fails(self.replace_once(before, after), message)

    def test_tp4_engine_ports_and_scrape_instances_are_unique_and_unchanged(self) -> None:
        text = TARGET.read_text()
        for port in ("127.0.0.1:29510", "127.0.0.1:29511"):
            self.assertEqual(text.count(f"--dist-init-addr {port}\n"), 1, port)
        ports = re.findall(r"--dist-init-addr (127\.0\.0\.1:\d+)\n", text)
        self.assertEqual(len(ports), len(set(ports)))
        self.assertEqual(len(ports), 6)
        instances = re.findall(r"^                      instance: \"([0-9a-z]+)\"$", text, flags=re.MULTILINE)
        self.assertEqual(sorted(instances), sorted(["1", "2", "1a", "1b", "2a", "2b"]))

    def test_tp2_rollout_services_are_required(self) -> None:
        for name in (R1A, R1B):
            start, end = tp2_bounds(self.valid, name)
            with self.subTest(service=name):
                self.assert_fails(self.valid[:start] + self.valid[end:], "is missing services")

    def test_tp2_canary_rejects_missing_scrape_job_and_scrape_label_drift(self) -> None:
        job = generator.tp2_scrape_job("2a")
        self.assertEqual(self.valid.count(job), 1)
        self.assert_fails(self.valid.replace(job, "", 1), f"missing sglang-{R2A} scrape job")
        for before, after, message in (
            (f"                      instance: \"2a\"\n", "                      instance: \"2\"\n", "scrape label instance"),
            ('                      gpu_pair: "4-5"\n', '                      gpu_pair: "0-1"\n', "scrape label gpu_pair"),
            (f"                      config_variant: \"{TP2_VARIANT}\"\n", '                      config_variant: "x"\n', "scrape label config_variant"),
            (f"['{R2A}:8000']", "['model-sg-glm53-w4afp8-tp4-r2:8000']", f"sglang-{R2A} must scrape"),
        ):
            with self.subTest(message=message):
                self.assert_fails(replace_nth(self.valid, before, 0, after), message)

    def test_tp2_canary_services_are_required(self) -> None:
        start, end = tp2_bounds(self.valid, R2B)
        self.assert_fails(self.valid[:start] + self.valid[end:], "is missing services")

    def test_rejects_missing_or_inconsistent_observability(self) -> None:
        required = "(opt-in observability is required on every replica)"
        cases = (
            # SGLANG_GHOST_CACHE dropped from one replica only: r1 (the anchor) or r2 (its own list).
            ("\n    - SGLANG_GHOST_CACHE=1\n", "\n", 0, f"model-sg-glm53-w4afp8-tp4-r1 must set SGLANG_GHOST_CACHE=1 {required}"),
            ("\n      - SGLANG_GHOST_CACHE=1\n", "\n", 0, f"model-sg-glm53-w4afp8-tp4-r2 must set SGLANG_GHOST_CACHE=1 {required}"),
            ("\n      - SGLANG_KV_TIER_METRICS=1\n", "\n", 0, f"model-sg-glm53-w4afp8-tp4-r2 must set SGLANG_KV_TIER_METRICS=1 {required}"),
            ("SGLANG_GHOST_CACHE_REPLICA=r2\n", "SGLANG_GHOST_CACHE_REPLICA=r1\n", 0, "must use distinct SGLANG_GHOST_CACHE_REPLICA names"),
            ("SGLANG_GHOST_CACHE_KEY_FILE=/ghost/key\n", "SGLANG_GHOST_CACHE_KEY_FILE=/ghost/r2-key\n", 1,
             "replicas must share one SGLANG_GHOST_CACHE_KEY_FILE"),
            # The gpu02 TP2 pair: r2b loses the ghost cache, r2a reuses r2b's name.
            ("\n      - SGLANG_GHOST_CACHE=1\n", "\n", 2, f"{R2B} must set SGLANG_GHOST_CACHE=1 {required}"),
            ("SGLANG_GHOST_CACHE_REPLICA=r2a\n", "SGLANG_GHOST_CACHE_REPLICA=r2b\n", 0, "must use distinct SGLANG_GHOST_CACHE_REPLICA names"),
            (f'nearai.otel.config_variant: "{TP2_VARIANT}"', f'nearai.otel.config_variant: "{TP2_VARIANT.removesuffix("-obs-v1")}"', 1,
             "nearai.otel.config_variant must be"),
            ("\n    - ghost:/ghost\n", "\n", 0, "must mount ghost:/ghost"),
            ('"--port", "9464"', '"--port", "9465"', 0, "glm53-ghost-aggregator must run python3 -m"),
            (f"    image: {generator.IMAGE}\n    container_name: glm53-ghost-aggregator\n",
             f"    image: {UNKNOWN_IMAGE}\n    container_name: glm53-ghost-aggregator\n", 0, "glm53-ghost-aggregator image must be the engines' image"),
            ("      type: tmpfs\n", "      type: none\n", 0, "must declare the ghost volume as tmpfs"),
            ("['glm53-ghost-aggregator:9464']", "['glm53-ghost-aggregator:9465']", 0, "must scrape glm53-ghost-aggregator:9464"),
            (f'nearai.otel.config_variant: "{R2_VARIANT}"', f'nearai.otel.config_variant: "{R2_VARIANT.removesuffix("-obs-v1")}"', 0,
             "nearai.otel.config_variant must be"),
        )
        for before, after, index, message in cases:
            with self.subTest(mutation=message, occurrence=index):
                self.assert_fails(replace_nth(self.valid, before, index, after), message)

    def test_dcgm_validator_covers_the_file(self) -> None:
        needle = "image: nvcr.io/nvidia/k8s/dcgm-exporter@sha256:613ab03c11d442fd960ff515f547e9921537454a712d08160bc8f677f89f1c35"
        mutated = self.replace_once(needle, "image: nvcr.io/nvidia/k8s/dcgm-exporter@sha256:" + "0" * 64)
        self.assert_fails(mutated, "GLM-5.3-Flash-SGL-TP4-W4AFP8-LongContext.yaml", DCGM_VALIDATOR)


CANARY_DOC = ROOT / "docs/gpu02-glm53-2xtp2-memopt-canary.md"
ROLLOUT_DOC = ROOT / "docs/long-context-glm53-2xtp2-rollout.md"
R1_TP4 = "http://model-sg-glm53-w4afp8-tp4-r1:8000"
R2_TP4 = "http://model-sg-glm53-w4afp8-tp4-r2:8000"
R1A_URL = "http://model-sg-glm53-w4afp8-tp2-r1a:8000"
R1B_URL = "http://model-sg-glm53-w4afp8-tp2-r1b:8000"
R2A_URL = "http://model-sg-glm53-w4afp8-tp2-r2a:8000"
R2B_URL = "http://model-sg-glm53-w4afp8-tp2-r2b:8000"


def _defined_services(text: str) -> set[str]:
    services = text[text.index("\nservices:\n") :]
    return set(re.findall(r"^  ([a-z0-9-]+):$", services, flags=re.MULTILINE))


def _interpolate(value: str, env: dict[str, str]) -> str:
    """Compose's ${NAME:-default}: the default when NAME is unset or empty."""

    def resolve(match: re.Match[str]) -> str:
        return env.get(match.group(1)) or match.group(2)

    return re.sub(r"\$\{([A-Z0-9_]+):-([^}]*)\}", resolve, value)


class ProxyPoolRenderTest(unittest.TestCase):
    """The effective proxy pool on each host and stage (override unset = TP4 r1 + r2)."""

    def setUp(self) -> None:
        self.text = TARGET.read_text()
        lines = [line.strip() for line in self.text.splitlines() if line.strip().startswith("- VLLM_BACKEND_URLS=")]
        self.assertEqual(len(lines), 1, "exactly one proxy pool line")
        self.expression = lines[0].removeprefix("- VLLM_BACKEND_URLS=")
        self.services = _defined_services(self.text)

    def assert_pool(self, pool: str, expected: str) -> None:
        self.assertEqual(pool, expected)
        for url in pool.split(","):
            host = re.fullmatch(r"http://([a-z0-9-]+):8000", url)
            self.assertIsNotNone(host, url)
            self.assertIn(host.group(1), self.services, f"{url} must name a service in {generator.TARGET}")

    def test_unset_or_empty_override_is_the_tp4_pool(self) -> None:
        # A host that has not converted (gpu23 today, gpu02 before #332, and any host in rollback)
        # leaves the variable unset or empty and keeps exactly r1 + r2.
        for env in ({}, {"GLM53_BACKEND_URLS": ""}):
            self.assert_pool(_interpolate(self.expression, env), generator.BACKEND_URLS_R1_R2)

    def test_every_host_and_stage_pool_names_defined_services_and_matches_its_gpus(self) -> None:
        expected = {
            ("gpu02", "r2-pair"): (R1_TP4, R2A_URL, R2B_URL),
            ("gpu02", "all-tp2"): (R1A_URL, R1B_URL, R2A_URL, R2B_URL),
            ("gpu23", "r2-pair"): (R1_TP4, R2A_URL, R2B_URL),
            ("gpu23", "all-tp2"): (R1A_URL, R1B_URL, R2A_URL, R2B_URL),
        }
        seen = {(host, stage) for host, stages in generator.HOST_POOLS.items() for stage in stages}
        self.assertEqual(seen, set(expected))
        self.assertNotIn("gpu13", generator.HOST_POOLS)  # gpu13 deploys small-models.yaml with a fixed pool, not this file
        for (host, stage), urls in expected.items():
            with self.subTest(host=host, stage=stage):
                value = generator.HOST_POOLS[host][stage]
                self.assert_pool(_interpolate(self.expression, {"GLM53_BACKEND_URLS": value}), ",".join(urls))
                # No pool may contain both a TP4 replica and the TP2 pair on the same GPUs.
                self.assertFalse(R1_TP4 in urls and R1A_URL in urls)
                self.assertFalse(R2_TP4 in urls and R2A_URL in urls)

    def test_named_pool_constants_match_host_pools(self) -> None:
        self.assertEqual(generator.GPU02_BACKEND_URLS, generator.HOST_POOLS["gpu02"]["r2-pair"])
        self.assertEqual(generator.GPU23_R2_PAIR_BACKEND_URLS, generator.HOST_POOLS["gpu23"]["r2-pair"])
        self.assertEqual(generator.GPU02_ALL_TP2_BACKEND_URLS, generator.HOST_POOLS["gpu02"]["all-tp2"])
        self.assertEqual(generator.GPU23_ALL_TP2_BACKEND_URLS, generator.HOST_POOLS["gpu23"]["all-tp2"])

    def test_runbooks_set_the_tested_values_and_scope_the_services(self) -> None:
        canary = CANARY_DOC.read_text()
        self.assertIn(f"GLM53_BACKEND_URLS={generator.GPU02_BACKEND_URLS}", canary)
        self.assertIn(
            'services: ["model-sg-glm53-w4afp8-tp2-r2a", "model-sg-glm53-w4afp8-tp2-r2b", "proxy-glm53", "otelcol-contrib"]',
            canary,
        )
        self.assertIn('services: ["model-sg-glm53-w4afp8-tp4-r2"]', canary)
        rollout = ROLLOUT_DOC.read_text()
        for host, stages in generator.HOST_POOLS.items():
            for stage, value in stages.items():
                with self.subTest(host=host, stage=stage):
                    self.assertIn(f"`{host}` `{stage}`: `GLM53_BACKEND_URLS={value}`", rollout)
        for services in (
            'services: ["model-sg-glm53-w4afp8-tp4-r1"]',
            'services: ["model-sg-glm53-w4afp8-tp2-r1a", "model-sg-glm53-w4afp8-tp2-r1b", "proxy-glm53", "otelcol-contrib"]',
            'services: ["model-sg-glm53-w4afp8-tp2-r2a", "model-sg-glm53-w4afp8-tp2-r2b", "proxy-glm53", "otelcol-contrib"]',
            'services: ["model-sg-glm53-w4afp8-tp4-r2"]',
        ):
            with self.subTest(services=services):
                self.assertIn(services, rollout)
        # Never an unscoped or empty services list.
        self.assertNotIn('services: []', rollout)

    @unittest.skipUnless(shutil.which("docker"), "docker CLI not available")
    def test_docker_compose_render(self) -> None:
        version = subprocess.run(["docker", "compose", "version"], capture_output=True, text=True, check=False)
        if version.returncode != 0:
            self.skipTest("docker compose plugin not available")

        def rendered_pool(override: str | None) -> str:
            env = {k: v for k, v in os.environ.items() if k != "GLM53_BACKEND_URLS"}
            # Deploy-time secrets marked ${VAR:?...}: dummy values, as the CI compose render does.
            for name in set(re.findall(r"\$\{([A-Z0-9_]+):?\?", self.text)):
                env.setdefault(name, "ci-dummy")
            if override is not None:
                env["GLM53_BACKEND_URLS"] = override
            result = subprocess.run(
                ["docker", "compose", "-f", str(TARGET), "config", "--format", "json"],
                capture_output=True,
                text=True,
                check=False,
                env=env,
            )
            self.assertEqual(result.returncode, 0, result.stderr[-2000:])
            return json.loads(result.stdout)["services"]["proxy-glm53"]["environment"]["VLLM_BACKEND_URLS"]

        self.assert_pool(rendered_pool(None), generator.BACKEND_URLS_R1_R2)
        for host, stages in generator.HOST_POOLS.items():
            for stage, value in stages.items():
                with self.subTest(host=host, stage=stage):
                    self.assert_pool(rendered_pool(value), value)


if __name__ == "__main__":
    _ = unittest.main()
