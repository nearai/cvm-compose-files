#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: python3 -m unittest scripts.test_glm53_w4afp8_long_context (needs ruby for the validator cases)
"""The generated W4AFP8 + HiCache long-context file, its generator and its validator contract."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts import prepare_glm53_w4afp8_long_context as generator

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
V1_IMAGE = "docker.io/nearaidev/sglang@sha256:fde25985aea3ebabf1eb581ae21d53be8540e32933eef942ee8b962a1bfbea20"
UNKNOWN_IMAGE = "docker.io/nearaidev/sglang@sha256:" + "0" * 64
R1_VARIANT = "fc91d24-long-context-w4afp8-c8192-qsplit-offloop-v3-hicache-cuda-host-pooled-v1-admission-reserve-disabled-pool-clamp-pdi1-h200-tp4-ep4-eagle-adaptive-5-1-6-strict-budget8192"
# r2 also carries the host650g marker of the HiCache host-tier canary (its 650 GiB budget).
R2_VARIANT = "fc91d24-long-context-w4afp8-c8192-qsplit-offloop-v3-hicache-cuda-host-pooled-v1-host650g-admission-reserve-disabled-pool-clamp-pdi2-h200-tp4-ep4-eagle-adaptive-5-1-6-strict-budget8192"
R1_WITHOUT_OFFLOOP_VARIANT = R1_VARIANT.replace("-offloop-v3", "")
R2_WITHOUT_OFFLOOP_VARIANT = R2_VARIANT.replace("-offloop-v3", "")


PROXY_POOL = "VLLM_BACKEND_URLS=${GLM53_BACKEND_URLS:-http://model-sg-glm53-w4afp8-tp4-r1:8000,http://model-sg-glm53-w4afp8-tp4-r2:8000}"
TP2_VARIANT = generator.TP2_VARIANT
R2A = "model-sg-glm53-w4afp8-tp2-r2a"
R2B = "model-sg-glm53-w4afp8-tp2-r2b"


def replace_nth(text: str, needle: str, index: int, replacement: str) -> str:
    positions = []
    start = text.find(needle)
    while start != -1:
        positions.append(start)
        start = text.find(needle, start + 1)
    target = positions[index]
    return text[:target] + replacement + text[target + len(needle):]


def rendered_engine_sections() -> tuple[str, str, str]:
    rendered = generator.generate((ROOT / generator.SOURCE).read_text())
    return rendered, generator.section(rendered, "x-sg-glm53-flash-common: &sg-glm53-flash-common\n", "\nx-dcgm-common: &dcgm-common\n", "shared engine anchor")[2], generator.section(rendered, "  model-sg-glm53-w4afp8-tp4-r2:\n", "\n  # --- GLM-5.3-Flash 2xTP2 memory-optimized canary replica 2a", "replica 2 service")[2]


class GeneratedFileTest(unittest.TestCase):
    def test_both_replicas_use_the_released_v3_image_identity(self) -> None:
        self.assertEqual(generator.REPLICA_IMAGE, {1: RELEASED_V3_IMAGE, 2: RELEASED_V3_IMAGE})
        self.assertEqual(generator.REPLICA_IMAGE_LABEL, {1: RELEASED_V3_LABEL, 2: RELEASED_V3_LABEL})

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
        self.assertIn(f"\n  image: {RELEASED_V3_IMAGE}\n", anchor)
        self.assertNotIn(PREVIOUS_R1_V2_IMAGE, anchor)
        self.assertIn(f"\n    image: {RELEASED_V3_IMAGE}\n", r2)
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
            service = generator.section(
                rendered, f"  {generator.TP2_SERVICE_PREFIX}{suffix}:\n", "\n  # --- GLM-5.3-Flash 2xTP2" if suffix == "2a" else "\n  # Explicit operator-only", suffix
            )[2]
            with self.subTest(replica=suffix):
                for flag in ("--tp-size 2", "--ep-size 2", "--mem-fraction-static 0.86", "--max-mamba-cache-size 330", "--max-running-requests 24",
                             "--max-queued-requests 8", "--speculative-num-steps 4", "--speculative-eagle-topk 1", "--speculative-num-draft-tokens 5",
                             "--mamba-ssm-dtype bfloat16", "--chunked-prefill-size 8192", "--hicache-write-policy write_through",
                             f"--dist-init-addr {spec['dist_init']}"):
                    self.assertIn(f"\n        {flag}\n", service)
                for forbidden in ("--speculative-adaptive", "--disable-overlap-schedule", "ADMISSION_RESERVE"):
                    self.assertNotIn(forbidden, service)
                self.assertIn(f"${{{spec['budget_var']}:-325GiB}}", service)
                self.assertEqual(2 * 325, 650)  # r2 default is 650 GiB



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
        self.assertEqual(self.valid.count(before), 3, before)
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
            (candidate.replace(f"    image: {RELEASED_V3_IMAGE}\n", f"    image: {PREVIOUS_R1_V2_IMAGE}\n", 1), "model-sg-glm53-w4afp8-tp4-r2 image must be"),
            (candidate.replace(f"  image: {RELEASED_V3_IMAGE}\n", f"  image: {PREVIOUS_R1_V2_IMAGE}\n", 1), "model-sg-glm53-w4afp8-tp4-r1 image must be"),
            (candidate.replace(f"  image: {RELEASED_V3_IMAGE}\n", f"  image: {V1_IMAGE}\n", 1), "does not run an approved split-capable image"),
            (candidate.replace(f"  image: {RELEASED_V3_IMAGE}\n", f"  image: {UNKNOWN_IMAGE}\n", 1), "does not run an approved split-capable image"),
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
            ('      nearai.otel.model_path: "graphistry/GLM-5.3-Flash-W4AFP8"\n', '      nearai.otel.model_path: "zai-org/GLM-5.3-Flash"\n', 4, "dcgm-glm53 nearai.otel.model_path"),
        )
        for needle, replacement, index, message in cases:
            with self.subTest(mutation=message):
                self.assert_fails(replace_nth(self.valid, needle, index, replacement), message)

    def test_tp2_canary_rejects_flag_and_memory_drift(self) -> None:
        argv = "argv must be the memory-optimized TP2 argv exactly"
        for name in (R2A, R2B):
            start = self.valid.index(f"  {name}:\n")
            end = self.valid.index("\n  # ---", start + 10) if name == R2A else self.valid.index("\n  # Explicit operator-only", start)
            block = self.valid[start:end]
            cases = (
                ("--mem-fraction-static 0.86", "--mem-fraction-static 0.80", argv),
                ("--max-mamba-cache-size 330", "--max-mamba-cache-size 165", argv),
                ("--max-running-requests 24", "--max-running-requests 32", argv),
                ("--max-queued-requests 8", "--max-queued-requests 16", argv),
                ("--speculative-num-steps 4", "--speculative-num-steps 5", argv),
                ("--speculative-num-draft-tokens 5", "--speculative-num-draft-tokens 6", argv),
                ("--tp-size 2", "--tp-size 4", argv),
                ("--hicache-write-policy write_through", "--hicache-write-policy write_through_selective", argv),
                ("--chunked-prefill-size 8192", "--chunked-prefill-size 32768", "must not enable 32K prefill chunks"),
                ("--mamba-ssm-dtype bfloat16\n", "--mamba-ssm-dtype bfloat16\n        --speculative-adaptive\n", "must not set --speculative-adaptive"),
                ("--mamba-ssm-dtype bfloat16\n", "--mamba-ssm-dtype bfloat16\n        --disable-overlap-schedule\n", "must not set --disable-overlap-schedule"),
            )
            for before, after, message in cases:
                with self.subTest(replica=name, mutation=after.strip()[:50]):
                    self.assertEqual(block.count(before), 1, before)
                    self.assert_fails(self.valid[:start] + block.replace(before, after, 1) + self.valid[end:], message)

    def test_tp2_canary_rejects_environment_pinning_and_telemetry_drift(self) -> None:
        for name, devices, wrong_devices, budget, instance in (
            (R2A, '["4","5"]', '["4","6"]', "${GLM53_R2A_HICACHE_RAM_BUDGET:-325GiB}", "2a"),
            (R2B, '["6","7"]', '["0","1"]', "${GLM53_R2B_HICACHE_RAM_BUDGET:-325GiB}", "2b"),
        ):
            start = self.valid.index(f"  {name}:\n")
            end = self.valid.index("\n  # ---", start + 10) if name == R2A else self.valid.index("\n  # Explicit operator-only", start)
            block = self.valid[start:end]
            cases = (
                (f"device_ids: {devices}", f"device_ids: {wrong_devices}", "must use GPU device_ids"),
                ("    environment:\n", "    environment:\n      - SGLANG_CHUNKED_PREFILL_ADMISSION_RESERVE=4096\n", "must not set admission-reserve environment"),
                (budget, budget.replace("325GiB", "650GiB"), "environment must be the long-context environment plus per-replica"),
                ("      - SGLANG_DSA_INDEXER_QSPLIT=1\n", "", "environment must be the long-context environment plus per-replica"),
                (f'nearai.otel.instance: "{instance}"', 'nearai.otel.instance: "9"', "nearai.otel.instance must be"),
                ("nearai.otel.gpu_pair:", "nearai.otel.gpu_pairx:", "nearai.otel.gpu_pair must be"),
                (f'nearai.otel.config_variant: "{TP2_VARIANT}"', 'nearai.otel.config_variant: "x"', "nearai.otel.config_variant must be"),
                ('nearai.otel.deployment: "glm53-flash-sgl-tp4"', 'nearai.otel.deployment: "other"', "nearai.otel.deployment"),
                (f"image: {RELEASED_V3_IMAGE}\n", f"image: {PREVIOUS_R1_V2_IMAGE}\n", "image must be"),
                ("    container_name: " + name + "\n", "    container_name: " + name + '\n    restart: "no"\n', "must share one runtime configuration outside command and environment"),
            )
            for before, after, message in cases:
                with self.subTest(replica=name, message=message):
                    self.assertGreaterEqual(block.count(before), 1, before)
                    self.assert_fails(self.valid[:start] + block.replace(before, after, 1) + self.valid[end:], message)

    def test_tp2_canary_rejects_duplicate_dist_init_port_and_gpu_overlap(self) -> None:
        self.assert_fails(self.replace_once("        --dist-init-addr 127.0.0.1:29513\n", "        --dist-init-addr 127.0.0.1:29512\n"), "is used by more than one engine")
        self.assert_fails(self.replace_once("        --dist-init-addr 127.0.0.1:29512\n", "        --dist-init-addr 127.0.0.1:29510\n"), "is used by more than one engine")
        self.assert_fails(self.replace_once('device_ids: ["6","7"]', 'device_ids: ["4","5"]'), "must not share GPUs")

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
        start = self.valid.index(f"  {R2B}:\n")
        end = self.valid.index("\n  # Explicit operator-only", start)
        self.assert_fails(self.valid[:start] + self.valid[end:], "is missing services")

    def test_dcgm_validator_covers_the_file(self) -> None:
        needle = "image: nvcr.io/nvidia/k8s/dcgm-exporter@sha256:613ab03c11d442fd960ff515f547e9921537454a712d08160bc8f677f89f1c35"
        mutated = self.replace_once(needle, "image: nvcr.io/nvidia/k8s/dcgm-exporter@sha256:" + "0" * 64)
        self.assert_fails(mutated, "GLM-5.3-Flash-SGL-TP4-W4AFP8-LongContext.yaml", DCGM_VALIDATOR)


if __name__ == "__main__":
    _ = unittest.main()
