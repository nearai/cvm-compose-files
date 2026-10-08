#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: python3 -m unittest scripts.test_glm53_v7_fleet (needs ruby for the validator cases)
"""The two v7 FLEET files (base tier, long tier): one config per tier, profiling off, the dashboard labels untouched, the runbook.

Imported by test_glm53_w4afp8_tp2x4.py and test_glm53_w4afp8_long_context.py so the existing CI steps run it.
"""

import re
import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts import prepare_glm53_w4afp8_long_context as long_generator
from scripts import prepare_glm53_w4afp8_tp2x4 as base_generator

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = Path("scripts/validate_glm53_prod_config.rb")
DCGM_VALIDATOR = Path("scripts/validate_glm53_dcgm_metrics.rb")
RUNBOOK = ROOT / "docs/glm53-v7-fleet-rollout.md"

V7_IMAGE = "docker.io/nearaidev/sglang@sha256:fa730e6e62b2ae8058114ce540487ade33ab93bc42b1179ae78edc92bd563fc5"
V6_IMAGE = "docker.io/nearaidev/sglang@sha256:9c6ddd4319c4ab00e351d8650459e68b8830e36ffcc029d67fa5e19d0ac3ed17"
FLEET_ENV = [
    "SGLANG_PREPROCESS_WORKERS=4",
    "SGLANG_PREPROCESS_TIMEOUT_S=60",
    "SGLANG_PREPROCESS_LOG_SLOW_S=5",
    "SGLANG_TOOL_SCHEMA_MAX_DEPTH=32",
    "SGLANG_TOOL_SCHEMA_MAX_NODES=25000",
]
# gpu03 r3 (arm B', prod/GLM-5.3-Flash-SGL-TP2x4-W4AFP8-V7-HiCacheOff-r3.yaml on pranavraja99/glm53-kvshare-ab), live argv.
R3_ARGV = shlex.split("""
    sglang serve
    --model-path /root/.cache/huggingface/hub/models--graphistry--GLM-5.3-Flash-W4AFP8/snapshots/99f1fa70408c52b007d4fd69e02e5a522422e755
    --served-model-name z-ai/glm-5.3-flash --tp-size 2 --ep-size 2 --mem-fraction-static 0.86
    --max-running-requests 64 --max-queued-requests 8 --enable-priority-scheduling --disable-priority-preemption
    --chunked-prefill-size 8192 --max-prefill-tokens 32768 --prefill-decode-interval 2 --cuda-graph-max-bs-decode 64
    --dsa-prefill-backend flashmla_kv --dsa-decode-backend flashmla_kv --kv-cache-dtype fp8_e4m3
    --speculative-algorithm EAGLE --speculative-num-steps 4 --speculative-eagle-topk 1 --speculative-num-draft-tokens 5
    --reasoning-parser glm45 --enable-strict-thinking --grammar-backend xgrammar --tool-call-parser glm47
    --chat-template /root/.cache/huggingface/hub/models--zai-org--GLM-5.3-Flash/snapshots/3f1971b7b5f7a528c9c4ef6212c8785298a8c24a/chat_template.jinja
    --context-length 1048576 --dist-init-addr 127.0.0.1:29510 --watchdog-timeout 1800 --host 0.0.0.0 --port 8000
    --enable-metrics --enable-cache-report --log-requests-level 0 --disable-fast-image-processor
    --limit-mm-data-per-request '{"image": 64}' --max-mamba-cache-size 380 --mamba-ssm-dtype bfloat16 --disable-overlap-schedule
""")
LONG_ARGV = """
    sglang serve
    --model-path /root/.cache/huggingface/hub/models--graphistry--GLM-5.3-Flash-W4AFP8/snapshots/99f1fa70408c52b007d4fd69e02e5a522422e755
    --served-model-name z-ai/glm-5.3-flash --tp-size 2 --ep-size 2 --mem-fraction-static 0.86
    --max-running-requests 16 --max-queued-requests 4 --enable-priority-scheduling --disable-priority-preemption
    --chunked-prefill-size 8192 --max-prefill-tokens 32768 --prefill-decode-interval 2 --cuda-graph-max-bs-decode 16
    --dsa-prefill-backend flashmla_kv --dsa-decode-backend flashmla_kv --kv-cache-dtype fp8_e4m3
    --speculative-algorithm EAGLE --speculative-num-steps 4 --speculative-eagle-topk 1 --speculative-num-draft-tokens 5
    --reasoning-parser glm45 --enable-strict-thinking --grammar-backend xgrammar --tool-call-parser glm47
    --chat-template /root/.cache/huggingface/hub/models--zai-org--GLM-5.3-Flash/snapshots/3f1971b7b5f7a528c9c4ef6212c8785298a8c24a/chat_template.jinja
    --context-length 1048576 --dist-init-addr 127.0.0.1:29512 --watchdog-timeout 1800 --host 0.0.0.0 --port 8000
    --enable-metrics --enable-cache-report --log-requests-level 0 --disable-fast-image-processor
    --limit-mm-data-per-request '{"image": 64}' --enable-hierarchical-cache --hicache-write-policy write_through
    --hicache-io-backend direct --hicache-mem-layout page_first_direct --max-mamba-cache-size 330 --mamba-ssm-dtype bfloat16
    --disable-overlap-schedule
"""
BASE_FILE = ROOT / base_generator.TARGET
LONG_FILE = ROOT / long_generator.TARGET
BASE_FIXTURE = ROOT / "scripts/fixtures/pre-v7/GLM-5.3-Flash-SGL-TP2x4-W4AFP8.pre-v7.yaml"
LONG_FIXTURE = ROOT / "scripts/fixtures/pre-v7/GLM-5.3-Flash-SGL-TP4-W4AFP8-LongContext.pre-v7.yaml"
BASE_ENGINES = [f"model-sg-glm53-w4afp8-tp2-r{n}" for n in (1, 2, 3, 4)]
LONG_ENGINES = [f"model-sg-glm53-w4afp8-tp2-r{n}" for n in ("1a", "1b", "2a", "2b")]
LABEL_KEYS = ("precision", "engine_image", "config_variant")


def code(text: str) -> str:
    """The text without comment lines."""
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))


def service_block(text: str, name: str) -> str:
    start = text.index(f"\n  {name}:\n") + 1
    match = re.search(r"\n  [^ \n#]", text[start + 1:])
    return text[start: start + 1 + match.start()] if match else text[start:]


def anchor_block(text: str, anchor: str) -> str:
    text = "\n" + text
    start = text.index(f"\n{anchor}: &") + 1
    match = re.search(r"\nx-|\nservices:", text[start + 1:])
    return text[start: start + 1 + match.start()]


def argv_of(block: str) -> list[str]:
    body = block.split("command: >\n", 1)[1].splitlines()
    lines = []
    for line in body:
        if line.strip() and len(line) - len(line.lstrip(" ")) < 6:
            break
        lines.append(line.strip())
    return shlex.split(" ".join(line for line in lines if line))


def env_of(block: str) -> list[str]:
    env = block.split("    environment:\n", 1)[1].split("\n    depends_on:", 1)[0]
    return [line.strip()[2:] for line in env.splitlines() if line.strip().startswith("- ")]


def scrub(text: str) -> str:
    """Drop the three telemetry values the rollout is allowed to change, everywhere they appear."""
    lines = []
    for line in text.splitlines():
        if re.match(r"^\s+(precision|engine_image|config_variant): ", line) or re.match(r"^\s+nearai\.otel\.(engine_image|config_variant): ", line):
            continue
        lines.append(re.sub(r'"(precision|engine_image|config_variant):[^"]*",?', "", line))
    return "\n".join(lines)


def jobs_of(text: str) -> dict[str, str]:
    return {m.group(1): m.group(0) for m in re.finditer(r"- job_name: (\S+)\n(?:(?! {14}- job_name:).*\n)*", text)}


def selector_lines(text: str) -> list[str]:
    """Every label line a dashboard can select on, in file order, minus the three allowed telemetry values."""
    return [line for line in scrub(text).splitlines() if "nearai.otel." in line or "datadoghq.ad.logs" in line]


class FleetFilesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.base = BASE_FILE.read_text()
        self.long = LONG_FILE.read_text()

    def test_profiling_is_off_everywhere(self) -> None:
        for text in (self.base, self.long):
            self.assertNotRegex(code(text), r"NEAR_SELF_PROFILE|NEAR_PROFILE")

    def test_base_engines_are_the_gpu03_r3_config_without_hicache(self) -> None:
        anchor = anchor_block(self.base, "x-sg-glm53-flash-candidate")
        self.assertEqual(argv_of(anchor), R3_ARGV)
        self.assertEqual(sum(1 for t in R3_ARGV if t.startswith(("--hicache", "--enable-hierarchical-cache"))), 0)
        self.assertEqual(self.base.count("    <<: *sg-glm53-flash-candidate\n"), 4)
        self.assertEqual(self.base.count("    <<: *sg-glm53-flash-common\n"), 0)
        # No replica overrides the image or the command, so all four inherit the v7 anchor.
        for name in BASE_ENGINES:
            block = service_block(self.base, name)
            self.assertNotIn("    image:", block)
            self.assertNotIn("    command:", block)
        self.assertEqual(anchor_block(self.base, "x-sg-glm53-flash-common").count(f"  image: {V7_IMAGE}\n"), 1)
        self.assertNotIn(V6_IMAGE.split("@")[1], code(self.base).replace(f"# {V6_IMAGE}", ""))

    def test_base_engines_carry_the_five_v7_variables_and_the_hicache_variables_stay_unread(self) -> None:
        for name in BASE_ENGINES:
            env = env_of(service_block(self.base, name)) if "    environment:\n" in service_block(self.base, name) else None
            if env is None:  # r1 inherits the anchor environment
                anchor = anchor_block(self.base, "x-sg-glm53-flash-common")
                env = [line.strip()[2:] for line in anchor.split("  environment:\n", 1)[1].split("  restart:", 1)[0].splitlines() if line.strip().startswith("- ")]
            with self.subTest(replica=name):
                self.assertEqual(env[-len(FLEET_ENV):], FLEET_ENV)
                self.assertEqual([e for e in env if e.startswith("SGLANG_PREPROCESS_") or e.startswith("SGLANG_TOOL_SCHEMA_")], FLEET_ENV)
                self.assertIn("SGLANG_HICACHE_RAM_BUDGET=${GLM53_HICACHE_RAM_BUDGET:-325GiB}", env)

    def test_long_tp2_engines_are_the_r2a_canary_config(self) -> None:
        # The argv the r2a canary ran in production (tee-bench exp 32), with the rendezvous port removed.
        canary_argv = shlex.split(LONG_ARGV)
        for name in LONG_ENGINES:
            block = service_block(self.long, name)
            argv = argv_of(block)
            with self.subTest(replica=name):
                self.assertEqual([t for t in argv if not t.startswith("127.0.0.1")], [t for t in canary_argv if not t.startswith("127.0.0.1")])
                self.assertIn(f"    image: {V7_IMAGE}\n", block)
                value = lambda flag: argv[argv.index(flag) + 1]
                self.assertEqual((value("--kv-cache-dtype"), value("--dsa-prefill-backend"), value("--dsa-decode-backend")), ("fp8_e4m3", "flashmla_kv", "flashmla_kv"))
                self.assertEqual((value("--max-running-requests"), value("--max-queued-requests"), value("--cuda-graph-max-bs-decode")), ("16", "4", "16"))
                self.assertEqual(argv.count("--disable-overlap-schedule"), 1)
                self.assertIn("--enable-hierarchical-cache", argv)
                self.assertEqual(value("--hicache-write-policy"), "write_through")
                env = env_of(block)
                self.assertEqual(env[-len(FLEET_ENV):], FLEET_ENV)
        for tp4 in ("model-sg-glm53-w4afp8-tp4-r1", "model-sg-glm53-w4afp8-tp4-r2"):
            block = service_block(self.long, tp4)
            self.assertNotIn("SGLANG_PREPROCESS", block)
            self.assertNotIn("fa730e6e62b2", block)

    def test_fleet_telemetry_values(self) -> None:
        engine = "fa730e6e62b2"
        self.assertEqual(engine, "fa730e6e62b2")
        for text, names in ((self.base, BASE_ENGINES), (self.long, LONG_ENGINES)):
            for name in names:
                block = service_block(text, name)
                job = jobs_of(text)[f"sglang-{name}"]
                with self.subTest(replica=name):
                    self.assertIn('"precision:int4-weights-fp8-activations-fp8-kv"', block)
                    self.assertIn(f'"engine_image:{engine}"', block)
                    self.assertIn(f'nearai.otel.engine_image: "{engine}"', block)
                    self.assertIn('precision: "int4-weights-fp8-activations-fp8-kv"', job)
                    self.assertIn(f'engine_image: "{engine}"', job)
        self.assertEqual(self.base.count(base_generator.CANDIDATE_VARIANT), 12)  # 4 x (log tag, label, scrape job)
        for token in ("hicacheoff", "mamba380", "fp8kv", "mr64", "-v7"):
            self.assertIn(token, base_generator.CANDIDATE_VARIANT)
        self.assertEqual(self.long.count(long_generator.TP2_VARIANT), 12)
        for token in ("fp8kv", "mr16q4", "-v7"):
            self.assertIn(token, long_generator.TP2_VARIANT)

    def test_the_dashboard_labels_are_untouched_for_every_service_and_scrape_job(self) -> None:
        # Grafana glm53-flash-prod selects on deployment, host_machine, host, service, model, instance and server_address
        # (the scrape target's host:port), plus service=dcgm-exporter / vllm-proxy. Only precision, engine_image and
        # config_variant may differ from the files as they were before the rollout.
        for new, fixture in ((self.base, BASE_FIXTURE), (self.long, LONG_FIXTURE)):
            old = fixture.read_text()
            self.assertEqual(selector_lines(new), selector_lines(old))
            new_jobs, old_jobs = jobs_of(new), jobs_of(old)
            self.assertEqual(list(new_jobs), list(old_jobs))
            for key in old_jobs:
                self.assertEqual(scrub(new_jobs[key]), scrub(old_jobs[key]), key)
            self.assertEqual(
                [m for m in re.findall(r"^  ([a-z0-9-]+):$", new[new.index("\nservices:\n"):], re.MULTILINE)],
                [m for m in re.findall(r"^  ([a-z0-9-]+):$", old[old.index("\nservices:\n"):], re.MULTILINE)],
            )
        for text in (self.base, self.long):
            for needle in ('service: "dcgm-exporter"', 'service: "vllm-proxy"', 'service: "sglang"', 'service: "sglang-ghost-aggregator"',
                           "- job_name: dcgm-dcgm-glm53", "- job_name: inference-proxy-proxy-glm53", "- job_name: ghost-aggregator-glm53-ghost-aggregator",
                           "- job_name: otelcol-app", 'host_machine: "${CVM_HOST}"', 'model: "z-ai/glm-5.3-flash"'):
                self.assertIn(needle, text)
        self.assertIn('deployment: "glm53-flash-sgl-tp2x4"', self.base)
        self.assertNotIn('deployment: "glm53-flash-sgl-tp4"', self.base)
        self.assertIn('deployment: "glm53-flash-sgl-tp4"', self.long)

    def test_everything_outside_the_engines_is_unchanged_from_the_pre_rollout_files(self) -> None:
        # Routing, nginx, registrar, proxy, DCGM, collector pipelines, volumes: textually identical outside the engine anchors,
        # engine services, their scrape jobs, the ghost sidecar's image and the header.
        def strip(text: str, names: list[str], anchors: list[str]) -> str:
            text = text[text.index("x-logging-conf:"):]
            for name in names:
                text = text.replace(service_block(text, name), "")
            for anchor in anchors:
                text = text.replace(anchor_block(text, anchor), "")
            for key in list(jobs_of(text)):
                if key.startswith("sglang-model-sg-glm53"):
                    text = text.replace(jobs_of(text)[key], "")
            text = text.replace(V7_IMAGE, "IMAGE").replace(V6_IMAGE, "IMAGE")
            return text
        base_names = BASE_ENGINES
        self.assertEqual(
            strip(self.base, base_names, ["x-sg-glm53-flash-common", "x-sg-glm53-flash-candidate"]),
            strip(BASE_FIXTURE.read_text(), base_names, ["x-sg-glm53-flash-common", "x-sg-glm53-flash-candidate"]),
        )
        long_names = LONG_ENGINES
        self.assertEqual(strip(self.long, long_names, []), strip(LONG_FIXTURE.read_text(), long_names, []))


SMALL = ROOT / "prod/small-models.yaml"
SMALL_FIXTURE = ROOT / "scripts/fixtures/pre-v7/small-models.pre-v7.yaml"
GPU13_GLM_SERVICES = ("proxy-glm53", "model-sg-glm53-w4afp8-tp2-r1a", "model-sg-glm53-w4afp8-tp2-r1b", "glm53-ghost-aggregator", "dcgm-glm53")


class Gpu13FleetTest(unittest.TestCase):
    """gpu13 deploys prod/small-models.yaml: its two GLM replicas must be the long file's v7 TP2 replicas, and nothing else may move."""

    def setUp(self) -> None:
        self.small = SMALL.read_text()
        self.old = SMALL_FIXTURE.read_text()
        self.long = LONG_FILE.read_text()

    def gpu13(self, name: str) -> tuple[list[str], list[str], str]:
        """(argv, env, image) of a gpu13 GLM replica; r1a inherits the shared anchor, r1b carries its own copy."""
        block = service_block(self.small, name)
        anchor = anchor_block(self.small, "x-sg-glm53-flash-common")
        source = block if "    command: >" in block else anchor
        argv = argv_of(source)
        env_text = block if "    environment:\n" in block else anchor
        key = "    environment:\n" if "    environment:\n" in block else "  environment:\n"
        env = [line.strip()[2:] for line in env_text.split(key, 1)[1].split("restart:" if source is anchor else "\n    deploy:", 1)[0].splitlines() if line.strip().startswith("- ")]
        image = (re.search(r"^ *image: (\S+)$", block, re.MULTILINE) or re.search(r"^ *image: (\S+)$", anchor, re.MULTILINE)).group(1)
        return argv, env, image

    def test_gpu13_glm_argv_and_env_equal_the_long_file_tp2_replicas_except_host_items(self) -> None:
        def host_free(argv: list[str]) -> list[str]:
            return [t for t in argv if not t.startswith("127.0.0.1:")]

        def env_free(env: list[str]) -> list[str]:
            return sorted(e for e in env if not e.startswith("SGLANG_GHOST_CACHE_REPLICA="))
        long_argv = host_free(argv_of(service_block(self.long, "model-sg-glm53-w4afp8-tp2-r2a")))
        long_env = env_free(env_of(service_block(self.long, "model-sg-glm53-w4afp8-tp2-r2a")))
        for name in ("model-sg-glm53-w4afp8-tp2-r1a", "model-sg-glm53-w4afp8-tp2-r1b"):
            argv, env, image = self.gpu13(name)
            with self.subTest(replica=name):
                self.assertEqual(host_free(argv), long_argv)
                self.assertEqual(env_free(env), env_free(env_of(service_block(self.long, name))))
                self.assertEqual(env_free([e for e in env if not e.startswith("SGLANG_HICACHE_RAM_BUDGET=")]), env_free([e for e in env_of(service_block(self.long, "model-sg-glm53-w4afp8-tp2-r2a")) if not e.startswith("SGLANG_HICACHE_RAM_BUDGET=")]))
                self.assertEqual(image, V7_IMAGE)
                self.assertEqual(env[-len(FLEET_ENV):], FLEET_ENV)
                self.assertFalse([e for e in env if "NEAR_SELF_PROFILE" in e])
        self.assertIn("--disable-overlap-schedule", long_argv)
        self.assertEqual(long_env, env_free(env_of(service_block(self.long, "model-sg-glm53-w4afp8-tp2-r2a"))))

    def test_gpu13_keeps_its_host_specific_bits(self) -> None:
        for name, budget in (("model-sg-glm53-w4afp8-tp2-r1a", "GLM53_R1A_HICACHE_RAM_BUDGET"), ("model-sg-glm53-w4afp8-tp2-r1b", "GLM53_R1B_HICACHE_RAM_BUDGET")):
            argv, env, _ = self.gpu13(name)
            self.assertIn(f"SGLANG_HICACHE_RAM_BUDGET=${{{budget}:-325GiB}}", env)
            self.assertIn("SGLANG_DSA_INDEXER_QSPLIT=1", env)
        self.assertIn("127.0.0.1:29510", self.gpu13("model-sg-glm53-w4afp8-tp2-r1a")[0])
        self.assertIn("127.0.0.1:29511", self.gpu13("model-sg-glm53-w4afp8-tp2-r1b")[0])
        self.assertIn('device_ids: ["4","5"]', service_block(self.small, "model-sg-glm53-w4afp8-tp2-r1a"))
        self.assertIn('device_ids: ["6","7"]', service_block(self.small, "model-sg-glm53-w4afp8-tp2-r1b"))

    def test_non_glm_services_and_everything_outside_the_glm_blocks_are_byte_identical(self) -> None:
        names = re.findall(r"^  ([a-z0-9-]+):$", self.old[self.old.index("\nservices:\n"):], re.MULTILINE)
        self.assertEqual(names, re.findall(r"^  ([a-z0-9-]+):$", self.small[self.small.index("\nservices:\n"):], re.MULTILINE))
        for name in names:
            if name not in GPU13_GLM_SERVICES:
                with self.subTest(service=name):
                    self.assertEqual(service_block(self.small, name), service_block(self.old, name))
        for anchor in re.findall(r"^(x-[a-z0-9-]+): &", self.old, re.MULTILINE):
            if anchor != "x-sg-glm53-flash-common":
                self.assertEqual(anchor_block(self.small, anchor), anchor_block(self.old, anchor), anchor)
        # Scrape jobs of every non-GLM target are identical.
        old_jobs, new_jobs = jobs_of(self.old), jobs_of(self.small)
        self.assertEqual(list(old_jobs), list(new_jobs))
        for key in old_jobs:
            if "glm53" not in key:
                self.assertEqual(new_jobs[key], old_jobs[key], key)

    def test_dashboard_labels_are_unchanged_on_gpu13(self) -> None:
        def selectors(text: str) -> list[str]:
            return [l for l in selector_lines(text) if "max_running_requests" not in l]
        self.assertEqual(selectors(self.small), selectors(self.old))
        old_jobs, new_jobs = jobs_of(self.old), jobs_of(self.small)
        for key in old_jobs:
            self.assertEqual(
                "\n".join(l for l in scrub(new_jobs[key]).splitlines() if "max_running_requests" not in l),
                "\n".join(l for l in scrub(old_jobs[key]).splitlines() if "max_running_requests" not in l), key)

    def test_gpu13_fleet_telemetry_values(self) -> None:
        for name in ("model-sg-glm53-w4afp8-tp2-r1a", "model-sg-glm53-w4afp8-tp2-r1b"):
            block = service_block(self.small, name)
            job = jobs_of(self.small)[f"sglang-{name}"]
            self.assertIn('"precision:int4-weights-fp8-activations-fp8-kv"', block)
            self.assertIn('"engine_image:fa730e6e62b2"', block)
            self.assertIn('precision: "int4-weights-fp8-activations-fp8-kv"', job)
            self.assertIn('engine_image: "fa730e6e62b2"', job)
            self.assertIn('max_running_requests: "16"', job)
            # The log tag (com.datadoghq.ad.logs) must advertise the same cap as the engine argv, the OTel label and the scrape job.
            argv, _, _ = self.gpu13(name)
            cap = argv[argv.index("--max-running-requests") + 1]
            self.assertEqual(cap, "16")
            self.assertIn(f'"max_running_requests:{cap}"', block)
            self.assertNotIn('"max_running_requests:12"', block)
            self.assertIn(f'nearai.otel.max_running_requests: "{cap}"', block)
            self.assertIn("-fp8kv-", block)
            self.assertIn("mr16q4", block)
        self.assertEqual(service_block(self.small, "glm53-ghost-aggregator").count(f"image: {V7_IMAGE}"), 1)


class FleetValidatorContractTest(unittest.TestCase):
    """The production validator must reject profiling, a HiCache flag on the base tier, and a v6 engine in either fleet file."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        names = [VALIDATOR, DCGM_VALIDATOR, base_generator.TARGET, long_generator.TARGET, Path("prod/GLM-5.3-Flash-SGL-TP4.yaml"),
                 Path("prod/GLM-5.3-Flash-SGL-TP4-HiCache.yaml"), Path("prod/GLM-5.3-Flash-SGL-TP4-LongContext.yaml"),
                 Path("prod/GLM-5.3-Flash-SGL-TP4-W4AFP8.yaml"), Path("docker/sglang-glm53-hicache/RELEASED_IMAGE")]
        for name in names:
            (self.root / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, self.root / name)
        self.files = {"base": self.root / base_generator.TARGET, "long": self.root / long_generator.TARGET}
        self.valid = {kind: path.read_text() for kind, path in self.files.items()}

    def run_ruby(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(["ruby", str(self.root / VALIDATOR)], capture_output=True, text=True, check=False)

    def assert_fails(self, kind: str, mutated: str, message: str) -> None:
        self.assertNotEqual(mutated, self.valid[kind])
        self.files[kind].write_text(mutated)
        try:
            result = self.run_ruby()
            output = result.stdout + result.stderr
            self.assertEqual(result.returncode, 1, output)
            self.assertIn(message, output)
            self.assertNotRegex(output, r"\.rb:\d+:in")
        finally:
            self.files[kind].write_text(self.valid[kind])

    def test_committed_files_pass(self) -> None:
        result = self.run_ruby()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_rejects_profiling_a_hicache_flag_on_the_base_tier_and_a_v6_engine(self) -> None:
        base, long = self.valid["base"], self.valid["long"]
        env = "      - SGLANG_KV_TIER_METRICS=1\n"
        self.assert_fails("base", base.replace(env, env + "      - NEAR_SELF_PROFILE=1\n", 1), "must not enable profiling anywhere")
        self.assert_fails("long", long.replace(env, env + "      - NEAR_SELF_PROFILE=1\n", 1), "must not enable profiling anywhere")
        self.assert_fails("base", base.replace("      --disable-overlap-schedule\n\nx-dcgm-common", "      --disable-overlap-schedule\n      --enable-hierarchical-cache\n\nx-dcgm-common", 1), "must run with HiCache off")
        self.assert_fails("base", base.replace("      - SGLANG_PREPROCESS_WORKERS=4\n", "", 1), "environment must be the W4AFP8 base engine environment plus")
        self.assert_fails("long", long.replace("      - SGLANG_PREPROCESS_WORKERS=4\n", "", 1), "environment must be the long-context environment plus")
        self.assert_fails("base", base.replace("  image: " + V7_IMAGE + "\n", "  image: " + V6_IMAGE + "\n", 1), "image must be")
        self.assert_fails("long", long.replace(f"    image: {V7_IMAGE}\n", f"    image: {V6_IMAGE}\n", 1), "image must be")


class FleetRunbookTest(unittest.TestCase):
    def setUp(self) -> None:
        self.runbook = RUNBOOK.read_text()
        self.lowered = self.runbook.lower()

    def test_it_names_the_files_services_image_and_the_scoped_calls(self) -> None:
        for text in (base_generator.TARGET.name, long_generator.TARGET.name, V7_IMAGE.split("@")[1], V6_IMAGE.split("@")[1], "glm53-hicache-w4afp8-v7",
                     "gpu02", "gpu03", "gpu04", "gpu23", "gpu13", "otelcol-contrib", "dry_run", "force_recreate", "compose/up",
                     *BASE_ENGINES, *LONG_ENGINES):
            self.assertIn(text, self.runbook)
        self.assertNotIn('"services": []', self.runbook)
        self.assertNotIn("services: []", self.runbook)

    def test_it_carries_the_user_rules_and_checks(self) -> None:
        for text in ("never more than one long replica down at a time", "at most one base replica down fleet-wide", "cannot be recalled", "glm53-flash-prod",
                     "preprocess pool started: workers=4 timeout=60s", "kv_cache_dtype", "fp8_e4m3", "enable_hierarchical_cache", "NEAR_PROFILE", "rollback",
                     "no kms", "env map"):
            self.assertIn(text.lower(), self.lowered)

    def test_it_paces_one_replica_per_lane_with_a_five_minute_bake(self) -> None:
        for text in ("one replica per lane per step", "5-minute bake", "exactly one recreate", "compose/down` is never used", "explicit `services` list"):
            self.assertIn(text.lower(), self.lowered)
        # The old pair-wise pacing must be gone.
        for text in ("one base pair at a time", "(one call, two services)", "`r1`, `r2` (one call"):
            self.assertNotIn(text, self.runbook)
        # Both lanes, in order, one replica per step.
        base = " -> ".join(f"gpu0{h} r{n}" for h in (4, 3) for n in (1, 2, 3, 4))
        long = " -> ".join(f"{h} r{n}" for h, ns in (("gpu23", ("1a", "1b", "2a", "2b")), ("gpu02", ("1a", "1b", "2a", "2b")), ("gpu13", ("1a", "1b"))) for n in ns)
        self.assertIn(base, self.runbook)
        self.assertIn(long, self.runbook)
        # gpu13 lists only its GLM engine; the per-host collector/aggregator recreate and the metrics-visible-before note are kept.
        for text in ("only that one glm engine", "once per host after its last replica", "before this recreate the metrics are already visible",
                     "glm53-ghost-aggregator", "otelcol-contrib", "pranav"):
            self.assertIn(text.lower(), self.lowered)
        # The bake checklist items.
        for text in ("/health", "/backends/list", "restartcount", "server_address", "sglang_num_requests_total", "sglang_num_running_reqs", "sglang_generation_tokens_total",
                     "ttft p95", "itl p95", "20%", "xid", "cuda error", "traceback", "preprocess", "worker died", "not-yet-migrated", "restart loop",
                     "roll that one replica back", "stop the lane", "enable_hierarchical_cache=false", "grep -c near_profile"):
            self.assertIn(text.lower(), self.lowered)
        # Gateway caps stay a post-rollout step with the computed values.
        for text in ("672", "160", "64", "256", "only after both lanes have finished"):
            self.assertIn(text.lower(), self.lowered)

    def test_it_carries_the_evidence_gateway_values_and_the_gpu13_flag(self) -> None:
        for text in ("exp 26", "exp 27", "exp 29", "exp 30", "exp 31", "exp 32", "84%", "VLLM_PROXY_ADMISSION_LONG_MAX_INFLIGHT_PER_HOST", "cvm-ansible-playbooks",
                     "small-models.yaml", "owner decision", "+11-21%"):
            self.assertIn(text, self.runbook)
