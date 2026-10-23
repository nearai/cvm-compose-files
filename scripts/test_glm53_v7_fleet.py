#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: python3 -m unittest scripts.test_glm53_v7_fleet (needs ruby for the validator cases)
"""The two FLEET files (base tier on v7, long tier on v12): one config per tier, profiling off, the dashboard labels untouched, the runbook.

Imported by test_glm53_w4afp8_tp2x4.py and test_glm53_w4afp8_long_context.py so the existing CI steps run it.
"""

import ast
import contextlib
import gc
import hashlib
import io
import json
import re
import shlex
import shutil
import subprocess
import tempfile
import textwrap
import threading
import types
import unittest
import weakref
from pathlib import Path

from scripts import prepare_glm53_w4afp8_long_context as long_generator
from scripts import prepare_glm53_w4afp8_tp2x4 as base_generator

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = Path("scripts/validate_glm53_prod_config.rb")
DCGM_VALIDATOR = Path("scripts/validate_glm53_dcgm_metrics.rb")
RUNBOOK = ROOT / "docs/glm53-v7-fleet-rollout.md"

V7_IMAGE = "docker.io/nearaidev/sglang@sha256:fa730e6e62b2ae8058114ce540487ade33ab93bc42b1179ae78edc92bd563fc5"
V12_IMAGE = "docker.io/nearaidev/sglang@sha256:da8d4e4b5ec667499b58a21e56050aae9bcfc4df9669b3ac27e2e20af033d681"
V12_LABEL = "da8d4e4b5ec6"
V6_IMAGE = "docker.io/nearaidev/sglang@sha256:9c6ddd4319c4ab00e351d8650459e68b8830e36ffcc029d67fa5e19d0ac3ed17"
OLD_PROXY_IMAGE = "nearaidev/vllm-proxy-rs@sha256:d61357da39918a57126864a451eaf054f06a6989c03fe9a1666f7e6374ba6907"
NEW_PROXY_IMAGE = "nearaidev/vllm-proxy-rs@sha256:e4a42b327523e304c0fc389f656ba7a02fd7b3d4a38cdc822f9288fd193a525a"
FLEET_ENV = [
    "SGLANG_PREPROCESS_WORKERS=4",
    "SGLANG_PREPROCESS_TIMEOUT_S=60",
    "SGLANG_PREPROCESS_LOG_SLOW_S=5",
    "SGLANG_TOOL_SCHEMA_MAX_DEPTH=32",
    "SGLANG_TOOL_SCHEMA_MAX_NODES=25000",
]
# gpu03 r3 (arm B', prod/GLM-5.3-Flash-SGL-TP2x4-W4AFP8-V7-HiCacheOff-r3.yaml on pranavraja99/glm53-kvshare-ab), live argv,
# except --max-queued-requests: the fleet base tier raises it from the live 8 to 32 (queue-full 503s; see the runbook's queue section).
R3_ARGV = shlex.split("""
    sglang serve
    --model-path /root/.cache/huggingface/hub/models--graphistry--GLM-5.3-Flash-W4AFP8/snapshots/99f1fa70408c52b007d4fd69e02e5a522422e755
    --served-model-name z-ai/glm-5.3-flash --tp-size 2 --ep-size 2 --mem-fraction-static 0.86
    --max-running-requests 64 --max-queued-requests 32 --enable-priority-scheduling --disable-priority-preemption
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
    --max-running-requests 16 --max-queued-requests 8 --enable-priority-scheduling --disable-priority-preemption
    --chunked-prefill-size 16384 --max-prefill-tokens 32768 --prefill-decode-interval 2 --cuda-graph-max-bs-decode 16
    --dsa-prefill-backend flashmla_sparse_q8 --dsa-decode-backend flashmla_kv --kv-cache-dtype fp8_e4m3
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
        # The long-tier TP2 argv of the v12 fleet (v7 plus Q8 DSA prefill and 16K chunks), compared without the rendezvous port.
        canary_argv = shlex.split(LONG_ARGV)
        for name in LONG_ENGINES:
            block = service_block(self.long, name)
            argv = argv_of(block)
            with self.subTest(replica=name):
                self.assertEqual([t for t in argv if not t.startswith("127.0.0.1")], [t for t in canary_argv if not t.startswith("127.0.0.1")])
                self.assertIn(f"    image: {V12_IMAGE}\n", block)
                value = lambda flag: argv[argv.index(flag) + 1]
                self.assertEqual((value("--kv-cache-dtype"), value("--dsa-prefill-backend"), value("--dsa-decode-backend")), ("fp8_e4m3", "flashmla_sparse_q8", "flashmla_kv"))
                self.assertEqual(value("--chunked-prefill-size"), "16384")
                self.assertEqual((value("--max-running-requests"), value("--max-queued-requests"), value("--cuda-graph-max-bs-decode")), ("16", "8", "16"))
                self.assertEqual(argv.count("--disable-overlap-schedule"), 1)
                self.assertIn("--enable-hierarchical-cache", argv)
                self.assertEqual(value("--hicache-write-policy"), "write_through")
                env = env_of(block)
                self.assertEqual(env[-len(FLEET_ENV):], FLEET_ENV)
        for tp4 in ("model-sg-glm53-w4afp8-tp4-r1", "model-sg-glm53-w4afp8-tp4-r2"):
            block = service_block(self.long, tp4)
            self.assertNotIn("SGLANG_PREPROCESS", block)
            self.assertNotIn("fa730e6e62b2", block)
            self.assertNotIn(V12_LABEL, block)

    def test_fleet_telemetry_values(self) -> None:
        for text, names, engine in ((self.base, BASE_ENGINES, "fa730e6e62b2"), (self.long, LONG_ENGINES, V12_LABEL)):
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
        for token in ("fp8kv", "mr16q8", "c16384", "-v12-q8"):
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
        # Routing, nginx, registrar, proxy config, DCGM, collector pipelines, volumes: textually identical outside the engine anchors,
        # engine services, their scrape jobs, the ghost sidecar's image, the proxy image and the header.
        def strip(text: str, names: list[str], anchors: list[str]) -> str:
            text = text[text.index("x-logging-conf:"):]
            for name in names:
                text = text.replace(service_block(text, name), "")
            for anchor in anchors:
                text = text.replace(anchor_block(text, anchor), "")
            for key in list(jobs_of(text)):
                if key.startswith("sglang-model-sg-glm53"):
                    text = text.replace(jobs_of(text)[key], "")
            text = text.replace(V7_IMAGE, "IMAGE").replace(V12_IMAGE, "IMAGE").replace(V6_IMAGE, "IMAGE")
            text = text.replace(NEW_PROXY_IMAGE, OLD_PROXY_IMAGE)
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
QWEN38_GPU0_SERVICE = "model-sg-qwen38-27b-fp8-tp1-r1"
QWEN38_GPU0_ANCHOR = "x-qwen3-8-27b-common"
QWEN38_GPU0_JOB = f"sglang-{QWEN38_GPU0_SERVICE}"
PRIVACY_FILTER_TRANSFORMERS_PIN = '"transformers==5.16.1"'
PRIVACY_FILTER_OLD_TRANSFORMERS = '"transformers>=4.46"'
PRIVACY_FILTER_PIN_COMMENTS = (
    "      # transformers 5.20+ needs PyTorch >=2.6; this 2.5.1 base crash-loops on fresh builds.\n",
    "      # 5.16.1 matches gpu13's working image built on 2026-09-02.\n",
)
COLOC_ENGINE = "model-sg-qwen38-27b-fp8-tp1-coloc"
COLOC_ENGINE_JOB = f"sglang-{COLOC_ENGINE}"
COLOC_BLOCK_HASHES = {
    "x-privacy-filter-common": "b4a275a42bf94b58f24a624c37a9b4b9abb8378f76e79fc5becb49e469f13776",
    COLOC_ENGINE: "670273cabb20266ebab017a7c46a9a2860504cac0e2299a11f32a632c801d7d7",
    COLOC_ENGINE_JOB: "b5c730ba7b0196a959960be2a6eadec8024524e95852245ea2b43b84f85775ad",
}
# Levanto Sage moved in from prod/sage.yaml (gpu04 GPU 6), unchanged except the GPU: GPU 2 of the
# small-models host (gpu23), which Qwen3.8 r2 and its DCGM exporter gave up.
SAGE_FILE = ROOT / "prod/sage.yaml"
SAGE_SERVICES = ("sage-dstack", "sage")
SAGE_HEADING = "  # ── Levanto Sage on GPU 2 of the small-models host"
SAGE_GPU04_GPU = "GPU-bafbdde0-cd41-8379-fa93-149864963329"
SAGE_GPU = "GPU-2f40e72c-80c6-c9b4-b65c-d6ec155ecc8d"
QWEN38_R2 = "model-sg-qwen38-27b-fp8-tp1-r2"
QWEN38_R2_DCGM = "dcgm-qwen38-27b"
QWEN38_R2_JOBS = (f"sglang-{QWEN38_R2}", f"dcgm-{QWEN38_R2_DCGM}")
SAGE_BLOCK_HASHES = {
    "sage-dstack": "e3020a51e7f4f6c6a02868004618e880da664a55abc6c875c1da860595f16518",
    "sage": "652c4833c281db47955a64f64cd25d57b44aa621dcbe8ca9a01a346182cf0ac5",
    "networks": "f83c2c4fabfffcf0587e644bb575e4cc4c602f0fe8a032061fac8215403be193",
}


def without_privacy_filter_pin(block: str) -> str:
    # The deliberate 5.16.1 pin keeps PyTorch 2.5 working after transformers 5.20 dropped it.
    for comment in PRIVACY_FILTER_PIN_COMMENTS:
        block = block.replace(comment, "")
    return block.replace(PRIVACY_FILTER_TRANSFORMERS_PIN, PRIVACY_FILTER_OLD_TRANSFORMERS)


def without_qwen38_gpu0_stage_b(text: str) -> str:
    """Apply only the Qwen3.8 GPU 0 retirement to the frozen pre-v7 fixture."""
    old_intro = (
        "# GPU 0 transition replica, shared with the privacy filter. The 0.60/128\n"
        "# recipe reached ~122 GiB on 2026-10-05, leaving too little room for the\n"
        "# filter's ~22 GB batches. Stage B retires this replica and anchor.\n"
    )
    if text.count(old_intro) != 1:
        raise AssertionError("expected one Qwen3.8 GPU 0 fixture introduction")
    text = text.replace(old_intro, "", 1)
    old_anchor = anchor_block(text, QWEN38_GPU0_ANCHOR)
    if text.count(old_anchor) != 1:
        raise AssertionError("expected one Qwen3.8 GPU 0 fixture anchor")
    text = text.replace(old_anchor, (
        "# Qwen3.8 runs alone on GPU 2; GPU 0 is reserved for the privacy filter.\n"
        "# Dedicated recipe: prod/qwen36-qwen38.yaml with gpu13's cache volume."
    ), 1)
    old_service = service_block(text, QWEN38_GPU0_SERVICE)
    if text.count("\n" + old_service) != 1:
        raise AssertionError("expected one Qwen3.8 GPU 0 fixture service")
    text = text.replace("\n" + old_service, "", 1)
    old_heading = (
        "  # ── Qwen3.8-27B transition: GPU 0 r1 and dedicated GPU 2 r2 ──\n"
        "  # gpu02 maintenance evacuation; see x-qwen3-8-27b-common. Uses its own"
    )
    if text.count(old_heading) != 1:
        raise AssertionError("expected one Qwen3.8 GPU 0 fixture heading")
    text = text.replace(old_heading, (
        "  # ── Qwen3.8-27B on dedicated GPU 2; GPU 0 is privacy-filter only ──\n"
        "  # See x-qwen3-8-27b-dedicated. Uses its own"
    ), 1)
    old_job = jobs_of(text)[QWEN38_GPU0_JOB]
    if text.count("              " + old_job) != 1:
        raise AssertionError("expected one Qwen3.8 GPU 0 scrape job")
    text = text.replace("              " + old_job, "", 1)

    # Match the five proxy text edits and the collector's two labels exactly.
    for original, replacements in (
        (service_block(text, "proxy-qwen38-27b"), (
            ("VLLM_BACKEND_URLS=http://model-sg-qwen38-27b-fp8-tp1-r1:8000,http://model-sg-qwen38-27b-fp8-tp1-r2:8000",
             "VLLM_BACKEND_URLS=http://model-sg-qwen38-27b-fp8-tp1-r2:8000"),
            ('"gpu_pair:0,2"', '"gpu_pair:2"'),
            ('"config_variant:qwen38-gpu13-gpu0-gpu2-transition"', '"config_variant:qwen38-gpu13-gpu2-dedicated"'),
            ('nearai.otel.gpu_pair: "0,2"', 'nearai.otel.gpu_pair: "2"'),
            ('nearai.otel.config_variant: "qwen38-gpu13-gpu0-gpu2-transition"', 'nearai.otel.config_variant: "qwen38-gpu13-gpu2-dedicated"'),
        )),
        (jobs_of(text)["inference-proxy-proxy-qwen38-27b"], (
            ('gpu_pair: "0,2"', 'gpu_pair: "2"'),
            ('config_variant: "qwen38-gpu13-gpu0-gpu2-transition"', 'config_variant: "qwen38-gpu13-gpu2-dedicated"'),
        )),
    ):
        updated = original
        for old, new in replacements:
            if updated.count(old) != 1:
                raise AssertionError(f"expected one Qwen3.8 stage-B fixture value: {old}")
            updated = updated.replace(old, new, 1)
        text = text.replace(original, updated, 1)
    return text


def with_qwen38_coloc_pool(text: str) -> str:
    """Apply only the serving-pool changes to the retired-GPU-0 fixture."""
    for original, replacements in (
        (service_block(text, "proxy-qwen38-27b"), (
            ("VLLM_BACKEND_URLS=http://model-sg-qwen38-27b-fp8-tp1-r2:8000",
             "VLLM_BACKEND_URLS=http://model-sg-qwen38-27b-fp8-tp1-r2:8000,http://model-sg-qwen38-27b-fp8-tp1-coloc:8000"),
            ('"gpu_pair:2"', '"gpu_pair:0,2"'),
            ('"config_variant:qwen38-gpu13-gpu2-dedicated"', '"config_variant:qwen38-gpu13-gpu2-plus-gpu0-coloc"'),
            ('nearai.otel.gpu_pair: "2"', 'nearai.otel.gpu_pair: "0,2"'),
            ('nearai.otel.config_variant: "qwen38-gpu13-gpu2-dedicated"', 'nearai.otel.config_variant: "qwen38-gpu13-gpu2-plus-gpu0-coloc"'),
        )),
        (jobs_of(text)["inference-proxy-proxy-qwen38-27b"], (
            ('gpu_pair: "2"', 'gpu_pair: "0,2"'),
            ('config_variant: "qwen38-gpu13-gpu2-dedicated"', 'config_variant: "qwen38-gpu13-gpu2-plus-gpu0-coloc"'),
        )),
    ):
        updated = original
        for old, new in replacements:
            if updated.count(old) != 1:
                raise AssertionError(f"expected one Qwen3.8 pooling fixture value: {old}")
            updated = updated.replace(old, new, 1)
        text = text.replace(original, updated, 1)
    return text


def with_sage_on_gpu2(text: str) -> str:
    """Apply only the Qwen3.8 r2 retirement (Levanto Sage takes GPU 2) to the coloc-pool fixture."""
    old_heading = "  # ── Qwen3.8-27B on dedicated GPU 2; GPU 0 is privacy-filter only ──\n"
    if text.count(old_heading) != 1:
        raise AssertionError("expected one Qwen3.8 GPU 2 fixture heading")
    text = text.replace(old_heading, "  # ── Qwen3.8-27B on GPU 0 beside the privacy filter; GPU 2 is Levanto Sage ──\n", 1)
    for name in (QWEN38_R2, QWEN38_R2_DCGM):
        old_service = service_block(text, name)
        if text.count("\n" + old_service) != 1:
            raise AssertionError(f"expected one {name} fixture service")
        text = text.replace("\n" + old_service, "", 1)
    for job in QWEN38_R2_JOBS:
        old_job = jobs_of(text)[job]
        if text.count("              " + old_job) != 1:
            raise AssertionError(f"expected one {job} scrape job")
        text = text.replace("              " + old_job, "", 1)
    for original, replacements in (
        (service_block(text, "proxy-qwen38-27b"), (
            ("VLLM_BACKEND_URLS=http://model-sg-qwen38-27b-fp8-tp1-r2:8000,http://model-sg-qwen38-27b-fp8-tp1-coloc:8000",
             "VLLM_BACKEND_URLS=http://model-sg-qwen38-27b-fp8-tp1-coloc:8000"),
            ('"gpu_pair:0,2"', '"gpu_pair:0"'),
            ('"config_variant:qwen38-gpu13-gpu2-plus-gpu0-coloc"', '"config_variant:qwen38-gpu0-coloc-only"'),
            ('nearai.otel.gpu_pair: "0,2"', 'nearai.otel.gpu_pair: "0"'),
            ('nearai.otel.config_variant: "qwen38-gpu13-gpu2-plus-gpu0-coloc"', 'nearai.otel.config_variant: "qwen38-gpu0-coloc-only"'),
        )),
        (jobs_of(text)["inference-proxy-proxy-qwen38-27b"], (
            ('gpu_pair: "0,2"', 'gpu_pair: "0"'),
            ('config_variant: "qwen38-gpu13-gpu2-plus-gpu0-coloc"', 'config_variant: "qwen38-gpu0-coloc-only"'),
        )),
    ):
        updated = original
        for old, new in replacements:
            if updated.count(old) != 1:
                raise AssertionError(f"expected one Qwen3.8 r2-retirement fixture value: {old}")
            updated = updated.replace(old, new, 1)
        text = text.replace(original, updated, 1)
    return text


class Gpu13FleetTest(unittest.TestCase):
    """gpu13 deploys prod/small-models.yaml: its two GLM replicas must be the long file's v12 TP2 replicas, and nothing else may move."""

    def setUp(self) -> None:
        self.small = SMALL.read_text()
        self.old = SMALL_FIXTURE.read_text()
        self.long = LONG_FILE.read_text()

    def privacy_filter_function(self, name: str) -> types.CodeType:
        block = anchor_block(self.small, "x-privacy-filter-common")
        source = textwrap.dedent(block.split("COPY <<'PYEOF' /app/server.py\n", 1)[1].split("\n      PYEOF", 1)[0])
        function = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == name)
        function.decorator_list = []
        return compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), "server.py", "exec")

    def privacy_filter_oom_namespace(self, primary, retry, retry_max_length=16384, exit_after=3):
        class OutOfMemoryError(Exception):
            pass

        class JSONResponse:
            def __init__(self, status_code, headers, content, background=None):
                self.status_code = status_code
                self.headers = headers
                self.content = content
                self.background = background

        class BackgroundTask:
            def __init__(self, func, *args):
                self.func = func
                self.args = args

        exit_calls = []
        exit_stub = lambda code: exit_calls.append(code)
        namespace = {
            "torch": types.SimpleNamespace(cuda=types.SimpleNamespace(OutOfMemoryError=OutOfMemoryError, empty_cache=lambda: None)),
            "gc": gc,
            "json": json,
            "os": types.SimpleNamespace(_exit=exit_stub),
            "sys": types.SimpleNamespace(stderr=io.StringIO()),
            "JSONResponse": JSONResponse,
            "BackgroundTask": BackgroundTask,
            "HTTPException": Exception,
            "ClassifyRequest": object,
            "clf": lambda *args, **kwargs: primary(OutOfMemoryError),
            "clf_retry": None if retry is None else lambda *args, **kwargs: retry(OutOfMemoryError),
            "tokenizer": lambda texts: types.SimpleNamespace(input_ids=[[0] for _ in texts]),
            "_merge_adjacent_spans": lambda spans: spans,
            "MODEL_ID": "openai/privacy-filter",
            "BATCH_SIZE": 1,
            "STRIDE": 256,
            "retry_max_length": retry_max_length,
            "OOM_EXIT_AFTER": exit_after,
            "_oom_lock": threading.Lock(),
            "_consecutive_oom_503": 0,
        }
        exec(self.privacy_filter_function("classify"), namespace)
        return namespace, exit_calls

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
                self.assertEqual(image, V12_IMAGE)
                self.assertEqual(env[-len(FLEET_ENV):], FLEET_ENV)
                self.assertFalse([e for e in env if "NEAR_SELF_PROFILE" in e])
        self.assertIn("--disable-overlap-schedule", long_argv)
        self.assertEqual(long_env, env_free(env_of(service_block(self.long, "model-sg-glm53-w4afp8-tp2-r2a"))))

    def test_gpu13_keeps_its_host_specific_bits(self) -> None:
        for name, budget in (("model-sg-glm53-w4afp8-tp2-r1a", "GLM53_R1A_HICACHE_RAM_BUDGET"), ("model-sg-glm53-w4afp8-tp2-r1b", "GLM53_R1B_HICACHE_RAM_BUDGET")):
            argv, env, _ = self.gpu13(name)
            self.assertIn(f"SGLANG_HICACHE_RAM_BUDGET=${{{budget}:-325GiB}}", env)
            self.assertIn("SGLANG_DSA_INDEXER_QSPLIT=1", env)
        for name in ("model-sg-glm53-w4afp8-tp2-r1a", "model-sg-glm53-w4afp8-tp2-r1b"):
            argv = self.gpu13(name)[0]
            self.assertEqual(argv[argv.index("--chunked-prefill-size") + 1], "16384", name)
            self.assertEqual(argv[argv.index("--dsa-prefill-backend") + 1], "flashmla_sparse_q8", name)
            self.assertEqual(argv[argv.index("--dsa-decode-backend") + 1], "flashmla_kv", name)
        self.assertIn("127.0.0.1:29510", self.gpu13("model-sg-glm53-w4afp8-tp2-r1a")[0])
        self.assertIn("127.0.0.1:29511", self.gpu13("model-sg-glm53-w4afp8-tp2-r1b")[0])
        self.assertIn('device_ids: ["4","5"]', service_block(self.small, "model-sg-glm53-w4afp8-tp2-r1a"))
        self.assertIn('device_ids: ["6","7"]', service_block(self.small, "model-sg-glm53-w4afp8-tp2-r1b"))

    def test_non_glm_services_and_everything_outside_the_glm_blocks_are_byte_identical(self) -> None:
        old = with_sage_on_gpu2(with_qwen38_coloc_pool(without_qwen38_gpu0_stage_b(self.old)))
        names = re.findall(r"^  ([a-z0-9-]+):$", self.old[self.old.index("\nservices:\n"):], re.MULTILINE)
        expected_names = [name for name in names if name != QWEN38_GPU0_SERVICE]
        expected_names.insert(expected_names.index(QWEN38_R2) + 1, COLOC_ENGINE)
        # Sage took GPU 2: r2 and its DCGM exporter are gone.
        expected_names = [name for name in expected_names if name not in (QWEN38_R2, QWEN38_R2_DCGM)]
        # Sage (pinned separately): its two services before the collector, its network and its volume.
        otelcol = expected_names.index("otelcol-contrib")
        expected_names[otelcol:otelcol] = list(SAGE_SERVICES)
        expected_names.insert(expected_names.index("default"), "sage")
        expected_names.insert(expected_names.index("certs"), "sage-data")
        self.assertEqual(
            expected_names,
            re.findall(r"^  ([a-z0-9-]+):$", self.small[self.small.index("\nservices:\n"):], re.MULTILINE),
        )
        for name in names:
            if name not in GPU13_GLM_SERVICES and name not in (QWEN38_GPU0_SERVICE, QWEN38_R2, QWEN38_R2_DCGM):
                with self.subTest(service=name):
                    new_block = service_block(self.small, name)
                    if name == "proxy-qwen38-27b":
                        new_block = new_block.split("  # Production co-location trial", 1)[0].rstrip("\n") + "\n"
                    if name == "dcgm-shared-gpu3":
                        new_block = new_block.split(SAGE_HEADING, 1)[0].rstrip("\n") + "\n"
                    expected = service_block(old, name)
                    if name == "proxy-qwen38-27b":
                        expected = expected.replace(OLD_PROXY_IMAGE, NEW_PROXY_IMAGE)
                    self.assertEqual(new_block, expected)
        anchors = re.findall(r"^(x-[a-z0-9-]+): &", self.old, re.MULTILINE)
        self.assertEqual(
            [anchor for anchor in anchors if anchor != QWEN38_GPU0_ANCHOR],
            re.findall(r"^(x-[a-z0-9-]+): &", self.small, re.MULTILINE),
        )
        for anchor in anchors:
            if anchor not in ("x-sg-glm53-flash-common", QWEN38_GPU0_ANCHOR, "x-privacy-filter-common"):
                new_block = anchor_block(self.small, anchor)
                expected = anchor_block(old, anchor)
                if anchor == "x-vllm-proxy-common":
                    expected = expected.replace(OLD_PROXY_IMAGE, NEW_PROXY_IMAGE)
                self.assertEqual(new_block, expected, anchor)
        # The coloc scrape job is pinned separately; every old non-GLM job stays identical.
        old_jobs, new_jobs = jobs_of(old), jobs_of(self.small)
        expected_jobs = [key for key in jobs_of(self.old) if key != QWEN38_GPU0_JOB]
        index = expected_jobs.index(f"sglang-{QWEN38_R2}") + 1
        expected_jobs[index:index] = [COLOC_ENGINE_JOB]
        expected_jobs = [key for key in expected_jobs if key not in QWEN38_R2_JOBS]
        self.assertEqual(expected_jobs, list(new_jobs))
        for key in old_jobs:
            if "glm53" not in key:
                self.assertEqual(new_jobs[key], old_jobs[key], key)

    def test_gpu0_coloc_blocks_are_exactly_the_reviewed_additions(self) -> None:
        blocks = {
            "x-privacy-filter-common": anchor_block(self.small, "x-privacy-filter-common"),
            COLOC_ENGINE: service_block(self.small, COLOC_ENGINE),
            COLOC_ENGINE_JOB: jobs_of(self.small)[COLOC_ENGINE_JOB],
        }
        for name, block in blocks.items():
            with self.subTest(block=name):
                self.assertEqual(hashlib.sha256(block.encode()).hexdigest(), COLOC_BLOCK_HASHES[name])
        old_volumes = self.old.split("\nvolumes:\n", 1)[1].split("\nconfigs:\n", 1)[0]
        new_volumes = self.small.split("\nvolumes:\n", 1)[1].split("\nconfigs:\n", 1)[0]
        self.assertEqual(new_volumes.replace("  qwen38_coloc_kernel_cache:\n", "").replace("  sage-data:\n", ""), old_volumes)
        self.assertEqual(new_volumes.count("  qwen38_coloc_kernel_cache:\n"), 1)
        self.assertEqual(new_volumes.count("  sage-data:\n"), 1)

    def test_sage_moves_in_from_prod_sage_on_gpu2_of_the_small_models_host_unchanged_except_the_gpu(self) -> None:
        networks = self.small.split("\nnetworks:\n", 1)[1].split("\nvolumes:\n", 1)[0]
        blocks = {name: service_block(self.small, name) for name in SAGE_SERVICES}
        for name, block in {**blocks, "networks": networks}.items():
            with self.subTest(block=name):
                self.assertEqual(hashlib.sha256(block.encode()).hexdigest(), SAGE_BLOCK_HASHES[name])
        # Deployed the way prod/sage.yaml was: no profile, nothing added.
        for name, block in blocks.items():
            self.assertNotIn("profiles:", block, name)
        self.assertEqual(blocks["sage"].count(SAGE_GPU), 2)
        self.assertNotIn(SAGE_GPU04_GPU, self.small)
        # The services are prod/sage.yaml's (#380) line for line, comments aside, except the GPU.
        old = SAGE_FILE.read_text()
        for name, block in blocks.items():
            with self.subTest(service=name):
                old_lines = [l.replace(SAGE_GPU04_GPU, SAGE_GPU) for l in service_block(old, name).split("\nnetworks:", 1)[0].splitlines()
                             if l.strip() and not l.lstrip().startswith("#")]
                new_lines = [l for l in block.splitlines()
                             if l.strip() and not l.lstrip().startswith("#")]
                self.assertEqual(new_lines, old_lines)

    def test_privacy_filter_pins_transformers(self) -> None:
        self.assertIn(PRIVACY_FILTER_TRANSFORMERS_PIN, anchor_block(self.small, "x-privacy-filter-common"))

    def test_privacy_filter_retry_window_is_strictly_smaller(self) -> None:
        namespace = {}
        exec(self.privacy_filter_function("_retry_window"), namespace)
        for max_length, cap, expected in (
            (131072, 16384, 16384),
            (4096, 16384, 2048),
            (32768, 16384, 16384),
            (16384, 16384, 8192),
            (15, 16384, None),
            (8, 16384, None),
        ):
            with self.subTest(max_length=max_length, cap=cap):
                result = namespace["_retry_window"](max_length, cap)
                self.assertEqual(result, expected)
                if result is not None:
                    self.assertLess(result, max_length)

    def test_privacy_filter_oom_recovery_runs_after_the_failed_forward_is_released(self) -> None:
        code = self.privacy_filter_function("classify")

        class OutOfMemoryError(Exception):
            pass

        class HTTPException(Exception):
            pass

        class JSONResponse:
            def __init__(self, status_code, headers, content, background=None):
                self.status_code = status_code
                self.headers = headers
                self.content = content
                self.background = background

        class BackgroundTask:
            def __init__(self, func, *args):
                self.func = func
                self.args = args

        class Sentinel:
            pass

        for retry_fails in (False, True):
            with self.subTest(retry_fails=retry_fails):
                sentinels = []
                cache_checks = []

                def fail_forward():
                    sentinel = Sentinel()
                    sentinels.append(weakref.ref(sentinel))
                    raise OutOfMemoryError("simulated failed forward")

                def primary(texts, batch_size, stride):
                    return fail_forward()

                def retry(texts, batch_size, stride):
                    self.assertIsNone(sentinels[0]())
                    if retry_fails:
                        return fail_forward()
                    return [[]]

                def empty_cache():
                    cache_checks.append(all(ref() is None for ref in sentinels))

                namespace = {
                    "torch": types.SimpleNamespace(cuda=types.SimpleNamespace(OutOfMemoryError=OutOfMemoryError, empty_cache=empty_cache)),
                    "gc": gc,
                    "json": json,
                    "os": types.SimpleNamespace(_exit=lambda code: None),
                    "sys": types.SimpleNamespace(stderr=io.StringIO()),
                    "JSONResponse": JSONResponse,
                    "BackgroundTask": BackgroundTask,
                    "HTTPException": HTTPException,
                    "ClassifyRequest": object,
                    "clf": primary,
                    "clf_retry": retry,
                    "tokenizer": lambda texts: types.SimpleNamespace(input_ids=[[0] for _ in texts]),
                    "_merge_adjacent_spans": lambda spans: spans,
                    "MODEL_ID": "openai/privacy-filter",
                    "BATCH_SIZE": 1,
                    "STRIDE": 256,
                    "retry_max_length": 16384,
                    "OOM_EXIT_AFTER": 3,
                    "_oom_lock": threading.Lock(),
                    "_consecutive_oom_503": 0,
                }
                exec(code, namespace)
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    result = namespace["classify"](types.SimpleNamespace(input="x", threshold=0.0))

                self.assertEqual(len(cache_checks), 2 if retry_fails else 1)
                self.assertTrue(all(cache_checks))
                self.assertEqual([json.loads(line)["event"] for line in stdout.getvalue().splitlines()], ["filter_oom"])
                if retry_fails:
                    self.assertIsInstance(result, JSONResponse)
                    self.assertEqual(result.status_code, 503)
                    self.assertEqual(result.headers, {"Retry-After": "2"})
                else:
                    self.assertEqual(result["data"][0]["spans"], [])

    def test_privacy_filter_oom_without_retry_returns_503(self) -> None:
        def primary(oom_error):
            raise oom_error("primary OOM")

        namespace, exit_calls = self.privacy_filter_oom_namespace(primary, None, retry_max_length=None)
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            result = namespace["classify"](types.SimpleNamespace(input="x", threshold=0.0))
        self.assertEqual(result.status_code, 503)
        self.assertEqual(result.headers, {"Retry-After": "2"})
        self.assertIsNone(result.background)
        self.assertEqual(json.loads(stdout.getvalue().splitlines()[0]), {"event": "filter_oom", "retry_max_length": None})
        self.assertEqual(exit_calls, [])

    def test_privacy_filter_exits_after_three_consecutive_oom_503s(self) -> None:
        def oom(oom_error):
            raise oom_error("simulated OOM")

        namespace, exit_calls = self.privacy_filter_oom_namespace(oom, oom)
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            results = [namespace["classify"](types.SimpleNamespace(input="x", threshold=0.0)) for _ in range(3)]
        self.assertEqual([response.status_code for response in results], [503, 503, 503])
        self.assertEqual([response.headers for response in results], [{"Retry-After": "2"}] * 3)
        self.assertEqual([response.background is not None for response in results], [False, False, True])
        self.assertIs(results[2].background.func, namespace["os"]._exit)
        self.assertEqual(results[2].background.args, (1,))
        self.assertEqual(exit_calls, [])
        self.assertEqual([json.loads(line) for line in stdout.getvalue().splitlines() if 'filter_oom_exit' in line],
                         [{"event": "filter_oom_exit", "consecutive_503": 3}])

    def test_privacy_filter_success_resets_consecutive_oom_503s(self) -> None:
        def primary(oom_error):
            raise oom_error("primary OOM")

        retry_outcomes = iter((False, False, True, False, False))

        def retry(oom_error):
            if next(retry_outcomes):
                return [[]]
            raise oom_error("retry OOM")

        namespace, exit_calls = self.privacy_filter_oom_namespace(primary, retry)
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            results = [namespace["classify"](types.SimpleNamespace(input="x", threshold=0.0)) for _ in range(5)]
        self.assertEqual([response.status_code if hasattr(response, "status_code") else 200 for response in results],
                         [503, 503, 200, 503, 503])
        self.assertTrue(all(getattr(response, "background", None) is None for response in results))
        self.assertEqual(namespace["_consecutive_oom_503"], 2)
        self.assertNotIn("filter_oom_exit", stdout.getvalue())
        self.assertEqual(exit_calls, [])

    def test_coloc_engine_joins_customer_routing_without_synthetic_load(self) -> None:
        names = re.findall(r"^  ([a-z0-9-]+):$", self.small[self.small.index("\nservices:\n"):], re.MULTILINE)
        anchors = re.findall(r"^(x-[a-z0-9-]+): &", self.small, re.MULTILINE)
        self.assertNotIn(QWEN38_GPU0_SERVICE, names)
        self.assertNotIn(QWEN38_GPU0_ANCHOR, anchors)
        self.assertEqual(
            [e for e in env_of(service_block(self.small, "proxy-qwen38-27b")) if e.startswith("VLLM_BACKEND_URLS=")],
            ["VLLM_BACKEND_URLS=http://model-sg-qwen38-27b-fp8-tp1-coloc:8000"],
        )
        # Since Sage took GPU 2 the GPU 0 engine is the only private Qwen3.8 replica.
        self.assertNotIn(QWEN38_R2, names)
        self.assertNotIn(QWEN38_R2_DCGM, names)
        self.assertIn("    <<: *privacy-filter-common", service_block(self.small, "model-privacy-filter"))
        self.assertEqual(anchor_block(self.small, "x-privacy-filter-common").count('device_ids: ["0"]'), 1)
        self.assertIn('device_ids: ["0"]', service_block(self.small, COLOC_ENGINE))
        for anchor in anchors:
            if anchor != "x-privacy-filter-common":
                self.assertNotRegex(anchor_block(self.small, anchor), r'device_ids:\s*\[[^\]]*"0"')
        for name in names:
            if name not in ("model-privacy-filter", COLOC_ENGINE) and not name.startswith("dcgm-"):
                block = service_block(self.small, name)
                with self.subTest(service=name):
                    self.assertNotIn("<<: *privacy-filter-common", block)
                    self.assertNotRegex(block, r'device_ids:\s*\[[^\]]*"0"')

    def test_dashboard_labels_are_unchanged_on_gpu13(self) -> None:
        def selectors(text: str) -> list[str]:
            return [l for l in selector_lines(text) if "max_running_requests" not in l and "max_queued_requests" not in l]
        old = with_sage_on_gpu2(with_qwen38_coloc_pool(without_qwen38_gpu0_stage_b(self.old)))
        without_coloc = self.small
        for name in (COLOC_ENGINE,):
            without_coloc = without_coloc.replace(service_block(without_coloc, name), "", 1)
        for name in (COLOC_ENGINE_JOB,):
            without_coloc = without_coloc.replace("              " + jobs_of(without_coloc)[name], "", 1)
        self.assertEqual(selectors(without_coloc), selectors(old))
        old_jobs, new_jobs = jobs_of(old), jobs_of(self.small)
        expected_jobs = [key for key in jobs_of(self.old) if key != QWEN38_GPU0_JOB]
        index = expected_jobs.index(f"sglang-{QWEN38_R2}") + 1
        expected_jobs[index:index] = [COLOC_ENGINE_JOB]
        expected_jobs = [key for key in expected_jobs if key not in QWEN38_R2_JOBS]
        self.assertEqual(expected_jobs, list(new_jobs))
        for key in old_jobs:
            self.assertEqual(
                "\n".join(l for l in scrub(new_jobs[key]).splitlines() if "max_running_requests" not in l and "max_queued_requests" not in l),
                "\n".join(l for l in scrub(old_jobs[key]).splitlines() if "max_running_requests" not in l and "max_queued_requests" not in l), key)

    def test_gpu13_fleet_telemetry_values(self) -> None:
        for name in ("model-sg-glm53-w4afp8-tp2-r1a", "model-sg-glm53-w4afp8-tp2-r1b"):
            block = service_block(self.small, name)
            job = jobs_of(self.small)[f"sglang-{name}"]
            self.assertIn('"precision:int4-weights-fp8-activations-fp8-kv"', block)
            self.assertIn(f'"engine_image:{V12_LABEL}"', block)
            self.assertIn('precision: "int4-weights-fp8-activations-fp8-kv"', job)
            self.assertIn(f'engine_image: "{V12_LABEL}"', job)
            self.assertIn('max_running_requests: "16"', job)
            self.assertIn('max_queued_requests: "8"', job)
            self.assertIn('nearai.otel.max_queued_requests: "8"', block)
            # The log tag (com.datadoghq.ad.logs) must advertise the same cap as the engine argv, the OTel label and the scrape job.
            argv, _, _ = self.gpu13(name)
            cap = argv[argv.index("--max-running-requests") + 1]
            self.assertEqual(cap, "16")
            self.assertIn(f'"max_running_requests:{cap}"', block)
            self.assertNotIn('"max_running_requests:12"', block)
            queued = argv[argv.index("--max-queued-requests") + 1]
            self.assertEqual(queued, "8")
            self.assertIn(f'"max_queued_requests:{queued}"', block)
            self.assertNotIn('"max_queued_requests:4"', block)
            self.assertIn(f'nearai.otel.max_running_requests: "{cap}"', block)
            self.assertIn("-fp8kv-", block)
            self.assertIn("mr16q8", block)
        self.assertEqual(service_block(self.small, "glm53-ghost-aggregator").count(f"image: {V12_IMAGE}"), 1)


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
        self.assert_fails("long", long.replace(f"    image: {V12_IMAGE}\n", f"    image: {V6_IMAGE}\n", 1), "image must be")


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
