#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: python3 -m unittest scripts.test_gpu13_glm53_base
"""gpu13 GPUs 4-7: two GLM-5.3 BASE replicas that match gpu04's v12 base replicas r1/r2."""

import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Final

from scripts import prepare_gpu13_glm53_base as gen
from scripts.test_glm53_gpu04_3b1l import services

ROOT: Final = Path(__file__).resolve().parents[1]
VALIDATOR: Final = Path("scripts/validate_ds4f_migration.rb")
# Lines a gpu13 engine block may differ in from gpu04's (GPU ids, headings, the public-port log tag).
ENGINE_DELTAS: Final = ("device_ids:", "# --- GLM-5.3-Flash engine replica", '"port:', "CUDA_DEVICE_ORDER", "# gpu13: CUDA's device order")


def argv(block: str) -> list[str]:
    return block.split("command: >\n", 1)[1].split("\n  volumes:\n", 1)[0].split()


def jobs(text: str) -> dict[str, str]:
    body = text.split("scrape_configs:\n", 1)[1].split("\n  registrar_script:\n", 1)[0]
    parts = re.split(r"(?m)^              - job_name: (\S+)\n", body)
    return {parts[i]: parts[i + 1] for i in range(1, len(parts), 2)}


def anchor(text: str, name: str) -> str:
    match = re.search(rf"^{re.escape(name)}:.*?(?=^x-[\w-]+:|^services:)", text, re.MULTILINE | re.DOTALL)
    if match is None:
        raise AssertionError(f"missing anchor {name}")
    return match.group()


class Gpu13GlmBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = (ROOT / gen.SOURCE).read_text()
        cls.glm = (ROOT / gen.GLM_SOURCE).read_text()
        cls.text = (ROOT / gen.TARGET).read_text()
        cls.svc = services(cls.text)
        cls.src_svc = services(cls.source)
        cls.glm_svc = services(cls.glm)

    def test_committed_file_is_generated(self) -> None:
        self.assertEqual(self.text, gen.generate(self.source, self.glm))

    def test_service_set(self) -> None:
        expected = (set(self.src_svc) - set(gen.DS_SERVICES)) | set(gen.GLM_SERVICES)
        self.assertEqual(set(self.svc), expected)

    def test_other_services_unchanged(self) -> None:
        changed = {"model-downloader", "nginx", "model-proxy-registrar", "otelcol-contrib"}
        for name in set(self.src_svc) - set(gen.DS_SERVICES) - changed:
            with self.subTest(service=name):
                self.assertEqual(self.svc[name], self.src_svc[name])
        # otelcol-contrib only loses the DS4.1 extra_hosts.
        self.assertEqual(self.svc["otelcol-contrib"], re.sub(r"(?m)^    extra_hosts:\n(?:      - \"model-vllm-dsv41-dp\d:host-gateway\"\n)+", "", self.src_svc["otelcol-contrib"]))
        for name in ("nginx-qwen38", "qwen38-registrar", "qwen38-abliterated-registrar", "sage", "sage-dstack"):
            self.assertIn(name, self.svc)
        for config in ("nginx_qwen38_conf", "qwen38_registrar_script", "qwen38_abliterated_registrar_script"):
            with self.subTest(config=config):
                self.assertEqual(self.text.split(f"  {config}:\n", 1)[1][:4000], self.source.split(f"  {config}:\n", 1)[1][:4000])

    def test_engines_match_gpu04_r1_r2(self) -> None:
        self.assertEqual(argv(anchor(self.text, "x-sg-glm53-flash-candidate")), argv(anchor(self.glm, "x-sg-glm53-flash-candidate")))
        self.assertIn("--max-running-requests 64", anchor(self.text, "x-sg-glm53-flash-candidate"))
        self.assertNotIn("--enable-hierarchical-cache", anchor(self.text, "x-sg-glm53-flash-candidate"))
        common = anchor(self.text, "x-sg-glm53-flash-common")
        self.assertIn("    - hugginface_cache:/root/.cache/huggingface\n", common)
        self.assertIn("    - CUDA_DEVICE_ORDER=PCI_BUS_ID\n", common)
        for name, gpus in ((gen.R1, '["4","5"]'), (gen.R2, '["6","7"]')):
            with self.subTest(engine=name):
                block = self.svc[name]
                self.assertIn(f"device_ids: {gpus}", block)
                self.assertIn("<<: *sg-glm53-flash-candidate", block)
                # Two-space comments are section headings that follow a service; they belong to the next one.
                keep = lambda b: [l for l in b.splitlines() if not l.startswith("  # ") and not any(d in l for d in ENGINE_DELTAS)]
                self.assertEqual(keep(block), keep(self.glm_svc[name]))
        self.assertIn("      - CUDA_DEVICE_ORDER=PCI_BUS_ID\n", self.svc[gen.R2])
        self.assertIn("SGLANG_GHOST_CACHE_REPLICA=r2", self.svc[gen.R2])

    def test_proxy_pool_and_ports(self) -> None:
        proxy = self.svc["proxy-glm53"]
        self.assertIn(f"VLLM_BACKEND_URLS=http://{gen.R1}:8000,http://{gen.R2}:8000\n", proxy)
        self.assertIn("MODEL_NAME=z-ai/glm-5.3-flash\n", proxy)
        self.assertIn('"port:8009"', proxy)
        self.assertIn("      - proxy-glm53\n", self.svc["nginx"])
        self.assertIn('      - "8009:8009"\n', self.svc["nginx"])

    def test_nginx_base_only(self) -> None:
        conf = self.text.split("  nginx_conf:\n", 1)[1].split("  nginx_qwen38_conf:\n", 1)[0]
        plain = conf.split("          listen 8009;\n", 1)[1].split("      }\n", 1)[0]
        self.assertEqual(conf.count("listen 8009;"), 1)
        self.assertIn("location / { proxy_pass http://proxy-glm53:8000; }", plain)
        self.assertNotIn("auth_request", plain)
        self.assertNotIn("glm-5-3-flash-long", conf)
        self.assertNotIn("default_server", conf)
        vhost = conf.split("server_name glm-5-3-flash.completions.near.ai\n", 1)[1].split("      }\n", 1)[0]
        self.assertIn(r'"~^glm-5-3-flash-b[0-9a-f]{12}\.completions(-stg)?\.near\.ai$$"', vhost)
        self.assertIn(r"~^glm-5-3-flash-i\d+\.completions(-stg)?\.near\.ai$$", vhost)
        self.assertIn("location / { proxy_pass http://proxy-glm53:8000; }", vhost)

    def test_registrar_base_domain_only(self) -> None:
        script = self.text.split("  registrar_script:\n", 1)[1].split("  nginx_conf:\n", 1)[0]
        self.assertIn('register_model "z-ai/glm-5.3-flash" "glm-5-3-flash.completions.near.ai"', script)
        self.assertNotIn("glm-5.3-flash-long", script)
        self.assertNotIn("glm-5-3-flash-long", script)
        self.assertIn('8009) check_chat_saturated "$${HOST_IP}:8009" "z-ai/glm-5.3-flash"', script)
        self.assertIn("for port in 8002 8003 8004 8005 8006 8007 8008 8009; do", script)

    def test_scrape_jobs(self) -> None:
        new, old = jobs(self.text), jobs(self.source)
        self.assertEqual([k for k in new if "glm53" not in k], [k for k in old if "dsv41" not in k])
        for name in (f"sglang-{gen.R1}", f"sglang-{gen.R2}", "ghost-aggregator-glm53-ghost-aggregator", "dcgm-dcgm-glm53", "inference-proxy-proxy-glm53"):
            self.assertIn(name, new)
        for key in old:
            if "dsv41" not in key:
                self.assertEqual(new[key], old[key], key)

    def test_no_deepseek_left(self) -> None:
        body = self.text.replace("(MODEL_NAME=deepseek-ai/DeepSeek-V4.1-Flash-1017)", "").replace("The DeepSeek-V4.1 checkpoint", "")
        for needle in ("dsv41", "DeepSeek", "deepseek", "humming", "network_mode: host"):
            self.assertNotIn(needle, body)

    @unittest.skipUnless(shutil.which("ruby"), "ruby required")
    def test_validator_rejects_gpu_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(ROOT / "prod", root / "prod")
            (root / "scripts").mkdir()
            shutil.copyfile(ROOT / VALIDATOR, root / VALIDATOR)
            cases = (
                (self.text.replace('device_ids: ["4","5"]', 'device_ids: ["3","5"]', 1), "must use GPUs 4,5"),
                (self.text.replace("    - CUDA_DEVICE_ORDER=PCI_BUS_ID\n", "", 1), "must set CUDA_DEVICE_ORDER=PCI_BUS_ID"),
                (self.text.replace("z-ai/glm-5.3-flash", "deepseek-v4-flash", 1), "retired DS4F identities"),
            )
            for mutated, message in cases:
                with self.subTest(message=message):
                    self.assertNotEqual(mutated, self.text)
                    (root / gen.TARGET).write_text(mutated)
                    result = subprocess.run(["ruby", str(root / VALIDATOR)], cwd=root, capture_output=True, text=True, check=False)
                    self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn(message, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
