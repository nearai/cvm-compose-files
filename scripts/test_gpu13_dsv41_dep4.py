#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: python3 -m unittest scripts.test_gpu13_dsv41_dep4
"""gpu13 DS4.1 DEP4 parity with gpu02's generated r2 replica."""

import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Final

from scripts.test_dsv41_lora_dep8 import service_blocks

ROOT: Final = Path(__file__).resolve().parents[1]
SMALL: Final = Path("prod/small-models.yaml")
PACK: Final = Path("prod/DeepSeek-V4.1-Flash-1017-vLLM-DEP4x2-Abliterated.yaml")
VALIDATOR: Final = Path("scripts/validate_ds4f_migration.rb")
RANKS: Final = tuple(f"model-vllm-dsv41-dp{gpu}" for gpu in range(4, 8))
BACKENDS: Final = ",".join(f"http://{name}:{9104 + index}" for index, name in enumerate(RANKS))
MODELS: Final = (
    ("deepseek-ai/DeepSeek-V4.1-Flash-1017", "deepseek-v4-1-flash.completions.near.ai"),
    ("deepseek-ai/DeepSeek-V4.1-Flash-1017-abliterated", "deepseek-v4-1-flash-abliterated.completions.near.ai"),
)


def anchor(text: str, name: str) -> str:
    """Return one top-level Compose anchor and its body."""
    match = re.search(rf"^{re.escape(name)}:.*?(?=^x-[\w-]+:|^services:)", text, re.MULTILINE | re.DOTALL)
    if match is None:
        raise AssertionError(f"missing anchor {name}")
    return match.group().rstrip()


def config(text: str, name: str) -> str:
    """Return one inline Compose config and its body."""
    match = re.search(rf"^  {re.escape(name)}:\n.*?(?=^  [\w-]+:\n|^volumes:|\Z)", text[text.index("\nconfigs:\n"):], re.MULTILINE | re.DOTALL)
    if match is None:
        raise AssertionError(f"missing config {name}")
    return match.group()


def environment(block: str) -> list[str]:
    """Return environment entries without their YAML indentation or comments."""
    env = block.split("    environment:\n", 1)[1].split("\n    labels:", 1)[0]
    return [line.strip()[2:] for line in env.splitlines() if line.strip().startswith("- ")]


class Gpu13Dsv41Dep4Test(unittest.TestCase):
    def setUp(self) -> None:
        self.small = (ROOT / SMALL).read_text()
        self.pack = (ROOT / PACK).read_text()

    def test_rank_blocks_match_gpu02_r2_with_device_order(self) -> None:
        gpu13 = service_blocks(self.small)
        gpu02 = service_blocks(self.pack)
        for name in RANKS:
            with self.subTest(rank=name):
                block = gpu13[name]
                self.assertEqual(block.count("      - CUDA_DEVICE_ORDER=PCI_BUS_ID\n"), 1)
                self.assertRegex(block, r"(?m)^      - CUDA_VISIBLE_DEVICES=[4-7]\n      # gpu13 only:")
                stripped = re.sub(r"      # gpu13 only:.*\n(?:      #.*\n)*      - CUDA_DEVICE_ORDER=PCI_BUS_ID\n", "", block, count=1)
                stripped = stripped.split("\n  # ---", 1)[0]
                self.assertEqual(stripped.rstrip(), gpu02[name].rstrip())

    def test_proxy_environments_and_labels_match_gpu02_except_ports(self) -> None:
        gpu13 = service_blocks(self.small)
        gpu02 = service_blocks(self.pack)
        for name, gpu02_port in (("proxy-dsv41", "8000"), ("proxy-dsv41-abliterated", "8002")):
            with self.subTest(proxy=name):
                actual = environment(gpu13[name])
                expected = environment(gpu02[name])
                self.assertEqual([entry for entry in actual if not entry.startswith("VLLM_BACKEND_URLS=")],
                                 [entry for entry in expected if not entry.startswith("VLLM_BACKEND_URLS=")])
                self.assertEqual([entry for entry in actual if entry.startswith("VLLM_BACKEND_URLS=")],
                                 [f"VLLM_BACKEND_URLS={BACKENDS}"])
                actual_labels = gpu13[name].split("    labels:\n", 1)[1].split("\n  ", 1)[0].rstrip()
                expected_labels = gpu02[name].split("    labels:\n", 1)[1].split("\n  ", 1)[0].rstrip()
                self.assertEqual(expected_labels.count(f'"port:{gpu02_port}"'), 1)
                self.assertEqual(actual_labels, expected_labels.replace(f'"port:{gpu02_port}"', '"port:8009"'))

    def test_engine_anchor_matches_gpu02_except_cache_volume(self) -> None:
        expected = anchor(self.pack, "x-dsv41-engine-common").replace("huggingface_cache:/root/.cache/huggingface", "hugginface_cache:/root/.cache/huggingface")
        self.assertEqual(anchor(self.small, "x-dsv41-engine-common"), expected)

    def test_downloader_revisions_match_gpu02(self) -> None:
        pattern = r"uvx --from 'huggingface_hub\[hf_xet\]' hf download [^\n]+"
        self.assertEqual(re.findall(pattern, service_blocks(self.small)["model-downloader"]),
                         re.findall(pattern, service_blocks(self.pack)["model-downloader"]))
        self.assertEqual(len(re.findall(pattern, service_blocks(self.small)["model-downloader"])), 2)

    def test_registrar_registers_only_ds41_models(self) -> None:
        registrar = config(self.small, "registrar_script")
        calls = re.findall(r'register_model "([^"$]+)" "([^"$]+)"', registrar)
        self.assertEqual([pair for pair in calls if "DeepSeek-V4.1" in pair[0]], list(MODELS))
        self.assertFalse(any("glm" in model.lower() for model, _ in calls))

    def test_nginx_discovery_and_sni(self) -> None:
        nginx = config(self.small, "nginx_conf")
        server = re.search(r"(?m)^      server \{\n\s+listen 8009;.*?^      \}", nginx, re.DOTALL | re.MULTILINE)
        self.assertIsNotNone(server)
        stub = server.group() if server else ""
        self.assertIn("auth_request", stub)
        self.assertIn("location = /v1/models", stub)
        for model, domain in MODELS:
            with self.subTest(model=model):
                self.assertIn(f'"id":"{model}"', stub)
                stem = domain.removesuffix(".completions.near.ai")
                vhosts = [block for block in re.findall(r"(?m)^      server \{\n.*?^      \}", nginx, re.MULTILINE | re.DOTALL)
                          if "listen 443 ssl http2;" in block and f"server_name {domain}" in block]
                self.assertEqual(len(vhosts), 1)
                names = vhosts[0] if vhosts else ""
                self.assertIn(f"{stem}.completions-stg.near.ai", names)
                self.assertIn(rf"~^{stem}-i\d+\.completions(-stg)?\.near\.ai$$", names)
                self.assertIn(rf"~^{stem}-b[0-9a-f]{{12}}\.completions(-stg)?\.near\.ai$$", names)
                proxy = "proxy-dsv41-abliterated" if model.endswith("-abliterated") else "proxy-dsv41"
                self.assertIn(f"proxy_pass http://{proxy}:8000;", names)
        self.assertNotIn("glm", nginx.lower())

    @unittest.skipUnless(shutil.which("ruby"), "ruby required")
    def test_validator_rejects_gpu_and_identity_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(ROOT / "prod", root / "prod")
            (root / "scripts").mkdir()
            shutil.copyfile(ROOT / VALIDATOR, root / VALIDATOR)
            original = self.small
            cases = (
                (original.replace("CUDA_VISIBLE_DEVICES=4", "CUDA_VISIBLE_DEVICES=3", 1), "must use GPU 4"),
                (original.replace("      - CUDA_DEVICE_ORDER=PCI_BUS_ID\n", "", 1), "must set CUDA_DEVICE_ORDER=PCI_BUS_ID"),
                (original.replace("DeepSeek-V4.1-Flash-1017", "DeepSeek-V4-Flash-0731", 1), "retired DS4F identities"),
            )
            for mutated, message in cases:
                with self.subTest(message=message):
                    self.assertNotEqual(mutated, original)
                    (root / SMALL).write_text(mutated)
                    result = subprocess.run(["ruby", str(root / VALIDATOR)], cwd=root, capture_output=True, text=True, check=False)
                    self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn(message, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
