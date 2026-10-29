#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: python3 -m scripts.test_dsv41_lora_dep8
"""Contracts for the generated DS4.1 DEP8 staging and production packs."""

import unittest
from pathlib import Path

from scripts import prepare_dsv41_lora_dep8 as generator


ROOT = Path(__file__).resolve().parents[1]
TARGETS = {
    "staging": ROOT / "prod/DeepSeek-V4.1-Flash-1017-vLLM-DEP8-Abliterated.yaml",
    "prod": ROOT / "prod/DeepSeek-V4.1-Flash-1017-vLLM-DEP8-Abliterated-Prod.yaml",
}
FORBIDDEN_PROD = ("completions-stg", "stg_certs", "stg-cert-init", "/stg-certs", "DSV41_STG_")
ALLOWLIST = "VLLM_PROXY_ALLOWED_ORG_IDS=${DSV41_ABLITERATED_ALLOWED_ORG_IDS:-__no_organizations__}"


def service_blocks(rendered: str) -> dict[str, str]:
    """Extract two-space service blocks without parsing Compose dependencies."""
    section = rendered.split("\nservices:\n", 1)[1].split("\nnetworks:\n", 1)[0]
    blocks: dict[str, str] = {}
    name = ""
    lines: list[str] = []
    for line in section.splitlines(keepends=True):
        if line.startswith("  ") and not line.startswith("   ") and line.rstrip().endswith(":"):
            if name:
                blocks[name] = "".join(lines)
            name = line.strip()[:-1]
            lines = [line]
        else:
            lines.append(line)
    if name:
        blocks[name] = "".join(lines)
    return blocks


class GeneratedPackTest(unittest.TestCase):
    def test_committed_files_match_generator(self) -> None:
        for zone, target in TARGETS.items():
            with self.subTest(zone=zone):
                self.assertEqual(target.read_text(), generator.render(zone))

    def test_engines_downloader_and_telemetry_are_identical(self) -> None:
        staging = service_blocks(generator.render("staging"))
        prod = service_blocks(generator.render("prod"))
        self.assertEqual(staging.keys() - {"stg-cert-init"}, prod.keys())
        edge = {"proxy-dsv41", "proxy-dsv41-abliterated", "nginx", "model-proxy-registrar", "stg-cert-init"}
        for name in staging.keys() - edge:
            with self.subTest(service=name):
                self.assertEqual(staging[name], prod[name])
        for name in ("model-downloader", *(f"model-vllm-dsv41-dp{rank}" for rank in range(8))):
            self.assertIn(name, prod)

    def test_prod_routing_and_certificate(self) -> None:
        prod = generator.render("prod")
        blocks = service_blocks(prod)
        self.assertIn('PROXY_URL="https://completions.near.ai"', prod)
        self.assertIn('8000) echo "deepseek-v4-1-flash.completions.near.ai"', prod)
        self.assertIn('8002) echo "deepseek-v4-1-flash-abliterated.completions.near.ai"', prod)
        for model, domain in (
            (generator.BASE_ID, "deepseek-v4-1-flash.completions.near.ai"),
            (generator.ADAPTER_ID, "deepseek-v4-1-flash-abliterated.completions.near.ai"),
        ):
            self.assertIn(f'register_model "{model}" "{domain}"', prod)
        for forbidden in FORBIDDEN_PROD:
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, prod)
        for name in ("proxy-dsv41", "proxy-dsv41-abliterated"):
            self.assertIn("TLS_CERT_PATH=/etc/letsencrypt/live/completions.near.ai/fullchain.pem", blocks[name])

    def test_staging_routing_and_certificate(self) -> None:
        staging = generator.render("staging")
        blocks = service_blocks(staging)
        self.assertIn('PROXY_URL="https://completions-stg.near.ai"', staging)
        for domain in (
            "deepseek-v4-1-flash.completions-stg.near.ai",
            "deepseek-v4-1-flash-abliterated.completions-stg.near.ai",
        ):
            self.assertIn(domain, staging)
        for name in ("proxy-dsv41", "proxy-dsv41-abliterated"):
            self.assertIn("TLS_CERT_PATH=/stg-certs/fullchain.pem", blocks[name])

    def test_abliterated_allowlist_only(self) -> None:
        for zone in TARGETS:
            with self.subTest(zone=zone):
                rendered = generator.render(zone)
                blocks = service_blocks(rendered)
                self.assertEqual(rendered.count(ALLOWLIST), 1)
                self.assertIn(ALLOWLIST, blocks["proxy-dsv41-abliterated"])
                self.assertNotIn("VLLM_PROXY_ALLOWED_ORG_IDS", blocks["proxy-dsv41"])
                for name in ("proxy-dsv41", "proxy-dsv41-abliterated"):
                    self.assertIn("VLLM_PROXY_STRICT_REQUEST_MODEL=1", blocks[name])


if __name__ == "__main__":
    unittest.main()
