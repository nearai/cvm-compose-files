#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: python3 -m scripts.test_dsv41_lora_dep8
"""Contracts for the generated DS4.1 DEP8 prod and staging pack."""

import unittest
from pathlib import Path
from typing import Final

from scripts import prepare_dsv41_lora_dep8 as generator


ROOT: Final = Path(__file__).resolve().parents[1]
TARGET: Final = ROOT / "prod/DeepSeek-V4.1-Flash-1017-vLLM-DEP8-Abliterated.yaml"
ALLOWLIST: Final = "VLLM_PROXY_ALLOWED_ORG_IDS=${DSV41_ABLITERATED_ALLOWED_ORG_IDS:-__no_organizations__}"
CERT_PATH: Final = "TLS_CERT_PATH=/etc/letsencrypt/live/completions.near.ai/fullchain.pem"


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
    def test_committed_file_matches_generator(self) -> None:
        self.assertEqual(TARGET.read_text(), generator.render())

    def test_registrar_registers_only_prod_domains(self) -> None:
        rendered = TARGET.read_text()
        self.assertTrue('PROXY_URL="https://completions.near.ai"' in rendered, "registrar must use prod proxy")
        self.assertFalse('PROXY_URL="https://completions-stg.near.ai"' in rendered, "no staging registrar")
        for port, model, domain in (
            (8000, generator.BASE_ID, "deepseek-v4-1-flash.completions.near.ai"),
            (8002, generator.ADAPTER_ID, "deepseek-v4-1-flash-abliterated.completions.near.ai"),
        ):
            with self.subTest(port=port):
                self.assertTrue(f'{port}) echo "{domain}"' in rendered, f"port {port} must map to {domain}")
                self.assertTrue(
                    f'register_model "{model}" "{domain}"' in rendered,
                    f"{model} must register only on {domain}",
                )

    def test_proxies_use_cvm_certificate_without_staging_certificate_setup(self) -> None:
        rendered = TARGET.read_text()
        for forbidden in ("stg_certs", "stg-cert-init", "/stg-certs", "DSV41_STG_"):
            with self.subTest(forbidden=forbidden):
                self.assertFalse(forbidden in rendered, f"{forbidden} must be absent from the pack")
        blocks = service_blocks(rendered)
        for name in ("proxy-dsv41", "proxy-dsv41-abliterated"):
            with self.subTest(proxy=name):
                self.assertTrue(CERT_PATH in blocks[name], f"{name} must attest the CVM cert")

    def test_one_tls_vhost_per_model_serves_both_zones(self) -> None:
        rendered = TARGET.read_text()
        nginx = rendered.split("  nginx_conf:\n", 1)[1]
        server_blocks = [block.split("\n      }", 1)[0] for block in nginx.split("      server {\n")[1:]]
        for port, subdomain in (
            (8001, "deepseek-v4-1-flash"),
            (8003, "deepseek-v4-1-flash-abliterated"),
        ):
            with self.subTest(port=port):
                tls_blocks = [block for block in server_blocks if f"listen {port} ssl http2;" in block]
                self.assertEqual(len(tls_blocks), 1)
                block = tls_blocks[0]
                self.assertIn(
                    f"server_name {subdomain}.completions.near.ai\n"
                    f"                    {subdomain}.completions-stg.near.ai\n"
                    f"                    ~^{subdomain}-i\\d+\\.completions(-stg)?\\.near\\.ai$$;",
                    block,
                )
                self.assertNotIn("ssl_certificate", block)

    def test_abliterated_allowlist_only_and_strict_model_routing(self) -> None:
        rendered = TARGET.read_text()
        blocks = service_blocks(rendered)
        self.assertEqual(rendered.count(ALLOWLIST), 1)
        self.assertIn(ALLOWLIST, blocks["proxy-dsv41-abliterated"])
        self.assertNotIn("VLLM_PROXY_ALLOWED_ORG_IDS", blocks["proxy-dsv41"])
        for name in ("proxy-dsv41", "proxy-dsv41-abliterated"):
            with self.subTest(proxy=name):
                self.assertIn("VLLM_PROXY_STRICT_REQUEST_MODEL=1", blocks[name])

    def test_all_eight_ranks_keep_queue_limit(self) -> None:
        rendered = TARGET.read_text()
        self.assertEqual(rendered.count("--max-num-queued-reqs 96"), 8)
        blocks = service_blocks(rendered)
        for rank in range(8):
            with self.subTest(rank=rank):
                self.assertEqual(blocks[f"model-vllm-dsv41-dp{rank}"].count("--max-num-queued-reqs 96"), 1)

    def test_priority_scheduling_needs_the_negating_proxy(self) -> None:
        # Without the negation vLLM would run the OpenRouter lane (-1) and reseller orgs (-2) before cloud-api (0).
        rendered = TARGET.read_text()
        blocks = service_blocks(rendered)
        for rank in range(8):
            with self.subTest(rank=rank):
                self.assertEqual(blocks[f"model-vllm-dsv41-dp{rank}"].count("--scheduling-policy priority"), 1)
        for name in ("proxy-dsv41", "proxy-dsv41-abliterated"):
            with self.subTest(proxy=name):
                self.assertIn("VLLM_PROXY_PRIORITY_LOWER_FIRST=1", blocks[name])
        # 5faff1b predates VLLM_PROXY_PRIORITY_LOWER_FIRST and would ignore it.
        self.assertFalse(
            "4937ad68d86dcc79871bd52341f2030a71e49bcafdf69d82ca287171aeb43c7b" in rendered,
            "pin a proxy build that supports VLLM_PROXY_PRIORITY_LOWER_FIRST",
        )


if __name__ == "__main__":
    unittest.main()
