#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: python3 -m scripts.test_dsv41_lora_dep8
"""Contracts for the generated DS4.1 prod and staging packs (DEP8 and 2xDEP4)."""

import unittest
from pathlib import Path
from typing import Final

from scripts import prepare_dsv41_lora_dep8 as generator


ROOT: Final = Path(__file__).resolve().parents[1]
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
    layout = generator.DEP8

    def rendered(self) -> str:
        return (ROOT / self.layout.target).read_text()

    def test_committed_file_matches_generator(self) -> None:
        self.assertEqual(self.rendered(), generator.render(self.layout))

    def test_registrar_registers_only_prod_domains(self) -> None:
        rendered = self.rendered()
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
        rendered = self.rendered()
        for forbidden in ("stg_certs", "stg-cert-init", "/stg-certs", "DSV41_STG_"):
            with self.subTest(forbidden=forbidden):
                self.assertFalse(forbidden in rendered, f"{forbidden} must be absent from the pack")
        blocks = service_blocks(rendered)
        for name in ("proxy-dsv41", "proxy-dsv41-abliterated"):
            with self.subTest(proxy=name):
                self.assertTrue(CERT_PATH in blocks[name], f"{name} must attest the CVM cert")

    def test_one_tls_vhost_per_model_serves_both_zones(self) -> None:
        rendered = self.rendered()
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
        rendered = self.rendered()
        blocks = service_blocks(rendered)
        self.assertEqual(rendered.count(ALLOWLIST), 1)
        self.assertIn(ALLOWLIST, blocks["proxy-dsv41-abliterated"])
        self.assertNotIn("VLLM_PROXY_ALLOWED_ORG_IDS", blocks["proxy-dsv41"])
        for name in ("proxy-dsv41", "proxy-dsv41-abliterated"):
            with self.subTest(proxy=name):
                self.assertIn("VLLM_PROXY_STRICT_REQUEST_MODEL=1", blocks[name])

    def test_all_eight_ranks_keep_queue_limit(self) -> None:
        rendered = self.rendered()
        self.assertEqual(rendered.count("--max-num-queued-reqs 96"), 8)
        blocks = service_blocks(rendered)
        for rank in range(8):
            with self.subTest(rank=rank):
                self.assertEqual(blocks[f"model-vllm-dsv41-dp{rank}"].count("--max-num-queued-reqs 96"), 1)

    def test_priority_scheduling_needs_the_negating_proxy(self) -> None:
        # Without the negation vLLM would run the OpenRouter lane (-1) and reseller orgs (-2) before cloud-api (0).
        rendered = self.rendered()
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


class GeneratedDep4x2PackTest(GeneratedPackTest):
    """Every DEP8 contract above, plus what keeps the two DEP4 replicas independent."""

    layout = generator.DEP4X2

    def test_two_independent_dep4_replicas(self) -> None:
        blocks = service_blocks(self.rendered())
        for replica, ranks, master, rpc in (("r1", range(4), 29600, 13345), ("r2", range(4, 8), 29700, 13445)):
            for local, rank in enumerate(ranks):
                block = blocks[f"model-vllm-dsv41-dp{rank}"]
                with self.subTest(rank=rank):
                    self.assertIn(f"CUDA_VISIBLE_DEVICES={rank}\n", block)
                    self.assertIn("--data-parallel-size 4\n", block)
                    self.assertIn(f"--data-parallel-rank {local}\n", block)
                    self.assertIn(f"--data-parallel-rpc-port {rpc}\n", block)
                    self.assertIn(f"VLLM_DP_MASTER_PORT={master}\n", block)
                    self.assertIn('"replica:' + replica + '"', block)
                    # A replica's ranks wait only for its own first rank, never for the other replica.
                    others = [f"model-vllm-dsv41-dp{r}:" for r in range(8) if r not in ranks]
                    self.assertFalse(any(o in block for o in others), f"dp{rank} must not depend on the other replica")

    def test_shared_host_engram_tables_read_by_cpu_gather(self) -> None:
        rendered = self.rendered()
        blocks = service_blocks(rendered)
        for rank in range(8):
            block = blocks[f"model-vllm-dsv41-dp{rank}"]
            with self.subTest(rank=rank):
                self.assertIn("VLLM_NEAR_ENGRAM_CPU_GATHER=1\n", block)
                self.assertIn("VLLM_NEAR_DISABLE_UVA=1\n", block)
                self.assertIn("""--engram-config '{"cpu_offload":true,"dp_shared_memory":true}'""", block)
                self.assertIn("""--compilation-config '{"cudagraph_mode":"PIECEWISE"}'""", block)
        # Patch 05 is only in the CC image; the #393 image would read the host tables through UVA.
        self.assertIn(generator.ENGINE_IMAGE, rendered)
        self.assertNotIn("0956859f52d262fa17084d61320c5b1ea22845825c7aab57d49ed8064cc8a396", rendered)

    def test_proxies_fail_over_from_a_stopped_replica(self) -> None:
        blocks = service_blocks(self.rendered())
        for name in ("proxy-dsv41", "proxy-dsv41-abliterated"):
            with self.subTest(proxy=name):
                self.assertIn("VLLM_BACKEND_CONNECT_FAILOVER=1\n", blocks[name])
                urls = ",".join(f"http://model-vllm-dsv41-dp{r}:{9100 + r}" for r in range(8))
                self.assertIn(f"VLLM_BACKEND_URLS={urls}\n", blocks[name])


if __name__ == "__main__":
    unittest.main()
