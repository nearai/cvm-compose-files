#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: python3 -m scripts.test_dsv41_lora_dep8
"""Contracts for the generated DS4.1 prod and staging packs (DEP8, 2xDEP4) and the one-host DEP8 Profile pack."""

import difflib
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

    def test_dep8_runs_combo2x_and_dep4x2_keeps_dspark3(self) -> None:
        dep8 = service_blocks(generator.render(generator.DEP8))
        dep4x2 = service_blocks(generator.render(generator.DEP4X2))
        for rank in range(8):
            with self.subTest(rank=rank):
                block = dep8[f"model-vllm-dsv41-dp{rank}"]
                self.assertIn('"num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block"', block)
                self.assertEqual(block.count("      - OMP_NUM_THREADS=8\n"), 1)
                self.assertIn("dspark5pb-mr64-engram-hbm-omp8", block)
                other = dep4x2[f"model-vllm-dsv41-dp{rank}"]
                self.assertIn('''--speculative-config \'{"method":"dspark","num_speculative_tokens":3}\'''', other)
                self.assertNotIn("OMP_NUM_THREADS", other)

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



PROFILER_FLAG: Final = (
    """--profiler-config '{"profiler":"torch","torch_profiler_dir":"/traces","torch_profiler_with_stack":false,"""
    """"torch_profiler_record_shapes":false,"ignore_frontend":true}'"""
)
NGINX_DENY: Final = "        location ~ ^/(v1/)?(start|stop)_profile { return 403; }\n"
RANK_TRACE_VOLUMES: Final = (
    "    volumes:\n      - huggingface_cache:/root/.cache/huggingface\n"
    "      - dsv41_kernel_cache:/root/.cache/vllm\n      - humming_cache:/root/.cache/humming\n"
    "      - dsv41-traces:/traces\n"
)


class GeneratedProfilePackTest(GeneratedPackTest):
    """Every DEP8 contract, plus: the Profile pack differs from DEP8 only by the profiling additions."""

    layout = generator.DEP8_PROFILE

    def test_profiler_flag_traces_volume_and_nccl_init_log_on_all_ranks(self) -> None:
        blocks = service_blocks(self.rendered())
        for rank in range(8):
            block = blocks[f"model-vllm-dsv41-dp{rank}"]
            with self.subTest(rank=rank):
                self.assertEqual(block.count(PROFILER_FLAG), 1)
                self.assertEqual(block.count(RANK_TRACE_VOLUMES), 1)
                self.assertIn("      - NCCL_DEBUG=INFO\n      - NCCL_DEBUG_SUBSYS=INIT\n", block)
                self.assertNotIn("NCCL_DEBUG=WARN", block)
        self.assertIn("\nvolumes:\n", self.rendered())
        self.assertIn("\n  dsv41-traces:\n", self.rendered())

    def test_normal_packs_have_no_profiler(self) -> None:
        for layout in (generator.DEP8, generator.DEP4X2):
            rendered = generator.render(layout)
            for absent in ("--profiler-config", "dsv41-traces", "dsv41-profiler", "NCCL_DEBUG=INFO", "_profile"):
                with self.subTest(layout=layout.deployment, absent=absent):
                    self.assertNotIn(absent, rendered)

    def test_nginx_denies_profile_routes_in_every_server_block(self) -> None:
        nginx = self.rendered().split("  nginx_conf:\n", 1)[1]
        server_blocks = nginx.split("      server {\n")[1:]
        self.assertEqual(len(server_blocks), 4)
        for block in server_blocks:
            with self.subTest(block=block.split("\n", 1)[0]):
                self.assertEqual(block.count(NGINX_DENY), 1)

    def test_profiler_service_is_a_one_shot_on_the_engine_image(self) -> None:
        block = service_blocks(self.rendered())["dsv41-profiler"]
        self.assertIn(f"    image: {generator.ENGINE_IMAGE}\n", block)
        self.assertIn('    restart: "no"\n', block)
        self.assertIn("    network_mode: host\n", block)
        self.assertIn('    entrypoint: ["python3", "/profile_summary.py"]\n', block)
        self.assertIn("      - dsv41-traces:/traces\n", block)
        self.assertIn("      - PROFILE_DELAY_S=900\n      - PROFILE_DUR_S=3\n", block)
        self.assertIn("logging: *logging-conf", block)
        self.assertIn('"source":"dsv41-profiler"', block)
        for absent in ("runtime:", "nvidia", "reservations", "<<:"):
            self.assertNotIn(absent, block)

    def test_profile_summary_config_embeds_the_script(self) -> None:
        rendered = self.rendered()
        self.assertIn("      - source: profile_summary\n        target: /profile_summary.py\n", rendered)
        config = rendered.split("  profile_summary:\n    content: |\n", 1)[1].split("  nginx_conf:\n", 1)[0]
        script = (ROOT / "scripts/dsv41_profile_summary.py").read_text()
        self.assertEqual(config.replace("$$", "$"), "".join(f"      {l}" if l.strip() else "\n" for l in script.splitlines(True)) + "\n")

    def test_only_the_profiling_additions_differ_from_dep8(self) -> None:
        dep8 = generator.render(generator.DEP8)
        profile = self.rendered()
        # The two whole-block additions are checked on their own; the rest is diffed line by line.
        block = service_blocks(profile)["dsv41-profiler"]
        self.assertEqual(profile.count(block), 1)
        profile = profile.replace(block, "")
        start = profile.index("  profile_summary:\n")
        profile = profile[:start] + profile[profile.index("  nginx_conf:\n"):]
        removed: list[str] = []
        added: list[str] = []
        for line in difflib.unified_diff(dep8.splitlines(True), profile.splitlines(True), n=0):
            if line.startswith(("---", "+++", "@@")):
                continue
            (added if line[0] == "+" else removed).append(line[1:])
        variant, profile_variant = generator.DEP8.variant, generator.DEP8_PROFILE.variant
        self.assertEqual(profile_variant, variant + "-profile")
        rank_volume_lines = set(RANK_TRACE_VOLUMES.splitlines(True))
        allowed_added = {
            "      - NCCL_DEBUG=INFO\n", "      - NCCL_DEBUG_SUBSYS=INIT\n", NGINX_DENY, "  dsv41-traces:\n",
            "      # Profile pack: the boot log names NCCL's transport (P2P/CUMEM vs SHM/NET).\n",
            f"      {PROFILER_FLAG}\n",
        } | rank_volume_lines
        for line in added:
            with self.subTest(added=line[:80]):
                self.assertTrue(
                    line in allowed_added or line.startswith("#") or profile_variant in line,
                    f"unexpected added line: {line!r}",
                )
        for line in removed:
            with self.subTest(removed=line[:80]):
                self.assertTrue(
                    line == "      - NCCL_DEBUG=WARN\n" or line.startswith("#") or variant in line,
                    f"unexpected removed line: {line!r}",
                )
        self.assertEqual(added.count(NGINX_DENY), 4)
        self.assertEqual(added.count(f"      {PROFILER_FLAG}\n"), 8)
        self.assertEqual(removed.count("      - NCCL_DEBUG=WARN\n"), 8)
        self.assertEqual(added.count("  dsv41-traces:\n"), 1)
        self.assertEqual(added.count("      - dsv41-traces:/traces\n"), 8)
        # Variant-labelled lines change only by the suffix (config_variant labels and Datadog tags).
        self.assertEqual(
            sorted(l.replace(profile_variant, variant) for l in added if profile_variant in l),
            sorted(l for l in removed if variant in l),
        )
        self.assertEqual(profile.replace(profile_variant, variant).count("--profiler-config"), 8)


if __name__ == "__main__":
    unittest.main()
