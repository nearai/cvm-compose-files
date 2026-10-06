#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: uv run scripts/prepare_glm53_w4afp8_tp2x4.py --write
"""Generate the 4x TP2 base-tier canary file from the W4AFP8 base-tier file.

The two TP4/EP4 replicas become four TP2/EP2 replicas, one per NVLink GPU pair, on the
HiCache + W4AFP8 image the long tier runs, with a 325 GiB write_through_selective host tier
per replica and the DSA indexer split. Everything else (domains, nginx, registrar, proxy
behavior, DCGM, the OTel pipeline) stays byte-identical to the source apart from the replica
names, the per-replica fan-out of the proxy pool, scrape jobs, perception check and soak
relay, and the deployment label that splits the canary host on dashboards. `--write`
regenerates the target; `--check` prints a diff and exits non-zero when the committed
target is stale.
"""

import argparse
import difflib
import sys
from pathlib import Path
from typing import Final

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("prod/GLM-5.3-Flash-SGL-TP4-W4AFP8.yaml")
TARGET = Path("prod/GLM-5.3-Flash-SGL-TP2x4-W4AFP8.yaml")

SOURCE_IMAGE: Final = "docker.io/nearaidev/sglang@sha256:8bce6a7cc872a80faded3bd1ef0a64873a1d7abae34c94e5358775ca21f133cc"
SOURCE_ENGINE_IMAGE_LABEL: Final = "8bce6a7cc872"
# The image both gpu02 long-tier replicas run (prod/GLM-5.3-Flash-SGL-TP4-W4AFP8-LongContext.yaml).
IMAGE: Final = "docker.io/nearaidev/sglang@sha256:47aff791090003a37f893e998c44794c410d3f7bdfc7fdd2dfab5eb5592b30bb"
ENGINE_IMAGE_LABEL: Final = "47aff7910900"
SOURCE_SERVICE_PREFIX: Final = "model-sg-glm53-w4afp8-tp4-r"
SERVICE_PREFIX: Final = "model-sg-glm53-w4afp8-tp2-r"
SOURCE_DEPLOYMENT: Final = "glm53-flash-sgl-tp4"
DEPLOYMENT: Final = "glm53-flash-sgl-tp2x4"
SOURCE_VARIANT: Final = "fc91d24-w4afp8-c4096-admission-reserve-v10-pool-clamp-pdi1-h200-tp4-ep4-eagle-adaptive-5-1-6-strict-budget8192"
VARIANT: Final = "hicache-w4afp8-qsplit-selective325-mamba165-bf16state-c8192-admission-reserve-v10-pdi1-h200-tp2-ep2-eagle-adaptive-5-1-6-strict-budget8192"
REPLICAS: Final = (1, 2, 3, 4)
# Base-tier memory-optimized config (tee-bench exp 25/25b/25c), promoted from the r3/r4 canary
# (#339) to all four replicas after the gpu03 30 min same-host bake (2026-10-06). Every replica
# runs the candidate argv below; the shared engine anchor keeps the previous prod argv only as
# the base the candidate edits are derived from.
CANDIDATE_REPLICAS: Final = (1, 2, 3, 4)
CANDIDATE_PDI: Final = "2"
CANDIDATE_VARIANT: Final = f"hicache-w4afp8-qsplit-selective325-mamba330-bf16state-memopt086-mr48-c8192-admission-reserve-v10-pdi{CANDIDATE_PDI}-h200-tp2-ep2-eagle-fixed-4-1-5-strict-budget8192"
# Token-for-token edits of the control argv. Each old token must occur exactly once.
CANDIDATE_EDITS: Final = (
    ("--mem-fraction-static 0.80", "--mem-fraction-static 0.86"),
    ("--max-running-requests 32", "--max-running-requests 48"),
    ("--prefill-decode-interval 1", f"--prefill-decode-interval {CANDIDATE_PDI}"),
    ("--cuda-graph-max-bs-decode 32", "--cuda-graph-max-bs-decode 48"),
    ("--speculative-num-steps 5", "--speculative-num-steps 4"),
    ("--speculative-num-draft-tokens 6", "--speculative-num-draft-tokens 5"),
    ("--speculative-adaptive", None),
    ("--max-mamba-cache-size 165", "--max-mamba-cache-size 330"),
)
# Each pair sits inside one four-GPU NVLink island.
DEVICE_IDS: Final = {1: ("0", "1"), 2: ("2", "3"), 3: ("4", "5"), 4: ("6", "7")}
SOAK_PORTS: Final = {1: 8008, 2: 8009, 3: 8010, 4: 8011}
HICACHE_FLAGS: Final = (
    "--enable-hierarchical-cache",
    "--hicache-write-policy write_through_selective",
    "--hicache-io-backend direct",
    "--hicache-mem-layout page_first_direct",
)
# 165 mamba state slots (5 per request) lift the per-replica running cap from 15 to 32. BF16
# SSM state halves the state memory so the VRAM KV pool stays at ~1.0M tokens (FP32 state at
# 165 slots left 67K tokens; raising --mem-fraction-static to 0.88 instead OOMed under load).
MAMBA_FLAGS: Final = (
    "--max-mamba-cache-size 165",
    "--mamba-ssm-dtype bfloat16",
)

HEADER: Final = (
    "# GLM-5.3 Flash base-tier 4x TP2 CANARY (one 8x H200 host: gpu03 or gpu04), generated from\n"
    "# prod/GLM-5.3-Flash-SGL-TP4-W4AFP8.yaml by scripts/prepare_glm53_w4afp8_tp2x4.py.\n"
    "# Four independent TP2/EP2 replicas, one per NVLink GPU pair (0-1, 2-3, 4-5, 6-7), replace\n"
    "# the two TP4/EP4 replicas. Same checkpoint (graphistry/GLM-5.3-Flash-W4AFP8@99f1fa7) and\n"
    "# the same admission reserve (4096, max fraction 0.75), plus 8192-token prefill chunks\n"
    "# (the lab-qualified TP2 argv), HiCache with a write_through_selective 325 GiB host tier\n"
    "# per replica (4 x 325 GiB = 1,300 GiB) and the DSA indexer split. All four pin the\n"
    "# HiCache + W4AFP8 image the long tier runs:\n"
    f"#   {IMAGE}\n"
    "# (docker/sglang-glm53-hicache-w4afp8, original #308 offloop-v3; source\n"
    "# aff61fca1798512dcaec8cc88756ee0f83bb78be; workflow 36210851936 SUCCESS).\n"
    "#\n"
    "# EVIDENCE (lab, gpu31/gpu32, CC off, 2026-10-01; 1,300 GiB DRAM per 8 GPUs in both arms;\n"
    "# base-tier synth traffic from the 2026-09-17 prod snapshot, least-conn + affinity router):\n"
    "#   - Peak served throughput per 8 GPUs: 4xTP2 8.18 req/s / 1,871 output tok/s vs 2xTP4\n"
    "#     6.49 req/s / 1,545 tok/s (+26% / +21%). Unserved backlog 4.7% vs 18% at 8 req/s\n"
    "#     offered, 16% vs 37% at 10 req/s. 0 HTTP errors in every arm.\n"
    "#   - Lower TTFT for prompts <=10K at every load. Long prompts (30K+) are slower at low load:\n"
    "#     TP2 prefills with half the GPUs.\n"
    "#   - Hit rate unchanged: 84-85% synth; agent-trace replay 93.1% (4xTP2) vs 93.2% (2xTP4)\n"
    "#     at 96 agents.\n"
    "#   - TP2: 81.7 GiB weights per GPU, 1.08M-token VRAM KV pool per replica, and a 5.87M-token\n"
    "#     DRAM tier per replica at 325 GiB.\n"
    "#   - write_through_selective (vs write_through) cut recomputed tokens 17-25% on the replay\n"
    "#     with no throughput/ITL cost.\n"
    "#   - Running cap: 165 mamba state slots with BF16 SSM state (--max-mamba-cache-size 165\n"
    "#     --mamba-ssm-dtype bfloat16) let each TP2 replica run 32 requests instead of 15, with a\n"
    "#     1.0M-token VRAM pool and a 5.4M-token DRAM tier. Lab at 10 req/s offered per 8 GPUs vs\n"
    "#     the 15-running canary: 8.83 vs 7.74 req/s served (+14%), 2,064 vs 1,771 output tok/s\n"
    "#     (+16%), TTFT p50/p90 0.39/3.20 vs 1.60/5.56 s, backlog 228 vs 925, hit rate 76.8% vs\n"
    "#     71.5%. Cost: TPOT p90 188 vs 70 ms (bigger decode batches). Saturates at ~8.8 req/s.\n"
    "#\n"
    "# MEMORY-OPTIMIZED CONFIG (docs/glm53-base-tier-memopt-canary.md): all four replicas run the\n"
    "# candidate argv (mem 0.86, EAGLE fixed 4/1/5, 330 mamba slots, 48 running, pdi 2; tee-bench exp\n"
    "# 25/25b/25c), promoted from the r3/r4 canary after the gpu03 same-host bake on 2026-10-06. On\n"
    "# gpu32 bare metal under overload it served +16% (1.75 conv/s) / +25% (2.5 conv/s) more requests.\n"
    "#\n"
    "# GATES before any deploy (docs/glm53-tp2x4-base-canary.md): (1) quality with BF16 mamba\n"
    "# state at parity - PASSED 2026-10-02 (GSM8K 97.8% vs 97.4% FP32 state, perception 7/7); (2) a prod-CVM\n"
    "# test on a drained host agreed with Lloyd: CC-on startup of 4 x 325 GiB pinned host tiers\n"
    "# (cudaMallocHost), TP2 NCCL under PPCIe, CVM MemAvailable headroom, cold-start time;\n"
    "# (3) the image already pulled on the host; (4) Pranav + Lloyd go.\n"
    "#\n"
    "# None of the four engine names exists in the TP4 file (and vice versa): stop the old\n"
    "# engines with a scoped compose/down first; never let --remove-orphans do the switch.\n"
    "# ROLLBACK: prod/GLM-5.3-Flash-SGL-TP4-W4AFP8.yaml, under the same orphan rule.\n"
    "# Do not hand-edit this file.\n"
)

SOURCE_HEADER_START: Final = "# GLM-5.3 Flash base tier on W4AFP8 (gpu03, gpu04, gpu23), generated from\n"
SOURCE_HEADER_END: Final = "# Do not hand-edit this file.\n"

TEXT_REPLACEMENTS: Final = (
    (
        "# W4AFP8 variant of the canonical GLM-5.3 Flash deployment for 8x H200 base-tier hosts.\n"
        "# Run the exact same committed tag and file on every base-tier host: two independent\n"
        "# TP4/EP4 replicas, one per four-GPU NVLink island, behind one inference proxy.\n",
        "# 4x TP2 variant of the W4AFP8 GLM-5.3 Flash base-tier deployment for one 8x H200 canary\n"
        "# host: four independent TP2/EP2 replicas, one per NVLink GPU pair, behind one inference\n"
        "# proxy.\n",
        "file header",
    ),
    (
        "# Production GLM-5.3-Flash on two independent H200 TP4/EP4 replicas. Each\n"
        "# replica stays within one four-GPU NVLink island; one inference-proxy uses its native\n"
        "# least-connections backend pool for health-aware balancing and failover.\n",
        "# GLM-5.3-Flash on four independent H200 TP2/EP2 replicas. Each replica stays\n"
        "# within one NVLink GPU pair; one inference-proxy uses its native\n"
        "# least-connections backend pool for health-aware balancing and failover.\n",
        "topology comment",
    ),
    (
        "# DSA import-cycle fixes. The production digest is the docker/sglang-glm53-w4afp8\n"
        "# derivative: the docker/sglang-glm53-admission-reserve build (opt-in chunked-prefill\n"
        "# admission-reserve v10 patch, nearai/inference-optimizer@fb94472) plus the W4AFP8\n"
        "# loader fix and the unconditional chunked-prefill pool clamp. The reserve is enabled\n",
        "# DSA import-cycle fixes. All four replicas pin the published split-capable\n"
        "# docker/sglang-glm53-hicache-w4afp8 derivative the long tier runs (original #308 v3):\n"
        "# the HCC-safe HiCache image (CUDA-owned host memory, carrying the opt-in chunked-prefill\n"
        "# admission-reserve v10 patch, nearai/inference-optimizer@fb94472) plus the W4AFP8\n"
        "# loader fix and the unconditional chunked-prefill pool clamp. The reserve is enabled\n",
        "image provenance comment",
    ),
    (
        "# intentionally conservative: BF16 KV, 0.80 static memory, 32 running requests, a\n",
        "# intentionally conservative: BF16 KV, 0.80 static memory, 32 running requests (an\n"
        "# effective 32 per TP2 replica with 165 BF16 mamba state slots, see the file header), a\n",
        "running requests comment",
    ),
    (
        "# bounded 8-request queue, 4096-token prefill chunks with --max-prefill-tokens 32768,\n",
        "# bounded 8-request queue, 8192-token prefill chunks with --max-prefill-tokens 32768,\n",
        "serving envelope comment",
    ),
    (
        "    # Admission-reserve v10 (docker/sglang-glm53-admission-reserve): reserve part of each\n"
        "    # 4096-token prefill chunk for waiting requests, sized on their uncached extend length,\n",
        "    # Admission-reserve v10 (carried by this image): reserve up to 4096 tokens of each\n"
        "    # 8192-token prefill chunk for waiting requests, sized on their uncached extend length,\n",
        "admission reserve comment",
    ),
    (
        "Registered z-ai/glm-5.3-flash (2x TP4/EP4) at",
        "Registered z-ai/glm-5.3-flash (4x TP2/EP2) at",
        "registrar log line",
    ),
)

ANCHOR_ENV_OLD: Final = "    - SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=1\n"
ANCHOR_ENV_NEW: Final = (
    "    - SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=1\n"
    "    # HiCache host tier: a fixed 325 GiB per replica across its two TP ranks, 1,300 GiB for\n"
    "    # the four replicas (the lab arm's DRAM total per 8 GPUs; ~5.87M tokens per replica). A\n"
    "    # percentage would resolve against MemAvailable at each start, so the replicas started\n"
    "    # later would get less; startup fails if it exceeds available RAM minus 10 GiB.\n"
    "    - SGLANG_HICACHE_RAM_BUDGET=${GLM53_HICACHE_RAM_BUDGET:-325GiB}\n"
    "    # CUDA-owned host memory (cudaMallocHost). Must stay 1 on TEE hosts: 0 selects\n"
    "    # cudaHostRegister, which fails with CUDA error 801 under HCC/PPCIe.\n"
    "    - SGLANG_HICACHE_CUDA_HOST_MEMORY=${GLM53_HICACHE_CUDA_HOST_MEMORY:-1}\n"
    "    - SGLANG_HICACHE_POOLED_TRANSFERS=1\n"
    "    - SGLANG_HICACHE_STAGING_PAGES=64\n"
    "    # DSA indexer query split. Each TP rank scores 1/TP of a prefill chunk's rows instead of\n"
    "    # all of them, so the fp32 logits buffer -- sized new_tokens x total_context -- shrinks\n"
    "    # by TP (halves at TP2). gpu02's long-tier r1 crashed on 2026-09-25 without it:\n"
    "    # deep_gemm.fp8_mqa_logits asked for 11.96 GiB with 11.88 GiB free at ~392K context\n"
    "    # (dsa_indexer_kpool.py:919). This is a MITIGATION, not the fix: it divides the buffer\n"
    "    # by TP and does not bound it. The real fix is calling _should_chunk_mqa_logits\n"
    "    # (defined, unused, at dsa_indexer_kpool.py:862) and chunking against free memory.\n"
    "    - SGLANG_DSA_INDEXER_QSPLIT=1\n"
)


class GenerationError(ValueError):
    pass


def replace_exact(text: str, old: str, new: str, expected: int, label: str) -> str:
    actual = text.count(old)
    if actual != expected:
        raise GenerationError(f"{label}: expected {expected} matches, found {actual}")
    return text.replace(old, new)


def section(text: str, start_marker: str, end_marker: str, label: str) -> tuple[int, int, str]:
    try:
        start = text.index(start_marker)
        end = text.index(end_marker, start + len(start_marker))
    except ValueError as error:
        raise GenerationError(f"{label}: source structure changed") from error
    return start, end, text[start:end]


def engine_arguments(source: list[str]) -> list[str]:
    """The W4AFP8 base argv turned into the lab-qualified TP2 argv, order preserved."""
    if not source or source[0] != "sglang serve":
        raise GenerationError("engine command must start with sglang serve")
    replaced = {
        "--tp-size 4": "--tp-size 2",
        "--ep-size 4": "--ep-size 2",
        "--chunked-prefill-size 4096": "--chunked-prefill-size 8192",
    }
    kept = ("--max-prefill-tokens 32768", "--prefill-decode-interval 1", "--max-running-requests 32", "--max-queued-requests 8")
    missing = sorted(argument for argument in (*replaced, *kept) if source.count(argument) != 1)
    if missing:
        raise GenerationError(f"source engine command changed: {missing}")
    owned = ("--hicache", "--enable-hierarchical-cache", "--max-mamba-cache-size", "--mamba-ssm-dtype")
    present = sorted(argument for argument in source if argument.startswith(owned))
    if present:
        raise GenerationError(f"source engine command already carries HiCache or mamba-cache options this generator adds: {present}")
    return [replaced.get(argument, argument) for argument in source] + list(HICACHE_FLAGS) + list(MAMBA_FLAGS)


def candidate_arguments(control: list[str]) -> list[str]:
    """The control TP2 argv turned into the memory-optimized candidate argv, order preserved."""
    missing = sorted(old for old, _ in CANDIDATE_EDITS if control.count(old) != 1)
    if missing:
        raise GenerationError(f"control engine command changed, cannot derive the candidate argv: {missing}")
    edits = dict(CANDIDATE_EDITS)
    candidate = [edits[argument] if argument in edits else argument for argument in control]
    return [argument for argument in candidate if argument is not None]


def candidate_anchor(arguments: list[str]) -> str:
    return (
        "x-sg-glm53-flash-candidate: &sg-glm53-flash-candidate\n"
        "  # Memory-optimized argv for every replica (tee-bench exp 25/25b/25c): the previous argv with\n"
        "  # mem 0.86, EAGLE fixed 4/1/5 (no adaptive), 330 mamba slots, 48 running / graph batch 48.\n"
        "  # Everything else, including the environment, is inherited from the common anchor.\n"
        "  <<: *sg-glm53-flash-common\n"
        "  command: >\n" + "".join(f"      {argument}\n" for argument in arguments) + "\n"
    )


def render_replica(template: str, replica: int) -> str:
    """Render one engine service from the source r1 service block."""
    block = replace_exact(
        template,
        "  # --- GLM-5.3-Flash engine replica 1 (SGLang TP4, GPUs 0-3) ---\n",
        f"  # --- GLM-5.3-Flash engine replica {replica} (SGLang TP2, GPUs {'-'.join(DEVICE_IDS[replica])}) ---\n",
        1,
        "replica comment",
    )
    block = replace_exact(block, f"{SOURCE_SERVICE_PREFIX}1", f"{SERVICE_PREFIX}{replica}", 3, "replica name")
    if replica in CANDIDATE_REPLICAS:
        block = replace_exact(block, "    <<: *sg-glm53-flash-common\n", "    <<: *sg-glm53-flash-candidate\n", 1, "candidate anchor")
        block = replace_exact(block, VARIANT, CANDIDATE_VARIANT, 2, "candidate config_variant")
    devices = ",".join(f'"{device}"' for device in DEVICE_IDS[replica])
    block = replace_exact(block, 'device_ids: ["0","1","2","3"]', f"device_ids: [{devices}]", 1, "replica devices")
    block = replace_exact(block, '"instance:1"', f'"instance:{replica}"', 1, "log instance")
    return replace_exact(block, 'nearai.otel.instance: "1"\n', f'nearai.otel.instance: "{replica}"\n', 1, "otel instance")


def render_scrape_job(template: str, replica: int) -> str:
    job = replace_exact(template, f"{SOURCE_SERVICE_PREFIX}1", f"{SERVICE_PREFIX}{replica}", 3, "scrape job name")
    if replica in CANDIDATE_REPLICAS:
        job = replace_exact(job, VARIANT, CANDIDATE_VARIANT, 1, "candidate scrape config_variant")
    return replace_exact(job, '                      instance: "1"\n', f'                      instance: "{replica}"\n', 1, "scrape instance")


def render_soak_server(template: str, replica: int) -> str:
    server = replace_exact(template, "          listen 8008 ssl;\n", f"          listen {SOAK_PORTS[replica]} ssl;\n", 1, "soak listen")
    return replace_exact(server, f"{SOURCE_SERVICE_PREFIX}1:8000", f"{SERVICE_PREFIX}{replica}:8000", 1, "soak backend")


def generate(source: str) -> str:
    if SERVICE_PREFIX in source or DEPLOYMENT in source:
        raise GenerationError("source compose already contains TP2 engines")
    if not source.startswith(SOURCE_HEADER_START) or source.count(SOURCE_HEADER_END) != 1:
        raise GenerationError("source generated header changed")
    updated = source[source.index(SOURCE_HEADER_END) + len(SOURCE_HEADER_END):]

    for old, new, label in TEXT_REPLACEMENTS:
        updated = replace_exact(updated, old, new, 1, label)

    # Shared engine anchor: image, argv and environment.
    anchor_start, anchor_end, anchor = section(
        updated, "x-sg-glm53-flash-common: &sg-glm53-flash-common\n", "\nx-dcgm-common: &dcgm-common\n", "shared engine anchor"
    )
    anchor = replace_exact(anchor, f"\n  image: {SOURCE_IMAGE}\n", f"\n  image: {IMAGE}\n", 1, "anchor image")
    command_start, command_end, command = section(anchor, "  command: >\n", "  volumes:\n", "anchor command")
    arguments = engine_arguments([line.strip() for line in command.splitlines()[1:] if line.strip()])
    anchor = anchor[:command_start] + "  command: >\n" + "".join(f"      {argument}\n" for argument in arguments) + anchor[command_end:]
    candidate = candidate_anchor(candidate_arguments(arguments))
    for required in ("    - SGLANG_CHUNKED_PREFILL_ADMISSION_RESERVE=4096\n", "    - SGLANG_ADMISSION_RESERVE_MAX_FRACTION=0.75\n"):
        if anchor.count(required) != 1:
            raise GenerationError(f"source engine environment changed: {required.strip()}")
    if "SGLANG_HICACHE_" in anchor or "SGLANG_DSA_INDEXER_QSPLIT" in anchor:
        raise GenerationError("source engine environment already carries HiCache or the indexer split")
    anchor = replace_exact(anchor, ANCHOR_ENV_OLD, ANCHOR_ENV_NEW, 1, "anchor environment")
    updated = updated[:anchor_start] + anchor + "\n" + candidate.rstrip("\n") + "\n" + updated[anchor_end:]

    # Telemetry identity shared by every replica, before the per-replica fan-out.
    for old, new, count, label in (
        (SOURCE_VARIANT, VARIANT, 6, "config_variant"),
        (f'"engine_image:{SOURCE_ENGINE_IMAGE_LABEL}"', f'"engine_image:{ENGINE_IMAGE_LABEL}"', 2, "log engine_image"),
        (f'engine_image: "{SOURCE_ENGINE_IMAGE_LABEL}"\n', f'engine_image: "{ENGINE_IMAGE_LABEL}"\n', 4, "metric engine_image"),
        (f'deployment:{SOURCE_DEPLOYMENT}"', f'deployment:{DEPLOYMENT}"', 10, "log deployment"),
        (f'deployment: "{SOURCE_DEPLOYMENT}"\n', f'deployment: "{DEPLOYMENT}"\n', 8, "metric deployment"),
    ):
        updated = replace_exact(updated, old, new, count, label)
    if SOURCE_DEPLOYMENT + '"' in updated or SOURCE_ENGINE_IMAGE_LABEL in updated:
        raise GenerationError("a source deployment or engine_image label survived")

    # Engine services: r1's block rendered once per replica.
    services_start, services_end, services = section(
        updated,
        "  # --- GLM-5.3-Flash engine replica 1 (SGLang TP4, GPUs 0-3) ---\n",
        "  # Explicit operator-only semantic check;",
        "engine services",
    )
    _, _, r1_block = section(
        services, "  # --- GLM-5.3-Flash engine replica 1", "  # --- GLM-5.3-Flash engine replica 2 (SGLang TP4, GPUs 4-7) ---\n", "r1 service"
    )
    if services.count(f"  {SOURCE_SERVICE_PREFIX}") != 2:
        raise GenerationError("source must define exactly two TP4 engine services")
    updated = updated[:services_start] + "".join(render_replica(r1_block, replica) for replica in REPLICAS) + updated[services_end:]

    # Proxy pool.
    updated = replace_exact(
        updated,
        f"VLLM_BACKEND_URLS=http://{SOURCE_SERVICE_PREFIX}1:8000,http://{SOURCE_SERVICE_PREFIX}2:8000\n",
        "VLLM_BACKEND_URLS=" + ",".join(f"http://{SERVICE_PREFIX}{replica}:8000" for replica in REPLICAS) + "\n",
        1,
        "proxy backends",
    )

    # Perception check loops over every replica.
    updated = replace_exact(updated, "        for replica in (1, 2):\n", "        for replica in (1, 2, 3, 4):\n", 1, "perception replicas")
    updated = replace_exact(
        updated,
        f'            base = f"http://{SOURCE_SERVICE_PREFIX}{{replica}}:8000"\n',
        f'            base = f"http://{SERVICE_PREFIX}{{replica}}:8000"\n',
        1,
        "perception base",
    )

    # Soak relay: one published port and one TLS server per replica.
    updated = replace_exact(
        updated,
        '      - "8008:8008"\n      - "8009:8009"\n',
        "".join(f'      - "{SOAK_PORTS[replica]}:{SOAK_PORTS[replica]}"\n' for replica in REPLICAS),
        1,
        "soak relay ports",
    )
    servers_start, servers_end, servers = section(updated, "        server {\n          listen 8008 ssl;\n", "      }\n\n  dcgm_h200_metrics:\n", "soak servers")
    server_one = servers[: servers.index("        server {\n          listen 8009 ssl;\n")]
    if servers.count("        server {\n") != 2 or not server_one.endswith("        }\n"):
        raise GenerationError("soak relay must have exactly two servers")
    updated = updated[:servers_start] + "".join(render_soak_server(server_one, replica) for replica in REPLICAS) + updated[servers_end:]

    # Scrape jobs: r1's job rendered once per replica.
    jobs_start, jobs_end, jobs = section(
        updated, f"              - job_name: sglang-{SOURCE_SERVICE_PREFIX}1\n", "              - job_name: dcgm-dcgm-glm53\n", "scrape jobs"
    )
    r1_job = jobs[: jobs.index(f"              - job_name: sglang-{SOURCE_SERVICE_PREFIX}2\n")]
    updated = updated[:jobs_start] + "".join(render_scrape_job(r1_job, replica) for replica in REPLICAS) + updated[jobs_end:]

    if SOURCE_SERVICE_PREFIX in updated or "tp4-r" in updated:
        raise GenerationError("a TP4 engine reference survived")
    return HEADER + updated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--write", action="store_true", help="Write the target file")
    _ = parser.add_argument("--check", action="store_true", help="Fail with a diff when the target is stale")
    args = parser.parse_args()
    if args.write and args.check:
        parser.error("--write and --check are mutually exclusive")

    expected = generate((ROOT / SOURCE).read_text())
    actual = (ROOT / TARGET).read_text() if (ROOT / TARGET).exists() else ""
    if args.check:
        if actual == expected:
            return 0
        diff = difflib.unified_diff(
            actual.splitlines(keepends=True), expected.splitlines(keepends=True), fromfile=f"a/{TARGET}", tofile=f"b/{TARGET}"
        )
        print("".join(diff), end="")
        return 1
    if args.write:
        _ = (ROOT / TARGET).write_text(expected)
        return 0
    print(expected, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
