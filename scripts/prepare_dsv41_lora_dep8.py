#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: uv run scripts/prepare_dsv41_lora_dep8.py --write
"""Generate the DeepSeek-V4.1-Flash-1017 packs for prod and staging on an 8x H200 CVM.

Two layouts of the same engine stack: one DEP8 replica, and two DEP4 replicas (2xDEP4).
The telemetry collector config is copied verbatim from the multi-model pack
prod/qwen36-qwen38.yaml; everything else is written here.
`--write` regenerates both targets; `--check` diffs a stale target.
Design and open gates: docs/dsv41-lora-dep4x2-design.md ("DEP8 in CVMs" section).
"""

import argparse
import difflib
import json
import sys
from pathlib import Path
from typing import Final, NamedTuple

ROOT = Path(__file__).resolve().parents[1]
OTEL_SOURCE = Path("prod/qwen36-qwen38.yaml")

BASE_ID: Final = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
BASE_REVISION: Final = "1bae0e1088993b9b82de08f59de5be5a34e42170"
ADAPTER_ID: Final = "deepseek-ai/DeepSeek-V4.1-Flash-1017-abliterated"
ADAPTER_REPO: Final = "UnderTides/DeepSeek-V4.1-Flash-1017-heretic-lora"
ADAPTER_REVISION: Final = "196c06e3c2aad43e9688d16fa993b4333f69358a"
# docker/vllm-dsv41-lora with the CC patches 04 (VLLM_NEAR_DISABLE_UVA) and 05 (#401), published by
# .github/workflows/publish-vllm-dsv41-lora.yaml run 40627835048 from main c877152, tag
# dsv41-lora-v0.32.0-near-v2-cc. The #393 image lacks patch 04 and returns garbage in a TDX+PPCIe CVM.
ENGINE_IMAGE: Final = "docker.io/nearaidev/vllm@sha256:5eaa0d4637080c10958b6cd8a371ae2ebb749b7a1e19c85bcb1abb89ee2540e4"
# inference-proxy main 166a7b7 (#335: VLLM_PROXY_PRIORITY_LOWER_FIRST; also carries #329:
# VLLM_PROXY_STRICT_REQUEST_MODEL), Build & Deploy run 41119935413, signed and attested. An older proxy
# ignores either variable: without #329 it serves the adapter on the base endpoint at the base price,
# without #335 vLLM runs the OpenRouter lane and reseller orgs ahead of cloud-api.
PROXY_IMAGE: Final = "nearaidev/vllm-proxy-rs@sha256:a99b89aa41a7c416cd2cb3e83f92d18afa66fcb1444780e35e6cf7fa9dc19a9d"
GPUS: Final = 8


class Layout(NamedTuple):
    target: Path
    deployment: str
    variant: str
    title: str
    # (replica name, global ranks); rank i always runs on GPU i and serves on RANK_PORT0 + i.
    replicas: tuple[tuple[str, tuple[int, ...]], ...]
    # Engram tables in host RAM (one /dev/shm copy shared by every replica, read by image patch 05's CPU
    # gather) instead of HBM.
    host_engram: bool


# Engram tables (186 GiB FP8) in HBM, sharded across the eight DP ranks (~23 GiB each). DEP8 leaves 18M KV
# tokens per rank at gpu-memory 0.90. GPU reads of host memory are unsafe in TDX+PPCIe CVMs (stale UVA reads,
# the nvidia_uvm CC fault-decrypt panic of 10-27); see tee-bench #16.
DEP8: Final = Layout(
    Path("prod/DeepSeek-V4.1-Flash-1017-vLLM-DEP8-Abliterated.yaml"),
    "dsv41-flash-1017-vllm-dep8-abliterated",
    "vllm0320-lora3p-cc2-w4a8-dep8-c8192-dspark3-mr64-engram-hbm",
    "ONE vLLM DEP8 replica",
    (("r1", tuple(range(GPUS))),),
    False,
)
# A DEP4 replica cannot hold the tables next to its weights (82 GiB/GPU; tables in HBM left no KV pool on
# v0.0.517), so both replicas share one host copy. Patch 05 gathers rows on the CPU and copies them to the
# GPU, so the GPU never reads host memory. Pierre 10-30: trade DEP8's throughput (+8-15% in the gpu02 A/B of
# 10-28) for a second replica that keeps serving while the other restarts.
DEP4X2: Final = Layout(
    Path("prod/DeepSeek-V4.1-Flash-1017-vLLM-DEP4x2-Abliterated.yaml"),
    "dsv41-flash-1017-vllm-dep4x2-abliterated",
    "vllm0320-lora3p-cc2-w4a8-dep4x2-c8192-dspark3-mr64-engram-host-cpugather",
    "TWO vLLM DEP4 replicas (r1 GPUs 0-3, r2 GPUs 4-7)",
    (("r1", (0, 1, 2, 3)), ("r2", (4, 5, 6, 7))),
    True,
)
LAYOUTS: Final = (DEP8, DEP4X2)
RANK_PORT0: Final = 9100  # rank i serves on 9100 + i on the CVM host network
# The engines run on the host network (NCCL over NVLink P2P, see x-dsv41-engine-common); the services on
# dstack_default reach rank i as model-vllm-dsv41-dp<i>:<9100+i>, each name mapped to host-gateway.

HF_HUB = "/root/.cache/huggingface/hub"
MAX_MODEL_LEN: Final = 1048576
BASE_PATH = f"{HF_HUB}/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/{BASE_REVISION}"
ADAPTER_PATH = f"{HF_HUB}/models--UnderTides--DeepSeek-V4.1-Flash-1017-heretic-lora/snapshots/{ADAPTER_REVISION}"

# (proxy service, served model id, http port, tls port, routing-port env var, domain)
PROXIES: Final = (
# gpu02's CVM forwards only 8000-8010, 8080 and 8444 (vm-manifest port_map); v0.0.519's TLS ports 8011/8013
# were unreachable from outside.
    ("proxy-dsv41", BASE_ID, 8000, 8001, "TLS_PORT_DSV41", "deepseek-v4-1-flash.completions.near.ai"),
    ("proxy-dsv41-abliterated", ADAPTER_ID, 8002, 8003, "TLS_PORT_DSV41_ABLITERATED",
     "deepseek-v4-1-flash-abliterated.completions.near.ai"),
)


def rank_service(rank: int) -> str:
    return f"model-vllm-dsv41-dp{rank}"


def rank_url(rank: int) -> str:
    return f"http://{rank_service(rank)}:{RANK_PORT0 + rank}"


def rank_host_aliases(indent: str) -> str:
    return "".join(f'{indent}- "{rank_service(i)}:host-gateway"\n' for i in range(GPUS))


def replica_of(layout: Layout, rank: int) -> tuple[int, str, tuple[int, ...]]:
    """(replica index, replica name, the replica's global ranks) for a global rank."""
    for index, (name, ranks) in enumerate(layout.replicas):
        if rank in ranks:
            return index, name, ranks
    raise ValueError(f"rank {rank} is in no replica")


def engine_command(layout: Layout, rank: int) -> str:
    index, _, ranks = replica_of(layout, rank)
    lora = json.dumps({"name": ADAPTER_ID, "path": ADAPTER_PATH, "base_model_name": BASE_ID}, separators=(",", ":"))
    # The image entrypoint is ["vllm", "serve"]: the command starts at the model path.
    args = [
        BASE_PATH,
        f"--served-model-name {BASE_ID}",
        "--trust-remote-code",
        "--tensor-parallel-size 1",
        f"--data-parallel-size {len(ranks)}",
        f"--data-parallel-rank {ranks.index(rank)}",
        "--enable-expert-parallel",
        "--data-parallel-address 127.0.0.1",
        # One coordinator port per replica: every rank shares the host network.
        f"--data-parallel-rpc-port {13345 + 100 * index}",
        f"--max-model-len {MAX_MODEL_LEN}",
        "--max-num-batched-tokens 8192",
        "--max-num-seqs 64",
        # Bounded queue, like the GLM base replicas (64 running + 32 queued): each rank's API server counts its
        # running + waiting requests and answers 503 "engine is currently busy" past 96. The proxy forwards the
        # 503, and cloud-api retries other backends and then returns 429 service_overloaded. Without it the
        # queue is unbounded and an overload ends in client timeouts.
        "--max-num-queued-reqs 96",
        # cloud-api ahead of the OpenRouter lane, like GLM: the proxy sets every request's `priority` and vLLM runs
        # the lowest value first, so the proxies negate it (VLLM_PROXY_PRIORITY_LOWER_FIRST=1) and the lane's -1
        # and reseller orgs' -2 run after cloud-api's 0. Only reorders the waiting queue and picks KV-pressure
        # preemption victims; the queue limit above refuses every priority alike, so the gateway's budget for
        # this model keeps the lane from filling it.
        "--scheduling-policy priority",
        # 2xDEP4 keeps the lab's 0.85: host-Engram ranks need room outside vLLM's pool for the gathered rows.
        f"--gpu-memory-utilization {'0.85' if layout.host_engram else '0.90'}",
        "--enable-prompt-tokens-details",
        "--tokenizer-mode deepseek_v41",
        "--reasoning-parser deepseek_v41",
        "--tool-call-parser deepseek_v41",
        "--enable-auto-tool-choice",
        """--speculative-config '{"method":"dspark","num_speculative_tokens":3}'""",
        "--enable-lora --max-loras 1 --max-cpu-loras 1 --max-lora-rank 8 --lora-target-modules wo_b",
        f"--lora-modules '{lora}'",
    ]
    if layout.host_engram:
        args += [
            """--engram-config '{"cpu_offload":true,"dp_shared_memory":true}'""",
            # The CPU gather syncs on the hash ids, so it must run eagerly between piecewise graphs.
            """--compilation-config '{"cudagraph_mode":"PIECEWISE"}'""",
        ]
    else:
        args.append("""--engram-config '{"cpu_offload":false,"embedding_across_dp":true}'""")
    args += ["--host 0.0.0.0", f"--port {RANK_PORT0 + rank}"]
    return "\n      ".join(args)


def tags(layout: Layout, extra: dict[str, str]) -> str:
    base = {"deployment": layout.deployment, "env": "${ENV}", "host": "${CVM_HOST}", "ip": "${HOST_IP}",
            "ppcie": "true", "config_variant": layout.variant}
    return ",".join(f'"{k}:{v}"' for k, v in {**extra, **base}.items())


def otel_labels(layout: Layout, service: str, job: str, source: str, container: str, port: str, model: str) -> str:
    rows = {
        "nearai.otel.scrape": "true", "nearai.otel.job": job, "nearai.otel.service": service,
        "nearai.otel.source": source, "nearai.otel.container_name": container, "nearai.otel.port": port,
        "nearai.otel.path": "/metrics", "nearai.otel.model": model, "nearai.otel.deployment": layout.deployment,
        "nearai.otel.env": "${ENV}", "nearai.otel.host": "${CVM_HOST}", "nearai.otel.host_machine": "${CVM_HOST}",
        "nearai.otel.cvm_name": "${CVM_NAME}", "nearai.otel.ip": "${HOST_IP}", "nearai.otel.ppcie": "true",
        "nearai.otel.config_variant": layout.variant,
    }
    return "".join(f'      {k}: "{v}"\n' for k, v in rows.items())


def scrape_job(job: str, target: str, labels: dict[str, str]) -> str:
    rows = "".join(f'                      {k}: "{v}"\n' for k, v in labels.items())
    return (
        f"              - job_name: {job}\n"
        "                scrape_interval: 15s\n"
        "                metrics_path: /metrics\n"
        "                static_configs:\n"
        f"                  - targets: ['{target}']\n"
        "                    labels:\n" + rows
    )


def otel_config_block(layout: Layout, source: str) -> str:
    """The collector config of OTEL_SOURCE with its scrape jobs replaced by this file's services."""
    block = source[source.index("  otelcol_app_config:\n"):source.index("  registrar_script:\n")]
    marker = "            scrape_configs:\n"
    head, rest = block.split(marker, 1)
    tail = "\n" + rest[rest.index("        prometheus/self:\n"):]
    common = {"env": "${ENV}", "host": "${CVM_HOST}", "host_machine": "${CVM_HOST}", "cvm_name": "${CVM_NAME}",
              "ip": "${HOST_IP}", "ppcie": "true", "deployment": layout.deployment, "config_variant": layout.variant}
    jobs = []
    for rank in range(GPUS):
        svc = rank_service(rank)
        _, replica, ranks = replica_of(layout, rank)
        jobs.append(scrape_job(f"vllm-{svc}", f"{svc}:{RANK_PORT0 + rank}", {
            "service": "vllm", "source": "vllm", "container_name": svc, "model": BASE_ID, "served_model": BASE_ID,
            "model_revision": BASE_REVISION, "replica": replica, "dp_rank": str(ranks.index(rank)),
            "gpu": str(rank), "port": str(PROXIES[0][2]), **common}))
    jobs.append(scrape_job("dcgm-dcgm-dsv41", "dcgm-dsv41:9400", {
        "service": "dcgm-exporter", "source": "dcgm-exporter", "container_name": "dcgm-dsv41", "model": BASE_ID,
        "port": "9400", **common}))
    for name, model, http, *_ in PROXIES:
        jobs.append(scrape_job(f"inference-proxy-{name}", f"{name}:8000", {
            "service": "vllm-proxy", "source": "vllm-proxy", "container_name": name, "model": model,
            "port": str(http), **common}))
    return head + marker + "".join(jobs) + tail


LAYOUT_NOTES: Final = {
    DEP8.target: """\
# Layout: eight `vllm serve` processes, one per GPU (DP8 attention + EP8 experts, TP1, external DP
# load balancing), Engram tables in HBM. In the gpu02 CVM A/B (10-28) this beat 2x DEP4 on
# throughput (+15-23% at rate x8) and TTFT. Two proxies, one per model id, both over all eight
# ranks; each refuses the other id (VLLM_PROXY_STRICT_REQUEST_MODEL=1).
""",
    DEP4X2.target: """\
# Layout: eight `vllm serve` processes, one per GPU, in two independent DEP4 replicas (DP4 attention +
# EP4 experts, TP1, external DP load balancing): r1 = dp0-3 on GPUs 0-3, r2 = dp4-7 on GPUs 4-7, each
# with its own DP coordinator and master port. Engram tables: one host copy in /dev/shm shared by both
# replicas (image patch 05: flock-elected loader, reused as-is when a replica restarts; CPU gather + H2D
# copy). DEP8 was 8-15% faster in the gpu02 A/B (10-28), but a rank failure or redeploy there takes the
# model down; here the other replica keeps serving. Two proxies, one per model id, both over all eight
# ranks; each refuses the other id (VLLM_PROXY_STRICT_REQUEST_MODEL=1) and retries a refused connection
# on another rank (VLLM_BACKEND_CONNECT_FAILOVER=1) until health checks drop a stopped replica.
""",
}


def header(layout: Layout) -> str:
    return f"""# DeepSeek-V4.1-Flash-1017 + abliterated LoRA: {layout.title} on one 8x H200 TDX+PPCIe CVM (gpu02).
# Generated by scripts/prepare_dsv41_lora_dep8.py (do not hand-edit). Design, evidence and open gates:
# docs/dsv41-lora-dep4x2-design.md; CVM bring-up and the DEP8-vs-2xDEP4 A/B: tee-bench #16.
#
# One deployment serves prod and staging, like GLM and small-models. The registrar registers
# deepseek-v4-1-flash[-abliterated].completions.near.ai only on https://completions.near.ai.
# The staging model-proxy copies the prod registry and rewrites names to *.completions-stg.near.ai.
# nginx answers both zones' names with the CVM certbot cert for *.completions.near.ai;
# both proxies attest that cert. cloud-api checks hostnames, so catalog rows on BOTH cloud-api
# and cloud-stg-api must use the *.completions.near.ai URLs. A -stg URL fails NotValidForName.
#
# Direct sk- keys to the abliterated proxy are refused unless DSV41_ABLITERATED_ALLOWED_ORG_IDS
# lists the organization. Set it before the first deploy of this tag if infra-tests or another
# direct caller needs the abliterated proxy. cloud-api traffic is governed by its catalog row,
# which must be unlisted and per-org (cloud-api #1237). promote_model.py does not copy isListed:
# never promote the abliterated id with it.
#
{LAYOUT_NOTES[layout.target]}# TDX+PPCIe requirements, all found on the gpu02 dev CVM (10-27):
#   - VLLM_NEAR_DISABLE_UVA=1 (image patch 04): the GPU reads stale pinned host memory, so vLLM's
#     UVA input buffers corrupt decode (NaN logits, garbage text).
#   - Host network and all GPUs visible per rank: in per-container namespaces NCCL ran over TCP sockets.
#   - Humming compiled with nvcc and cached on a volume: its nvrtc helper opens a CUDA context per
#     kernel, which queues for hours under CC.
# Engine argv = tee-bench exp 50 best stack (W4A8 Humming, 8K prefill chunk, DSpark 3, 64 running)
# plus the wo_b LoRA of exp 51 arm l9 ({ADAPTER_REPO} rev {ADAPTER_REVISION[:8]}).
#
"""

def render(layout: Layout = DEP8) -> str:
    source = (ROOT / OTEL_SOURCE).read_text()
    out: list[str] = [header(layout)]
    out.append(f"""x-logging-conf: &logging-conf
  driver: "json-file"
  options:
    max-size: "100m"
    max-file: "10"
    labels: "com.datadoghq.ad.logs,com.docker.compose.service"

x-nvidia: &nvidia
  runtime: nvidia
  ipc: host
  ulimits:
    memlock: -1
    nofile:
      soft: 65535
      hard: 65535

x-vllm-proxy-common: &vllm-proxy-common
  image: {PROXY_IMAGE}
  user: root
  privileged: true
  <<: *nvidia
  extra_hosts:
    - "compose-manager:host-gateway"
{rank_host_aliases("    ")}  volumes:
    - /var/run/dstack.sock:/var/run/dstack.sock
    - certs:/etc/letsencrypt:ro
  restart: unless-stopped
  logging: *logging-conf

x-dsv41-engine-common: &dsv41-engine-common
  <<: *nvidia
  init: true
  image: {ENGINE_IMAGE}
  shm_size: 64g
  # One network namespace for all ranks: in per-container bridge namespaces NCCL treats each rank
  # as its own host and runs every collective over TCP sockets instead of NVLink.
  network_mode: host
  volumes:
    - huggingface_cache:/root/.cache/huggingface
    - dsv41_kernel_cache:/root/.cache/vllm
    - humming_cache:/root/.cache/humming
  # Every rank sees all GPUs (NCCL P2P needs its peers) and picks its own with CUDA_VISIBLE_DEVICES.
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: all
            capabilities: [gpu]
  restart: unless-stopped
  # Under CC, killing a rank mid-kernel can leave its GPU unusable until the VM is rebuilt.
  stop_grace_period: 10m
  logging: *logging-conf

services:
  model-downloader:
    image: ghcr.io/astral-sh/uv:python3.11-bookworm-slim@sha256:4f5d923c9dcea037f57bda425dd209f3ec643da2f0b74227f68d09dab0b3bb36
    container_name: model-downloader
    restart: "no"
    entrypoint: ["sh", "-c"]
    # ~475 GiB for the base checkpoint (incl. 186 GiB of Engram tables) plus a 4 MB adapter.
    command:
      - |
        set -e
        echo "Downloading {BASE_ID}..."
        uvx --from 'huggingface_hub[hf_xet]' hf download {BASE_ID} --revision {BASE_REVISION}
        echo "Downloading {ADAPTER_REPO}..."
        uvx --from 'huggingface_hub[hf_xet]' hf download {ADAPTER_REPO} --revision {ADAPTER_REVISION}
        echo "Download complete."
    volumes:
      - huggingface_cache:/root/.cache/huggingface
    environment:
      - HF_TOKEN=${{HUGGING_FACE_HUB_TOKEN}}
    logging: *logging-conf
    labels:
      com.datadoghq.ad.logs: '[{{"source":"model-downloader","service":"model-downloader","tags":[{tags(layout, {"model": "multi"})}]}}]'

""")
    out.append(f"""  nginx:
    image: nginx@sha256:1d13701a5f9f3fb01aaa88cef2344d65b6b5bf6b7d9fa4cf0dca557a8d7702ba
    container_name: nginx
    depends_on:
""")
    for name, *_ in PROXIES:
        out.append(f"      {name}:\n        condition: service_started\n")
    out.append(f"""    command: /bin/sh -c 'certs="/etc/letsencrypt/live/completions.near.ai/fullchain.pem"; last=$$(stat -L -c %Y $$certs 2>/dev/null); while :; do sleep 60; cur=$$(stat -L -c %Y $$certs 2>/dev/null); if [ -n "$$cur" ] && [ "$$cur" != "$$last" ]; then last=$$cur; echo "certs changed, reloading nginx"; nginx -s reload; fi; done & while :; do sleep 6h; nginx -s reload; done & nginx -g "daemon off;"'
    ports:
""")
    for _, _, http, tls, *_ in PROXIES:
        out.append(f'      - "{http}:{http}"\n      - "{tls}:{tls}"\n')
    out.append(f"""    volumes:
      - certs:/etc/letsencrypt:ro
    configs:
      - source: nginx_conf
        target: /etc/nginx/conf.d/default.conf
        mode: 0644
    restart: unless-stopped
    logging: *logging-conf
    labels:
      com.datadoghq.ad.logs: '[{{"source":"nginx","service":"nginx","tags":[{tags(layout, {"model": "multi"})}]}}]'

  model-proxy-registrar:
    image: curlimages/curl@sha256:d94d07ba9e7d6de898b6d96c1a072f6f8266c687af78a74f380087a0addf5d17
    container_name: model-proxy-registrar
    entrypoint: ["sh", "/register.sh"]
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "[ $$(( $$(date +%s) - $$(cat /tmp/registrar_alive 2>/dev/null || echo 0) )) -lt 180 ]"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 1800s
    environment:
      - HOST_IP=${{HOST_IP}}
""")
    for _, _, _, tls, env, _ in PROXIES:
        out.append(f"      - {env}=${{{env}:-{tls}}}\n")
    out.append(f"""      - MODEL_PROXY_TOKEN=${{MODEL_PROXY_TOKEN}}
      - PROXY_TOKEN=${{PROXY_TOKEN}}
    configs:
      - source: registrar_script
        target: /register.sh
        mode: 0755
    logging: *logging-conf
    labels:
      com.datadoghq.ad.logs: '[{{"source":"model-proxy-registrar","service":"model-proxy-registrar","tags":[{tags(layout, {"model": "multi"})}]}}]'

""")
    backends = ",".join(rank_url(i) for i in range(GPUS))
    failover = (
        "      # A stopped replica refuses connections until health checks drop it: retry those on another rank.\n"
        "      - VLLM_BACKEND_CONNECT_FAILOVER=1\n"
        if len(layout.replicas) > 1 else ""
    )
    for name, model, http, *_ in PROXIES:
        allowlist = (
            "      # Deny customer API keys sent directly to this proxy until the intended organization IDs are supplied.\n"
            "      # cloud-api's pool token and the registrar probe (config tokens) are not affected.\n"
            "      - VLLM_PROXY_ALLOWED_ORG_IDS=${DSV41_ABLITERATED_ALLOWED_ORG_IDS:-__no_organizations__}\n"
            if model == ADAPTER_ID else ""
        )
        out.append(f"""  {name}:
    <<: *vllm-proxy-common
    container_name: {name}
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - CLOUD_API_URL=https://cloud-api.near.ai
      - CLOUD_API_USAGE_TOKEN=${{CLOUD_API_USAGE_TOKEN}}
      - COMPOSE_MANAGER_URL=http://compose-manager:8080
      - LOG_FORMAT=json
      - MODEL_NAME={model}
      # Refuse the other model id: this proxy attests, bills and access-controls only {model}.
      - VLLM_PROXY_STRICT_REQUEST_MODEL=1
      # vLLM runs the lowest priority first: write the negated value (see --scheduling-policy on the ranks).
      - VLLM_PROXY_PRIORITY_LOWER_FIRST=1
{allowlist}      - OHTTP_ENABLED=true
      - TOKEN=${{PROXY_TOKEN}}
      - VLLM_BACKEND_URLS={backends}
      # Keep append-only conversations on the rank that already holds their prefix cache.
      - VLLM_BACKEND_CONVERSATION_AFFINITY=1
{failover}      - VLLM_PROXY_MAX_REQUEST_SIZE=104857600
      - VLLM_PROXY_POOL_IDLE_TIMEOUT_SECS=3
      - TLS_CERT_PATH=/etc/letsencrypt/live/completions.near.ai/fullchain.pem
      - USE_NV_ATTESTATION_SDK=true
      - WEB_CONTEXT_SEARCH_URL=${{WEB_CONTEXT_SEARCH_URL}}
      - WEB_CONTEXT_SEARCH_API_KEY=${{WEB_CONTEXT_SEARCH_API_KEY}}
    labels:
      com.datadoghq.ad.logs: '[{{"source":"vllm-proxy","service":"vllm-proxy","tags":[{tags(layout, {"model": model, "port": str(http)})}]}}]'
{otel_labels(layout, "vllm-proxy", "inference-proxy", "vllm-proxy", name, "8000", model)}
""")
    for rank in range(GPUS):
        svc = rank_service(rank)
        index, replica, ranks = replica_of(layout, rank)
        depends = "      model-downloader:\n        condition: service_completed_successfully\n"
        if rank != ranks[0]:
            depends += f"      {rank_service(ranks[0])}:\n        condition: service_started\n"
        replica_env = ""
        if layout.host_engram:
            replica_env += (
                "      # Image patch 05: Engram rows gathered on the CPU from the shared /dev/shm copy.\n"
                "      - VLLM_NEAR_ENGRAM_CPU_GATHER=1\n"
            )
        if len(layout.replicas) > 1:
            replica_env += f"      - VLLM_DP_MASTER_PORT={29600 + 100 * index}\n"
        out.append(f"""  {svc}:
    <<: *dsv41-engine-common
    container_name: {svc}
    depends_on:
{depends}    environment:
      - CUDA_VISIBLE_DEVICES={rank}
      - HF_HUB_OFFLINE=1
      - PYTHONUNBUFFERED=1
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
      - NCCL_DEBUG=WARN
      # W4A8: MXFP4 expert weights with FP8 activations on Hopper FP8 tensor cores (exp 50).
      - 'VLLM_HUMMING_INPUT_QUANT_CONFIG={{"dtype":"float8e4m3","quant_mode":"dynamic_token"}}'
      # Image patch 04: copy per-step inputs to the GPU; under TDX+PPCIe the GPU reads stale host memory.
      - VLLM_NEAR_DISABLE_UVA=1
{replica_env}      # nvcc on the CPU instead of humming's nvrtc helper (a CUDA context per kernel, which queues under CC).
      - HUMMING_COMPILER=nvcc
      - HUMMING_CACHE_DIR=/root/.cache/humming/dp{rank}
      - HUMMING_MAX_PARALLEL_BUILD_WORKERS=8
    command: >
      {engine_command(layout, rank)}
    labels:
      com.datadoghq.ad.logs: '[{{"source":"vllm","service":"vllm","tags":[{tags(layout, {"model": BASE_ID, "replica": replica, "dp_rank": str(ranks.index(rank)), "gpu": str(rank)})}]}}]'
{otel_labels(layout, "vllm", "vllm", "vllm", svc, str(RANK_PORT0 + rank), BASE_ID)}
""")
    out.append(f"""  dcgm-dsv41:
    image: nvcr.io/nvidia/k8s/dcgm-exporter@sha256:ed594cf53fe6942e84b07b0740cdcbb249fa4b39cb21feeebf93881ae51f0b5e
    container_name: dcgm-dsv41
    runtime: nvidia
    cap_add:
      - SYS_ADMIN
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["0","1","2","3","4","5","6","7"]
              capabilities: [gpu]
    restart: unless-stopped
    logging: *logging-conf
    labels:
      com.datadoghq.ad.logs: '[{{"source":"dcgm-exporter","service":"dcgm-exporter","tags":[{tags(layout, {"model": BASE_ID})}]}}]'
{otel_labels(layout, "dcgm-exporter", "dcgm", "dcgm-exporter", "dcgm-dsv41", "9400", BASE_ID)}
  otelcol-contrib:
    image: otel/opentelemetry-collector-contrib@sha256:85ac41c2db88d0df9bd6145e608a3cb023f5d8443868adbfbbf66efb51087917
    container_name: otelcol-contrib
    command: ["--config=/etc/otelcol-contrib/config.yaml"]
    user: "0:0"
    extra_hosts:
{rank_host_aliases("      ")}
    mem_limit: "768m"
    environment:
      - MONITORING_INGEST_TOKEN=${{MONITORING_INGEST_TOKEN:-}}
      - CVM_NAME=${{CVM_NAME:-unknown}}
      - CVM_HOST=${{CVM_HOST:-unknown}}
      - DD_HOSTNAME=${{DD_HOSTNAME:-unknown}}
      - ENV=${{ENV:-unknown}}
      - HOST_IP=${{HOST_IP:-unknown}}
    expose:
      - "4317"
      - "4318"
    volumes:
      - /var/lib/docker/containers:/var/lib/docker/containers:ro
      - otelcol_app_storage:/var/lib/otelcol-contrib
    configs:
      - source: otelcol_app_config
        target: /etc/otelcol-contrib/config.yaml
        mode: 0444
    restart: unless-stopped
    logging:
      driver: "local"
      options:
        max-size: "20m"
        max-file: "3"
    labels:
      com.datadoghq.ad.logs: '[{{"source":"otelcol-contrib","service":"otelcol-contrib","tags":[{tags(layout, {"model": "multi"})}]}}]'

networks:
  default:
    external: true
    name: dstack_default

volumes:
  otelcol_app_storage:
  huggingface_cache:
  dsv41_kernel_cache:
  humming_cache:
  certs:
    external: true
    name: certs

configs:
""")
    out.append(otel_config_block(layout, source))
    out.append(registrar_block())
    out.append(nginx_block())
    return "".join(out)


def models_location(model: str) -> str:
    # The model-proxy routes a domain to the endpoints whose GET /v1/models lists its model. Both
    # proxies pass the engine list through, which holds both ids, so it sent one id to the proxy that
    # refuses it (404). Each port pair lists only the id its proxy serves. The registrar's chat probe
    # still unregisters an endpoint whose engines are down.
    body = json.dumps(
        {"object": "list", "data": [{"id": model, "object": "model", "owned_by": "vllm", "max_model_len": MAX_MODEL_LEN}]},
        separators=(",", ":"),
    )
    return f"        location = /v1/models {{ default_type application/json; return 200 '{body}'; }}\n"


def registrar_block() -> str:
    ports = " ".join(str(http) for _, _, http, *_ in PROXIES)
    routing = "".join(f"          {http}) echo \"$${{{env}}}\" ;;\n" for _, _, http, _, env, _ in PROXIES)
    models = "".join(f"          {http}) echo \"{model}\" ;;\n" for _, model, http, *_ in PROXIES)
    domains = "".join(f"          {http}) echo \"{domain}\" ;;\n" for _, _, http, _, _, domain in PROXIES)
    register = "".join(
        f'        register_model "{model}" "{domain}"\n' for _, model, _, _, _, domain in PROXIES
    )
    source = (ROOT / OTEL_SOURCE).read_text()
    block = source[source.index("  registrar_script:\n"):source.index("  nginx_conf:\n")]
    prod_url = '      PROXY_URL="https://completions.near.ai"\n'
    assert prod_url in block
    head, rest = block.split('      PORTS="8002 8006"\n', 1)
    rest = rest[rest.index("      register_endpoint() {"):]
    tail_start = rest.index('        register_model "Qwen/Qwen3.8-27B"')
    tail_end = rest.index("        date +%s > /tmp/registrar_alive\n        sleep 60")
    rest = rest[:tail_start] + register + rest[tail_end:]
    middle = (
        f'      PORTS="{ports}"\n\n'
        "      for port in $$PORTS; do\n"
        '        eval "REG_$$port=false"\n'
        '        eval "FAIL_COUNT_$$port=0"\n'
        "      done\n\n"
        "      routing_port_for() {\n        case $$1 in\n" + routing + "          *) return 1 ;;\n        esac\n      }\n\n"
        "      model_for() {\n        case $$1 in\n" + models + "          *) return 1 ;;\n        esac\n      }\n\n"
        "      domain_for() {\n        case $$1 in\n" + domains + "          *) return 1 ;;\n        esac\n      }\n\n"
    )
    return head + middle + rest


def nginx_block() -> str:
    source = (ROOT / OTEL_SOURCE).read_text()
    block = source[source.index("  nginx_conf:\n"):]
    head = block[:block.index("      server {\n        listen 8002;")]
    plain = "".join(
        f"      server {{\n        listen {http};\n{models_location(model)}"
        f"        location / {{ proxy_pass http://{name}:8000; }}\n      }}\n\n"
        for name, model, http, *_ in PROXIES
    )
    tls_head = (
        # deepseek-v4-1-flash-abliterated.completions-stg.near.ai (55 bytes) overflows the default
        # 64-byte server_names bucket and nginx -t fails.
        "      server_names_hash_bucket_size 128;\n"
        "      ssl_certificate /etc/letsencrypt/live/completions.near.ai/fullchain.pem;\n"
        "      ssl_certificate_key /etc/letsencrypt/live/completions.near.ai/privkey.pem;\n"
        "      ssl_protocols TLSv1.2 TLSv1.3;\n\n"
    )
    tls = []
    for name, model, _, tls_port, _, domain in PROXIES:
        sub = domain.split(".", 1)[0]
        tls.append(
            f"      server {{\n        listen {tls_port} ssl http2;\n"
            f"        server_name {domain}\n"
            f"                    {sub}.completions-stg.near.ai\n"
            f"                    ~^{sub}-i\\d+\\.completions(-stg)?\\.near\\.ai$$;\n"
            "        keepalive_timeout 1h;\n        keepalive_requests 1000000;\n"
            "        location ^~ /metrics { return 404; }\n        location ^~ /v1/metrics { return 404; }\n"
            + models_location(model)
            + f"        location / {{ proxy_pass http://{name}:8000; }}\n      }}\n"
        )
    return head + plain + tls_head + "\n".join(tls)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    stale = 0
    for layout in LAYOUTS:
        rendered = render(layout)
        path = ROOT / layout.target
        if args.write:
            path.write_text(rendered)
            continue
        current = path.read_text() if path.exists() else ""
        if current != rendered:
            sys.stdout.writelines(
                difflib.unified_diff(current.splitlines(True), rendered.splitlines(True), str(layout.target), "rendered")
            )
            stale = 1
    return stale


if __name__ == "__main__":
    sys.exit(main())
