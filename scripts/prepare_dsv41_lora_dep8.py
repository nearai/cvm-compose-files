#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: uv run scripts/prepare_dsv41_lora_dep8.py --write
"""Generate the DeepSeek-V4.1-Flash-1017 + abliterated LoRA file (one vLLM DEP8 replica, one 8x H200 CVM).

The telemetry collector config is copied verbatim from the multi-model pack
prod/qwen36-qwen38.yaml; everything else is written here. `--write` regenerates the target,
`--check` prints a diff and exits non-zero when the committed target is stale.
Design and open gates: docs/dsv41-lora-dep4x2-design.md ("DEP8 in CVMs" section).
"""

import argparse
import difflib
import json
import sys
from pathlib import Path
from typing import Final

ROOT = Path(__file__).resolve().parents[1]
OTEL_SOURCE = Path("prod/qwen36-qwen38.yaml")
TARGET = Path("prod/DeepSeek-V4.1-Flash-1017-vLLM-DEP8-Abliterated.yaml")

BASE_ID: Final = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
BASE_REVISION: Final = "1bae0e1088993b9b82de08f59de5be5a34e42170"
ADAPTER_ID: Final = "deepseek-ai/DeepSeek-V4.1-Flash-1017-abliterated"
ADAPTER_REPO: Final = "UnderTides/DeepSeek-V4.1-Flash-1017-heretic-lora"
ADAPTER_REVISION: Final = "196c06e3c2aad43e9688d16fa993b4333f69358a"
# docker/vllm-dsv41-lora with the CC patches 04 (VLLM_NEAR_DISABLE_UVA) and 05 (#401), published by
# .github/workflows/publish-vllm-dsv41-lora.yaml run 40627835048 from main c877152, tag
# dsv41-lora-v0.32.0-near-v2-cc. The #393 image lacks patch 04 and returns garbage in a TDX+PPCIe CVM.
ENGINE_IMAGE: Final = "docker.io/nearaidev/vllm@sha256:5eaa0d4637080c10958b6cd8a371ae2ebb749b7a1e19c85bcb1abb89ee2540e4"
# inference-proxy main 5faff1b (#329: VLLM_PROXY_STRICT_REQUEST_MODEL), Build & Deploy run
# 40425504976, pushed as :staging. An older proxy ignores the variable and would serve the adapter
# on the base endpoint at the base price.
PROXY_IMAGE: Final = "nearaidev/vllm-proxy-rs@sha256:4937ad68d86dcc79871bd52341f2030a71e49bcafdf69d82ca287171aeb43c7b"
# Engram tables (186 GiB FP8) in HBM, sharded across the eight DP ranks (~23 GiB each). DEP8 leaves 18M KV
# tokens per rank at gpu-memory 0.90. Host-memory layouts are unsafe in TDX+PPCIe CVMs (stale UVA reads, the
# nvidia_uvm CC fault-decrypt panic of 10-27); see tee-bench #16.
DEPLOYMENT: Final = "dsv41-flash-1017-vllm-dep8-abliterated"
VARIANT: Final = "vllm0320-lora3p-cc2-w4a8-dep8-c8192-dspark3-mr64-engram-hbm"
DP_SIZE: Final = 8
RANK_PORT0: Final = 9100  # rank i serves on 9100 + i on the CVM host network
# The engines run on the host network (NCCL over NVLink P2P, see x-dsv41-engine-common); the services on
# dstack_default reach rank i as model-vllm-dsv41-dp<i>:<9100+i>, each name mapped to host-gateway.

HF_HUB = "/root/.cache/huggingface/hub"
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
    return "".join(f'{indent}- "{rank_service(i)}:host-gateway"\n' for i in range(DP_SIZE))


def engine_command(rank: int) -> str:
    lora = json.dumps({"name": ADAPTER_ID, "path": ADAPTER_PATH, "base_model_name": BASE_ID}, separators=(",", ":"))
    # The image entrypoint is ["vllm", "serve"]: the command starts at the model path.
    args = [
        BASE_PATH,
        f"--served-model-name {BASE_ID}",
        "--trust-remote-code",
        "--tensor-parallel-size 1",
        f"--data-parallel-size {DP_SIZE}",
        f"--data-parallel-rank {rank}",
        "--enable-expert-parallel",
        "--data-parallel-address 127.0.0.1",
        "--data-parallel-rpc-port 13345",
        "--max-model-len 1048576",
        "--max-num-batched-tokens 8192",
        "--max-num-seqs 64",
        "--gpu-memory-utilization 0.90",
        "--enable-prompt-tokens-details",
        "--tokenizer-mode deepseek_v41",
        "--reasoning-parser deepseek_v41",
        "--tool-call-parser deepseek_v41",
        "--enable-auto-tool-choice",
        """--speculative-config '{"method":"dspark","num_speculative_tokens":3}'""",
        "--enable-lora --max-loras 1 --max-cpu-loras 1 --max-lora-rank 8 --lora-target-modules wo_b",
        f"--lora-modules '{lora}'",
        """--engram-config '{"cpu_offload":false,"embedding_across_dp":true}'""",
    ]
    args += ["--host 0.0.0.0", f"--port {RANK_PORT0 + rank}"]
    return "\n      ".join(args)


def tags(extra: dict[str, str]) -> str:
    base = {"deployment": DEPLOYMENT, "env": "${ENV}", "host": "${CVM_HOST}", "ip": "${HOST_IP}",
            "ppcie": "true", "config_variant": VARIANT}
    return ",".join(f'"{k}:{v}"' for k, v in {**extra, **base}.items())


def otel_labels(service: str, job: str, source: str, container: str, port: str, model: str) -> str:
    rows = {
        "nearai.otel.scrape": "true", "nearai.otel.job": job, "nearai.otel.service": service,
        "nearai.otel.source": source, "nearai.otel.container_name": container, "nearai.otel.port": port,
        "nearai.otel.path": "/metrics", "nearai.otel.model": model, "nearai.otel.deployment": DEPLOYMENT,
        "nearai.otel.env": "${ENV}", "nearai.otel.host": "${CVM_HOST}", "nearai.otel.host_machine": "${CVM_HOST}",
        "nearai.otel.cvm_name": "${CVM_NAME}", "nearai.otel.ip": "${HOST_IP}", "nearai.otel.ppcie": "true",
        "nearai.otel.config_variant": VARIANT,
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


def otel_config_block(source: str) -> str:
    """The collector config of OTEL_SOURCE with its scrape jobs replaced by this file's services."""
    block = source[source.index("  otelcol_app_config:\n"):source.index("  registrar_script:\n")]
    marker = "            scrape_configs:\n"
    head, rest = block.split(marker, 1)
    tail = "\n" + rest[rest.index("        prometheus/self:\n"):]
    common = {"env": "${ENV}", "host": "${CVM_HOST}", "host_machine": "${CVM_HOST}", "cvm_name": "${CVM_NAME}",
              "ip": "${HOST_IP}", "ppcie": "true", "deployment": DEPLOYMENT, "config_variant": VARIANT}
    jobs = []
    for rank in range(DP_SIZE):
        svc = rank_service(rank)
        jobs.append(scrape_job(f"vllm-{svc}", f"{svc}:{RANK_PORT0 + rank}", {
            "service": "vllm", "source": "vllm", "container_name": svc, "model": BASE_ID, "served_model": BASE_ID,
            "model_revision": BASE_REVISION, "replica": "r1", "dp_rank": str(rank), "gpu": str(rank),
            "port": str(PROXIES[0][2]), **common}))
    jobs.append(scrape_job("dcgm-dcgm-dsv41", "dcgm-dsv41:9400", {
        "service": "dcgm-exporter", "source": "dcgm-exporter", "container_name": "dcgm-dsv41", "model": BASE_ID,
        "port": "9400", **common}))
    for name, model, http, *_ in PROXIES:
        jobs.append(scrape_job(f"inference-proxy-{name}", f"{name}:8000", {
            "service": "vllm-proxy", "source": "vllm-proxy", "container_name": name, "model": model,
            "port": str(http), **common}))
    return head + marker + "".join(jobs) + tail


HEADER = f"""# DeepSeek-V4.1-Flash-1017 + abliterated LoRA: ONE vLLM DEP8 replica on one 8x H200 TDX+PPCIe CVM (gpu02).
# Generated by scripts/prepare_dsv41_lora_dep8.py (do not hand-edit). Design, evidence and open gates:
# docs/dsv41-lora-dep4x2-design.md; CVM bring-up and the DEP8-vs-2xDEP4 A/B: tee-bench #16.
#
# STAGING ONLY, NOT LISTED. Register only the staging inferenceUrl
# (https://<sub>.completions-stg.near.ai) in cloud-stg-api; never PATCH prod cloud-api. Note that
# any active PATCH to cloud-stg-api fires validate_model_staging -> promote_model_prod; the
# promote's zone guard refuses a completions-stg URL, but the chain still auto-merges infra-tests
# PRs, so coordinate with Henry before that PATCH.
#
# Layout: eight `vllm serve` processes, one per GPU (DP8 attention + EP8 experts, TP1, external DP
# load balancing), Engram tables in HBM. In the gpu02 CVM A/B (10-28) this beat 2x DEP4 on
# throughput (+15-23% at rate x8) and TTFT. Two proxies, one per model id, both over all eight
# ranks; each refuses the other id (VLLM_PROXY_STRICT_REQUEST_MODEL=1).
# TDX+PPCIe requirements, all found on the gpu02 dev CVM (10-27):
#   - VLLM_NEAR_DISABLE_UVA=1 (image patch 04): the GPU reads stale pinned host memory, so vLLM's
#     UVA input buffers corrupt decode (NaN logits, garbage text).
#   - Host network and all GPUs visible per rank: in per-container namespaces NCCL ran over TCP sockets.
#   - Humming compiled with nvcc and cached on a volume: its nvrtc helper opens a CUDA context per
#     kernel, which queues for hours under CC.
# Engine argv = tee-bench exp 50 best stack (W4A8 Humming, 8K prefill chunk, DSpark 3, 64 running)
# plus the wo_b LoRA of exp 51 arm l9 ({ADAPTER_REPO} rev {ADAPTER_REVISION[:8]}).
#
"""


def render() -> str:
    source = (ROOT / OTEL_SOURCE).read_text()
    out: list[str] = [HEADER]
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
      com.datadoghq.ad.logs: '[{{"source":"model-downloader","service":"model-downloader","tags":[{tags({"model": "multi"})}]}}]'

  nginx:
    image: nginx@sha256:1d13701a5f9f3fb01aaa88cef2344d65b6b5bf6b7d9fa4cf0dca557a8d7702ba
    container_name: nginx
    depends_on:
""")
    for name, *_ in PROXIES:
        out.append(f"      - {name}\n")
    out.append("""    command: /bin/sh -c 'cert=/etc/letsencrypt/live/completions.near.ai/fullchain.pem; last=$$(stat -L -c %Y "$$cert" 2>/dev/null); while :; do sleep 60; cur=$$(stat -L -c %Y "$$cert" 2>/dev/null); if [ -n "$$cur" ] && [ "$$cur" != "$$last" ]; then last=$$cur; echo "$$cert changed, reloading nginx"; nginx -s reload; fi; done & while :; do sleep 6h; nginx -s reload; done & nginx -g "daemon off;"'
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
      com.datadoghq.ad.logs: '[{{"source":"nginx","service":"nginx","tags":[{tags({"model": "multi"})}]}}]'

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
      com.datadoghq.ad.logs: '[{{"source":"model-proxy-registrar","service":"model-proxy-registrar","tags":[{tags({"model": "multi"})}]}}]'

""")
    backends = ",".join(rank_url(i) for i in range(DP_SIZE))
    for name, model, http, *_ in PROXIES:
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
      - OHTTP_ENABLED=true
      - TOKEN=${{PROXY_TOKEN}}
      - VLLM_BACKEND_URLS={backends}
      # Keep append-only conversations on the rank that already holds their prefix cache.
      - VLLM_BACKEND_CONVERSATION_AFFINITY=1
      - VLLM_PROXY_MAX_REQUEST_SIZE=104857600
      - VLLM_PROXY_POOL_IDLE_TIMEOUT_SECS=3
      - TLS_CERT_PATH=/etc/letsencrypt/live/completions.near.ai/fullchain.pem
      - USE_NV_ATTESTATION_SDK=true
      - WEB_CONTEXT_SEARCH_URL=${{WEB_CONTEXT_SEARCH_URL}}
      - WEB_CONTEXT_SEARCH_API_KEY=${{WEB_CONTEXT_SEARCH_API_KEY}}
    labels:
      com.datadoghq.ad.logs: '[{{"source":"vllm-proxy","service":"vllm-proxy","tags":[{tags({"model": model, "port": str(http)})}]}}]'
{otel_labels("vllm-proxy", "inference-proxy", "vllm-proxy", name, "8000", model)}
""")
    for rank in range(DP_SIZE):
        svc = rank_service(rank)
        depends = "      model-downloader:\n        condition: service_completed_successfully\n"
        if rank:
            depends += f"      {rank_service(0)}:\n        condition: service_started\n"
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
      # nvcc on the CPU instead of humming's nvrtc helper (a CUDA context per kernel, which queues under CC).
      - HUMMING_COMPILER=nvcc
      - HUMMING_CACHE_DIR=/root/.cache/humming/dp{rank}
      - HUMMING_MAX_PARALLEL_BUILD_WORKERS=8
    command: >
      {engine_command(rank)}
    labels:
      com.datadoghq.ad.logs: '[{{"source":"vllm","service":"vllm","tags":[{tags({"model": BASE_ID, "replica": "r1", "dp_rank": str(rank), "gpu": str(rank)})}]}}]'
{otel_labels("vllm", "vllm", "vllm", svc, str(RANK_PORT0 + rank), BASE_ID)}
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
      com.datadoghq.ad.logs: '[{{"source":"dcgm-exporter","service":"dcgm-exporter","tags":[{tags({"model": BASE_ID})}]}}]'
{otel_labels("dcgm-exporter", "dcgm", "dcgm-exporter", "dcgm-dsv41", "9400", BASE_ID)}
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
      com.datadoghq.ad.logs: '[{{"source":"otelcol-contrib","service":"otelcol-contrib","tags":[{tags({"model": "multi"})}]}}]'

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
    out.append(otel_config_block(source))
    out.append(registrar_block())
    out.append(nginx_block())
    return "".join(out)


def registrar_block() -> str:
    ports = " ".join(str(http) for _, _, http, *_ in PROXIES)
    routing = "".join(f"          {http}) echo \"$${{{env}}}\" ;;\n" for _, _, http, _, env, _ in PROXIES)
    models = "".join(f"          {http}) echo \"{model}\" ;;\n" for _, model, http, *_ in PROXIES)
    domains = "".join(f"          {http}) echo \"{domain}\" ;;\n" for _, _, http, _, _, domain in PROXIES)
    register = "".join(f'        register_model "{model}" "{domain}"\n' for _, model, _, _, _, domain in PROXIES)
    source = (ROOT / OTEL_SOURCE).read_text()
    block = source[source.index("  registrar_script:\n"):source.index("  nginx_conf:\n")]
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
        f"      server {{\n        listen {http};\n        location / {{ proxy_pass http://{name}:8000; }}\n      }}\n\n"
        for name, _, http, *_ in PROXIES
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
    for name, _, _, tls_port, _, domain in PROXIES:
        sub = domain.split(".", 1)[0]
        tls.append(
            f"      server {{\n        listen {tls_port} ssl http2;\n"
            f"        server_name {domain}\n"
            f"                    {sub}.completions-stg.near.ai\n"
            f"                    ~^{sub}-i\\d+\\.completions(-stg)?\\.near\\.ai$$;\n"
            "        keepalive_timeout 1h;\n        keepalive_requests 1000000;\n"
            "        location ^~ /metrics { return 404; }\n        location ^~ /v1/metrics { return 404; }\n"
            f"        location / {{ proxy_pass http://{name}:8000; }}\n      }}\n"
        )
    return head + plain + tls_head + "\n".join(tls)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = render()
    target = ROOT / TARGET
    if args.write:
        target.write_text(rendered)
        return 0
    current = target.read_text() if target.exists() else ""
    if current == rendered:
        return 0
    sys.stdout.writelines(difflib.unified_diff(current.splitlines(True), rendered.splitlines(True), str(TARGET), "rendered"))
    return 1


if __name__ == "__main__":
    sys.exit(main())
