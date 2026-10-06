#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: uv run scripts/prepare_glm53_pd_qualification.py --write
"""Generate the GLM-5.3 Flash prefill/decode (P/D) qualification file from the 4x TP2 base file.

The target is a DRAINED-HOST qualification file: it has no registrar, no nginx and no
inference proxy, so it can never take production traffic. Load reaches it only through an
authenticated TLS relay (same pattern and token as glm53-soak-relay). It defines a prefill
and a decode variant of the TP2 engine on every NVLink-pair of GPUs plus three PD routers, so
one `compose/up` with a service subset selects the layout (3P:1D, 1P:3D or 2P:2D).

Everything else (logging, DCGM, the OTel pipeline, the downloader, volumes, networks) is
copied from the source file. `--write` regenerates the target; `--check` prints a diff and
exits non-zero when the committed target is stale.
"""

import argparse
import difflib
import sys
from pathlib import Path
from typing import Final

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("prod/GLM-5.3-Flash-SGL-TP2x4-W4AFP8.yaml")
TARGET = Path("prod/GLM-5.3-Flash-SGL-PD-W4AFP8.yaml")
PATCH_SCRIPT = Path("scripts/glm53_pd_startup_patch.py")

# Stock SGLang v0.5.21 (cu130). It carries #39340 (non-2048 DSA top-k in the page-table
# transform), the fix for the multi-chunk prefill assert that blocked P/D on the fork.
IMAGE: Final = "docker.io/lmsysorg/sglang@sha256:b1259f3ea3275f66237c498ea388919729018bc9f01c3d638391e06e2cf3f469"
ENGINE_IMAGE_LABEL: Final = "b1259f3ea327"
DEPLOYMENT: Final = "glm53-flash-sgl-pd-qual"
SOURCE_DEPLOYMENT: Final = "glm53-flash-sgl-tp2x4"
SOURCE_PREFIX: Final = "model-sg-glm53-w4afp8-tp2-r"
PAIRS: Final = {1: 0, 2: 2, 3: 4, 4: 6}  # replica slot -> first GPU of its NVLink pair
ROUTERS: Final = {
    # name: (prefill slots, decode slots, relay port, prometheus port)
    "pd-router-3p1d": ((1, 2, 3), (4,), 8008, 29001),
    "pd-router-1p3d": ((1,), (2, 3, 4), 8009, 29002),
    "pd-router-2p2d": ((1, 3), (2, 4), 8010, 29003),
}
METRICS_PORT: Final = 8011  # relay: /m/<engine>/metrics -> that engine's /metrics

COMMON_ARGS: Final = (
    "--model-path /root/.cache/huggingface/hub/models--graphistry--GLM-5.3-Flash-W4AFP8/snapshots/99f1fa70408c52b007d4fd69e02e5a522422e755",
    "--served-model-name z-ai/glm-5.3-flash",
    "--tp-size 2",
    "--ep-size 2",
    "--max-running-requests 32",
    "--max-queued-requests 8",
    "--enable-priority-scheduling",
    "--disable-priority-preemption",
    "--prefill-decode-interval 1",
    "--cuda-graph-max-bs-decode 32",
    "--dsa-prefill-backend tilelang",
    "--dsa-decode-backend tilelang",
    "--kv-cache-dtype bfloat16",
    # Fixed-step EAGLE. --speculative-adaptive captures CUDA graphs for every step count
    # (~7.5 GB) and its 26.5 GB/rank verify buffer scales with draft tokens; fixed 3/4 lets
    # the decode pool grow from 383K to 1.64M tokens (lab, GSM8K 98.7%).
    "--speculative-algorithm EAGLE",
    "--speculative-num-steps 3",
    "--speculative-eagle-topk 1",
    "--speculative-num-draft-tokens 4",
    "--reasoning-parser glm45",
    "--enable-strict-thinking",
    "--grammar-backend xgrammar",
    "--tool-call-parser glm47",
    "--chat-template /root/.cache/huggingface/hub/models--zai-org--GLM-5.3-Flash/snapshots/3f1971b7b5f7a528c9c4ef6212c8785298a8c24a/chat_template.jinja",
    "--context-length 1048576",
    "--watchdog-timeout 1800",
    "--host 0.0.0.0",
    "--port 8000",
    "--enable-metrics",
    "--enable-cache-report",
    "--log-requests-level 0",
    "--disable-fast-image-processor",
    "--limit-mm-data-per-request '{\"image\": 64}'",
    "--max-mamba-cache-size 165",
    "--mamba-ssm-dtype bfloat16",
    "--disaggregation-transfer-backend nixl",
)
# Prefill nodes never decode, so big chunks cost no decode stalls.
PREFILL_ARGS: Final = ("--mem-fraction-static 0.80", "--chunked-prefill-size 32768", "--max-prefill-tokens 65536")
DECODE_ARGS: Final = ("--mem-fraction-static 0.84", "--chunked-prefill-size 8192", "--max-prefill-tokens 32768")

HEADER: Final = """\
# GLM-5.3 Flash prefill/decode (P/D) QUALIFICATION file for ONE DRAINED base-tier host, generated
# from prod/GLM-5.3-Flash-SGL-TP2x4-W4AFP8.yaml by scripts/prepare_glm53_pd_qualification.py.
# Do not hand-edit this file.
#
# NOT A SERVING CONFIG: no registrar, nginx or inference proxy, so the host never takes
# production traffic while this file runs. Load reaches it only through glm53-pd-relay
# (TLS, Bearer ${PROXY_TOKEN}, same scheme as glm53-soak-relay):
#   :8008 -> pd-router-3p1d   :8009 -> pd-router-1p3d   :8010 -> pd-router-2p2d
#   :8011 -> /m/<engine>/metrics
# Pick the layout with the compose/up service list (each GPU pair runs at most one engine):
#   3P:1D  model-sg-glm53-w4afp8-tp2-pf-r1,-pf-r2,-pf-r3, -dc-r4, pd-router-3p1d
#   1P:3D  model-sg-glm53-w4afp8-tp2-pf-r1, -dc-r2,-dc-r3,-dc-r4, pd-router-1p3d
#   2P:2D  model-sg-glm53-w4afp8-tp2-pf-r1,-pf-r3, -dc-r2,-dc-r4, pd-router-2p2d
#   plus glm53-pd-relay, otelcol-contrib, dcgm-glm53 (model-downloader runs first).
#
# What makes P/D work (lab, gpu31/gpu32, 2026-10-03/04; docs/glm53-pd-qualification.md):
#   - GPU->GPU KV over NIXL/UCX CUDA IPC needs ALL THREE: every engine sees all 8 GPUs (each
#     pinned with --base-gpu-id), pid: host, and PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
#     (VMM memory cannot be CUDA-IPC exported without fabric handles -> silent host staging at
#     0.3 GB/s). With all three: 18 GB/s for SGLang's scattered 35K x 64 KB shape, cached 189K
#     turn 0.45 s vs 6-11 s.
#   - On H200 NVL lab hosts cross-NVLink-island CUDA IPC corrupted KV (GSM8K 0.0); prod hosts
#     report "NVIDIA H200" (SXM/HGX, NVSwitch), which this qualification verifies first.
#   - Fixed-step EAGLE (3/4) and decode mf 0.84: decode pool 1.64M tokens.
# Not carried vs the base file: HiCache (upstream v0.5.21 registers host memory with
# cudaHostRegister, CUDA error 801 under HCC/PPCIe), the admission reserve and the DSA indexer
# query split (fork patches). Startup patches (configs: glm53_pd_startup_patch) restore the
# W4AFP8 loader fix and the 8192 default thinking budget.
# ROLLBACK: compose/down this file's services, then prod/GLM-5.3-Flash-SGL-TP2x4-W4AFP8.yaml
# (full service list including model-proxy-registrar), per docs/glm53-tp2x4-base-canary.md.
"""


def section(text: str, start_marker: str, end_marker: str, label: str) -> str:
    start = text.find(start_marker)
    end = text.find(end_marker, start + len(start_marker))
    if start < 0 or end < 0:
        raise SystemExit(f"{label}: markers not found in {SOURCE}")
    return text[start:end]


def engine_service(role: str, slot: int) -> str:
    name = f"model-sg-glm53-w4afp8-tp2-{role}-r{slot}"
    gpu = PAIRS[slot]
    args = list(COMMON_ARGS) + list(PREFILL_ARGS if role == "pf" else DECODE_ARGS)
    args += [f"--base-gpu-id {gpu}", f"--dist-init-addr 127.0.0.1:{29510 + 10 * slot + (0 if role == 'pf' else 5)}"]
    if role == "pf":
        args += ["--disaggregation-mode prefill", f"--disaggregation-bootstrap-port {8997 + slot}"]
    else:
        args += ["--disaggregation-mode decode"]
    variant = f"pd-{'prefill' if role == 'pf' else 'decode'}-v0521-w4afp8-tp2-ep2-eagle-fixed-3-1-4-mf{'080-c32768' if role == 'pf' else '084-c8192'}-nixl-ipc"
    cmd = " ".join(args)
    return f"""  # --- P/D {'prefill' if role == 'pf' else 'decode'} engine on GPUs {gpu}-{gpu + 1} (TP2) ---
  {name}:
    <<: *sg-glm53-pd-common
    container_name: {name}
    command:
      - |
        python3 /etc/glm53/pd_startup_patch.py
        exec sglang serve {cmd}
    depends_on:
      model-downloader:
        condition: service_completed_successfully
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["0","1","2","3","4","5","6","7"]
              capabilities: [gpu]
    labels:
      com.datadoghq.ad.logs: '[{{"source":"sglang","service":"sglang","tags":["model:z-ai/glm-5.3-flash","model_path:graphistry/GLM-5.3-Flash-W4AFP8","served_model:z-ai/glm-5.3-flash","precision:int4-weights-fp8-activations-bf16-kv","deployment:{DEPLOYMENT}","config_variant:{variant}","request_logging:disabled","engine_image:{ENGINE_IMAGE_LABEL}","env:${{ENV}}","host:${{CVM_HOST}}","ip:${{HOST_IP}}","port:8000","instance:{role}{slot}"]}}]'
      nearai.otel.scrape: "true"
      nearai.otel.job: "sglang"
      nearai.otel.service: "sglang"
      nearai.otel.source: "sglang"
      nearai.otel.container_name: "{name}"
      nearai.otel.port: "8000"
      nearai.otel.path: "/metrics"
      nearai.otel.model: "z-ai/glm-5.3-flash"
      nearai.otel.model_path: "graphistry/GLM-5.3-Flash-W4AFP8"
      nearai.otel.served_model: "z-ai/glm-5.3-flash"
      nearai.otel.deployment: "{DEPLOYMENT}"
      nearai.otel.config_variant: "{variant}"
      nearai.otel.thinking_budget_policy: "default8192-startup-patch"
      nearai.otel.request_logging: "disabled"
      nearai.otel.engine_image: "{ENGINE_IMAGE_LABEL}"
      nearai.otel.instance: "{role}{slot}"
      nearai.otel.env: "${{ENV}}"
      nearai.otel.host: "${{CVM_HOST}}"
      nearai.otel.host_machine: "${{CVM_HOST}}"
      nearai.otel.cvm_name: "${{CVM_NAME}}"
      nearai.otel.ip: "${{HOST_IP}}"

"""


def router_service(name: str) -> str:
    pf, dc, _port, prom = ROUTERS[name]
    prefill = " ".join(f"--prefill http://model-sg-glm53-w4afp8-tp2-pf-r{s}:8000 {8997 + s}" for s in pf)
    decode = " ".join(f"--decode http://model-sg-glm53-w4afp8-tp2-dc-r{s}:8000" for s in dc)
    # sglang_router refuses power_of_two with a single worker on a side.
    ppol = "power_of_two" if len(pf) > 1 else "round_robin"
    dpol = "power_of_two" if len(dc) > 1 else "round_robin"
    return f"""  {name}:
    image: {IMAGE}
    container_name: {name}
    runtime: runc
    entrypoint: ["python3", "-m", "sglang_router.launch_router"]
    command: ["--pd-disaggregation", {", ".join(f'"{t}"' for t in f"{prefill} {decode}".split())}, "--prefill-policy", "{ppol}", "--decode-policy", "{dpol}", "--host", "0.0.0.0", "--port", "8000", "--prometheus-port", "{prom}", "--request-timeout-secs", "3600"]
    restart: unless-stopped
    logging: *logging-conf
    labels:
      com.datadoghq.ad.logs: '[{{"source":"sglang-router","service":"sglang-router","tags":["model:z-ai/glm-5.3-flash","deployment:{DEPLOYMENT}","layout:{name[10:]}","env:${{ENV}}","host:${{CVM_HOST}}","ip:${{HOST_IP}}"]}}]'

"""


def relay_conf() -> str:
    servers = []
    for name, (_pf, _dc, port, _prom) in ROUTERS.items():
        servers.append(f"""    server {{
      listen {port} ssl;
      server_name _;
      if ($$pd_authorized = 0) {{ return 401; }}
      set $$backend http://{name}:8000;
      location ~ ^/(health|v1/models|v1/chat/completions|v1/completions)$$ {{
        limit_except GET POST {{ deny all; }}
        proxy_pass $$backend$$request_uri;
      }}
      location / {{ return 404; }}
    }}""")
    servers.append(f"""    server {{
      listen {METRICS_PORT} ssl;
      server_name _;
      if ($$pd_authorized = 0) {{ return 401; }}
      location ~ ^/m/(model-sg-glm53-w4afp8-tp2-(pf|dc)-r[1-4])/metrics$$ {{
        limit_except GET {{ deny all; }}
        proxy_pass http://$$1:8000/metrics;
      }}
      location / {{ return 404; }}
    }}""")
    body = "\n".join(servers)
    return f"""  glm53_pd_relay_conf:
    content: |
      user nginx;
      worker_processes 1;
      pid /var/run/nginx.pid;
      error_log /dev/null;
      events {{ worker_connections 256; }}
      http {{
        # Qualification traffic only: no request/response logging or disk buffering.
        access_log off;
        client_max_body_size 100m;
        client_body_timeout 3600s;
        send_timeout 3600s;
        resolver 127.0.0.11 valid=5s ipv6=off;
        map_hash_bucket_size 256;
        map $$http_authorization $$pd_authorized {{
          default 0;
          "Bearer ${{PROXY_TOKEN:?PROXY_TOKEN required for authenticated qualification access}}" 1;
        }}
        ssl_certificate /etc/letsencrypt/live/completions.near.ai/fullchain.pem;
        ssl_certificate_key /etc/letsencrypt/live/completions.near.ai/privkey.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        proxy_http_version 1.1;
        proxy_request_buffering off;
        proxy_buffering off;
        proxy_max_temp_file_size 0;
        proxy_connect_timeout 10s;
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
        proxy_set_header Authorization "";
        proxy_set_header Connection "";
        proxy_next_upstream off;
{chr(10).join("  " + ln if ln else ln for ln in body.splitlines())}
      }}

"""


def generate(source: str) -> str:
    logging_and_nvidia = section(source, "x-logging-conf:", "x-vllm-proxy-common:", "logging/nvidia anchors")
    dcgm_anchor = section(source, "x-dcgm-common:", "services:", "dcgm anchor")
    downloader = section(source, "  model-downloader:\n", "\n  hf-cleanup:", "model-downloader") + "\n\n"
    dcgm = section(source, "  dcgm-glm53:\n", "\n  otelcol-contrib:", "dcgm service") + "\n\n"
    otel = section(source, "  otelcol-contrib:\n", "\nnetworks:", "otelcol service") + "\n"
    tail_infra = section(source, "networks:\n", "configs:\n", "networks/volumes")
    dcgm_cfg = section(source, "  dcgm_h200_metrics:\n", "  otelcol_app_config:\n", "dcgm config")
    otel_cfg = section(source, "  otelcol_app_config:\n", "  registrar_script:\n", "otel config")
    # Scrape jobs: re-render the r1 job for every P/D engine; drop the inference-proxy job.
    job_start = otel_cfg.index(f"              - job_name: sglang-{SOURCE_PREFIX}1\n")
    job_end = otel_cfg.index(f"              - job_name: sglang-{SOURCE_PREFIX}2\n")
    template = otel_cfg[job_start:job_end]
    jobs_end = otel_cfg.index("              - job_name: dcgm-dcgm-glm53\n")
    jobs = []
    for role in ("pf", "dc"):
        for slot in PAIRS:
            name = f"model-sg-glm53-w4afp8-tp2-{role}-r{slot}"
            variant = f"pd-{'prefill' if role == 'pf' else 'decode'}-v0521-w4afp8-tp2-ep2-eagle-fixed-3-1-4-mf{'080-c32768' if role == 'pf' else '084-c8192'}-nixl-ipc"
            j = template.replace(f"{SOURCE_PREFIX}1", name)
            lines = []
            for ln in j.splitlines(keepends=True):
                s = ln.strip()
                if s.startswith("config_variant:"):
                    ln = ln[: ln.index("config_variant:")] + f'config_variant: "{variant}"\n'
                elif s.startswith("engine_image:"):
                    ln = ln[: ln.index("engine_image:")] + f'engine_image: "{ENGINE_IMAGE_LABEL}"\n'
                elif s.startswith("instance:"):
                    ln = ln[: ln.index("instance:")] + f'instance: "{role}{slot}"\n'
                elif s.startswith("thinking_budget_policy:"):
                    ln = ln[: ln.index("thinking_budget_policy:")] + 'thinking_budget_policy: "default8192-startup-patch"\n'
                lines.append(ln)
            jobs.append("".join(lines))
    otel_cfg = otel_cfg[:job_start] + "".join(jobs) + otel_cfg[jobs_end:]
    proxy_start = otel_cfg.index("              - job_name: inference-proxy-proxy-glm53\n")
    proxy_end = otel_cfg.index("        prometheus/self:\n")
    otel_cfg = otel_cfg[:proxy_start] + otel_cfg[proxy_end:]

    patch_src = (ROOT / PATCH_SCRIPT).read_text()
    patch_cfg = "  glm53_pd_startup_patch:\n    content: |\n" + "".join(
        ("      " + ln if ln.strip() else "\n") for ln in patch_src.replace("$", "$$").splitlines(keepends=True)
    ) + "\n"

    common = f"""x-sg-glm53-pd-common: &sg-glm53-pd-common
  <<: *nvidia
  init: true
  image: {IMAGE}
  # The fast GPU->GPU KV path needs a shared PID namespace (UCX cuda_ipc) and all GPUs visible.
  pid: host
  entrypoint: ["bash", "-c"]
  volumes:
    - kernel_cache:/root/.cache
    - huggingface_cache:/root/.cache/huggingface
  configs:
    - source: glm53_pd_startup_patch
      target: /etc/glm53/pd_startup_patch.py
      mode: 0444
  environment:
    - HF_TOKEN=${{HUGGING_FACE_HUB_TOKEN}}
    - HF_HUB_OFFLINE=1
    - TRANSFORMERS_OFFLINE=1
    - HF_HUB_DISABLE_TELEMETRY=1
    - DO_NOT_TRACK=1
    - NVIDIA_DRIVER_CAPABILITIES=compute,utility
    - CUDA_DEVICE_ORDER=PCI_BUS_ID
    - NCCL_DEBUG=WARN
    # MUST stay False: expandable segments put the KV pool in CUDA VMM memory, which UCX
    # cuda_ipc cannot export without fabric handles, so NIXL silently falls back to host
    # staging (0.28-0.39 GB/s). False gives 18-105 GB/s.
    - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
    - TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor
    - TRITON_CACHE_DIR=/root/.cache/triton
    - TILELANG_CACHE_DIR=/root/.cache/tilelang
    - SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0
    - SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=1
  restart: unless-stopped
  stop_grace_period: 5m
  logging: *logging-conf

"""
    services = ["services:\n", downloader]
    for role in ("pf", "dc"):
        for slot in PAIRS:
            services.append(engine_service(role, slot))
    for name in ROUTERS:
        services.append(router_service(name))
    services.append(f"""  glm53-pd-relay:
    image: nginx@sha256:1d13701a5f9f3fb01aaa88cef2344d65b6b5bf6b7d9fa4cf0dca557a8d7702ba
    container_name: glm53-pd-relay
    runtime: runc
    cap_drop: [ALL]
    cap_add: [CHOWN, SETUID, SETGID]
    security_opt: ["no-new-privileges:true"]
    restart: "no"
    mem_limit: 256m
    cpus: 1
    stop_grace_period: 5m
    ports: ["8008:8008", "8009:8009", "8010:8010", "8011:8011"]
    volumes:
      - certs:/etc/letsencrypt:ro
    tmpfs:
      - /var/cache/nginx:rw,noexec,nosuid,size=64m
      - /var/run:rw,noexec,nosuid,size=4m
    entrypoint: ["nginx", "-g", "daemon off;"]
    configs:
      - source: glm53_pd_relay_conf
        target: /etc/nginx/nginx.conf
        mode: 0600
    logging: *logging-conf

""")
    services.append(dcgm.replace(SOURCE_DEPLOYMENT, DEPLOYMENT))
    services.append(otel.replace(SOURCE_DEPLOYMENT, DEPLOYMENT))
    out = (HEADER + "\n" + logging_and_nvidia + common + dcgm_anchor + "".join(services) + "\n" + tail_infra
           + "configs:\n" + relay_conf() + patch_cfg + dcgm_cfg + otel_cfg.replace(SOURCE_DEPLOYMENT, DEPLOYMENT).rstrip("\n") + "\n")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    a = ap.parse_args()
    want = generate((ROOT / SOURCE).read_text())
    target = ROOT / TARGET
    if a.write:
        target.write_text(want)
        print(f"wrote {TARGET}")
        return 0
    have = target.read_text() if target.exists() else ""
    if have != want:
        sys.stdout.writelines(difflib.unified_diff(have.splitlines(True), want.splitlines(True), str(TARGET), "generated"))
        return 1
    print(f"{TARGET} is up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
