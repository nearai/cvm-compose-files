#!/usr/bin/env python3
# How to run: python3 scripts/kvq/prepare_kvq_qual.py --write   (or --check)
"""Generate prod/GLM-5.3-Flash-SGL-KVShare-Qual.yaml: in-host KV-sharing qualification under CC.

Lab-only stack for gpu13 GPUs 4-7 (GPUs 0-3 keep serving the small models). Deploy it as its own
compose project (`glm53kvq`, never `work`) with a `services` subset per step. It defines no
registrar, nginx, proxy or published port, so it can never take production traffic; every test is
a one-shot container that prints `KVQ {json}` lines to its log.

  kvq-probe            CC primitives: topology/CC mode, GPU->GPU peer + cross-process CUDA IPC with
                       checksums, pinned host + tmpfs restore path, cudaHostRegister (801 expected)
  kvq-pf / kvq-dc      Test B: 1P:1D NIXL GPU->GPU KV move (stock v0.5.21 + the #335 startup patch)
  kvq-router           PD router for kvq-pf/kvq-dc
  kvq-driver           GSM8K + cached long turn + recompute baseline against KVQ_TARGETS
Every GPU container sees exactly GPUs 4-7 (cuda:0-3) in the same order, so peers are visible.
"""
import argparse, difflib, hashlib, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET = Path("prod/GLM-5.3-Flash-SGL-KVShare-Qual.yaml")
PD_PATCH = Path("scripts/glm53_pd_startup_patch.py")
PROBE = Path("scripts/kvq/kvq_probe.py")
DRIVER = Path("scripts/kvq/kvq_driver.py")
SHARED_PATCH = Path("scripts/kvq/glm53_shared_kv_startup_patch.py")
NIXL_BENCH = Path("scripts/kvq/nixl_bench.py")  # lab bench from gpu31 pd-v0521-20261003, unchanged

V0521 = "docker.io/lmsysorg/sglang@sha256:b1259f3ea3275f66237c498ea388919729018bc9f01c3d638391e06e2cf3f469"
V6 = "docker.io/nearaidev/sglang@sha256:9c6ddd4319c4ab00e351d8650459e68b8830e36ffcc029d67fa5e19d0ac3ed17"
SNAP = "/root/.cache/huggingface/hub/models--graphistry--GLM-5.3-Flash-W4AFP8/snapshots/99f1fa70408c52b007d4fd69e02e5a522422e755"
TEMPLATE = "/root/.cache/huggingface/hub/models--zai-org--GLM-5.3-Flash/snapshots/3f1971b7b5f7a528c9c4ef6212c8785298a8c24a/chat_template.jinja"
GPUS = '["4","5","6","7"]'
DEPLOYMENT = "glm53-flash-sgl-kvshare-qual"

COMMON_ARGS = (
    f"--model-path {SNAP} --served-model-name z-ai/glm-5.3-flash --tp-size 2 --ep-size 2 "
    "--max-running-requests 32 --max-queued-requests 8 --enable-priority-scheduling --disable-priority-preemption "
    "--prefill-decode-interval 1 --cuda-graph-max-bs-decode 32 --dsa-prefill-backend tilelang --dsa-decode-backend tilelang "
    "--kv-cache-dtype bfloat16 --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 "
    "--speculative-num-draft-tokens 4 --reasoning-parser glm45 --enable-strict-thinking --grammar-backend xgrammar "
    f"--tool-call-parser glm47 --chat-template {TEMPLATE} --context-length 1048576 --watchdog-timeout 1800 "
    "--host 0.0.0.0 --port 8000 --enable-metrics --enable-cache-report --log-requests-level 0 --disable-fast-image-processor "
    "--limit-mm-data-per-request '{\"image\": 64}' --max-mamba-cache-size 165 --mamba-ssm-dtype bfloat16 "
    "--disaggregation-transfer-backend nixl"
)
PF_ARGS = "--mem-fraction-static 0.80 --chunked-prefill-size 32768 --max-prefill-tokens 65536 --base-gpu-id 0 --dist-init-addr 127.0.0.1:29540 --disaggregation-mode prefill --disaggregation-bootstrap-port 8998"
DC_ARGS = "--mem-fraction-static 0.84 --chunked-prefill-size 8192 --max-prefill-tokens 32768 --base-gpu-id 2 --dist-init-addr 127.0.0.1:29545 --disaggregation-mode decode"


def block(text, indent):
    pad = " " * indent
    return "\n".join((pad + l) if l.strip() else "" for l in text.replace("$", "$$").rstrip("\n").split("\n"))


def log_label(service):
    """Without this label the gpu13 log collector skips the container (results are read from Loki)."""
    return f"""    labels:
      com.datadoghq.ad.logs: '[{{"source":"{service}","service":"{service}","tags":["deployment:{DEPLOYMENT}","env:${{ENV}}","host:${{CVM_HOST}}"]}}]'
"""


def labels(name, variant, image_short, instance):
    return f"""    labels:
      com.datadoghq.ad.logs: '[{{"source":"sglang","service":"sglang","tags":["model:z-ai/glm-5.3-flash","deployment:{DEPLOYMENT}","config_variant:{variant}","engine_image:{image_short}","env:${{ENV}}","host:${{CVM_HOST}}","instance:{instance}"]}}]'
      nearai.otel.scrape: "true"
      nearai.otel.job: "sglang"
      nearai.otel.service: "sglang"
      nearai.otel.source: "sglang"
      nearai.otel.container_name: "{name}"
      nearai.otel.port: "8000"
      nearai.otel.path: "/metrics"
      nearai.otel.model: "z-ai/glm-5.3-flash"
      nearai.otel.served_model: "z-ai/glm-5.3-flash"
      nearai.otel.deployment: "{DEPLOYMENT}"
      nearai.otel.config_variant: "{variant}"
      nearai.otel.engine_image: "{image_short}"
      nearai.otel.instance: "{instance}"
      nearai.otel.env: "${{ENV}}"
      nearai.otel.host: "${{CVM_HOST}}"
      nearai.otel.host_machine: "${{CVM_HOST}}"
"""


def engine(name, args, variant, instance):
    return f"""  {name}:
    <<: *kvq-pd-engine
    container_name: {name}
    command:
      - |
        python3 /etc/glm53/pd_startup_patch.py
        exec sglang serve {COMMON_ARGS} {args}
{labels(name, variant, "b1259f3ea327", instance)}"""


def cfg(name, text):
    """Content-addressed config name: compose reuses a config whose name is unchanged."""
    return f"{name}_{hashlib.sha256(text.encode()).hexdigest()[:8]}"


# Test A: the gpu13 prod argv (prod/small-models.yaml r1a/r1b, v6 image) plus the L3 file tier on a
# tmpfs store both replicas mount. Lower max-running than prod is not needed: same shape.
# Test A = the prod A/B treatment engine: the base tier's candidate argv at origin/main 9bbcef0
# (prod/GLM-5.3-Flash-SGL-TP2x4-W4AFP8.yaml x-sg-glm53-flash-candidate, write_through_selective,
# mr48, 330 mamba slots), so the CC qualification covers exactly what the A/B deploys.
V6_ARGS = "--model-path /root/.cache/huggingface/hub/models--graphistry--GLM-5.3-Flash-W4AFP8/snapshots/99f1fa70408c52b007d4fd69e02e5a522422e755 --served-model-name z-ai/glm-5.3-flash --tp-size 2 --ep-size 2 --mem-fraction-static 0.86 --max-running-requests 48 --max-queued-requests 8 --enable-priority-scheduling --disable-priority-preemption --chunked-prefill-size 8192 --max-prefill-tokens 32768 --prefill-decode-interval 2 --cuda-graph-max-bs-decode 48 --dsa-prefill-backend tilelang --dsa-decode-backend tilelang --kv-cache-dtype bfloat16 --speculative-algorithm EAGLE --speculative-num-steps 4 --speculative-eagle-topk 1 --speculative-num-draft-tokens 5 --reasoning-parser glm45 --enable-strict-thinking --grammar-backend xgrammar --tool-call-parser glm47 --chat-template /root/.cache/huggingface/hub/models--zai-org--GLM-5.3-Flash/snapshots/3f1971b7b5f7a528c9c4ef6212c8785298a8c24a/chat_template.jinja --context-length 1048576 --watchdog-timeout 1800 --host 0.0.0.0 --port 8000 --enable-metrics --enable-cache-report --log-requests-level 0 --disable-fast-image-processor --limit-mm-data-per-request '{IMG}' --enable-hierarchical-cache --hicache-write-policy write_through_selective --hicache-io-backend direct --hicache-mem-layout page_first_direct --max-mamba-cache-size 330 --mamba-ssm-dtype bfloat16".replace("{IMG}", '{"image": 64}')
# Shared tier (only when KVQ_SHARED=1; KVQ_SHARED=0 gives the private-cache control arm P).
SHARED_ARGS = ("--hicache-host-memory-mode cache --hicache-storage-backend file "
               "--hicache-storage-prefetch-policy wait_complete")


def shared_engine(name, base_gpu, port, replica):
    return f"""  {name}:
    <<: *kvq-gpu
    image: {V6}
    container_name: {name}
    entrypoint: ["bash", "-c"]
    command:
      - |
        if [ "$${{KVQ_SHARED:-1}}" = "1" ]; then python3 /etc/kvq/shared_kv_patch.py || exit 1; EXTRA="{SHARED_ARGS}"; else EXTRA=""; fi
        exec sglang serve {V6_ARGS} --base-gpu-id {base_gpu} --dist-init-addr 127.0.0.1:{port} $$EXTRA
    configs:
      - source: {{c_shared}}
        target: /etc/kvq/shared_kv_patch.py
        mode: 0444
    volumes:
      - kvq_kernel_cache:/root/.cache
      - huggingface_cache:/root/.cache/huggingface
      - kvq_shared:/kvshared
      - kvq_ghost:/ghost
    environment:
      <<: *kvq-env
      KVQ_SHARED: ${{KVQ_SHARED:-1}}
      # Prod value; Test A needs no CUDA IPC.
      PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True
      SGLANG_HICACHE_RAM_BUDGET: ${{KVQ_RAM_BUDGET:-250GiB}}
      # Base-tier env parity (admission reserve v10 rides in the v6 image).
      SGLANG_CHUNKED_PREFILL_ADMISSION_RESERVE: "4096"
      SGLANG_ADMISSION_RESERVE_MAX_FRACTION: "0.75"
      SGLANG_KVSHARE_METRICS: "1"
      SGLANG_HICACHE_CUDA_HOST_MEMORY: "1"
      SGLANG_HICACHE_POOLED_TRANSFERS: "1"
      SGLANG_HICACHE_STAGING_PAGES: "64"
      SGLANG_DSA_INDEXER_QSPLIT: "1"
      SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR: /kvshared
      SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE: ${{KVQ_FILE_MAX:-60Gi}}
      SGLANG_HICACHE_SHARED_STORE_BUDGET: ${{KVQ_STORE_BUDGET:-150GiB}}
      SGLANG_GHOST_CACHE: "1"
      SGLANG_GHOST_CACHE_SAMPLE: "16"
      SGLANG_GHOST_CACHE_KEY_FILE: /ghost/key
      SGLANG_GHOST_CACHE_SOCKET: /ghost/aggregator.sock
      SGLANG_GHOST_CACHE_REPLICA: {replica}
      SGLANG_KV_TIER_METRICS: "1"
    restart: "no"
    stop_grace_period: 2m
{{labels(name, "kvq-shared-v6-basecand-selective-file-tier", "9c6ddd4319c4", replica)}}"""


def render():
    pd_patch = (ROOT / PD_PATCH).read_text()
    probe = (ROOT / PROBE).read_text()
    driver = (ROOT / DRIVER).read_text()
    shared = (ROOT / SHARED_PATCH).read_text()
    c_pd, c_probe, c_driver = cfg("glm53_pd_startup_patch", pd_patch), cfg("kvq_probe_py", probe), cfg("kvq_driver_py", driver)
    c_shared = cfg("kvq_shared_kv_patch_py", shared)
    nixl_bench = (ROOT / NIXL_BENCH).read_text()
    c_nixl = cfg("kvq_nixl_bench_py", nixl_bench)
    def nixl(role, gpu):
        return f"""  kvq-nixl-{role}:
    <<: *kvq-gpu
    image: {V0521}
    container_name: kvq-nixl-{role}
    entrypoint: ["bash", "-c"]
    command:
      - |
        python3 /etc/kvq/nixl_bench.py {role} --dir /x/$${{KVQ_NIXL_RUN:-r1}} --gpu {gpu} 2>&1 | sed 's/^/KVQ_NIXL /'
        echo KVQ_DONE
    configs:
      - source: {c_nixl}
        target: /etc/kvq/nixl_bench.py
        mode: 0444
    volumes:
      - kvq_xfer:/x
    environment:
      <<: *kvq-env
      UCX_TLS: ${{KVQ_NIXL_TLS:-all}}
      UCX_PROTO_INFO: "y"
    restart: "no"
{log_label("kvq-nixl-" + role)}"""
    sh1 = shared_engine("kvq-r1", 0, 29550, "kvq-r1").replace("{c_shared}", c_shared).replace("{labels(name, \"kvq-shared-v6-basecand-selective-file-tier\", \"9c6ddd4319c4\", replica)}", labels("kvq-r1", "kvq-shared-v6-basecand-selective-file-tier", "9c6ddd4319c4", "kvq-r1"))
    sh2 = shared_engine("kvq-r2", 2, 29551, "kvq-r2").replace("{c_shared}", c_shared).replace("{labels(name, \"kvq-shared-v6-basecand-selective-file-tier\", \"9c6ddd4319c4\", replica)}", labels("kvq-r2", "kvq-shared-v6-basecand-selective-file-tier", "9c6ddd4319c4", "kvq-r2"))
    return f"""# GENERATED by scripts/kvq/prepare_kvq_qual.py -- do not edit by hand.
# In-host KV-sharing qualification under NVIDIA CC (TDX + PPCIe) on gpu13 GPUs 4-7.
# LAB ONLY: deploy as compose project `glm53kvq` (never `work`) with a `services` subset; no
# registrar/nginx/proxy/ports, so it never takes production traffic. Runbook: docs/glm53-kvshare-qualification.md

x-logging-conf: &logging-conf
  driver: "json-file"
  options:
    max-size: "50m"
    max-file: "3"
    labels: "com.datadoghq.ad.logs,com.docker.compose.service"

x-kvq-gpu: &kvq-gpu
  runtime: nvidia
  ipc: host
  # UCX/NIXL cuda_ipc and torch CUDA IPC need a shared PID namespace across containers.
  pid: host
  init: true
  ulimits:
    memlock: -1
    nofile:
      soft: 65535
      hard: 65535
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            # gpu13 GPUs 4-7 only (0-3 serve the small models). Same set and order in every
            # container so cross-replica peers are visible (cuda:0-1 = GPUs 4-5, cuda:2-3 = 6-7).
            device_ids: {GPUS}
            capabilities: [gpu]
  volumes:
    - kvq_kernel_cache:/root/.cache
    - huggingface_cache:/root/.cache/huggingface
  logging: *logging-conf

x-kvq-env: &kvq-env
  HF_HUB_OFFLINE: "1"
  TRANSFORMERS_OFFLINE: "1"
  HF_HUB_DISABLE_TELEMETRY: "1"
  DO_NOT_TRACK: "1"
  NVIDIA_DRIVER_CAPABILITIES: compute,utility
  CUDA_DEVICE_ORDER: PCI_BUS_ID
  NCCL_DEBUG: WARN
  # MUST stay False: expandable segments put allocations in CUDA VMM memory, which legacy
  # cudaIpc / UCX cuda_ipc cannot export; NIXL then silently falls back to host staging.
  PYTORCH_CUDA_ALLOC_CONF: expandable_segments:False
  UCX_TLS: ${{KVQ_UCX_TLS:-all}}
  # Diagnostics, off by default: UCX_PROTO_INFO=y prints the protocol (cuda_ipc vs host staging
  # vs tcp) chosen for each transfer shape; a shorter KV wait surfaces a stuck transfer sooner.
  UCX_PROTO_INFO: ${{KVQ_UCX_PROTO_INFO:-n}}
  UCX_LOG_LEVEL: ${{KVQ_UCX_LOG_LEVEL:-warn}}
  SGLANG_DISAGGREGATION_WAITING_TIMEOUT: ${{KVQ_PD_WAIT_TIMEOUT:-300}}
  TORCHINDUCTOR_CACHE_DIR: /root/.cache/torchinductor
  TRITON_CACHE_DIR: /root/.cache/triton
  TILELANG_CACHE_DIR: /root/.cache/tilelang
  SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION: "0"
  SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE: "1"

x-kvq-pd-engine: &kvq-pd-engine
  <<: *kvq-gpu
  image: {V0521}
  entrypoint: ["bash", "-c"]
  configs:
    - source: {c_pd}
      target: /etc/glm53/pd_startup_patch.py
      mode: 0444
  environment:
    <<: *kvq-env
  restart: "no"
  stop_grace_period: 2m

services:
  # --- Step 1: CC primitives (one-shot, exits after printing KVQ_DONE) ---
  kvq-probe:
    <<: *kvq-gpu
    image: {V6}
    container_name: kvq-probe
    entrypoint: ["python3", "/etc/kvq/probe.py"]
    configs:
      - source: {c_probe}
        target: /etc/kvq/probe.py
        mode: 0444
    environment:
      <<: *kvq-env
    restart: "no"
{log_label("kvq-probe")}
  # --- Step 2 (Test B): 1P:1D GPU->GPU KV move over NIXL. Prefill on GPUs 4-5, decode on 6-7 ---
{engine("kvq-pf", PF_ARGS, "kvq-pd-prefill-v0521-gpu45-nixl-ipc", "kvq-pf")}
{engine("kvq-dc", DC_ARGS, "kvq-pd-decode-v0521-gpu67-nixl-ipc", "kvq-dc")}
  # One-shot NIXL VRAM->VRAM microbench between two containers: initiator cuda:0 (GPU 4) -> target
  # cuda:2 (GPU 6). KVQ_NIXL_RUN names the run dir, KVQ_NIXL_TLS sets UCX_TLS.
{nixl("target", 2)}
{nixl("initiator", 0)}
  # One-shot: which UCX transports (cuda_ipc? cuda_copy?) the NIXL engines' UCX sees, same GPU set.
  kvq-ucxinfo:
    <<: *kvq-gpu
    image: {V0521}
    container_name: kvq-ucxinfo
    entrypoint: ["bash", "-c"]
    command:
      - |
        U=$$(command -v ucx_info || find / -name ucx_info -type f 2>/dev/null | head -1)
        echo "KVQ_UCX bin=$$U"
        python3 -c "import nixl, os; print('KVQ_UCX nixl', os.path.dirname(nixl.__file__))" 2>&1 | head -2
        find / -name 'libuct_cuda*.so*' 2>/dev/null | head -5 | sed 's/^/KVQ_UCX lib /'
        [ -n "$$U" ] && $$U -v | sed 's/^/KVQ_UCX /'
        [ -n "$$U" ] && $$U -d | grep -E 'Transport|Device|Memory domain|Component|bandwidth' | sed 's/^/KVQ_UCX /'
        echo KVQ_DONE
    environment:
      <<: *kvq-env
    restart: "no"
{log_label("kvq-ucxinfo")}
  kvq-router:
    image: {V0521}
    container_name: kvq-router
    runtime: runc
    entrypoint: ["python3", "-m", "sglang_router.launch_router"]
    command: ["--pd-disaggregation", "--prefill", "http://kvq-pf:8000", "8998", "--decode", "http://kvq-dc:8000", "--prefill-policy", "round_robin", "--decode-policy", "round_robin", "--host", "0.0.0.0", "--port", "8000", "--prometheus-port", "29040", "--request-timeout-secs", "3600"]
    # The router loads the tokenizer from --model-path; without the model volume it treats the
    # local path as a HuggingFace id, 404s, and never registers the prefill worker.
    volumes:
      - huggingface_cache:/root/.cache/huggingface:ro
    restart: "no"
    logging: *logging-conf
{log_label("kvq-router")}
  # --- Step 3 (Test A): two prod-shaped v6 replicas sharing an L3 file tier on tmpfs. GPUs 4-5 / 6-7.
  # Never run together with kvq-pf/kvq-dc (same GPUs). KVQ_SHARED=0 is the private-cache control. ---
{sh1}
{sh2}
  kvq-ghost-aggregator:
    image: {V6}
    container_name: kvq-ghost-aggregator
    runtime: runc
    init: true
    cap_drop: [ALL]
    security_opt: ["no-new-privileges:true"]
    mem_limit: 3g
    cpus: 1
    environment:
      NVIDIA_VISIBLE_DEVICES: void
      PYTHONDONTWRITEBYTECODE: "1"
    entrypoint: ["python3", "-m", "sglang.srt.observability.ghost_aggregator"]
    command: ["--socket", "/ghost/aggregator.sock", "--port", "9464", "--sample", "16", "--model-name", "z-ai/glm-5.3-flash"]
    volumes:
      - kvq_ghost:/ghost
    restart: "no"
    logging: *logging-conf
{log_label("kvq-ghost-aggregator")}
  # --- Load driver (one-shot, stdlib only). KVQ_TESTS/KVQ_TARGETS come from the deploy env. ---
  kvq-driver:
    image: {V0521}
    container_name: kvq-driver
    runtime: runc
    entrypoint: ["python3", "/etc/kvq/driver.py"]
    configs:
      - source: {c_driver}
        target: /etc/kvq/driver.py
        mode: 0444
    environment:
      KVQ_TARGETS: ${{KVQ_TARGETS:-http://kvq-router:8000}}
      KVQ_TESTS: ${{KVQ_TESTS:-health,gsm8k,longturn,cold}}
      KVQ_GSM8K_N: ${{KVQ_GSM8K_N:-150}}
      # Keep <= the target's max-running + max-queued (prod GLM argv: 12+4 long tier, 48+8 base).
      KVQ_GSM8K_CONC: ${{KVQ_GSM8K_CONC:-8}}
      KVQ_GSM8K_SAMPLES: ${{KVQ_GSM8K_SAMPLES:-3}}
      KVQ_TURN2_MAX_TOKENS: ${{KVQ_TURN2_MAX_TOKENS:-512}}
      KVQ_LONG_TOKENS: ${{KVQ_LONG_TOKENS:-189000}}
      KVQ_LONG_SEED: ${{KVQ_LONG_SEED:-7}}
      KVQ_LONG_REPEAT: ${{KVQ_LONG_REPEAT:-1}}
      KVQ_METRICS_URLS: ${{KVQ_METRICS_URLS:-}}
      KVQ_METRICS_GREP: ${{KVQ_METRICS_GREP:-}}
      KVQ_TURN_GAP_S: ${{KVQ_TURN_GAP_S:-0}}
      KVQ_TURN1_REPEAT: ${{KVQ_TURN1_REPEAT:-1}}
      KVQ_CONVO_TURNS_A: ${{KVQ_CONVO_TURNS_A:-3}}
    restart: "no"
    logging: *logging-conf
{log_label("kvq-driver")}
networks:
  default:
    external: true
    name: dstack_default

volumes:
  # Model weights already on gpu13 (the `work` project's volume); mounted for reading.
  huggingface_cache:
    external: true
    name: work_hugginface_cache
  kvq_kernel_cache:
  # Test A shared L3 store: in-memory, visible only to this CVM's lab replicas.
  kvq_shared:
    driver: local
    driver_opts:
      type: tmpfs
      device: tmpfs
      o: "size=${{KVQ_TMPFS_SIZE:-170g}},mode=0700"
  kvq_xfer:
    driver: local
    driver_opts:
      type: tmpfs
      device: tmpfs
      o: "size=16m,mode=0700"
  kvq_ghost:
    driver: local
    driver_opts:
      type: tmpfs
      device: tmpfs
      o: "size=1m,mode=0700"

configs:
  {c_pd}:
    content: |
{block(pd_patch, 6)}

  {c_probe}:
    content: |
{block(probe, 6)}

  {c_driver}:
    content: |
{block(driver, 6)}

  {c_shared}:
    content: |
{block(shared, 6)}

  {c_nixl}:
    content: |
{block(nixl_bench, 6)}
"""


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    a = ap.parse_args()
    new = render()
    path = ROOT / TARGET
    if a.write:
        path.write_text(new); print(f"wrote {TARGET}")
        return
    old = path.read_text() if path.exists() else ""
    if old != new:
        sys.stdout.writelines(difflib.unified_diff(old.splitlines(True), new.splitlines(True), str(TARGET), "generated"))
        sys.exit(1)
    print("up to date")


if __name__ == "__main__":
    main()
