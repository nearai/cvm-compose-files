#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: python3 scripts/prepare_gpu13_glm53_base.py --write   (or --check)
"""Generate gpu13's small-models file with two GLM-5.3 BASE replicas on GPUs 4-7 instead of DeepSeek-V4.1 DEP4.

Why: DeepSeek-V4.1 moves to two identical DEP8 replicas (gpu02 + gpu04), so a canary on one can be compared like for
like with the other. gpu04 stops serving GLM, and gpu13 GPUs 4-7 take two of its base TP2 replicas back.

Input: scripts/fixtures/gpu13-glm53-base/small-models.dsv41-dep4.yaml (prod/small-models.yaml as deployed with the
DS4.1 DEP4 ranks, v0.0.547) and gpu04's base replicas r1/r2 in prod/GLM-5.3-Flash-SGL-TP2x4-W4AFP8-V12Q8-gpu04-3B1L.yaml.
Output: prod/small-models.yaml with ONLY:
- the DS4.1 anchors, ranks (dp4-dp7), proxies (base + abliterated), DCGM exporter, kernel/humming cache volumes,
  collector extra_hosts, scrape jobs, downloader lines, :8009 discovery stub, TLS vhosts and registrar mappings removed;
- gpu04's v12 base engine anchors and replicas r1 (GPUs 4,5) and r2 (GPUs 6,7) copied byte for byte, except the GPU
  ids, the shared cache volume's name on this host (`hugginface_cache`) and CUDA_DEVICE_ORDER=PCI_BUS_ID (every gpu13
  GPU service pins CUDA's order to NVML's); gpu04's proxy-glm53 pooling r1+r2; the ghost aggregator and its volume;
  a DCGM exporter on GPUs 4-7; gpu04's scrape jobs for them;
- :8009 as a plain proxy to proxy-glm53 (model-proxy discovers the canonical z-ai/glm-5.3-flash there, and the
  OpenRouter gateway probes /healthz and /v1/metrics), a TLS vhost for the BASE names (canonical, staging, -i<N>,
  -b<hex12>) and a registrar mapping z-ai/glm-5.3-flash -> glm-5-3-flash.completions.near.ai (the same mapping
  gpu03/gpu04 register, so last-writer-wins never flips it).

HiCache is OFF in the v12 base argv (gpu04 r1 has no --enable-hierarchical-cache), so the HiCache RAM variables
are unread and the two replicas add no host-RAM pinning; gpu13's RAM budget is unchanged for Sage and the small
models (the DS4.1 ranks' /dev/shm Engram copy, ~186 GiB, is freed).

`--write` regenerates the target; `--check` prints a diff and exits non-zero when the committed target is stale.
Runbook: docs/gpu13-glm53-base-gpu4-7.md.
"""

import argparse
import re
import difflib
import sys
from pathlib import Path
from typing import Final

ROOT = Path(__file__).resolve().parents[1]
SOURCE: Final = Path("scripts/fixtures/gpu13-glm53-base/small-models.dsv41-dep4.yaml")
GLM_SOURCE: Final = Path("prod/GLM-5.3-Flash-SGL-TP2x4-W4AFP8-V12Q8-gpu04-3B1L.yaml")
TARGET: Final = Path("prod/small-models.yaml")

PREFIX: Final = "model-sg-glm53-w4afp8-tp2-"
R1: Final = f"{PREFIX}r1"
R2: Final = f"{PREFIX}r2"
GLM_SERVICES: Final = ("proxy-glm53", R1, R2, "glm53-ghost-aggregator", "dcgm-glm53")
DS_SERVICES: Final = (
    "proxy-dsv41",
    "proxy-dsv41-abliterated",
    "model-vllm-dsv41-dp4",
    "model-vllm-dsv41-dp5",
    "model-vllm-dsv41-dp6",
    "model-vllm-dsv41-dp7",
    "dcgm-dsv41",
)
JOB: Final = "              - job_name: "
DEVICE_ORDER: Final = (
    "{i}# gpu13: CUDA's device order must equal the NVML order used by device_ids (as for every GPU service in this file).\n"
    "{i}- CUDA_DEVICE_ORDER=PCI_BUS_ID\n"
)

# Inside the anchor: a comment line between two anchors would join the previous anchor's block.
ANCHOR_NOTE: Final = """\
  # GLM-5.3 Flash BASE tier on GPUs 4-7: gpu04's v12 base engine (prod/GLM-5.3-Flash-SGL-TP2x4-W4AFP8-V12Q8-gpu04-3B1L.yaml),
  # copied by scripts/prepare_gpu13_glm53_base.py. Keep the argv byte-identical to the base fleet's.
"""

# Inside proxy-glm53's block: a comment line between two services would join the previous service's block.
SERVICES_NOTE: Final = """\
    # GLM-5.3 Flash BASE tier x2 (GPUs 4,5 and 6,7), behind nginx :8009 / :8444: gpu04's v12 base replicas r1/r2,
    # byte for byte except the GPUs. Registered on glm-5-3-flash.completions.near.ai (base) only.
    # Runbook and rollback: docs/gpu13-glm53-base-gpu4-7.md.
"""

NGINX_8009_DS: Final = '''\
      # :8009 carries both DS4.1 ids because all host ports 8000-8010 and 8444
      # are already taken. Discovery is gated on the real base serving path.
'''

NGINX_8009_GLM: Final = """\
      # :8009 - GLM-5.3 Flash base tier (proxy-glm53 pools r1 and r2 on GPUs 4-7). model-proxy discovers the
      # canonical z-ai/glm-5.3-flash here and routes the base domain to :8444; the OpenRouter gateway probes
      # /healthz and /v1/metrics on this port directly.
      server {
          listen 8009;
          location / {
              set $$proxy_glm53 http://proxy-glm53:8000;
              proxy_pass $$proxy_glm53;
          }
      }
"""

NGINX_TLS_GLM: Final = """\
      server {
          listen 443 ssl http2;
          # Base names only (canonical, staging, rotation -i<N>, per-backend -b<hex12>). The -long-* names are not
          # served here: gpu13 is not a long-tier host.
          server_name glm-5-3-flash.completions.near.ai
                      glm-5-3-flash.completions-stg.near.ai
                      ~^glm-5-3-flash-i\\d+\\.completions(-stg)?\\.near\\.ai$$
                      "~^glm-5-3-flash-b[0-9a-f]{12}\\.completions(-stg)?\\.near\\.ai$$";
          keepalive_timeout 1h;
          keepalive_requests 1000000;
          location ^~ /metrics { return 404; }
          location ^~ /v1/metrics { return 404; }
          location / {
              set $$proxy_glm53 http://proxy-glm53:8000;
              proxy_pass $$proxy_glm53;
          }
      }
"""

GLM_DOWNLOADS: Final = """\
        echo "Downloading zai-org/GLM-5.3-Flash..."
        uvx --from 'huggingface_hub[hf_xet]' hf download zai-org/GLM-5.3-Flash --revision 84c6a6aa9497188e15a635ba793b0f95a79b1033
        echo "Downloading corrected GLM-5.3-Flash chat template..."
        uvx --from 'huggingface_hub[hf_xet]' hf download zai-org/GLM-5.3-Flash chat_template.jinja --revision 3f1971b7b5f7a528c9c4ef6212c8785298a8c24a
        echo "Downloading graphistry/GLM-5.3-Flash-W4AFP8..."
        uvx --from 'huggingface_hub[hf_xet]' hf download graphistry/GLM-5.3-Flash-W4AFP8 --revision 99f1fa70408c52b007d4fd69e02e5a522422e755
"""

GHOST_VOLUME: Final = """\
  ghost:
    driver: local
    driver_opts:
      type: tmpfs
      device: tmpfs
      o: "size=1m,mode=0700"
"""


# SGLang v0.5.23 (docker/sglang-glm53-v0523), the image gpu03's base replicas run. Same argv as gpu04's v12 replicas
# plus the one flag v0.5.23 needs (its default prefill CUDA graphs OOM at warmup at --mem-fraction-static 0.86). The ghost
# aggregator stays on v12, as on gpu03.
V12_IMAGE: Final = "docker.io/nearaidev/sglang@sha256:da8d4e4b5ec667499b58a21e56050aae9bcfc4df9669b3ac27e2e20af033d681"
V0523_IMAGE: Final = "docker.io/nearaidev/sglang@sha256:050b1adf961c8621264d14696d24645f70d1a675509ad400a72ccfb1a3fddcb2"
V0523_FLAG: Final = "--cuda-graph-backend-prefill disabled"
V12_SUFFIX: Final = "-v12-q8"
V0523_SUFFIX: Final = "-v0523"


def once(text: str, old: str, new: str, what: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{what}: expected exactly one match, found {count}")
    return text.replace(old, new)


def block(text: str, start: str, end: str, what: str) -> str:
    """The text from `start` (inclusive) to `end` (exclusive); both must occur once."""
    if text.count(start) != 1 or text.count(end) != 1:
        raise SystemExit(f"{what}: block markers must occur exactly once")
    i = text.index(start)
    j = text.index(end)
    if j <= i:
        raise SystemExit(f"{what}: end marker precedes start marker")
    return text[i:j]


def replace_block(text: str, start: str, end: str, new: str, what: str) -> str:
    return once(text, block(text, start, end, what), new, what)


def glm_pieces(glm: str) -> dict[str, str]:
    """gpu04's GLM blocks, adapted to gpu13 (GPU ids, cache volume name, CUDA device order)."""
    anchors = block(glm, "x-sg-glm53-flash-common: &sg-glm53-flash-common\n", "x-dcgm-common: &dcgm-common\n", "glm anchors")
    anchors = once(anchors, "    - huggingface_cache:/root/.cache/huggingface\n", "    - hugginface_cache:/root/.cache/huggingface\n", "anchor cache volume")
    anchors = once(
        anchors,
        "    - NVIDIA_DRIVER_CAPABILITIES=compute,utility\n",
        "    - NVIDIA_DRIVER_CAPABILITIES=compute,utility\n" + DEVICE_ORDER.format(i="    "),
        "anchor device order",
    )
    proxy = block(glm, "  proxy-glm53:\n", "  # --- GLM-5.3-Flash LONG-CONTEXT proxy", "proxy-glm53")
    proxy = once(
        proxy,
        f"      - VLLM_BACKEND_URLS=http://{R1}:8000,http://{R2}:8000,http://{PREFIX}r3:8000\n",
        f"      - VLLM_BACKEND_URLS=http://{R1}:8000,http://{R2}:8000\n",
        "proxy pool",
    )
    r1 = block(glm, "  # --- GLM-5.3-Flash engine replica 1 (SGLang TP2, GPUs 0-1) ---\n", "  # --- GLM-5.3-Flash engine replica 2", "r1")
    r1 = once(r1, "GPUs 0-1) ---\n", "GPUs 4-5) ---\n", "r1 heading")
    r1 = once(r1, 'device_ids: ["0","1"]', 'device_ids: ["4","5"]', "r1 GPUs")
    r2 = block(glm, "  # --- GLM-5.3-Flash engine replica 2 (SGLang TP2, GPUs 2-3) ---\n", "  # --- GLM-5.3-Flash engine replica 3", "r2")
    r2 = once(r2, "GPUs 2-3) ---\n", "GPUs 6-7) ---\n", "r2 heading")
    r2 = once(r2, 'device_ids: ["2","3"]', 'device_ids: ["6","7"]', "r2 GPUs")
    r2 = once(
        r2,
        "      - NVIDIA_DRIVER_CAPABILITIES=compute,utility\n",
        "      - NVIDIA_DRIVER_CAPABILITIES=compute,utility\n" + DEVICE_ORDER.format(i="      "),
        "r2 device order",
    )
    # The public port is nginx :8009 on this host (validate_otel_labels.rb ties log tags and scrape labels to it).
    proxy = once(proxy, '"port:8000"]}]', '"port:8009"]}]', "proxy log port tag")
    r1 = once(r1, '"port:8000","instance:1"]}]', '"port:8009","instance:1"]}]', "r1 log port tag")
    r2 = once(r2, '"port:8000","instance:2"]}]', '"port:8009","instance:2"]}]', "r2 log port tag")
    aggregator = block(glm, "  # --- Ghost prefix cache aggregator: one per CVM, CPU only ---\n", "  # --- Full-host GPU telemetry ---\n", "aggregator")
    aggregator = once(aggregator, "  # All four engines in this CVM", "  # Both GLM engines in this CVM", "aggregator comment")
    jobs = block(glm, f"{JOB}sglang-{R1}\n", f"{JOB}sglang-{PREFIX}r3\n", "engine jobs")
    if jobs.count('                      port: "8000"\n') != 2:
        raise SystemExit("engine jobs: expected two port labels")
    # gpu13 is a Protected-PCIe board: every GPU 4-7 job carried ppcie: "true" (as the DS4.1 jobs did).
    jobs = jobs.replace('                      port: "8000"\n', '                      port: "8009"\n                      ppcie: "true"\n')
    jobs += block(glm, f"{JOB}ghost-aggregator-glm53-ghost-aggregator\n", f"{JOB}dcgm-dcgm-glm53\n", "aggregator job")
    dcgm_job = block(glm, f"{JOB}dcgm-dcgm-glm53\n", f"{JOB}inference-proxy-proxy-glm53\n", "dcgm job")
    dcgm_job += '                      gpu_pair: "4-7"\n                      ppcie: "true"\n'
    proxy_job = block(glm, f"{JOB}inference-proxy-proxy-glm53\n", f"{JOB}inference-proxy-proxy-glm53-long\n", "proxy job")
    proxy_job = once(proxy_job, '                      port: "8000"\n', '                      port: "8009"\n                      ppcie: "true"\n', "proxy job port")
    return {
        "anchors": once(anchors, "x-sg-glm53-flash-common: &sg-glm53-flash-common\n", "x-sg-glm53-flash-common: &sg-glm53-flash-common\n" + ANCHOR_NOTE, "anchor note"),
        "services": once(proxy, "  proxy-glm53:\n", "  proxy-glm53:\n" + SERVICES_NOTE, "proxy note") + r1 + r2 + aggregator,
        "jobs": jobs,
        "dcgm_job": dcgm_job,
        "proxy_job": proxy_job,
    }


def dcgm_service(ds_block: str) -> str:
    """dcgm-dsv41 (GPUs 4-7, this file's DCGM anchor) relabelled for GLM."""
    out = ds_block
    for old, new in (
        ("dcgm-dsv41", "dcgm-glm53"),
        ("deepseek-ai/DeepSeek-V4.1-Flash-1017", "z-ai/glm-5.3-flash"),
        ("dsv41-flash-1017-vllm-dep4x2-abliterated", "glm53-flash-sgl-tp2x4"),
    ):
        if old not in out:
            raise SystemExit(f"dcgm-dsv41: {old!r} not found")
        out = out.replace(old, new)
    variant = out.split('nearai.otel.config_variant: "', 1)[1].split('"', 1)[0]
    out = once(out, f'      nearai.otel.config_variant: "{variant}"\n', "", "dcgm config_variant label")
    out = once(out, f',"config_variant:{variant}"', "", "dcgm config_variant tag") if f',"config_variant:{variant}"' in out else out
    return out


def registrar(text: str) -> str:
    head, rest = text.split("  registrar_script:\n", 1)
    script, tail = rest.split("  nginx_conf:\n", 1)
    script = once(
        script,
        "      # A bounded vLLM queue can return 503 or close a connection under load.\n"
        "      # Keep the DS4.1 endpoint registered in either case, as gpu02 does.\n",
        "      # A bounded engine queue can return 503 or close a connection under load.\n"
        "      # Keep the GLM endpoint registered in either case: the OpenRouter gateway and the\n"
        "      # proxy's own saturation handling steer around a full replica.\n",
        "registrar saturated comment",
    )
    for what in ("hit connection-level backpressure", "returned HTTP 503", "failed (curl exit", "failed (HTTP"):
        old = f'echo "DS4.1 chat probe for $$model via $$endpoint {what}'
        script = once(script, old, f'echo "Chat probe for $$model via $$endpoint {what}', f"registrar message {what}")
    script = once(
        script,
        '          8009) check_chat_saturated "$${HOST_IP}:8009" "deepseek-ai/DeepSeek-V4.1-Flash-1017" \\\n'
        '                  && check_chat_saturated "$${HOST_IP}:8009" "deepseek-ai/DeepSeek-V4.1-Flash-1017-abliterated" "deepseek-v4-1-flash-abliterated.completions.near.ai" \\\n'
        '                  && check_tls_route "deepseek-v4-1-flash.completions.near.ai" \\\n'
        '                  && check_tls_route "deepseek-v4-1-flash-abliterated.completions.near.ai" ;;\n',
        '          8009) check_chat_saturated "$${HOST_IP}:8009" "z-ai/glm-5.3-flash" \\\n'
        '                  && check_tls_route "glm-5-3-flash.completions.near.ai" ;;\n',
        "registrar 8009 check",
    )
    script = once(
        script,
        "        # model-proxy discovers both ids on HOST_IP:8009 from nginx's gated stub.\n"
        "        # Both domains route to :8444, where nginx selects the proxy by SNI.\n"
        "        # These mappings are identical to gpu02's to avoid last-writer-wins drift.\n"
        '        register_model "deepseek-ai/DeepSeek-V4.1-Flash-1017" "deepseek-v4-1-flash.completions.near.ai"\n'
        '        register_model "deepseek-ai/DeepSeek-V4.1-Flash-1017-abliterated" "deepseek-v4-1-flash-abliterated.completions.near.ai"\n',
        "        # GLM-5.3 Flash (:8009, base tier). model-proxy's model->domain map is one domain per id,\n"
        "        # last-writer-wins: this is the same mapping gpu03 and gpu04 register, never the long domain.\n"
        '        register_model "z-ai/glm-5.3-flash" "glm-5-3-flash.completions.near.ai"\n',
        "registrar mappings",
    )
    return head + "  registrar_script:\n" + script + "  nginx_conf:\n" + tail


def nginx(text: str) -> str:
    head, rest = text.split("  nginx_conf:\n", 1)
    conf, tail = rest.split("  # ── TEMPORARY (gpu02 maintenance): dedicated ingress for Qwen3.8", 1)
    ds_8009 = block(conf, NGINX_8009_DS, "      # ── TLS server blocks (port 443 = host 8444) ──\n", "nginx :8009")
    conf = once(conf, ds_8009, NGINX_8009_GLM + "\n", "nginx :8009")
    # The two DS4.1 TLS vhosts are the conf's last blocks.
    start = "      server {\n          listen 443 ssl http2;\n          # These base names cannot match the abliterated -iN or -b<hex12> names.\n"
    if conf.count(start) != 1:
        raise SystemExit("nginx DS TLS: start marker not found exactly once")
    i = conf.index(start)
    ds_tls = conf[i:]
    if "proxy_pass $$proxy_dsv41_abliterated;\n          }\n      }\n" not in ds_tls or not ds_tls.endswith("      }\n\n"):
        raise SystemExit("nginx DS TLS: unexpected tail")
    conf = conf[:i] + NGINX_TLS_GLM + "\n"
    return head + "  nginx_conf:\n" + conf + "  # ── TEMPORARY (gpu02 maintenance): dedicated ingress for Qwen3.8" + tail


def to_v0523(out: str) -> str:
    """r1/r2 on the v0.5.23 image: common anchor image, candidate argv + V0523_FLAG, engine_image/config_variant labels."""
    common = block(out, "x-sg-glm53-flash-common: &sg-glm53-flash-common\n", "x-sg-glm53-flash-candidate: &sg-glm53-flash-candidate\n", "common anchor")
    out = once(out, common, once(common, f"  image: {V12_IMAGE}\n", f"  image: {V0523_IMAGE}\n", "common anchor image"), "common anchor")
    candidate = out[out.index("x-sg-glm53-flash-candidate: &sg-glm53-flash-candidate\n"):]
    candidate = candidate[: candidate.index("\n\n") + 1]
    if V0523_FLAG.split()[0] in candidate:
        raise SystemExit("candidate argv already sets --cuda-graph-backend-prefill")
    out = once(out, candidate, once(candidate, "      --disable-overlap-schedule\n", f"      --disable-overlap-schedule\n      {V0523_FLAG}\n", "candidate argv"), "candidate anchor")
    old, new = V12_IMAGE.split(":")[-1][:12], V0523_IMAGE.split(":")[-1][:12]
    for pattern in ('"engine_image:{}"', 'nearai.otel.engine_image: "{}"', 'engine_image: "{}"'):
        if out.count(pattern.format(old)) != 2:
            raise SystemExit(f"expected two r1/r2 {pattern.format(old)!r}, found {out.count(pattern.format(old))}")
        out = out.replace(pattern.format(old), pattern.format(new))
    variants = set(re.findall(r'config_variant[:"]\s*"?([\w.-]+{})"'.format(re.escape(V12_SUFFIX)), out))
    if len(variants) != 1:
        raise SystemExit(f"expected one v12 config_variant, found {sorted(variants)}")
    (variant,) = variants
    if out.count(variant) != 6:
        raise SystemExit(f"expected six r1/r2 config_variant labels, found {out.count(variant)}")
    out = out.replace(variant, variant[: -len(V12_SUFFIX)] + V0523_SUFFIX)
    if V12_IMAGE not in block(out, "  glm53-ghost-aggregator:\n", "    container_name: glm53-ghost-aggregator\n", "aggregator"):
        raise SystemExit("ghost aggregator must stay on v12")
    return out


def generate(text: str, glm: str) -> str:
    pieces = glm_pieces(glm)
    out = replace_block(text, "x-dsv41-proxy-common: &dsv41-proxy-common\n", "x-vllm-env:\n", "", "dsv41 proxy anchor")
    out = replace_block(out, "x-dsv41-engine-common: &dsv41-engine-common\n", "x-dcgm-common: &dcgm-common\n", pieces["anchors"], "dsv41 engine anchor")
    out = once(
        out,
        "    # ~475 GiB for the base checkpoint (incl. 186 GiB Engram tables) plus a 4 MB adapter.\n"
        "    # Keep >=500 GiB free on the CVM ZFS model-cache volume /dstack/persistent, not host /data;\n"
        '    # check system_filesystem_usage_bytes{host="gpu13-compose-manager",mountpoint="/dstack/persistent"}.\n'
        "    # Evict old zai-org/GLM-5.3-Flash and graphistry/GLM-5.3-Flash-W4AFP8\n"
        "    # checkpoints with hf-cleanup after the switch if cache headroom is needed.\n",
        "    # The GLM-5.3 Flash snapshots gpu04's base file fetches (W4AFP8 ~165 GiB plus the zai-org repo and its\n"
        "    # corrected chat template); a no-op when gpu13's cache still holds them. Keep >=250 GiB free on the CVM ZFS\n"
        "    # model-cache volume /dstack/persistent, not host /data;\n"
        '    # check system_filesystem_usage_bytes{host="gpu13-compose-manager",mountpoint="/dstack/persistent"}.\n'
        "    # The DeepSeek-V4.1 checkpoint (~475 GiB) can be evicted with hf-cleanup\n"
        "    # (MODEL_NAME=deepseek-ai/DeepSeek-V4.1-Flash-1017) once the DS4.1 ranks are gone.\n",
        "downloader comment",
    )
    out = once(
        out,
        '        echo "Downloading deepseek-ai/DeepSeek-V4.1-Flash-1017..."\n'
        "        uvx --from 'huggingface_hub[hf_xet]' hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 --revision 1bae0e1088993b9b82de08f59de5be5a34e42170\n"
        '        echo "Downloading UnderTides/DeepSeek-V4.1-Flash-1017-heretic-lora..."\n'
        "        uvx --from 'huggingface_hub[hf_xet]' hf download UnderTides/DeepSeek-V4.1-Flash-1017-heretic-lora --revision 196c06e3c2aad43e9688d16fa993b4333f69358a\n",
        GLM_DOWNLOADS,
        "downloader lines",
    )
    out = once(out, "      - proxy-dsv41\n      - proxy-dsv41-abliterated\n", "      - proxy-glm53\n", "nginx depends_on")
    out = replace_block(out, "  proxy-dsv41:\n", "  # --- FLUX x1 (GPU 3, shared with small models) ---\n", pieces["services"], "dsv41 services")
    ds_dcgm = block(out, "  dcgm-dsv41:\n", "  # GPU 3 is shared across FLUX", "dcgm-dsv41")
    out = once(out, ds_dcgm, dcgm_service(ds_dcgm), "dcgm-dsv41")
    out = once(
        out,
        "    extra_hosts:\n"
        '      - "model-vllm-dsv41-dp4:host-gateway"\n'
        '      - "model-vllm-dsv41-dp5:host-gateway"\n'
        '      - "model-vllm-dsv41-dp6:host-gateway"\n'
        '      - "model-vllm-dsv41-dp7:host-gateway"\n'
        "    expose:\n",
        "    expose:\n",
        "collector extra_hosts",
    )
    out = once(out, "  dsv41_kernel_cache:\n  humming_cache:\n", GHOST_VOLUME, "volumes")
    out = replace_block(out, f"{JOB}vllm-model-vllm-dsv41-dp4\n", f"{JOB}sglang-model-sg-flux2-klein-4b-tp1\n", pieces["jobs"], "dsv41 engine jobs")
    out = replace_block(out, f"{JOB}dcgm-dcgm-dsv41\n", f"{JOB}dcgm-dcgm-shared-gpu3\n", pieces["dcgm_job"], "dsv41 dcgm job")
    out = replace_block(out, f"{JOB}inference-proxy-proxy-dsv41\n", f"{JOB}inference-proxy-proxy-flux2-klein-4b\n", pieces["proxy_job"], "dsv41 proxy jobs")
    out = registrar(out)
    out = nginx(out)
    for needle in ("dsv41", "DeepSeek-V4.1", "deepseek-v4-1", "humming"):
        body = out.replace("(MODEL_NAME=deepseek-ai/DeepSeek-V4.1-Flash-1017)", "").replace("The DeepSeek-V4.1 checkpoint", "")
        if needle in body:
            raise SystemExit(f"leftover {needle!r} in the generated file")
    return to_v0523(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = generate((ROOT / SOURCE).read_text(), (ROOT / GLM_SOURCE).read_text())
    target = ROOT / TARGET
    if args.write:
        target.write_text(expected)
        print(f"wrote {TARGET}")
        return 0
    actual = target.read_text() if target.exists() else ""
    if actual != expected:
        sys.stdout.writelines(difflib.unified_diff(actual.splitlines(True), expected.splitlines(True), str(TARGET), "expected"))
        return 1
    print(f"{TARGET} is up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
