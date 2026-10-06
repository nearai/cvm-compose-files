"""Opt-in engine observability shared by the generated GLM-5.3 Flash prod files.

docker/sglang-glm53-hicache-w4afp8 v6 (tag glm53-hicache-w4afp8-v6, #340; v5 was never signed)
carries two observation-only patches, both inert unless enabled (see that recipe's README):

- ghost-prefix-cache.diff: SGLANG_GHOST_CACHE=1 makes TP rank 0 measure would-be prefix hits from
  keyed, sampled page digests (sglang:ghost_*). In shared mode every engine of one CVM reads one key
  from an in-memory volume and sends its sampled digests to one aggregator sidecar per CVM, which
  exports the pooled view (sglang:ghost_pool_*) on its own /metrics port.
- kv-tier-metrics.diff: SGLANG_KV_TIER_METRICS=1 makes TP rank 0 export sglang:kv_tier_*.

Older images ignore the engine variables. The sidecar runs the engines' image, so it only works once
the engines run v6 too. scripts/validate_glm53_prod_config.rb and scripts/validate_ds4f_migration.rb
hold the same values; change them together.
"""

from typing import Final

SERVICE: Final = "glm53-ghost-aggregator"
VOLUME: Final = "ghost"
MOUNT: Final = f"{VOLUME}:/ghost"
KEY_FILE: Final = "/ghost/key"
SOCKET: Final = "/ghost/aggregator.sock"
SAMPLE: Final = 16
PORT: Final = 9464
MODEL_NAME: Final = "z-ai/glm-5.3-flash"
JOB: Final = f"ghost-aggregator-{SERVICE}"
# Appended to every engine's config_variant so dashboards can tell the observed engines apart.
VARIANT_SUFFIX: Final = "-obs-v1"


def engine_environment(replica: str, indent: int) -> str:
    """The engine environment entries, as list items at `indent` spaces."""
    pad = " " * indent
    lines = (
        "# Opt-in observability (docker/sglang-glm53-hicache-w4afp8 v6; older images ignore it).",
        "# TP rank 0 only, off the serving path. The ghost prefix cache hashes each finished",
        f"# request's pages into keyed digests (a 1/{SAMPLE} sample; tokens never leave the engine)",
        "# on a background thread and exports sglang:ghost_*. Every engine in this CVM shares the",
        f"# key on the in-memory {VOLUME} volume and sends its digests to {SERVICE},",
        "# without ever blocking: if the aggregator is down the sends are dropped and counted.",
        "# The KV tier metrics export sglang:kv_tier_* (VRAM/DRAM evictions, load-backs, tier sizes).",
        "- SGLANG_GHOST_CACHE=1",
        f"- SGLANG_GHOST_CACHE_SAMPLE={SAMPLE}",
        f"- SGLANG_GHOST_CACHE_KEY_FILE={KEY_FILE}",
        f"- SGLANG_GHOST_CACHE_SOCKET={SOCKET}",
        f"- SGLANG_GHOST_CACHE_REPLICA={replica}",
        "- SGLANG_KV_TIER_METRICS=1",
    )
    return "".join(f"{pad}{line}\n" for line in lines)


def replica_line(replica: str) -> str:
    return f"- SGLANG_GHOST_CACHE_REPLICA={replica}\n"


def volume_declaration() -> str:
    """The top-level volume: tmpfs, so the key never touches disk and dies with the CVM."""
    return (
        f"  {VOLUME}:\n"
        "    driver: local\n"
        "    driver_opts:\n"
        "      type: tmpfs\n"
        "      device: tmpfs\n"
        '      o: "size=1m,mode=0700"\n'
    )


def sidecar_service(image: str, deployment: str, engines: str) -> str:
    """The per-CVM aggregator service. `engines` ("Both engines", ...) starts its comment."""
    return (
        "  # --- Ghost prefix cache aggregator: one per CVM, CPU only ---\n"
        f"  # {engines} in this CVM send their sampled page digests (keyed hashes, never tokens)\n"
        f"  # to {SOCKET} on the in-memory {VOLUME} volume. This process keeps one LRU across\n"
        f"  # them and exports sglang:ghost_pool_* on :{PORT}: per replica, the reuse one pooled cache\n"
        "  # would have served and the reuse seen only on another replica. It runs the engines' image so\n"
        "  # both move together. Serving never depends on it: the engines do not wait for it, and an\n"
        "  # image without ghost-prefix-cache.diff makes only this container exit and restart.\n"
        f"  {SERVICE}:\n"
        f"    image: {image}\n"
        f"    container_name: {SERVICE}\n"
        "    runtime: runc\n"
        "    init: true\n"
        "    cap_drop: [ALL]\n"
        '    security_opt: ["no-new-privileges:true"]\n'
        "    mem_limit: 3g\n"
        "    cpus: 1\n"
        "    environment:\n"
        "      - NVIDIA_VISIBLE_DEVICES=void\n"
        "      - PYTHONDONTWRITEBYTECODE=1\n"
        '    entrypoint: ["python3", "-m", "sglang.srt.observability.ghost_aggregator"]\n'
        f'    command: ["--socket", "{SOCKET}", "--port", "{PORT}", "--sample", "{SAMPLE}", "--model-name", "{MODEL_NAME}"]\n'
        "    volumes:\n"
        f"      - {MOUNT}\n"
        "    restart: unless-stopped\n"
        "    logging: *logging-conf\n"
        "    labels:\n"
        "      com.datadoghq.ad.logs: '[{\"source\":\"sglang-ghost-aggregator\",\"service\":\"sglang-ghost-aggregator\","
        f'"tags":["model:{MODEL_NAME}","served_model:{MODEL_NAME}","deployment:{deployment}",'
        '"env:${ENV}","host:${CVM_HOST}","ip:${HOST_IP}"]}]\'\n'
        '      nearai.otel.scrape: "true"\n'
        '      nearai.otel.job: "ghost-aggregator"\n'
        '      nearai.otel.service: "sglang-ghost-aggregator"\n'
        '      nearai.otel.source: "sglang-ghost-aggregator"\n'
        f'      nearai.otel.container_name: "{SERVICE}"\n'
        f'      nearai.otel.port: "{PORT}"\n'
        '      nearai.otel.path: "/metrics"\n'
        f'      nearai.otel.model: "{MODEL_NAME}"\n'
        f'      nearai.otel.served_model: "{MODEL_NAME}"\n'
        f'      nearai.otel.deployment: "{deployment}"\n'
        '      nearai.otel.env: "${ENV}"\n'
        '      nearai.otel.host: "${CVM_HOST}"\n'
        '      nearai.otel.host_machine: "${CVM_HOST}"\n'
        '      nearai.otel.cvm_name: "${CVM_NAME}"\n'
        '      nearai.otel.ip: "${HOST_IP}"\n'
        "\n"
    )


def scrape_job(deployment: str) -> str:
    """The collector's scrape job for the sidecar, labelled like every other target in the file."""
    return (
        f"              - job_name: {JOB}\n"
        "                scrape_interval: 15s\n"
        "                metrics_path: /metrics\n"
        "                static_configs:\n"
        f"                  - targets: ['{SERVICE}:{PORT}']\n"
        "                    labels:\n"
        '                      service: "sglang-ghost-aggregator"\n'
        '                      source: "sglang-ghost-aggregator"\n'
        f'                      container_name: "{SERVICE}"\n'
        f'                      model: "{MODEL_NAME}"\n'
        f'                      served_model: "{MODEL_NAME}"\n'
        f'                      deployment: "{deployment}"\n'
        '                      env: "${ENV}"\n'
        '                      host: "${CVM_HOST}"\n'
        '                      host_machine: "${CVM_HOST}"\n'
        '                      cvm_name: "${CVM_NAME}"\n'
        '                      ip: "${HOST_IP}"\n'
        f'                      port: "{PORT}"\n'
    )
