#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: python3 scripts/prepare_glm53_gpu04_3b1l.py --write   (or --check)
"""Generate gpu04's GLM-5.3 file with three base replicas and one long-context replica.

Why: since gpu13 left the GLM long tier (2026-11-03), gpu23 is the only long host, and the OpenRouter lane has refused
~20-25K long requests a day as `backend_queue` (vs ~0.2-0.4K a day from 10-31 to 11-02). Long demand itself is flat at
~160-220K requests a day. Moving one base TP2 replica to the long tier adds 16 long running slots (64 -> 80, +25%) for
64 base slots (512 -> 448, -12.5%).

The output is a full copy of prod/GLM-5.3-Flash-SGL-TP2x4-W4AFP8-V12Q8.yaml (the v12 base fleet file; gpu04 runs it
today and it stays the rollback) with ONLY:
- base replica r4 (GPUs 6-7) replaced by `model-sg-glm53-w4afp8-tp2-l4`, a copy of gpu23's long replica r2b from
  prod/GLM-5.3-Flash-SGL-TP4-W4AFP8-LongContext.yaml (same v12 image, Q8 prefill, 16K chunks, 16 running, and the
  host cache tier OFF since 2026-11-06 - see docs/glm53-long-hicache-off.md) with its own container, ghost-replica
  and instance names and ONE argv change: --max-queued-requests 8 -> 32 (see LONG_QUEUE below);
- proxy-glm53 pooling r1-r3, and a second proxy, proxy-glm53-long, pooling l4 only;
- nginx publishing :8001 (long discovery stub + the gateway's probe and health paths, proxied to proxy-glm53-long)
  and a TLS vhost that sends the glm-5-3-flash-long.* names to proxy-glm53-long; the base vhost becomes
  default_server, so every other name still lands on proxy-glm53;
- the registrar registering BOTH endpoints: base ip:8000 (canonical id, base domain, as today) and long ip:8001
  (synthetic z-ai/glm-5.3-flash-long id, long domain), each with routing port 8444 and its own health check; the
  one-shot "legacy :8001" unregister is removed (it would drop the long endpoint at every registrar start);
- the soak relay's :8011, the perception check and the OTel scrape jobs following r4 -> l4, plus a scrape job for
  proxy-glm53-long.

LONG_QUEUE: the OpenRouter gateway applies one long-host ceiling (56) to every long host and the long replica holds
16 running. gpu23's long replicas queue 8, so they never reach the gateway's saturated threshold (16 queued). With a
queue of 32 this replica is flagged saturated at 16 queued and the gateway steers new long conversations to gpu23
before the engine would refuse (16 + 16 < 56 and < 16 + 32), instead of answering 503s.

`--write` regenerates the target; `--check` prints a diff and exits non-zero when the committed target is stale.
Runbook: docs/gpu04-glm53-3base-1long.md.
"""

import argparse
import difflib
import sys
from pathlib import Path
from typing import Final

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import prepare_glm53_w4afp8_long_context as long_generator  # noqa: E402

SOURCE: Final = Path("prod/GLM-5.3-Flash-SGL-TP2x4-W4AFP8-V12Q8.yaml")
LONG_SOURCE: Final = Path("prod/GLM-5.3-Flash-SGL-TP4-W4AFP8-LongContext.yaml")
TARGET: Final = Path("prod/GLM-5.3-Flash-SGL-TP2x4-W4AFP8-V12Q8-gpu04-3B1L.yaml")

PREFIX: Final = "model-sg-glm53-w4afp8-tp2-"
BASE_R4: Final = f"{PREFIX}r4"
LONG_R2B: Final = f"{PREFIX}r2b"
LONG_L4: Final = f"{PREFIX}l4"
LONG_PROXY: Final = "proxy-glm53-long"
LONG_HTTP_PORT: Final = "8001"
LONG_QUEUE: Final = 32

HEADER: Final = """\
# GLM-5.3 Flash gpu04 MIXED-TIER FILE: three base replicas (r1-r3, GPUs 0-5) and one long-context replica (l4,
# GPUs 6-7), generated from prod/GLM-5.3-Flash-SGL-TP2x4-W4AFP8-V12Q8.yaml and gpu23's long replica r2b in
# prod/GLM-5.3-Flash-SGL-TP4-W4AFP8-LongContext.yaml by scripts/prepare_glm53_gpu04_3b1l.py (do not hand-edit).
# Deploy on gpu04 only. Runbook, routing and rollback: docs/gpu04-glm53-3base-1long.md.
#   - Base tier: proxy-glm53 pools r1-r3; the registrar keeps ip:8000 on glm-5-3-flash.completions.near.ai.
#   - Long tier: proxy-glm53-long pools l4; the registrar adds ip:8001 (a discovery-only /v1/models stub answering the
#     SYNTHETIC id z-ai/glm-5.3-flash-long, gated on proxy-glm53-long) on glm-5-3-flash-long.completions.near.ai.
#     Both endpoints route to :8444; nginx splits them by SNI (glm-5-3-flash-long.* -> proxy-glm53-long, anything
#     else -> proxy-glm53). The canonical id is never registered under the long domain.
#   - l4 is gpu23's long replica r2b (v12, Q8 prefill, 16K chunks, 16 running, host cache tier OFF since
#     2026-11-06 - see docs/glm53-long-hicache-off.md) with
#     --max-queued-requests 32 instead of 8, so the OpenRouter gateway's saturated threshold (16 queued) steers
#     new long conversations away from it before the engine refuses.
# ROLLBACK: redeploy prod/GLM-5.3-Flash-SGL-TP2x4-W4AFP8-V12Q8.yaml (r4 back on GPUs 6-7, proxy-glm53 with r1-r4,
# nginx/registrar/otel without :8001), after stopping l4 and proxy-glm53-long with a scoped compose/down.
#
"""


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


def quoted_value(text: str, marker: str, what: str) -> str:
    """The double-quoted value following `marker`; the marker must occur exactly once."""
    if text.count(marker) != 1:
        raise SystemExit(f"{what}: marker {marker!r} not found exactly once")
    return text.split(marker, 1)[1].split('"', 1)[0]


def long_replica(long_text: str) -> str:
    r2b = block(
        long_text,
        "  # --- GLM-5.3-Flash 2xTP2 memory-optimized canary replica 2b (SGLang TP2, GPUs 6-7) ---\n",
        "  # --- GLM-5.3-Flash 2xTP2 memory-optimized canary replica 1a (SGLang TP2, GPUs 0-1) ---\n",
        "r2b",
    )
    r2b = once(
        r2b,
        "  # --- GLM-5.3-Flash 2xTP2 memory-optimized canary replica 2b (SGLang TP2, GPUs 6-7) ---\n"
        "  # Started only by a scoped services list naming it (never by an unscoped up, which would collide with the TP4 replica on these GPUs).\n",
        "  # --- GLM-5.3-Flash LONG-CONTEXT replica l4 (SGLang TP2, GPUs 6-7), behind proxy-glm53-long ---\n"
        "  # gpu23's long replica r2b with its own names and --max-queued-requests 32 (header). Replaces base r4 on these GPUs:\n"
        "  # stop r4 with a scoped compose/down before the first up of l4.\n",
        "r2b comment",
    )
    r2b = r2b.replace(LONG_R2B, LONG_L4)
    r2b = once(r2b, "--max-queued-requests 8\n", f"--max-queued-requests {LONG_QUEUE}\n", "l4 queue")
    r2b = once(r2b, "SGLANG_GHOST_CACHE_REPLICA=r2b", "SGLANG_GHOST_CACHE_REPLICA=l4", "l4 ghost replica")
    r2b = once(r2b, "GLM53_R2B_HICACHE_RAM_BUDGET", "GLM53_L4_HICACHE_RAM_BUDGET", "l4 hicache budget var")
    r2b = once(r2b, '"instance:2b"', '"instance:l4"', "l4 log instance")
    r2b = once(r2b, 'nearai.otel.instance: "2b"', 'nearai.otel.instance: "l4"', "l4 otel instance")
    r2b = r2b.replace("-mr16q8-", f"-mr16q{LONG_QUEUE}-")
    if f"-mr16q{LONG_QUEUE}-" not in r2b:
        raise SystemExit("l4 config_variant: queue token not found")
    return r2b


def v12_long_text() -> str:
    """The long-context file as rendered WITHOUT gpu23's r2b v0.5.23 canary, i.e. r2b on the v12 fleet config.

    l4 copies gpu23's r2b, and r2b is now a single-replica v0.5.23 canary in the committed long file. gpu04's l4 must not
    follow that canary implicitly, so it is pinned to the v12 r2b until the long tier rolls v0.5.23 out on purpose.
    """
    return long_generator.generate((ROOT / long_generator.SOURCE).read_text(), v0523_canary=False)


def generate(text: str, long_text: str) -> str:
    l4 = long_replica(long_text)
    long_variant = quoted_value(l4, 'nearai.otel.config_variant: "', "l4 variant")

    # --- engines: r4 -> l4 ---
    r4 = block(
        text,
        "  # --- GLM-5.3-Flash engine replica 4 (SGLang TP2, GPUs 6-7) ---\n",
        "  # Explicit operator-only semantic check; never starts with a normal stack apply.\n",
        "r4",
    )
    text = text.replace(r4, l4)

    # --- base proxy pools r1-r3; long proxy pools l4 ---
    proxy = block(text, "  # --- GLM-5.3-Flash proxy ---\n", "  # --- GLM-5.3-Flash engine replica 1 ", "proxy")
    base_proxy = once(
        proxy,
        f",http://{BASE_R4}:8000\n",
        "\n",
        "base proxy pool",
    )
    long_proxy = base_proxy.replace("  # --- GLM-5.3-Flash proxy ---\n", "  # --- GLM-5.3-Flash LONG-CONTEXT proxy (l4 only; nginx :8001 and the glm-5-3-flash-long.* vhost) ---\n")
    long_proxy = once(long_proxy, "  proxy-glm53:\n", f"  {LONG_PROXY}:\n", "long proxy service")
    long_proxy = once(long_proxy, "container_name: proxy-glm53\n", f"container_name: {LONG_PROXY}\n", "long proxy name")
    long_proxy = once(
        long_proxy,
        f"VLLM_BACKEND_URLS=http://{PREFIX}r1:8000,http://{PREFIX}r2:8000,http://{PREFIX}r3:8000\n",
        f"VLLM_BACKEND_URLS=http://{LONG_L4}:8000\n",
        "long proxy pool",
    )
    long_proxy = once(long_proxy, '"port:8000"]}]', '"port:8000","tier:long"]}]', "long proxy log tag")
    long_proxy = once(long_proxy, 'nearai.otel.container_name: "proxy-glm53"', f'nearai.otel.container_name: "{LONG_PROXY}"', "long proxy otel name")
    text = text.replace(proxy, base_proxy + long_proxy)

    # --- nginx publishes :8001 and waits for both proxies ---
    text = once(
        text,
        '    depends_on:\n      - proxy-glm53\n    command: /bin/sh -c',
        f'    depends_on:\n      - proxy-glm53\n      - {LONG_PROXY}\n    command: /bin/sh -c',
        "nginx depends_on",
    )
    text = once(
        text,
        '      - "8000:8000"\n      - "8444:443"\n    volumes:\n      - certs:/etc/letsencrypt:ro\n    configs:\n      - source: nginx_conf',
        f'      - "8000:8000"\n      - "8444:443"\n      - "{LONG_HTTP_PORT}:{LONG_HTTP_PORT}"\n    volumes:\n      - certs:/etc/letsencrypt:ro\n    configs:\n      - source: nginx_conf',
        "nginx ports",
    )

    # --- registrar: base + long endpoints ---
    text = once(
        text,
        "      - HTTP_PORT=${HTTP_PORT:-8000}\n      - TLS_PORT=${TLS_PORT:-8444}\n",
        "      - HTTP_PORT=${HTTP_PORT:-8000}\n"
        f"      - LONG_HTTP_PORT=${{LONG_HTTP_PORT:-{LONG_HTTP_PORT}}}\n"
        "      - TLS_PORT=${TLS_PORT:-8444}\n",
        "registrar env",
    )
    text = registrar(text)

    # --- nginx conf: :8001 server and the long TLS vhost ---
    text = nginx(text)

    # --- verification sidecars ---
    text = once(
        text,
        f"          set $$backend http://{BASE_R4}:8000;\n",
        f"          set $$backend http://{LONG_L4}:8000;\n",
        "soak relay 8011",
    )
    text = once(
        text,
        '        for replica in (1, 2, 3, 4):\n            base = f"http://model-sg-glm53-w4afp8-tp2-r{replica}:8000"\n',
        '        for replica in ("r1", "r2", "r3", "l4"):\n            base = f"http://model-sg-glm53-w4afp8-tp2-{replica}:8000"\n',
        "perception check",
    )

    # --- OTel: r4 job -> l4 (long labels), plus the long proxy ---
    job = block(
        text,
        f"              - job_name: sglang-{BASE_R4}\n",
        "              - job_name: ghost-aggregator-glm53-ghost-aggregator\n",
        "r4 scrape job",
    )
    new_job = job.replace(BASE_R4, LONG_L4)
    new_job = once(new_job, 'deployment: "glm53-flash-sgl-tp2x4"', 'deployment: "glm53-flash-sgl-tp4"', "l4 job deployment")
    new_job = once(new_job, 'instance: "4"', 'instance: "l4"', "l4 job instance")
    base_variant = quoted_value(new_job, 'config_variant: "', "l4 job base variant")
    new_job = once(new_job, f'config_variant: "{base_variant}"', f'config_variant: "{long_variant}"', "l4 job variant")
    text = text.replace(job, new_job)
    proxy_job = block(
        text,
        "              - job_name: inference-proxy-proxy-glm53\n",
        "        prometheus/self:\n",
        "proxy scrape job",
    )
    long_job = proxy_job.replace("proxy-glm53", LONG_PROXY)
    long_job = once(long_job, '      port: "8000"\n', '      port: "8000"\n                      tier: "long"\n', "long proxy job tier")
    text = text.replace(proxy_job, proxy_job + long_job)

    return HEADER + text


REGISTRAR_LEGACY: Final = """\
      # The previous mixed stack published its retired co-resident engine on
      # port 8001. Remove only this host-local endpoint; the global model-domain
      # mapping must remain while other hosts may still serve that model.
      unregister_endpoint "$${HOST_IP}:8001" || \\
        echo "[GLM-5.3-Flash-SGL-TP4] WARN: legacy endpoint cleanup failed; continuing"

"""

REGISTRAR_LONG_FUNCS: Final = """\
      # Long-context tier (gpu04 l4): ip:$${LONG_HTTP_PORT} is nginx's discovery-only stub, which
      # answers the SYNTHETIC id z-ai/glm-5.3-flash-long (gated on proxy-glm53-long), so model-proxy
      # attaches this host's :8444 backend to the long domain too; nginx splits the two by SNI.
      # NEVER register the canonical z-ai/glm-5.3-flash under the long domain: model-proxy's
      # model->domain map is one domain per model id, last-writer-wins.
      LONG_ENDPOINT="$${HOST_IP}:$${LONG_HTTP_PORT}"
      LONG_REGISTERED=false
      LONG_FAILURE_COUNT=0

      mark_long_registered() {
        if ! register_endpoint "$$LONG_ENDPOINT" "$${TLS_PORT}"; then
          echo "[GLM-5.3-Flash-gpu04-long] Long endpoint registration failed; retrying on the next loop"
          return 1
        fi
        if ! register_model "z-ai/glm-5.3-flash-long" "glm-5-3-flash-long.completions.near.ai"; then
          echo "[GLM-5.3-Flash-gpu04-long] Long model registration failed; retrying on the next loop"
          return 1
        fi
        if [ "$$LONG_REGISTERED" = false ]; then
          echo "[GLM-5.3-Flash-gpu04-long] Registered z-ai/glm-5.3-flash-long (1x TP2 l4) at $$LONG_ENDPOINT -> :$${TLS_PORT}"
        fi
        LONG_REGISTERED=true
      }

      # Same probe as the base tier, through :$${LONG_HTTP_PORT} -> proxy-glm53-long -> l4. 0 healthy,
      # 2 alive but saturated (keep registered), 1 failed.
      check_long_inference() {
        status=$$(curl -sS --connect-timeout 10 --max-time 45 -o /dev/null -w "%{http_code}" \\
          -X POST "http://$${HOST_IP}:$${LONG_HTTP_PORT}/v1/chat/completions" \\
          -H "Authorization: Bearer $$INFERENCE_TOKEN" \\
          -H "Content-Type: application/json" \\
          -d '{"model":"z-ai/glm-5.3-flash","messages":[{"role":"user","content":"hi"}],"max_tokens":1}')
        curl_status=$$?
        if [ "$$curl_status" -ne 0 ]; then
          case "$$curl_status" in
            52|56) return 2 ;;
            28)
              if curl -sf --connect-timeout 5 --max-time 10 \\
                "http://$${HOST_IP}:$${LONG_HTTP_PORT}/healthz" > /dev/null 2>&1; then
                return 2
              fi
              return 1
              ;;
          esac
          return 1
        fi
        case "$$status" in
          2*) return 0 ;;
          503) return 2 ;;
          *)
            echo "[GLM-5.3-Flash-gpu04-long] Backend returned HTTP $$status"
            return 1
            ;;
        esac
      }

      long_tick() {
        check_long_inference
        case "$$?" in
          0|2)
            if mark_long_registered; then
              LONG_FAILURE_COUNT=0
            fi
            ;;
          *)
            LONG_FAILURE_COUNT=$$((LONG_FAILURE_COUNT + 1))
            echo "[GLM-5.3-Flash-gpu04-long] Health check failed ($$LONG_FAILURE_COUNT/$$MAX_RETRIES)"
            # model-proxy's endpoint cleanup matches by IP, so this unregister briefly (~1 discovery
            # cycle) drops the shared :8444 backend from the base domain too; it runs only after
            # MAX_RETRIES consecutive long failures, and the long stub's 5xx already drops the long
            # domain within a cycle on its own.
            if [ "$$LONG_REGISTERED" = true ] && [ "$$LONG_FAILURE_COUNT" -ge "$$MAX_RETRIES" ]; then
              echo "[GLM-5.3-Flash-gpu04-long] Retry limit reached, unregistering $$LONG_ENDPOINT"
              unregister_endpoint "$$LONG_ENDPOINT"
              LONG_REGISTERED=false
            fi
            ;;
        esac
      }

"""


def registrar(text: str) -> str:
    text = once(text, REGISTRAR_LEGACY, REGISTRAR_LONG_FUNCS, "registrar legacy cleanup")
    text = once(
        text,
        '        echo "SIGTERM received, unregistering $$ENDPOINT"\n        unregister_endpoint "$$ENDPOINT"\n',
        '        echo "SIGTERM received, unregistering $$ENDPOINT and $$LONG_ENDPOINT"\n'
        '        unregister_endpoint "$$LONG_ENDPOINT"\n'
        '        unregister_endpoint "$$ENDPOINT"\n',
        "registrar cleanup",
    )
    # The base endpoint registers first each cycle, so it stays the backend's first-registered probe address.
    text = once(
        text,
        "        esac\n        sleep 60\n      done\n",
        "        esac\n        long_tick\n        sleep 60\n      done\n",
        "registrar loop",
    )
    text = once(
        text,
        '              echo "[GLM-5.3-Flash-SGL-TP4] Retry limit reached, unregistering $$ENDPOINT"\n'
        '              unregister_endpoint "$$ENDPOINT"\n',
        '              echo "[GLM-5.3-Flash-SGL-TP4] Retry limit reached, unregistering $$ENDPOINT and $$LONG_ENDPOINT"\n'
        '              unregister_endpoint "$$LONG_ENDPOINT"\n'
        '              unregister_endpoint "$$ENDPOINT"\n',
        "registrar base failure",
    )
    return text


NGINX_LONG_PLAIN: Final = """\
      # :8001 - GLM-5.3-Flash LONG-CONTEXT replica l4 (proxy-glm53-long).
      # `= /v1/models` is a discovery-only stub: model-proxy probes it and maps the SYNTHETIC id
      # z-ai/glm-5.3-flash-long to glm-5-3-flash-long.completions.near.ai (routing targets :8444, where
      # the long vhost below sends the long names to proxy-glm53-long on the canonical id). The
      # synthetic id must never appear in SGLang, the catalog, or any serving-path config.
      # The stub is gated on the REAL long serving path via auth_request (a zero-body subrequest to
      # proxy-glm53-long's /v1/models): with l4 or its proxy down it returns 5xx and discovery drops
      # the long domain's backend. PHASE-ORDER TRAP: a bare `return` runs in the rewrite phase,
      # BEFORE auth_request's access phase, so the gate would be a no-op; try_files runs in the
      # precontent phase, AFTER access, and the always-missing file falls through to the named
      # location that holds the `return`.
      # `location /` is the OpenRouter gateway's probe path for this backend (/healthz, /v1/metrics)
      # and the registrar's long health check (chat), like gpu13's :8009 was.
      server {
        listen 8001;
        location = /v1/models {
          auth_request /_glm53_long_health;
          try_files /__synthetic_models__ @glm53_long_models;
        }
        location @glm53_long_models {
          default_type application/json;
          return 200 '{"object":"list","data":[{"id":"z-ai/glm-5.3-flash-long","object":"model","owned_by":"nearai"}]}';
        }
        location = /_glm53_long_health {
          internal;
          set $$proxy_glm53_long_health http://proxy-glm53-long:8000/v1/models;
          proxy_pass $$proxy_glm53_long_health;
          proxy_pass_request_body off;
          proxy_set_header Content-Length "";
          proxy_connect_timeout 2s;
          proxy_read_timeout 4s;
        }
        location / {
          set $$proxy_glm53_long http://proxy-glm53-long:8000;
          proxy_pass $$proxy_glm53_long;
        }
      }
"""

NGINX_LONG_TLS: Final = """\

      # Long-context names -> proxy-glm53-long (l4). The long `-i<N>` / `-b<hex12>` regexes cannot
      # match the base names and the base vhost is default_server, so every other SNI (the base
      # names, the base `-b<hex12>` handle) stays on proxy-glm53.
      server {
        listen 443 ssl http2;
        server_name glm-5-3-flash-long.completions.near.ai
                    glm-5-3-flash-long.completions-stg.near.ai
                    ~^glm-5-3-flash-long-i\\d+\\.completions(-stg)?\\.near\\.ai$$
                    "~^glm-5-3-flash-long-b[0-9a-f]{12}\\.completions(-stg)?\\.near\\.ai$$";
        keepalive_timeout 1h;
        keepalive_requests 1000000;
        location ^~ /metrics { return 404; }
        location ^~ /v1/metrics { return 404; }
        location / {
          set $$proxy_glm53_long http://proxy-glm53-long:8000;
          proxy_pass $$proxy_glm53_long;
        }
      }
"""


def nginx(text: str) -> str:
    if text.count("  nginx_conf:\n") != 1:
        raise SystemExit("nginx conf: '  nginx_conf:' section marker not found exactly once")
    conf_start = text.index("  nginx_conf:\n")
    head, conf = text[:conf_start], text[conf_start:]
    conf = once(
        conf,
        "      ssl_certificate /etc/letsencrypt/live/completions.near.ai/fullchain.pem;\n",
        NGINX_LONG_PLAIN + "      ssl_certificate /etc/letsencrypt/live/completions.near.ai/fullchain.pem;\n",
        "nginx plain :8001",
    )
    conf = once(conf, "        listen 443 ssl http2;\n", "        listen 443 ssl http2 default_server;\n", "nginx base default_server")
    if not conf.endswith("      }\n"):
        raise SystemExit("nginx conf: unexpected tail")
    conf = conf + NGINX_LONG_TLS
    return head + conf


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = generate((ROOT / SOURCE).read_text(), v12_long_text())
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
