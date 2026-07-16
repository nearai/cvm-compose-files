# cvm-compose-files

Docker Compose configurations for the inference CVMs that serve models on `*.completions.near.ai`. Each YAML defines the full stack for one (or more) models: model-downloader, the engine (vLLM / SGLang), `inference-proxy` (auth + signing + attestation + E2EE), nginx TLS termination, `model-proxy-registrar`, `dcgm-exporter`, and the OpenTelemetry collector.

Deployed by `compose-manager` (running inside each inference CVM) which checks out a tag of this repo and runs `docker compose -f <file> up`.

## Directory layout

```
.
├── prod/                  # Production-ready configs. Only these are deployable to prod.
│   ├── GLM-5.2-SGL-FP8-TP8.yaml
│   ├── GLM-5.2-W4AFP8-SGL-TP8.yaml
│   ├── small-models.yaml
│   └── dsv4-qwen36-gemma4.yaml
├── experiments/           # Work-in-progress: AWQ/int4/nvfp4 quantization sweeps,
│                          # otel test harnesses, alternative engine configs.
│                          # NOT deployable to prod. Soak in staging only.
└── scripts/               # CI validators (OTel label contract, compose syntax).
```

**`prod/` is the only directory a production deploy reads from.** A file must pass the [prod-ready checklist](#prod-ready-checklist) before it moves out of `experiments/`.

`cleanup-hf-model.yaml` is a standalone operational utility (deletes cached HF weights), not a model serving config. It lives at the repo root and is excluded from the OTel label validator.

## Prod-ready checklist

A config in `experiments/` graduates to `prod/` only when **all** of the following are true:

1. **Soaked in staging.** Ran on a real CVM under load for ≥30 min with 0 request failures, against the model-onboarding soak harness (see `ansible-playbooks/docs/model-onboarding-workflow.md`). The 30-min soak is the gate, not a suggestion — it catches the detokenizer-wedge / queue-saturation class of bugs that don't surface in a smoke test.
2. **Full monitoring stack present.** Every model config must include: `inference-proxy`, `proxy-nginx`, `model-proxy-registrar`, `model-downloader`, `dcgm-exporter`, `otelcol-contrib`, and the model engine service — all carrying the correct Datadog `com.datadoghq.ad.*` labels **and** `nearai.otel.*` scrape labels. The CI validator (`scripts/validate_otel_labels.rb`) enforces the label contract.
3. **Digest-pinned images.** Every `image:` field uses `@sha256:…`, never `:latest`, `:dev`, or a bare tag. The HuggingFace model is pinned to a specific `--revision` (commit SHA). This is required for attestation reproducibility — the compose hash is registered with the KMS contract.
4. **Registrar healthcheck + readiness probe.** The `model-proxy-registrar` service has the liveness healthcheck (`/tmp/registrar_alive` marker, 180s threshold) and the 30-attempt readiness probe before registration. See `prod/GLM-5.2-SGL-FP8-TP8.yaml` for the canonical pattern (added in cvm-compose-files#57 to surface silent registrar failures).
5. **Graceful drain in nginx.** The TLS `server` block sets `keepalive_timeout 1h` + `keepalive_requests 1000000` so H2 connections from cloud-api survive long idle gaps. Without this, the next request opens a new TCP via model-proxy's L4 LB, may land on a different backend, and the signature 404s.

If a config fails any item, it stays in `experiments/` with a header comment documenting what's missing and why.

## Deploying

> **Use the `/rolling-deploy` skill.** It automates one-host-at-a-time rollouts with health checks and graceful drain for slow-shutdown models (GLM-5.x, DeepSeek). Manual `compose/up` drops in-flight requests on models that take minutes to drain.

### Tag age gate

`compose-manager` enforces `MIN_TAG_AGE_HOURS` (default 48h), gated on the **commit** date, not the tag date. The `auto-tag.yaml` workflow creates `v0.0.N` annotated tags on every merge to main — but those tags point at fresh commits, so they only become deployable ~48h after merge.

For an immediate deploy of an urgent fix, use the **`backdate-tag`** skill, which commits the change, creates a backdated annotated tag, and pushes it. This is the sanctioned workaround; do not edit `MIN_TAG_AGE_HOURS` on production compose-manager instances.

### Deploy command shape

```bash
# Fetch the FULL env from gpu-manager (source of truth — never hand-pick a subset)
# TOK is the gpu-manager dashboard bearer token — do NOT hardcode it here.
DASH=https://gpu-manager.infra.near.ai
TOK="$GPU_MANAGER_TOKEN"
ENV_JSON=$(curl -sS "$DASH/api/instances" -H "Authorization: Bearer $TOK" \
  | jq -c --arg ip "$HOST" '.[] | select(.url|contains($ip)) | .env_vars')

# Rolling deploy via compose-manager
curl -sS -X POST -H "Authorization: Bearer $COMPOSE_MANAGER_TOKEN" \
  -H 'Content-Type: application/json' \
  "http://$HOST:8080/compose/up" \
  -d "{\"tag\":\"v0.0.X\",\"file\":\"prod/<Config>.yaml\",
       \"services\":[],\"env\":$ENV_JSON,\"force_recreate\":false}"
```

The `file` path is relative to the repo root and **includes the `prod/` prefix** (e.g. `"prod/GLM-5.2-SGL-FP8-TP8.yaml"`).

Full deploy recipes (graceful drain for slow-shutdown models, env-var fetch, force-recreate recovery) are in [`infra-docs/docs/deployment.md`](../infra-docs/docs/deployment.md).

## Current production configs

> **Verify against `/machines` before trusting this list.** Query `http://40.160.1.150:8088/machines` for the live host → port → model mapping. The table below drifts.

| Config | Models served | Notes |
|--------|---------------|-------|
| `prod/GLM-5.2-SGL-FP8-TP8.yaml` | `z-ai/glm-5.2` | SGLang TP8, 1M context, official FP8. gpu23 + gpu04. |
| `prod/GLM-5.2-W4AFP8-SGL-TP8.yaml` | `z-ai/glm-5.2` | SGLang TP8, 4-bit (PhalaCloud W4AFP8, 368GB vs 755GB FP8) — same model, serves the full 1M context FP8 can't fit, MTP on. **Alternative** to the FP8 config; deploy one or the other. Startup-hang under CC noted in header (under investigation). |
| `prod/GLM-5.1-SGL-AWQ-TP4.yaml` | `zai-org/GLM-5.1-FP8` | SGLang 2× TP4 (AWQ W4A16). gpu03 + gpu13 + gpu26. Uses `build:` (inline SGLang patch) — see header. Replaced the archived FP8 TP8 config. |
| `prod/Qwen3.5-122B.yaml` | `Qwen/Qwen3.5-122B-A10B` | SGLang 2× TP4 on the official FP8 checkpoint. gpu30. |
| `prod/qwen35-dsv4-flash.yaml` | `Qwen/Qwen3.5-122B-A10B`, `deepseek-ai/DeepSeek-V4-Flash` | Customer-requested 8-GPU mixed pack: 1× Qwen3.5 TP4 replica plus 2× DeepSeek-V4-Flash FP4 TP2 replicas. |
| `prod/small-models.yaml` | gpt-oss-120b, FLUX.2-klein, Qwen3-VL-30B, Qwen3-Embedding, Qwen3-Reranker, whisper-large-v3, privacy-filter, Qwen3.6-35B-A3B-FP8, gemma-4-31B-it | Multi-model pack, gpu07 + gpu11. 10 services across 8 GPUs. |
| `prod/dsv4-qwen36-gemma4.yaml` | DeepSeek-V4-Flash, Qwen3.6-27B-FP8, google/gemma-4-31B-it, Qwen3.6-35B-A3B-FP8 | Multi-model pack, gpu07 |

Configs in `experiments/` are WIP (AWQ/int4/nvfp4 quantization sweeps, otel test harnesses, alternative engine configs) and are not deployed to prod. See their header comments for status. `experiments/GLM-5.1-FP8-TP8-archived.yaml` is the previous GLM-5.1 prod config (SGLang TP8, official FP8) — superseded by `prod/GLM-5.1-SGL-AWQ-TP4.yaml`, kept for reference.

## Datadog decommission

`nearai.otel.logs` is the canonical Docker log metadata label. App collectors
with a `filelog/docker_containers` pipeline consume it; configs without that
pipeline carry the label in preparation for enabling OTel log collection.
During the migration, every log-collected service also carries an exact
`com.datadoghq.ad.logs` copy so Datadog and OTel can run in parallel while Loki
parity is verified. Pre-existing services excluded from both log backends use
`nearai.otel.logs.disabled: "true"`; CI allowlists those exact file/service
pairs, so changing that opt-out requires a separate collection-scope and
privacy review plus an allowlist change. Remove the `com.datadoghq.ad.logs` copy
only after the compose slice has completed its soak, every affected prod config
has an active OTel log pipeline, and Loki parity is verified. Remove Datadog
autodiscovery metric labels only after the corresponding OTel/Prometheus scrape
path is active and metric parity is verified separately.

## CI

`.github/workflows/validate-compose.yaml` runs on every push/PR:

1. **OTel label contract** (`scripts/validate_otel_labels.rb`) — enforces neutral `nearai.otel.logs` metadata or an explicit collection opt-out, temporary Datadog parity, OTel scrape labels, and collector env vars across every service. It also rejects collectors that still consume Datadog labels. `scripts/test_validate_otel_labels.rb` locks the decommission failure cases. `cleanup-hf-model.yaml` is excluded.
2. **Proxy dependency/env contracts** (`scripts/validate_proxy_dependencies.rb`, `scripts/validate_proxy_environment.rb`) — enforces the proxy dependency and required env var wiring.
3. **Prod registrar auth contract** (`scripts/validate_registrar_auth.rb`) — enforces that prod registrar health probes authenticate when probing inference-proxy endpoints.
4. **Compose syntax** — `docker compose -f <file> config` against every `*.yaml`, with dummy env vars for the `${VAR:?required}` fail-fast markers.
5. **Embedded OTel collector config** — validates the `otelcol_app_config` content with the actual collector image.

A PR that fails any of these cannot merge. Run locally before pushing:

```bash
# Compose syntax (fast)
for f in prod/*.yaml experiments/*.yaml; do
  docker compose -f "$f" config --format=yaml >/dev/null 2>&1 \
    || echo "INVALID: $f"
done

# OTel label contract (requires ruby)
ruby scripts/validate_otel_labels.rb

# Registrar auth contract (requires ruby)
ruby scripts/validate_registrar_auth.rb
```

## Adding a new model

The end-to-end onboarding flow (staging `PATCH /v1/admin/models` → auto-generated tests → 30-min soak → auto-promote to prod) is documented in `ansible-playbooks/docs/model-onboarding-workflow.md`. The compose-config side of that flow:

1. **Choose the served model name and SNI domain first** — see [`AGENT.md` → Model naming and SNI domains](AGENT.md#model-naming-and-sni-domains). These are external identifiers: prefer the OpenRouter slug, no quantization suffix, no variant info. Pick them before writing the config so they don't need changing later.
2. Start the config in `experiments/<Model>-<variant>.yaml`. Copy from the closest existing prod config as a template.
3. Pin the image digests and HF `--revision`.
4. Add the neutral OTel log/scrape labels and temporary Datadog parity labels (copy from `prod/GLM-5.2-SGL-FP8-TP8.yaml` — the label set is exact).
5. Soak in staging. Iterate.
6. When the checklist passes, move to `prod/` and open a PR. CI validates; the merge auto-tags.
