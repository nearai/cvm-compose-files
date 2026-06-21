# AGENT.md — cvm-compose-files

Operating rules for agents editing inference compose configs in this repo. Read this before adding or modifying any YAML.

## Directory layout

```
prod/           # Production-ready configs. Only these are deployable to prod.
experiments/    # WIP: quantization sweeps, alt engine configs, otel test harnesses.
                # NOT deployable to prod. Soak in staging only.
scripts/        # CI validators.
*.yaml @ root   # Only `cleanup-hf-model.yaml` (operational utility, validator-excluded).
```

**A file in `experiments/` must never be deployed to prod.** The `file` field in a `compose/up` payload must always reference `prod/<Config>.yaml`.

### Deploy path migration

Files were moved from the repo root into `prod/` and `experiments/` in this PR. The validator (`scripts/validate_otel_labels.rb`) and CI workflow (`.github/workflows/validate-compose.yaml`) now glob `prod/*.yaml` + `experiments/*.yaml` + root `*.yaml`. **Deploy commands must update their `file` field** — every existing `compose/up` payload, dashboard `env_vars` entry, and runbook that referenced `"file":"GLM-5.2-SGL-FP8-TP8.yaml"` must now use `"file":"prod/GLM-5.2-SGL-FP8-TP8.yaml"`. The gpu-manager dashboard's stored `env_vars` per instance does not store the file path (it's passed at deploy time), but any pinned deploy scripts or skills need the `prod/` prefix.

## File naming

- **Prod single-model**: `prod/<Model>.yaml` — e.g. `prod/GLM-5.2-SGL-FP8-TP8.yaml`. The name encodes model + engine + quant + TP. Be specific: `-SGL-FP8-TP8` is more useful than bare `GLM-5.2.yaml` when multiple configs for the same model exist.
- **Prod multi-model pack**: `prod/<model1>-<model2>-<model3>.yaml` — e.g. `prod/dsv4-qwen36-gemma4.yaml`. Lowercased short names, hyphen-joined.
- **Experiment**: `experiments/<Model>-<variant>-test.yaml` — e.g. `experiments/GLM-5.1-AWQ-4bit-test.yaml`. The `-test` suffix is mandatory for anything not meeting the prod-ready checklist.
- **Operational utility**: root-level, descriptive — `cleanup-hf-model.yaml`. These are added to `EXCLUDED_FILES` in the validator.

## Model naming and SNI domains

### The served name is the API contract — the checkpoint is an implementation detail

`MODEL_NAME` (inference-proxy env var) and `--served-model-name` (engine flag) are what API consumers see in requests and responses. They must be set to the **stable model identifier**, not the HuggingFace checkpoint path. The checkpoint (`--model-path`, `--revision`) is an internal detail: you can swap it for a different quantization without changing anything visible to callers.

GLM-5.2 is the canonical example: both `prod/GLM-5.2-SGL-FP8-TP8.yaml` (checkpoint `zai-org/GLM-5.2-FP8`) and `prod/GLM-5.2-W4AFP8-SGL-TP8.yaml` (checkpoint `PhalaCloud/GLM-5.2-W4AFP8`) set `MODEL_NAME=z-ai/glm-5.2` and `--served-model-name z-ai/glm-5.2`. A caller switching from one to the other sees no difference.

### Choosing a served model name

1. **Prefer the OpenRouter slug** — `z-ai/glm-5.2`, `openai/gpt-oss-120b`, `deepseek-ai/DeepSeek-V4-Flash`. Check `openrouter.ai/models` for the canonical slug before inventing one.
2. **Fall back to the HuggingFace org/model-name** when there is no OpenRouter listing — strip any quantization suffix from the model name (no `-FP8`, `-AWQ`, `-W4AFP8`, `-NVFP4`, etc.).
3. **No quantization in the served name, ever.** The served name describes the model; the checkpoint describes how it's stored. Callers must not need to know or care which quantization is running.

### Never change the served name of a prod model

`MODEL_NAME` is a breaking change. API clients — including cloud-api's usage reporter, external callers, and infra-tests — hardcode it. Changing it silently breaks:
- Usage reporting (`MODEL_NAME` must exactly match cloud-api's `model_name` column — mismatch → silent 404s).
- Client requests that pin a model ID.
- infra-tests assertions.

If you need a name correction, open an issue, coordinate the cloud-api model table update, and flip both atomically — never just one side. For a checkpoint swap (e.g., switching from FP8 to W4AFP8), keep `MODEL_NAME` and `--served-model-name` identical to what is already in prod.

### SNI domain naming

The SNI domain (`server_name` in nginx, first arg to `register_model`) is also a stable external identifier. Rules:

1. **No quantization, no variant suffix.** Both GLM-5.2 configs use `glm-5-2.completions.near.ai`. If a model has multiple alternative configs, they share one domain — only one is deployed at a time.
2. **Format**: lowercase, digits, hyphens — `<model-shortname>.completions.near.ai`. Use dots in version numbers as hyphens (e.g., `glm-5-2`, `qwen3-6-27b`).
3. **Derive from the model name, not the checkpoint.** `gpt-oss-120b.completions.near.ai` not `gpt-oss-120b-fp8.completions.near.ai`.
4. **Never reuse a domain for a different model.** Once a domain is live, callers cache it. Retiring a model means removing its domain registration; do not reassign it.
5. **Changing a domain is a breaking change** — it must be coordinated with model-proxy DNS, cloud-api's inference URL table, and any pinned clients. Treat it like renaming a model.

### Summary table

| Field | Owner | Changes allowed? | Rule |
|-------|-------|-----------------|------|
| `MODEL_NAME` / `--served-model-name` | API contract | Never for prod | Prefer OpenRouter slug; no quant suffix |
| `--model-path` / `--revision` | Checkpoint | Free to swap | Points to actual HF checkpoint |
| `server_name` / `register_model` domain | SNI routing | Never without coordination | `<model-shortname>.completions.near.ai`, no quant |
| `nearai.otel.model` DD label | Observability | Free to update | Use checkpoint path so DD shows what's actually running |

## Prod-ready checklist

A config graduates from `experiments/` to `prod/` only when **all** pass:

1. **Soaked ≥30 min in staging, 0 failures.** Against the model-onboarding soak harness. Non-negotiable — catches detokenizer-wedge and queue-saturation bugs that smoke tests miss.
2. **Full monitoring stack.** Every model config includes: `inference-proxy`, `proxy-nginx`, `model-proxy-registrar`, `model-downloader`, `dcgm-exporter`, `otelcol-contrib`, and the model engine. All carry Datadog `com.datadoghq.ad.*` labels **and** `nearai.otel.*` scrape labels. CI enforces the contract (see [Monitoring label contract](#monitoring-label-contract)).
3. **Digest-pinned images.** Every `image:` is `name@sha256:…`. No `:latest`, `:dev`, or bare tags. HuggingFace model pinned via `--revision <sha>`. Required for KMS attestation reproducibility.
4. **Registrar healthcheck + readiness probe.** `model-proxy-registrar` has the `/tmp/registrar_alive` liveness healthcheck (180s threshold, `start_period: 1200s`) and the 30-attempt readiness probe before entering the registration loop. Canonical pattern in `prod/GLM-5.2-SGL-FP8-TP8.yaml` (cvm-compose-files#57).
5. **Graceful drain in nginx.** TLS `server` block: `keepalive_timeout 1h; keepalive_requests 1000000;`. Prevents H2 connection churn that causes signature 404s when cloud-api's bucket-pinned connection lands on a different backend after a model-proxy L4 rebalance.

If a config fails any item, it stays in `experiments/` with a `# STATUS: EXPERIMENT — <reason>` header.

## Compose file structure

Every prod config follows this service set (copy from `prod/GLM-5.2-SGL-FP8-TP8.yaml` as the canonical template):

| Service | Role | Required in prod |
|---------|------|------------------|
| `model-downloader` | One-shot HF download via `uvx hf download … --revision <sha>`. `restart: "no"`. | yes |
| `proxy-nginx` | TLS termination (`:8444→443`, `:8000→80`), JSON access logs, `proxy_pass` to the inference-proxy. | yes |
| `model-proxy-registrar` | Curl loop registering the backend with model-proxy. Healthcheck on `/tmp/registrar_alive`. | yes |
| `proxy-<model>` | `nearaidev/vllm-proxy-rs@sha256:…` — auth, signing, attestation, E2EE. `privileged: true`, mounts `dstack.sock` + certs. | yes |
| `model-sg-<model>` or `model-vllm-<model>` | The engine (SGLang or vLLM). `depends_on: model-downloader: service_completed_successfully`. | yes |
| `dcgm-exporter` | NVIDIA GPU metrics on `:9400`. `runtime: nvidia`. | yes |
| `otelcol-contrib` | OpenTelemetry collector. Scrapes `nearai.otel.scrape: "true"` services, exports to the central ingest. | yes |

### Anchors (top of file)

Every file defines these anchors and reuses them via `<<: *anchor`:

- `x-logging-conf` — json-file driver, 100m×10 files, Datadog log labels enabled.
- `x-nvidia` — `runtime: nvidia`, `ipc: host`, memlock/nofile ulimits.
- `x-vllm-proxy-common` — the inference-proxy base: image digest, `user: root`, `privileged: true`, `extra_hosts: ["compose-manager:host-gateway"]`, mounts `dstack.sock` + `certs:ro`, `restart: unless-stopped`.
- `x-downloader-common` — the `uv:python3.11-bookworm-slim` image, `sh -c` entrypoint, `restart: "no"`.

### Configs (compose `configs:` block)

- `nginx_conf` — inline nginx config. JSON `log_format` with `request_id`, `org_id`, `workspace_id`. Two `server` blocks: `:80` (HTTP→proxy) and `:443` (TLS→proxy) with the H2 keepalive settings.
- `registrar_script` — inline shell. The health-check + registration loop. Parameterized by `HOST_IP`, `HTTP_PORT`, `TLS_PORT`, `MODEL_PROXY_TOKEN`.
- `otelcol_app_config` — inline OTel collector YAML. **Must not** export OTLP to `datadog-agent:4317` (validator rejects this — standalone inference configs export to the central ingest, not a local DD agent).

### Volumes

- `huggingface_cache` — HF model cache, shared between downloader and engine.
- `kernel_cache` — DeepGemm JIT cache (SGLang). Must be a named volume, not tmpfs.
- `certs` — Let's Encrypt certs, mounted `:ro` into nginx and the proxy.
- `otelcol_app_storage` — OTel collector storage. **Validator requires this volume to exist.**

## Monitoring label contract

CI (`scripts/validate_otel_labels.rb`) enforces a strict label contract. Every service with logs or metrics must carry matching Datadog + OTel labels.

### Required Datadog log tags (`com.datadoghq.ad.logs`)

Every service except `otelcol-contrib` must have `model`, `deployment`, `env`, `host`, `ip` tags. `port` tag must match the nginx public port for proxy/engine services. The JSON shape:

```yaml
labels:
  com.datadoghq.ad.logs: '[{"source":"sglang","service":"sglang","tags":["model:zai-org/GLM-5.2-FP8","deployment:GLM-5.2","env:${ENV}","host:${CVM_HOST}","ip:${HOST_IP}","port:8000"]}]'
```

`source` must equal `nearai.otel.service` when both are set. `service` must equal `nearai.otel.service`.

### Required OTel scrape labels (`nearai.otel.*`)

For any service with `nearai.otel.scrape: "true"`:

| Label | Value |
|-------|-------|
| `nearai.otel.job` | scraper job name (e.g. `sglang`, `vllm-proxy`, `dcgm`) |
| `nearai.otel.service` | service name (must match Datadog `source`/`service`) |
| `nearai.otel.port` | scrape port (e.g. `8000`, `9400`) |
| `nearai.otel.path` | metrics path (usually `/metrics`) |
| `nearai.otel.model` | full model id (e.g. `zai-org/GLM-5.2-FP8`) |
| `nearai.otel.deployment` | deployment name (e.g. `GLM-5.2`) |
| `nearai.otel.env` | `${ENV}` |
| `nearai.otel.host` | `${CVM_HOST}` |
| `nearai.otel.ip` | `${HOST_IP}` |

The collector's `otelcol_app_config` must have a `scrape_config` target matching `<service_name>:<nearai.otel.port>` for every scraped service, with labels mirroring the above. Extra scrape targets (no matching `scrape: "true"` service) fail validation. Missing scrape targets fail validation.

### OTel collector env vars (required, with soft defaults)

`MONITORING_INGEST_TOKEN`, `CVM_NAME`, `CVM_HOST`, `DD_HOSTNAME`, `ENV`, `HOST_IP` — each must appear as `VAR=${VAR:-...}` (soft-default marker) in `otelcol-contrib.environment`. The validator checks for the `${VAR:-` substring.

## Env vars (injected by compose-manager at deploy time)

The gpu-manager dashboard (`https://gpu-manager.infra.near.ai`) is the source of truth for per-instance env vars. **Fetch the full map and pass it whole** — never hand-pick a subset. The "enough" set rots as new vars get added; a partial deploy silently drops whatever's missing.

Key vars and what breaks if omitted:

- `PROXY_TOKEN` — Bearer auth for inference-proxy. Missing → 401.
- `CLOUD_API_USAGE_TOKEN` — usage reporting to `/v1/internal/usage`. Missing → reporting silently skipped (no fallback; legacy `/v1/usage` was removed).
- `MODEL_PROXY_TOKEN` — registrar auth for model-proxy. Missing → backend disappears from `/backends/count`.
- `WEB_CONTEXT_SEARCH_URL` / `WEB_CONTEXT_SEARCH_API_KEY` — Brave web search agent loop. Missing → 400s.
- `HUGGING_FACE_HUB_TOKEN` — gated model downloads.
- `MONITORING_INGEST_TOKEN` — OTel collector export auth. Missing → metrics don't ship.
- `HOST_IP`, `CVM_HOST`, `CVM_NAME`, `DD_HOSTNAME`, `ENV` — labeling. Missing → Datadog/OTel tags are empty.

See `infra-docs/docs/deployment.md` → "Env on compose/up/down" for the fetch-and-splice recipe.

## Validation

Run before pushing:

```bash
# Compose syntax (fast, ~5s per file)
for f in prod/*.yaml experiments/*.yaml; do
  docker compose -f "$f" config --format=yaml >/dev/null 2>&1 \
    && echo "OK: $f" || echo "INVALID: $f"
done

# OTel label contract (requires ruby)
ruby scripts/validate_otel_labels.rb

# Proxy dependency contract — proxy services must not depends_on model services (requires ruby)
ruby scripts/validate_proxy_dependencies.rb

# Proxy environment contract — inference-proxy services must explicitly pass required env vars
ruby scripts/validate_proxy_environment.rb

# Embedded OTel collector config validation (per file, slow)
for f in prod/*.yaml; do
  config="$(docker compose -f "$f" config --format=json | jq -r '.configs.otelcol_app_config.content // empty')"
  [ -z "$config" ] && continue
  image="$(docker compose -f "$f" config --format=json | jq -r '.services["otelcol-contrib"].image')"
  printf '%s' "$config" | docker run --rm -i "$image" validate --config=/dev/stdin
done
```

CI runs all four on every push/PR. A failing check blocks merge.

## Common pitfalls

- **Proxy services must not `depends_on` model services — CI-enforced.** `depends_on` makes `docker compose up <service>` implicitly start (or recreate) the listed dependencies. During operational tasks like a proxy-only restart or an nginx reload, this can unexpectedly recreate a healthy model container — dropping in-flight requests and triggering a 15-30 min cold start. `scripts/validate_proxy_dependencies.rb` (CI-enforced) rejects any proxy service (`proxy-*`, `vllm-proxy-*`) with a `depends_on` on a model service (`model-*`). The `model-* → model-downloader` dependency (download-before-serve) is allowed; only the proxy→model direction is banned. The readiness probe in `model-proxy-registrar` gates registration instead (the registrar polls until the backend is up before registering). If you ever need a proxy→model dependency for a special case, document it in the config header and pass explicit `services:` lists in `compose/up` calls to scope what gets touched — but the CI validator will block it until the rule is updated.
- **`build:` configs need `docker compose build` first.** `prod/GLM-5.1-SGL-AWQ-TP4.yaml` uses an inline Dockerfile (`x-awq-build`) to patch SGLang for the QuantTrio AWQ checkpoint. Deploy requires `docker compose build` before `up`, and compose-manager's `compose/up` may not trigger a build automatically. Pre-build on the host, then deploy.
- **Moving a service's `proxy_pass` upstream without restarting nginx.** CVM nginx resolves `proxy_pass` hostnames at boot only. If you recreate proxy containers (new Docker IPs), nginx keeps routing to the old IPs. Always include `nginx` in the services list when recreating proxy containers, or force-recreate nginx immediately after.
- **`--max-queued-requests` too high on SGLang.** The queue is a brief overflow buffer, not a backlog. If it's large enough to hold requests for seconds, TTFB explodes and cloud-api times out. Pick a value ≈ "how many requests can complete in the fail-over window." GLM-5.1 uses 8.
- **Forgetting `--enable-cache-report` (SGLang) / `--enable-prompt-tokens-details` (vLLM).** Without these, `cached_tokens` is silently null/absent and cache-hit billing is wrong. See `infra-docs/docs/inference.md` → Cached Token Reporting.
- **Changing `MODEL_NAME` without updating cloud-api's model table.** inference-proxy's usage reporter requires `MODEL_NAME` to exactly match cloud-api's `model_name`. cloud-api does NOT check aliases — a mismatch causes silent 404s on usage reporting.
- **Putting a quantization suffix in `MODEL_NAME`, `--served-model-name`, or the SNI domain.** These are external identifiers — callers must not need to know which checkpoint is running. Swap the checkpoint (`--model-path`/`--revision`) freely; keep the name and domain unchanged. See [Model naming and SNI domains](#model-naming-and-sni-domains).
- **Assigning a new SNI domain when swapping checkpoints.** All configs for the same model share one domain. Introducing a second domain for the same model splits traffic and breaks callers that pinned the original.
- **Using `:dev` or `:latest` image tags.** KMS attestation registers the compose hash; a mutable tag means the running image can drift from the registered hash. Always pin `@sha256:…`.
- **`force_recreate: true` on heavy model containers.** Can wedge the docker daemon's stop path (GPU memory pressure during teardown). Use the down-then-up graceful drain pattern for GLM/DeepSeek — see `infra-docs/docs/deployment.md` → "Graceful drain for GLM".
- **Adding a scrape target without `nearai.otel.scrape: "true"`.** The validator rejects extra targets. Every target in `otelcol_app_config` must have a matching scraped service.

## Tagging

- `auto-tag.yaml` creates `v0.0.N` annotated tags on every merge to main. Tags are gated on the **commit** date by `MIN_TAG_AGE_HOURS` (default 48h) in compose-manager — a fresh commit's tag isn't deployable for 48h.
- For urgent deploys, use the `backdate-tag` skill — it backdates the commit + tag so compose-manager accepts it immediately. This is the sanctioned workaround; do not lower `MIN_TAG_AGE_HOURS` on prod.
- The skill handles the commit + annotated tag + push. Do not manually `git tag` — the skill ensures the commit date is backdated, not just the tag date.
