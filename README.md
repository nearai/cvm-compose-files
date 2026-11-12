# cvm-compose-files

Docker Compose files for the NEAR AI inference stacks that run in production, plus the recipes for the engine
images they pin.

## Layout

```
prod/                 compose files that run in production (nothing else lives here)
docker/<recipe>/      engine image recipes: Dockerfile, patches, PROVENANCE, SHA256SUMS
adapters/             LoRA adapter fetched at runtime by prod/small-models.yaml
scripts/              CI validators
docs/                 runbook for prod/rotate-compose-manager-token.yaml
.github/workflows/    CI validation, image publishing, auto-tagging
```

## Production files

| File | What it runs |
|------|--------------|
| `prod/GLM-5.3-Flash-SGL-TP2x4-W4AFP8-V12Q8.yaml` | GLM-5.3-Flash base tier, 4x SGLang TP2 |
| `prod/GLM-5.3-Flash-SGL-TP4-W4AFP8-LongContext.yaml` | GLM-5.3-Flash long-context tier, 4x SGLang TP2 |
| `prod/DeepSeek-V4.1-Flash-1017-vLLM-DEP8-Abliterated.yaml` | DeepSeek-V4.1-Flash, vLLM DEP8, base + abliterated LoRA |
| `prod/small-models.yaml` | Sage, Qwen3.6/3.8, privacy filter, embeddings, reranker, Whisper, Qwen3-VL, FLUX.2, 2x GLM base TP2 |
| `prod/gpt-oss-120b-openrouter-baremetal.yaml` | gpt-oss-120b for OpenRouter (bare metal, plain `docker compose`) |
| `prod/qwen38-27b-openrouter-baremetal.yaml` | Qwen3.8-27B for OpenRouter (bare metal, plain `docker compose`) |
| `prod/rotate-compose-manager-token.yaml` | One-shot compose-manager token rotation |

CVM files are deployed by compose-manager from a `v0.0.N` tag on `main` (tags are created on merge by
`auto-tag.yaml`). Secrets are passed as environment variables at deploy time; files only hold `${VAR}` placeholders.

## CI

`validate-compose.yaml` runs on every push and PR:

- `docker compose config` on every `prod/*.yaml`, and `otelcol validate` on each embedded collector config.
- `scripts/validate_*.rb` (Ruby 3.3): OTel labels, proxy dependencies and environment, registrar auth, token
  rotation contract, GLM-5.3 DCGM metrics.
- `scripts/validate_public_metrics.py` and `scripts/validate_streaming_keepalive.py`: nginx public-metrics isolation
  and HTTP/2 keepalive.

Run them locally the same way the workflow does, e.g.
`docker run --rm -v "$PWD:/repo:ro" -w /repo ruby:3.3-alpine ruby scripts/validate_otel_labels.rb`.

## Images

Each `publish-*.yaml` workflow is a manual `workflow_dispatch` that builds one `docker/<recipe>` at a merged `main`
commit, scans it, attests it and pushes it to Docker Hub:

| Workflow | Recipe |
|----------|--------|
| `publish-glm53-v0523.yaml` | `docker/sglang-glm53-v0523` (GLM-5.3 engines) |
| `publish-glm53-admission-reserve.yaml` | `docker/sglang-glm53-hicache-w4afp8`, `docker/sglang-glm53-admission-reserve` |
| `publish-sglang-v0523-secure.yaml` | `docker/sglang-v0523-secure` (small-models SGLang engines) |
| `publish-vllm-dsv41-lora.yaml` | `docker/vllm-dsv41-lora` (DeepSeek-V4.1) |
| `publish-vllm-dsv41-v033.yaml` | `docker/vllm-dsv41-v033` (DeepSeek-V4.1 on vLLM 0.33) |

Compose files pin images by digest. To ship a new image: merge the recipe change, run the workflow with that commit,
then pin the printed digest in a separate PR.
