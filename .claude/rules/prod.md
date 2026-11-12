---
paths:
  - "prod/*.yaml"
  - "ci/templates/*.yaml"
---

# prod/ compose files

Names, labels and lifecycle of deployables. The hard rules in AGENTS.md apply on top of this file.

## Names
- New compose file: copy a file in `ci/templates/`; keep its line-1 header form, key and service order.
- Serving file: `prod/<checkpoint>-<engine>-<topology>[-<tier>].yaml`. `checkpoint` = lowercased HF repo
  name; `engine` = `sglang` | `vllm`; `topology` = `tp<N>` | `dep<N>` | `dp<N>`, `x<R>` for replicas,
  `_` between engine groups; `tier` only to tell two files apart (`base`, `long`). Examples: `tp4` one
  TP4 engine; `tp2x4` four TP2 engines; `dep8` one engine over 8 data-parallel ranks with expert
  parallelism; `tp2x4_tp4x2` four TP2 and two TP4 engines in one file.
- Pack: `prod/pack-<purpose>.yaml`. Utility or any non-serving project: `prod/util-<verb>-<object>.yaml`.
- Serving and pack files set no top-level `name:`; utility files set `name:` to the file stem.
- Service keys: `model-sg-*`, `model-vllm-*`, `proxy-*`, `model-proxy-registrar*`, `model-downloader*`,
  `nginx*`, `dcgm-*`, `sidecar-*`, `otelcol-contrib`, `hf-cleanup`.
- Engine labels: `served_model`, `model_path`, `deployment` (= file stem), `engine_image`, `instance`,
  `config_variant`; change `config_variant` when image, command or env changes.
- Model pin: the engine loads `models--<org>--<repo>/snapshots/<40hex>` offline; the downloader
  fetches the same repo at the same `--revision`. Env names `UPPER_SNAKE`.

## Lifecycle
- Status is `live` or `candidate`, only in the README table, never in a compose file.
- `live` runs in production. A `candidate` is being rolled out now. It has the same service keys as
  the live file it replaces (same served-name set), or an announced change with an exception entry.
- Everything else not in production is deleted. Rollback = redeploy an older tag.
- Never rename a legacy path in place. A functional change (the rendered config differs) to a legacy
  file ships as a new standard-named candidate. Comment- or doc-only edits stay in place.
- Moving a deployed file takes two PRs: add the new file as candidate; the promotion PR deletes the
  old file, its table row, its legacy line and its doc.
- Before deleting a file a host may run, check the deployment manager and say so in the PR.
