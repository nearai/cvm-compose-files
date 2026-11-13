---
paths:
  - "prod/*.yaml"
  - "ci/templates/*.yaml"
---

# prod/ compose files

Names, labels and lifecycle of deployables. The hard rules in AGENTS.md apply on top of this file.

## Names
CI derives the name from the rendered file (`ci/checks/names.py`); paths in `ci/data/legacy.txt` are exempt.
- New compose file: copy a file in `ci/templates/`; keep its line-1 header form, key and service order.
- One served model on one engine kind: `prod/<served>-<engine>-<topology>[-<tier>].yaml`.
  - `served` = the repo part of the served name, lowercased (`z-ai/glm-5.3-flash` -> `glm-5.3-flash`).
  - `engine` = `sglang` | `vllm`.
  - `topology` = one token per engine group, `tp<N>` | `dep<N>` | `dp<N>`, `x<R>` for R replicas,
    joined by `_` in ascending size (`tp2x4_tp4x2`). `tp4` one TP4 engine; `tp2x4` four TP2 engines;
    `dep8` one engine over 8 data-parallel ranks with expert parallelism.
  - `tier` only when two files would otherwise share a name; then every such file has a distinct one.
- Several served models or engine kinds: `prod/pack-<purpose>.yaml`. No engine: `prod/util-<verb>-<object>.yaml`.
- Every `deployment` label and `deployment:` log tag equals the file stem.
- Serving and pack files set no top-level `name:`; utility files set `name:` to the file stem.
- Service keys: `model-sg-*`, `model-vllm-*`, `proxy-*`, `model-proxy-registrar*`, `model-downloader*`,
  `nginx*`, `dcgm-*`, `sidecar-*`, `otelcol-contrib`.
- Engine labels: `served_model`, `model_path`, `deployment` (= file stem), `engine_image`, `instance`,
  `config_variant`; change `config_variant` when image, command or env changes.
- Model pin: the engine loads `models--<org>--<repo>/snapshots/<40hex>` offline; the downloader
  fetches the same repo at the same `--revision`. Env names `UPPER_SNAKE`.
- Comment wording follows [docs.md](docs.md).

## Lifecycle
- Status is `live` or `candidate`, only in the README table, never in a compose file.
- `live` runs in production. A `candidate` is being rolled out now. It has the same service keys as
  the live file it replaces (same served-name set), or an announced change with an exception entry.
- Everything else not in production is deleted. Rollback = redeploy an older tag.
- Never rename a deployed file in place: its labels and project name would change. A functional change
  (the rendered config differs) to a legacy file ships as a new standard-named candidate, except a fix
  that closes a waiver or known gap in docs/verifying.md, moves a literal to a deploy-time variable, or
  rewrites a script without changing what it does (shown in the PR). Comment- or doc-only edits stay.
- Moving a deployed file takes two PRs: add the new file as candidate; the promotion PR deletes the
  old file, its table row, its legacy line and its doc.
- Before deleting a file a host may run, check the deployment manager and say so in the PR.
