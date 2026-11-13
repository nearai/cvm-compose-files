# cvm-compose-files

Docker Compose files for confidential inference, and the build contexts of the images they pin.
Each deployable file describes one inference stack: model engines, the proxies in front of them, model
downloaders and monitoring. Hosts run a file from a release tag `v0.0.N`.

## Trust model

- This repo says what should run. It cannot prove what a host runs. The proof is the image signature and
  the host's attestation; [docs/verifying.md](docs/verifying.md) shows how to check them. The compose file itself
  is not part of the host's hardware measurement.
- Images are pinned by digest, except as listed in [Known gaps](docs/verifying.md#known-gaps). Images built
  here are published by publish workflows on `main` and signed with a keyless Sigstore identity; for
  older digests the repo does not record which identity signed them. Most also get a build provenance
  attestation.
- Privacy here means the serving containers must not log prompts or outputs. CI checks this through the
  flags and env of SGLang and vLLM services only; docs/verifying.md says exactly what is covered.
- Engines load model weights at a pinned Hugging Face commit revision, except where docs/verifying.md lists
  a gap.
- CI checks the compose files, build contexts, workflows and text in this repo. It does not inspect the
  contents of published images, and it cannot see what a host was told to deploy.
- Some live files predate these rules. Each deviation CI detects in them is a listed exception; see
  [Known gaps](docs/verifying.md#known-gaps).

## Layout

| path | contents |
|---|---|
| `prod/` | every deployable compose file, flat |
| `docker/<dir>/` | build context of one published image; its build inputs are frozen once merged (read it at the image's source commit) |
| `ci/` | the gate: `run.py`, guards in `checks/`, `lib/`, tool configs in `config/`, two templates in `templates/` |
| `ci/data/` | allowlists, waivers, the legacy list and expected findings; they change with what they record |
| `docs/` | docs for outside readers: how to verify a deployment |
| `.github/` | `ci.yaml`, `auto-tag.yaml`, one `publish-*.yaml` per image, PR template |
| `adapters/` | legacy: a LoRA adapter some deployables fetch at runtime |

Legacy paths are listed in `ci/data/legacy.txt` and only shrink.

## Names

| name | form | example |
|---|---|---|
| serving file | `prod/<checkpoint>-<engine>-<topology>[-<tier>].yaml` | `prod/GLM-5.3-Flash-SGL-TP2x4-W4AFP8-V12Q8.yaml` |
| other deployable | `prod/pack-<purpose>.yaml`, `prod/util-<verb>-<object>.yaml` | `prod/rotate-compose-manager-token.yaml` |
| build dir | `docker/<engine>-v<upstream>-<scope>-r<N>/` | `docker/sglang-v0.5.23-glm53-r1/` |
| image tag | build dir without `<engine>-` | `v0.5.23-glm53-r1` |
| publish workflow | `.github/workflows/publish-<build dir>.yaml` | `publish-sglang-v0.5.23-glm53-r1.yaml` |
| served name | the model's Hugging Face `org/repo` ID | `z-ai/glm-5.3-flash` |

`topology`: `tp4` one TP4 engine, `tp2x4` four TP2 engines, `dep8` one engine over 8 data-parallel
ranks with expert parallelism. Files and dirs that predate these forms keep their names; files are
marked `(legacy name)` in the status table. A config revision is a release tag plus a file path.

## Status

One row per served model name and engine configuration of each deployable; files without an engine
get one row. `python3 ci/run.py --write-status` generates every column except `status`, and CI
fails if a row differs from the rendered file. `model@revision` is the loaded checkpoint and the
first 12 hex of its pinned revision (`-`: not pinned). `engine image` is the build dir the pinned
digest comes from, a third-party image, or an image built on the host. `topology` follows the
file-name grammar in `.claude/rules/prod.md`. `privacy`: `flag set` (explicit no-log flag),
`waived` (listed in Known gaps), `n.a.` (no engine). `(legacy name)`: listed in `ci/data/legacy.txt`.

<!-- status:begin -->
| file | serves | model@revision | engine image | topology | privacy | status |
|---|---|---|---|---|---|---|
| `prod/DeepSeek-V4.1-Flash-1017-vLLM-DEP8-Abliterated.yaml` (legacy name) | deepseek-ai/DeepSeek-V4.1-Flash-1017 | deepseek-ai/DeepSeek-V4.1-Flash-1017@1bae0e108899 | docker/vllm-dsv41-v033 | dep8 | waived | live |
| `prod/GLM-5.3-Flash-SGL-TP2x4-W4AFP8-V12Q8.yaml` (legacy name) | z-ai/glm-5.3-flash | graphistry/GLM-5.3-Flash-W4AFP8@99f1fa70408c | docker/sglang-glm53-v0523 | tp2x4 | flag set | live |
| `prod/GLM-5.3-Flash-SGL-TP4-W4AFP8-LongContext.yaml` (legacy name) | z-ai/glm-5.3-flash | graphistry/GLM-5.3-Flash-W4AFP8@99f1fa70408c | docker/sglang-glm53-hicache-w4afp8 | tp4x2 | flag set | live |
| `prod/GLM-5.3-Flash-SGL-TP4-W4AFP8-LongContext.yaml` (legacy name) | z-ai/glm-5.3-flash | graphistry/GLM-5.3-Flash-W4AFP8@99f1fa70408c | docker/sglang-glm53-v0523 | tp2x4 | flag set | live |
| `prod/pack-small-models.yaml` | Qwen/Qwen3-Embedding-0.6B | Qwen/Qwen3-Embedding-0.6B@97b0c614be4d | third-party vllm/vllm-openai | tp1 | flag set | candidate |
| `prod/pack-small-models.yaml` | Qwen/Qwen3-Reranker-0.6B | Qwen/Qwen3-Reranker-0.6B@e61197ed4502 | third-party vllm/vllm-openai | tp1 | flag set | candidate |
| `prod/pack-small-models.yaml` | Qwen/Qwen3-VL-30B-A3B-Instruct | Qwen/Qwen3-VL-30B-A3B-Instruct-FP8@d9748a51ae66 | third-party vllm/vllm-openai | tp1 | flag set | candidate |
| `prod/pack-small-models.yaml` | Qwen/Qwen3.6-35B-A3B-FP8 | Qwen/Qwen3.6-35B-A3B-FP8@95a723d08a94 | docker/sglang-v0523-secure | tp1 | flag set | candidate |
| `prod/pack-small-models.yaml` | Qwen/Qwen3.8-27B | Qwen/Qwen3.8-27B-FP8@017b9c7af6b5 | docker/sglang-v0523-secure | tp1 | flag set | candidate |
| `prod/pack-small-models.yaml` | black-forest-labs/FLUX.2-klein-4B | black-forest-labs/FLUX.2-klein-4B@- | docker/sglang-v0523-secure | tp1 | flag set | candidate |
| `prod/pack-small-models.yaml` | openai/whisper-large-v3 | openai/whisper-large-v3@06f233fe06e7 | built on host | tp1 | flag set | candidate |
| `prod/pack-small-models.yaml` | z-ai/glm-5.3-flash | graphistry/GLM-5.3-Flash-W4AFP8@99f1fa70408c | docker/sglang-glm53-v0523 | tp2x2 | flag set | candidate |
| `prod/qwen38-27b-abliterated-cc-1gpu.yaml` (legacy name) | Qwen/Qwen3.8-27B | Qwen/Qwen3.8-27B-FP8@017b9c7af6b5 | third-party lmsysorg/sglang | tp1 | flag set | live |
| `prod/rotate-compose-manager-token.yaml` (legacy name) | - | - | - | - | n.a. | live |
| `prod/sage-pull.yaml` (legacy name) | - | - | - | - | n.a. | live |
| `prod/sage.yaml` (legacy name) | - | - | - | - | n.a. | live |
| `prod/small-models.yaml` (legacy name) | Qwen/Qwen3-Embedding-0.6B | Qwen/Qwen3-Embedding-0.6B@97b0c614be4d | third-party vllm/vllm-openai | tp1 | waived | live |
| `prod/small-models.yaml` (legacy name) | Qwen/Qwen3-Reranker-0.6B | Qwen/Qwen3-Reranker-0.6B@e61197ed4502 | third-party vllm/vllm-openai | tp1 | waived | live |
| `prod/small-models.yaml` (legacy name) | Qwen/Qwen3-VL-30B-A3B-Instruct | Qwen/Qwen3-VL-30B-A3B-Instruct-FP8@d9748a51ae66 | third-party vllm/vllm-openai | tp1 | waived | live |
| `prod/small-models.yaml` (legacy name) | Qwen/Qwen3.6-35B-A3B-FP8 | Qwen/Qwen3.6-35B-A3B-FP8@95a723d08a94 | docker/sglang-v0523-secure | tp1 | flag set | live |
| `prod/small-models.yaml` (legacy name) | Qwen/Qwen3.8-27B | Qwen/Qwen3.8-27B-FP8@017b9c7af6b5 | docker/sglang-v0523-secure | tp1 | flag set | live |
| `prod/small-models.yaml` (legacy name) | black-forest-labs/FLUX.2-klein-4B | black-forest-labs/FLUX.2-klein-4B@- | docker/sglang-v0523-secure | tp1 | flag set | live |
| `prod/small-models.yaml` (legacy name) | openai/whisper-large-v3 | openai/whisper-large-v3@06f233fe06e7 | built on host | tp1 | waived | live |
| `prod/small-models.yaml` (legacy name) | z-ai/glm-5.3-flash | graphistry/GLM-5.3-Flash-W4AFP8@99f1fa70408c | docker/sglang-glm53-v0523 | tp2x2 | flag set | live |
<!-- status:end -->

`live` runs in production. Files that run nowhere are deleted; older versions stay reachable through
release tags.

Models served by more than one live file:

- `z-ai/glm-5.3-flash`: `GLM-5.3-Flash-SGL-TP2x4-W4AFP8-V12Q8.yaml` runs four TP2 engines with an FP8 KV
  cache. `GLM-5.3-Flash-SGL-TP4-W4AFP8-LongContext.yaml` runs four TP2 engines with smaller batches and two
  TP4 engines on the hicache image with a bfloat16 KV cache and hierarchical cache. `small-models.yaml`
  runs two TP2 engines with the V12Q8 settings next to other models.
- `Qwen/Qwen3.8-27B`: `qwen38-27b-abliterated-cc-1gpu.yaml` runs it alone on one GPU on a third-party
  SGLang image; `small-models.yaml` runs it on the `sglang-v0523-secure` image with half the GPU memory
  and batch limits, next to other models. Both load the same checkpoint and LoRA adapter.

## Verify

[docs/verifying.md](docs/verifying.md): from a release tag and a file path to the image digest, its signature and
attestation, its build context and the model revision.

## Rules

[AGENTS.md](AGENTS.md) holds the rules for changing and reviewing this repo; `python3 ci/run.py` runs the
same guards as CI. Report vulnerabilities as described in [SECURITY.md](.github/SECURITY.md).
