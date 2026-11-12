# cvm-compose-files

Docker Compose files for confidential inference, and the build contexts of the images they pin.
Each deployable file describes one inference stack: model engines, the proxies in front of them, model
downloaders and monitoring. Hosts run a file from a release tag `v0.0.N`.

## Trust model

- This repo says what should run. It cannot prove what a host runs. The proof is the image signature and
  the host's attestation; [VERIFYING.md](VERIFYING.md) shows how to check them.
- Every image is pinned by digest. Images built here are published by a workflow run from `main` and
  signed with a keyless Sigstore identity tied to that workflow; most also get a build provenance
  attestation.
- Engines load model weights at a pinned Hugging Face commit revision, except where VERIFYING.md lists
  a gap.
- CI checks the compose files, build contexts, workflows and text in this repo. It does not inspect the
  contents of published images, and it cannot see what a host was told to deploy.
- Some live files predate these rules. Each deviation CI detects in them is a listed exception; see
  [Known gaps](VERIFYING.md#known-gaps).

## Layout

| path | contents |
|---|---|
| `prod/` | every deployable compose file, flat |
| `docker/<dir>/` | build context of one published image; never edited once published |
| `ci/` | guards (`check_*.py`, run by `run.py`), their data files, tool configs, two templates |
| `docs/` | usage docs, at most one per deployable |
| `.github/` | `ci.yaml`, `auto-tag.yaml`, one `publish-*.yaml` per image, PR template |
| `scripts/` | legacy validators, still run by CI |
| `adapters/` | legacy: a LoRA adapter some deployables fetch at runtime |
| `cleanup-hf-model.yaml` | legacy: a utility deployable outside `prod/` |

Legacy paths are listed in `ci/legacy.txt` and only shrink.

## Status

Every deployable file, the served model names it serves, and whether it is `live` or a `candidate`.

<!-- status:begin -->
| file | serves | status |
|---|---|---|
| `cleanup-hf-model.yaml` | - | live |
| `prod/DeepSeek-V4.1-Flash-1017-vLLM-DEP8-Abliterated.yaml` | deepseek-ai/DeepSeek-V4.1-Flash-1017 | live |
| `prod/GLM-5.3-Flash-SGL-TP2x4-W4AFP8-V12Q8.yaml` | z-ai/glm-5.3-flash | live |
| `prod/GLM-5.3-Flash-SGL-TP4-W4AFP8-LongContext.yaml` | z-ai/glm-5.3-flash | live |
| `prod/qwen38-27b-abliterated-cc-1gpu.yaml` | Qwen/Qwen3.8-27B | live |
| `prod/rotate-compose-manager-token.yaml` | - | live |
| `prod/sage-pull.yaml` | - | live |
| `prod/sage.yaml` | - | live |
| `prod/small-models.yaml` | Qwen/Qwen3-Embedding-0.6B,Qwen/Qwen3-Reranker-0.6B,Qwen/Qwen3-VL-30B-A3B-Instruct,Qwen/Qwen3.6-35B-A3B-FP8,Qwen/Qwen3.8-27B,black-forest-labs/FLUX.2-klein-4B,openai/whisper-large-v3,z-ai/glm-5.3-flash | live |
<!-- status:end -->

`live` runs in production. A `candidate` is being rolled out now. Files that run nowhere are deleted;
older versions stay reachable through release tags.

## Verify

[VERIFYING.md](VERIFYING.md): from a release tag and a file path to the image digest, its signature and
attestation, its build context and the model revision.

## Rules

[AGENTS.md](AGENTS.md) holds the rules for changing and reviewing this repo; `python3 ci/run.py` runs the
same guards as CI. Report vulnerabilities as described in [SECURITY.md](SECURITY.md).
