This repo cannot prove what a host runs; the image signature and attestation are the proof, the repo says what should run.

# Verifying a deployment

Start from what the host's attestation reports: a release tag `v0.0.N` and a file path. Always verify by
digest. Image tags on the registry can be moved; digests cannot.

## 1. Tag and path to file

Check that the tag's commit is on `main`, then read the file at that tag:

```sh
git fetch origin main --tags
git merge-base --is-ancestor "v0.0.N^{commit}" origin/main && echo on-main
git show v0.0.N:prod/<file>.yaml > deployed.yaml
sha256sum deployed.yaml
```

Compare the hash with the file hash in the host's attestation log (see "Compose files and attestation"
under Known gaps). The file at a tag can differ from the file at `HEAD`; verify the tagged bytes.

## 2. File to image digests

Every `image:` value ends in `@sha256:<64 hex>`. Render the file to see the final values after anchors
and overrides:

```sh
python3 ci/run.py --render prod/<file>.yaml
```

Each digest has a row in `ci/images.allow`: `repo@sha256:<digest> <tag> <build dir> <source commit>`,
with `-` for third-party images. A digest built here maps to one row of the table below.

## 3. Signature and provenance

Images built here are signed keylessly by the publish workflow that built them. The identity is the
workflow path on `refs/heads/main` (table below). Verify both the signature and the build provenance:

```sh
IMG=docker.io/nearaidev/<sglang|vllm>@sha256:<digest>
ID='https://github.com/nearai/cvm-compose-files/.github/workflows/<publish-workflow>.yaml@refs/heads/main'

cosign verify --certificate-identity "$ID" \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com "$IMG"

gh attestation verify "oci://$IMG" --repo nearai/cvm-compose-files --cert-identity "$ID"
```

A signature from any identity not in the table, or from a ref other than `refs/heads/main`, does not
count.

## 4. Image to build context

The image labels record the build recipe and the source commit:

```sh
docker buildx imagetools inspect "$IMG" --format '{{json .Image.Config.Labels}}'
```

Read `nearai.build.recipe` (the Dockerfile path) and `org.opencontainers.image.revision` (the commit).
Check out that commit and read the build dir there, not at `HEAD`; a build dir at `HEAD` can hold later,
unpublished changes:

```sh
git merge-base --is-ancestor <revision> origin/main && echo on-main
git show <revision>:docker/<dir>/Dockerfile
```

Each build dir has a `PROVENANCE` file with its base image digest and upstream source revision, and its
Dockerfile pins `FROM` by digest. Follow `FROM` digests down the chain in the table below.

## 5. Model revision

Engines load weights offline from `models--<org>--<repo>/snapshots/<40 hex>`; the model downloader in
the same file fetches the same repo at `--revision <40 hex>`. Check that the revision exists at
`https://huggingface.co/<org>/<repo>/commit/<40 hex>` and that both values match.

## Signing identities

Every workflow that has signed images for this repo. Rows are only added, never removed; a retired
workflow's existing signatures stay valid for the images it published.

| workflow | identity (prefix `https://github.com/nearai/cvm-compose-files/`) | images | state |
|---|---|---|---|
| `publish-glm53-admission-reserve.yaml` | `.github/workflows/publish-glm53-admission-reserve.yaml@refs/heads/main` | `nearaidev/sglang` | active |
| `publish-glm53-fc91d24.yaml` | `.github/workflows/publish-glm53-fc91d24.yaml@refs/heads/main` | `nearaidev/sglang` | active |
| `publish-glm53-hicache.yaml` | `.github/workflows/publish-glm53-hicache.yaml@refs/heads/main` | `nearaidev/sglang` | active |
| `publish-glm53-v0523.yaml` | `.github/workflows/publish-glm53-v0523.yaml@refs/heads/main` | `nearaidev/sglang` | active |
| `publish-sglang-v0523-secure.yaml` | `.github/workflows/publish-sglang-v0523-secure.yaml@refs/heads/main` | `nearaidev/sglang` | active |
| `publish-vllm-dsv41-lora.yaml` | `.github/workflows/publish-vllm-dsv41-lora.yaml@refs/heads/main` | `nearaidev/vllm` | active |
| `publish-vllm-dsv41-v033.yaml` | `.github/workflows/publish-vllm-dsv41-v033.yaml@refs/heads/main` | `nearaidev/vllm` | active |
| `publish-glm53-v0519.yaml` | `.github/workflows/publish-glm53-v0519.yaml@refs/heads/main` | `nearaidev/sglang` | retired |
| `publish-dsv4-0731.yaml` | `.github/workflows/publish-dsv4-0731.yaml@refs/heads/main` | `nearaidev/sglang` | retired |

`publish-glm53-admission-reserve.yaml` builds `sglang-glm53-admission-reserve`,
`sglang-glm53-hicache-admission-reserve` and `sglang-glm53-hicache-w4afp8`. Each other workflow builds
the one dir its name points to. These names predate the `publish-<build dir>.yaml` rule that new images
follow. This repo does not record which identity signed each older digest; if one in the table below
fails every identity here, treat it as unsigned.

## Digests and build dirs

Images built from the build dirs in `docker/`. "inferred" means no label or attestation ties the digest
to a source commit; the dir is matched by its `nearai.*` labels only.

| digest (`sha256:`) | tag | build dir | source commit | built `FROM` |
|---|---|---|---|---|
| `050b1adf961c8621264d14696d24645f70d1a675509ad400a72ccfb1a3fddcb2` | `glm53-v0523-near-v1` | `sglang-glm53-v0523` | `2cf337c2c5fc3fa78570a5ae0e3dd432abaefde8` | upstream SGLang |
| `6de632a5e9422be4b6e84e32cf475679c0f29c4bb6d58984d873ff9f7ba92a65` | `sglang-v0523-secure-near-v1` | `sglang-v0523-secure` | `a07dea8faf1523d85f20e328031977b7964be3e4` | upstream SGLang |
| `a744ada2a362e2548a88527432da1662aa027012226d9e0ed4fce9cda1e6f0bd` | `dsv41-lora-v0.33.0-near-v1` | `vllm-dsv41-v033` (inferred) | inferred | upstream vLLM |
| `5eaa0d4637080c10958b6cd8a371ae2ebb749b7a1e19c85bcb1abb89ee2540e4` | - | `vllm-dsv41-lora` | read the label | upstream vLLM |
| `9c6ddd4319c4ab00e351d8650459e68b8830e36ffcc029d67fa5e19d0ac3ed17` | `glm53-hicache-w4afp8-v6` | `sglang-glm53-hicache-w4afp8` | `678a6b5ee3e4e7b83340f810c76530958eecc499` | `3eccc307…` |
| `fa730e6e62b2ae8058114ce540487ade33ab93bc42b1179ae78edc92bd563fc5` | - | `sglang-glm53-hicache-w4afp8` | read the label | `3eccc307…` |
| `c15a33672ade5a9f2b4f7874a746eb4114952dae7230d5baa563a02eca081b78` | - | `sglang-glm53-hicache-w4afp8` | read the label | `3eccc307…` |
| `da8d4e4b5ec667499b58a21e56050aae9bcfc4df9669b3ac27e2e20af033d681` | `glm53-hicache-w4afp8-v12` | `sglang-glm53-hicache-w4afp8` | `79de066bbdce973b442242d44d3046791a059f2e` | `3eccc307…` |
| `3eccc30709f5719c81084d1264f03ca5354b3059ec4cff62f0c5ff88c309680c` | - | `sglang-glm53-hicache-admission-reserve` | read the label | `01f44c79…` |
| `01f44c790d43b3eae136f5f3bc930599947bd37fcea081d2dbd3271ff4fc3f8a` | - | `sglang-glm53-hicache` | read the label | `a7b7136a…` |
| `e9d29a1cb1cd65284392c4d62d5f2a36669628057e15c60fe93ea40cfe4fc7e7` | `glm53-admission-reserve-v10` | `sglang-glm53-admission-reserve` | `0244bab81cc0ec42f0aa24d5de57b804650c9cfa` | `a7b7136a…` |
| `a7b7136abcf5e07522289d96e96fec9b42a1a30f9dfda57e957f142680d2d67b` | - | `sglang-glm53-fc91d24` | read the label | upstream SGLang |

Digests pinned by a deployable at `HEAD` are also rows in `ci/images.allow`. The others are pinned at
earlier release tags or are bases of another image.

## Known gaps

CI checks compose files, build contexts, workflows and repo text. It does not check image contents, and
nothing in this repo binds a host to a file. Each gap below is a listed entry in `ci/exceptions.json`
(62 entries) unless marked "not an exception".

**Request logging.**
- 12 vLLM engine services set no request-logging flag and rely on the engine default (off in the pinned
  version); CI rejects any flag that enables it. New vLLM engines must set `--no-enable-log-requests`.
  Rule `privacy.no-log-flag`: `prod/DeepSeek-V4.1-Flash-1017-vLLM-DEP8-Abliterated.yaml`
  `model-vllm-dsv41-dp0` to `model-vllm-dsv41-dp7` (8); `prod/small-models.yaml`
  `model-vllm-qwen3vl-30b-a3b-fp8-tp1`, `model-vllm-qwen3-embedding-0.6b-tp1`,
  `model-vllm-qwen3-reranker-0.6b-tp1`, `model-vllm-whisper-large-v3-tp1` (4).
- Not an exception: engine privacy rules (logging, dump and profiling flags) apply to engine services
  only. `sage` (`prod/sage.yaml`, `prod/small-models.yaml`) and `model-privacy-filter`
  (`prod/small-models.yaml`) are marked `aux` and are not checked by them.

**Secrets.**
- 6 entries: secret variables with an empty default, so the service starts without them. Rule
  `privacy.secret`: `otelcol-contrib` in the five serving files; `qwen38-registrar` and
  `qwen38-abliterated-registrar` in `prod/qwen38-27b-abliterated-cc-1gpu.yaml`.
- 2 secrets are interpolated into an embedded proxy config. Rule `privacy.secret`: `glm53-soak-relay`
  in `prod/GLM-5.3-Flash-SGL-TP2x4-W4AFP8-V12Q8.yaml` and
  `prod/GLM-5.3-Flash-SGL-TP4-W4AFP8-LongContext.yaml`.

**Model pins.**
- 2 entries: one engine in `prod/small-models.yaml` (`model-sg-flux2-klein-4b-tp1`) loads a model
  without a pinned revision and with remote code enabled (`pins.hf-revision`, `pins.remote-code`).
- 4 entries: a LoRA adapter is fetched at runtime from this repo's `main` branch, not a pinned commit.
  The download is SHA-256-checked and fails closed (`pins.url` in
  `prod/qwen38-27b-abliterated-cc-1gpu.yaml` and `prod/small-models.yaml`; `layout.ext` and
  `layout.size` for the weights file in `adapters/`).

**Image pins.**
- 9 entries: image references that carry a tag next to their digest (`pins.digest`). The digest decides
  what runs.
- 1 entry: the utility image in `cleanup-hf-model.yaml` is pinned by tag only (`pins.digest`).
- 6 entries: two services in `prod/small-models.yaml` (`model-vllm-whisper-large-v3-tp1`,
  `model-privacy-filter`) build their image on the host from an inline Dockerfile, so no signature or
  attestation covers them (`pins.digest` 2, `pins.raw-key` 4).

**Internal references in live files.** Values CI flags in deployables that cannot change without a
functional change of the file:
- 5 entries: an internal telemetry endpoint inside the embedded collector config of each serving file
  (`public.domain`).
- 8 entries: host names inside config values, labels or service names: 1 in
  `prod/qwen38-27b-abliterated-cc-1gpu.yaml`, 6 in `prod/small-models.yaml`, 1 in
  `ci/services.allow.json` that names a service of a live file (`public.host`).
- 2 entries: private repository references inside an embedded script in `prod/small-models.yaml` (`public.repo`).

**Publishing.**
- 4 publish workflows run on a self-hosted runner (`public.runner`): `publish-glm53-admission-reserve`,
  `-fc91d24`, `-hicache`, `publish-vllm-dsv41-lora`.
- 1 publish workflow clones its engine source from a personal fork (`public.repo`,
  `publish-glm53-fc91d24.yaml`).
- Not an exception: `publish-vllm-dsv41-lora` and `publish-vllm-dsv41-v033` skip the provenance
  attestation when resuming from an already pushed digest; such an image has a signature only.
- Not an exception: a `v*` tag ruleset blocks moving and deleting tags, but maintainers can create
  tags. Check that a tag's commit is on `main` (step 1).
- Not an exception: build dirs that existed before the CI guards are not scanned by the public-text
  rules and may hold internal references.

**Compose files and attestation.** Not an exception. The host's measured configuration covers its base
services, not the model compose file. A deployed model file appears in the host's attestation log as an
action record with the tag, commit, file path and file hash; it is not part of the measurement. A
verifier must compare that record with step 1.
