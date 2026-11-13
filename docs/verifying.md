# Verifying a deployment

This repo cannot prove what a host runs. It says what should run; the image signatures and the host's
attestation are the proof.

## What the proof covers

- **Deployment record.** The compose file is not in the hardware measurement. The host's deployment
  manager, [compose-manager](https://github.com/nearai/compose-manager), records each file it applies:
  release tag, commit, file path and SHA-256 of the file bytes at that tag. Its
  `GET /v1/attestation/report` returns that record with a TDX quote whose report data commits to the
  record's hash and a nonce the caller may choose. The record is as trustworthy as the measured
  manager. Environment values supplied at deploy time are not part of it. The endpoint is served by
  each host's manager; this repo does not publish host addresses.
- **Images** are pinned by digest, except the cases counted under Known gaps. An image built here can
  be checked against its signature, its provenance and its build dir at the recorded source commit.
- **Privacy** here means one thing: the serving containers must not write prompts or outputs to their
  logs. It matters because the telemetry collector in each serving file ships container logs off the
  host. CI enforces it only through the flags and env of services it classifies as SGLang or vLLM (by
  service key, image name, command or `role: engine`), `aux` services included. It does not cover
  services on other images (nginx, the collector, GPU exporters, downloaders, registrars, `sage`,
  `model-privacy-filter`), and it cannot see what any image logs by default. The published
  `sglang-v0523-secure` and `sglang-glm53-v0523` images also carry log-redaction patches, which CI does
  not check: [14-admin-auth-redaction](../docker/sglang-v0523-secure/patches/14-admin-auth-redaction.diff)
  and [15-customer-data-log-redaction](../docker/sglang-v0523-secure/patches/15-customer-data-log-redaction.diff).
  Patch 26 in those dirs and the redaction patches of the former `sglang-glm53-hicache-w4afp8` dir are
  not in any published image; the `SERIES` or Dockerfile at an image's source commit records what it contains.

Start from the deployment record: a release tag `v0.0.N`, a file path and the file's SHA-256. Always
verify by digest. Image tags on the registry can be moved; digests cannot.

## 1. Tag and path to file

Check that the tag's commit is on `main`, then read the file at that tag:

```sh
git fetch origin main --tags
git merge-base --is-ancestor "v0.0.N^{commit}" origin/main && echo on-main
git show v0.0.N:prod/<file>.yaml > deployed.yaml
sha256sum deployed.yaml
```

Compare the hash with `file_sha256` in the deployment record. The file at a tag can differ from the
file at `HEAD`; verify the tagged bytes.

## 2. File to image digests

Render the file to see the final `image:` values after anchors and overrides:

```sh
python3 ci/run.py --render prod/<file>.yaml
```

Each digest has a row in `ci/data/images.allow`:
`repo@sha256:<digest> <tag> <build dir> <source commit>`, with `-` for third-party images. A digest
built here also has a row in the Digests and build dirs table.

## 3. Signature and provenance

Images built here are signed keylessly by a publish workflow. The identity is the workflow path on
`refs/heads/main` (Signing identities table). Verify the signature and, where one exists, the build
provenance attestation:

```sh
IMG=docker.io/nearaidev/<sglang|vllm>@sha256:<digest>
ID='https://github.com/nearai/cvm-compose-files/.github/workflows/<publish-workflow>.yaml@refs/heads/main'

cosign verify --certificate-identity "$ID" \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com "$IMG"

gh attestation verify "oci://$IMG" --repo nearai/cvm-compose-files --cert-identity "$ID"
```

A signature from any identity not in the Signing identities table, or from a ref other than
`refs/heads/main`, does not count.

## 4. Image to build context

Read the source commit and build dir from the BuildKit provenance stored in the image index under the
pinned digest. This works for SGLang and vLLM images alike; the vLLM images carry no
`org.opencontainers.image.revision` label, so do not rely on labels.

```sh
docker buildx imagetools inspect "$IMG" --format '{{json .Provenance.SLSA}}' \
  | jq '.metadata["https://mobyproject.org/buildkit@v1#metadata"].vcs'
```

`revision` is the source commit and `localdir:context` the build dir. Read the build dir at that
commit, not at `HEAD`; several images were built from one dir at different commits:

```sh
git merge-base --is-ancestor <revision> origin/main && echo on-main
git show <revision>:docker/<dir>/Dockerfile
```

Each build dir has a `PROVENANCE` file with its base image digest and upstream source revision, and its
Dockerfile pins `FROM` by digest. Follow `FROM` digests down the Digests and build dirs table.

## 5. Model revision

Engines load weights offline from `models--<org>--<repo>/snapshots/<40 hex>`; the model downloader in
the same file fetches the same repo at `--revision <40 hex>`. Check that the revision exists at
`https://huggingface.co/<org>/<repo>/commit/<40 hex>` and that both values match.

## Signing identities

Every workflow that has signed images for this repo. Rows are only added, never removed; a retired
workflow's existing signatures stay valid for the images it published.

| workflow | identity (prefix `https://github.com/nearai/cvm-compose-files/`) | images | state |
|---|---|---|---|
| `publish-glm53-v0523.yaml` | `.github/workflows/publish-glm53-v0523.yaml@refs/heads/main` | `nearaidev/sglang` | active |
| `publish-sglang-v0523-secure.yaml` | `.github/workflows/publish-sglang-v0523-secure.yaml@refs/heads/main` | `nearaidev/sglang` | active |
| `publish-vllm-dsv41-v033.yaml` | `.github/workflows/publish-vllm-dsv41-v033.yaml@refs/heads/main` | `nearaidev/vllm` | active |
| `publish-glm53-admission-reserve.yaml` | `.github/workflows/publish-glm53-admission-reserve.yaml@refs/heads/main` | `nearaidev/sglang` | retired |
| `publish-glm53-fc91d24.yaml` | `.github/workflows/publish-glm53-fc91d24.yaml@refs/heads/main` | `nearaidev/sglang` | retired |
| `publish-glm53-hicache.yaml` | `.github/workflows/publish-glm53-hicache.yaml@refs/heads/main` | `nearaidev/sglang` | retired |
| `publish-vllm-dsv41-lora.yaml` | `.github/workflows/publish-vllm-dsv41-lora.yaml@refs/heads/main` | `nearaidev/vllm` | retired |
| `publish-glm53-v0519.yaml` | `.github/workflows/publish-glm53-v0519.yaml@refs/heads/main` | `nearaidev/sglang` | retired |
| `publish-dsv4-0731.yaml` | `.github/workflows/publish-dsv4-0731.yaml@refs/heads/main` | `nearaidev/sglang` | retired |

`publish-glm53-admission-reserve.yaml` built `sglang-glm53-admission-reserve`,
`sglang-glm53-hicache-admission-reserve`, `sglang-glm53-hicache-w4afp8` and three earlier dirs whose
images have no row here. Each other workflow builds
the one dir its name points to. These names predate the `publish-<build dir>.yaml` rule that new images
follow. This repo does not record which identity signed each older digest; if a digest in the Digests
and build dirs table fails every identity in this table, treat it as unsigned.

## Digests and build dirs

Images built from the build dirs in `docker/`. The source commit and build dir come from each image's
BuildKit provenance (step 4); every source commit is on `main`. A short digest in `built FROM` is the
prefix of another row of this table. A build dir that is no longer in `docker/` at `HEAD` is read at its
source commit, like every other.
Rows with tag `-` are base images of other rows. If one is removed from the registry, the chain above it
can no longer be followed.

| digest (`sha256:`) | tag | build dir | source commit | built `FROM` |
|---|---|---|---|---|
| `050b1adf961c8621264d14696d24645f70d1a675509ad400a72ccfb1a3fddcb2` | `glm53-v0523-near-v1` | `sglang-glm53-v0523` | `2cf337c2c5fc3fa78570a5ae0e3dd432abaefde8` | upstream SGLang |
| `6de632a5e9422be4b6e84e32cf475679c0f29c4bb6d58984d873ff9f7ba92a65` | `sglang-v0523-secure-near-v1` | `sglang-v0523-secure` | `a07dea8faf1523d85f20e328031977b7964be3e4` | upstream SGLang |
| `a744ada2a362e2548a88527432da1662aa027012226d9e0ed4fce9cda1e6f0bd` | `dsv41-lora-v0.33.0-near-v1` | `vllm-dsv41-v033` | `7469ed1c47d216ceee4a5329515e6fa0901d90df` | upstream vLLM |
| `5eaa0d4637080c10958b6cd8a371ae2ebb749b7a1e19c85bcb1abb89ee2540e4` | - | `vllm-dsv41-lora` | `c87715224cc5712f5f302ac0e486f533c99ce5c5` | upstream vLLM |
| `9c6ddd4319c4ab00e351d8650459e68b8830e36ffcc029d67fa5e19d0ac3ed17` | `glm53-hicache-w4afp8-v6` | `sglang-glm53-hicache-w4afp8` | `678a6b5ee3e4e7b83340f810c76530958eecc499` | `3eccc307…` |
| `fa730e6e62b2ae8058114ce540487ade33ab93bc42b1179ae78edc92bd563fc5` | - | `sglang-glm53-hicache-w4afp8` | `29a7db96822caedff38c571dfef8942110702a23` | `3eccc307…` |
| `c15a33672ade5a9f2b4f7874a746eb4114952dae7230d5baa563a02eca081b78` | - | `sglang-glm53-hicache-w4afp8` | `913d6995be03562fbad6ef9c7f498343cfbaf449` | `3eccc307…` |
| `da8d4e4b5ec667499b58a21e56050aae9bcfc4df9669b3ac27e2e20af033d681` | `glm53-hicache-w4afp8-v12` | `sglang-glm53-hicache-w4afp8` | `79de066bbdce973b442242d44d3046791a059f2e` | `3eccc307…` |
| `3eccc30709f5719c81084d1264f03ca5354b3059ec4cff62f0c5ff88c309680c` | - | `sglang-glm53-hicache-admission-reserve` | `0244bab81cc0ec42f0aa24d5de57b804650c9cfa` | `01f44c79…` |
| `01f44c790d43b3eae136f5f3bc930599947bd37fcea081d2dbd3271ff4fc3f8a` | - | `sglang-glm53-hicache` | `74a1fb19aa5763eab4084ebb8a755a15691fddd0` | `a7b7136a…` |
| `e9d29a1cb1cd65284392c4d62d5f2a36669628057e15c60fe93ea40cfe4fc7e7` | `glm53-admission-reserve-v10` | `sglang-glm53-admission-reserve` | `0244bab81cc0ec42f0aa24d5de57b804650c9cfa` | `a7b7136a…` |
| `a7b7136abcf5e07522289d96e96fec9b42a1a30f9dfda57e957f142680d2d67b` | - | `sglang-glm53-fc91d24` | `ea48a48b3bfcaa57099252a19143efc414adff5b` | upstream SGLang |

Digests pinned by a deployable at `HEAD` are also rows in `ci/data/images.allow`. The others are pinned
at earlier release tags or are bases of another image.

## Known gaps

Known gaps, highest impact first:

1. `privacy.no-log-flag`: some vLLM engines set no explicit no-log flag.
2. `pins.raw-key`, `pins.digest`: some images are built on the host or pinned by tag.
3. `pins.hf-revision`, `pins.remote-code`: some models load without a pinned revision.
4. `layout.ext`, `layout.size`: `adapters/` holds a LoRA adapter that earlier release tags fetch from
   `main`; `HEAD` fetches it at a pinned commit.
5. `privacy.secret`: some secrets have empty defaults or sit inside embedded configs.
6. `public.*`: some config values, labels and workflows hold internal names.
7. The GPU a Sage service runs on is the deploy-time value `SAGE_GPU_UUID`, so the deployment record
   does not show it. The file fails to start without it.

CI checks compose files, build contexts, workflows and repo text. It does not check image contents, and
nothing in this repo binds a host to a file. Deviations CI detects in live files are entries in
`ci/data/exceptions.json`; each binds one rule, one file and the matched value. Counts per rule:

| rule | entries | what it waives |
|---|---|---|
| `privacy.no-log-flag` | 12 | vLLM engines that set no request-logging flag and rely on the engine default |
| `privacy.secret` | 9 | secret variables with an empty default, or used inside an embedded config |
| `pins.digest` | 12 | image references with a tag next to the digest, pinned by tag only, or built on the host |
| `pins.raw-key` | 8 | `build` or `dockerfile_inline` in a deployable (images built on the host, unsigned) |
| `pins.hf-revision` | 1 | a model loaded without a pinned revision |
| `pins.remote-code` | 1 | remote code enabled for that model |
| `layout.ext`, `layout.size` | 1 each | the adapter weights file in `adapters/` |
| `public.domain` | 6 | an internal telemetry endpoint |
| `public.host` | 27 | host names in config values, labels or service names |
| `public.repo` | 2 | references to repositories outside the public allowlist |
| `status.key-set` | 1 | a candidate with model downloaders its live file does not have |

Gaps that are not exceptions:
- The logging rules check flags and env only, on services the guards classify as SGLang or vLLM by
  key, image, command or role (see What the proof covers). Only engines must set an explicit no-log
  flag; other services on those images, such as every `role: aux` entry in
  `ci/data/services.allow.json`, are checked for flags and env that enable logging.
- actionlint and zizmor check a publish workflow only until it merges. Publish workflows that exist
  at the base commit are frozen and not linted, so any findings in them stay until the workflow is
  deleted with the last build dir it publishes. The three current ones check out the repository
  without `persist-credentials: false` (zizmor `artipacked`).
- Some pinned images are neither standard upstream images nor built here: `nearaidev/vllm-proxy-rs`
  and `nearaidev/compose-manager-launcher` are built outside this repo, and the image of
  `prod/sage.yaml` and `prod/sage-pull.yaml` comes from a vendor. This repo has no build record or
  signing identity for them; only their digests are pinned. The Whisper engine in
  `prod/small-models.yaml` and `prod/pack-small-models.yaml` is built on the host (`pins.raw-key`) and
  has no digest at all.
- The nginx checks read `location` blocks and explicit `proxy_pass`, `rewrite` and `return` targets on
  TLS inference listeners. They do not follow a `proxy_pass` that maps one prefix onto another or an
  `upstream` block; review reads those.
- `publish-vllm-dsv41-v033.yaml` skips the provenance attestation when it resumes from an already
  pushed digest; such an image has a signature only.
- Repository settings applied at merge: a `v*` tag ruleset blocks moving and deleting tags, but
  maintainers can create tags, so check that a tag's commit is on `main` (step 1).
- Build inputs of build dirs that existed before the CI guards are frozen, not scanned by the
  public-text rules, and hold internal references (host names, internal domains, repo names). Docs
  in those dirs that no build reads are scanned.
