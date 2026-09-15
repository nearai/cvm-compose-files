# DeepSeek-V4-Flash EAGLE prefill-fairness canary image

This directory reproduces the SGLang image selected by the 2026-07-28 GPU02
DeepSeek-V4-Flash optimization investigation. It applies one guarded scheduler
change to the immutable SGLang v0.5.16 base image. The behavior remains off
unless `SGLANG_EAGLE_PREFILL_FAIRNESS=1` is set.

## Provenance

- Base image:
  `lmsysorg/sglang@sha256:984699c298a95b73c469b2191403ddc85fd780506e13c39c4afff3845e27bc6c`
- SGLang source:
  `fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1` (`v0.5.16`)
- Patch SHA-256:
  `1f8e133308716f84d3301cb2636c217d0bddef7f85527b9cda0d06760235fcb4`
- Model revision used in testing:
  `553034d7dd9e06c2eeaee68cf85a17d6d4754cf0`

The GPU02 test image ID was
`sha256:438d86dc105a953f126d55ef2820bf20ff7232f550ab74976a24979deeb213c3`.
That value is a removed local image ID, not a registry digest, and must never
be placed in a production compose file.

## Build and promotion gate

Build the image locally from this directory:

```bash
docker build \
  --pull \
  --tag ds4f-eagle-prefill-fairness:v0.5.16-fdebc938 \
  docker/sglang-dsv4-eagle-fairness
```

Scan it, publish it through the approved registry workflow, and record all of
the following before changing a production compose file:

- immutable `repository@sha256:...` reference;
- source revision and patch checksum above;
- image scan result and SBOM;
- a successful `python3 -m py_compile` layer from this Dockerfile;
- semantic DS4F readiness and stream-completion checks on H200.

Do not use a mutable tag in production. This repository intentionally does not
include registry credentials or an automatic push workflow.

## One-replica canary delta

After an immutable registry digest exists, override only
`model-sg-dsv4-flash-fp4-tp2-r1` in
`prod/qwen35-dsv4-flash.yaml`. Keep replica 2 on the current production image
and command as the matched control.

Use:

```text
SGLANG_EAGLE_PREFILL_FAIRNESS=1
--speculative-num-steps 2
--speculative-eagle-topk 1
--speculative-num-draft-tokens 3
--max-prefill-tokens 16384
```

Retain the current TP2, Marlin, chunk 8192, running 64, queued 16,
schedule-conservativeness 1.3, CUDA graph 64, and memory fraction 0.83
settings. Do not apply these changes to the shared `x-dsv4-flash-common`
anchor: doing so would roll both replicas at once.

The exact rollback image for replica 1 is:

```text
lmsysorg/sglang@sha256:6bb5fee34b6c4537c09a4775e2292ac40350d5ad1218fcc835b2692142f443b1
```

Restore EAGLE 3 steps / 4 draft tokens and remove
`SGLANG_EAGLE_PREFILL_FAIRNESS` during rollback.
