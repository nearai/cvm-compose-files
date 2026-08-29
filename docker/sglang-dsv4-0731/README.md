# DeepSeek-V4-Flash-0731 nightly derivative

This directory builds the SGLang runtime candidate for the recurring DS4F TP
rank desynchronization. It moves the production derivative from SGLang v0.5.16
to the exact post-#33587 nightly validated as one arm of the GPU31/GPU32 test
matrix, while retaining the small downstream behaviors already required by the
gpu30 workload.

It deliberately keeps the stable public model name
`deepseek-ai/DeepSeek-V4-Flash`. The 0731 checkpoint is an implementation
detail and must not change `MODEL_NAME`, `--served-model-name`, or the
`dsv4-flash.completions.near.ai` SNI domain.

This build-context change is the first half of the release. The derivative must
be merged, published, and qualified before a separate deployment PR can pin its
registry digest in `prod/glm53-flash-dsv4-flash.yaml` for the gpu30 canary.

## Provenance

- Base image tag: `lmsysorg/sglang:nightly-dev-20260818-c0b6474b`
- Base image digest:
  `sha256:51e576f02368480c055c7aadb67590d82b172e2392123ce4cf4cc8251b2d8caf`
- SGLang source: `c0b6474b43363c2f4bc60fe3d7817d393fb51d32`
- WAR-fence fix: upstream PR
  [#33587](https://github.com/sgl-project/sglang/pull/33587), merge commit
  `717a559f02b3ad85ba4bb4623772a1672e9e3e9c`
- Bounded chunked-prefill admission patch SHA-256:
  `3f7111c472b583b0921246b38d54d09ab6bb41090efe8a6301b750612b98fc89`
- Pre-first-chunk abort patch SHA-256:
  `28bc45ba377e3616d7221a609e584272c765f4fc3e69e0620d1ed0003bd608e4`
- Model revision: `7872f01b1d1fe23eabc4c98b48bffcef5a386062`

The nightly already contains the official DSV4 reasoning-effort profile, the
earlier pre-stream admission work, and PR #33587. Those changes are no longer
carried as downstream backports.

## Retained downstream behavior

The official nightly is not a direct production replacement. This derivative
retains:

1. `SGLANG_CHUNKED_PREFILL_ADMISSION_RESERVE=4096`, which leaves an aligned
   slice of an 8192-token long-prefill chunk for a complete queued request;
2. pre-first-chunk scheduler abort propagation, so an overload or scheduler
   rejection can return its HTTP status before committing an SSE 200 response;
3. removal of the two unused Nsight EFA `nic_sampler` helpers covered by the
   existing `CVE-2025-68121` image-hardening control.

The gpu30 deployment candidate should also set:

```text
SGLANG_DSV4_COMPRESS_STATE_DTYPE=bf16
```

The matched H200 A/B increased the TP4 KV pool from 10.47M to 11.53M tokens
(10%) without an observed latency, throughput, or synthetic-quality regression.
BF16 compressed state is a capacity improvement, not the rank-desynchronization
fix.

Do not set `SGLANG_FORCE_COARSE_WAR_BARRIER=1` on this image. It was the
same-build v0.5.16 containment; this base includes the upstream WAR-fence fix.
`SGLANG_ENABLE_PREFILL_WAR_READ_DONE` remains at its upstream default until it
is independently qualified.

## Build and promotion

```bash
docker build \
  --pull \
  --tag ds4f-0731-sglang:nightly-c0b6474-near-v1 \
  docker/sglang-dsv4-0731
```

The build fails if the base source revision differs, either patch checksum
changes, either patch stops applying, the resulting diff has whitespace
errors, the modified Python files fail compilation, or either hardened helper
is absent before removal.

Publish only through `.github/workflows/publish-dsv4-0731.yaml`. The workflow:

1. checks out an exact merged commit on `main`;
2. validates base, patch, model, and hardening provenance;
3. builds and pushes with BuildKit provenance and an SBOM;
4. verifies the published labels and base-image material;
5. runs Trivy and rejects fixable critical vulnerabilities;
6. attests and signs the immutable digest.

The pinned nightly identifies itself to Python package scanners as
`0.0.0.dev1+gc0b6474b4`, which sorts below the advisory's fixed `0.5.10`
version. The narrowly scoped, expiring `trivy-ignore.yaml` entries cover only
that exact PURL and CVE-2026-3059/CVE-2026-3060. The build independently fails
unless the loopback-only multimodal broker and safe encoder-disaggregation
deserializer remediations remain in the source. All other fixable critical
findings still fail publication.

Production compose must use the resulting
`docker.io/nearaidev/sglang@sha256:...` reference, never the nightly tag or a
mutable registry tag.

## Evidence and remaining qualification

The unmodified pinned nightly completed the August 20 GPU31/GPU32 matrix:

- 4,329 strict streaming rank-churn requests;
- 56 semantic, reasoning, JSON, tool-call, and SSE cases;
- zero request errors, malformed streams, restarts, OOMs, NCCL/desync failures,
  XIDs, AER errors, or uncorrected ECC errors.

The intermittent production hang did not reproduce. This supports the image
candidate but does not prove the race eliminated. The derivative itself is not
qualified until the published digest passes. A local build of this derivative
also completed two simultaneous BF16-compressed TP4 arms on GPU32, with sparse
prefill on and off:

- 14/14 semantic, reasoning, JSON, tool-call, and strict-SSE checks passed;
- both 850k-token prefill/live-decode/new-short gates passed, with 1.2912s and
  1.2333s worst new-short TTFT/event gaps;
- each arm returned 17 complete strict streams and 79 clean pre-stream HTTP 503
  rejections under a 96-request overload burst, with no malformed stream;
- the 30-minute soak completed 1,117 strict streams with zero request or health
  failures; worst TTFT was 7.6121s;
- both replicas stayed at zero restarts/OOMs, and the exact window had no
  NCCL/desync failure, XID, AER error, or corrected/uncorrected ECC growth;
- each replica exposed 11.526M KV tokens with
  `SGLANG_DSV4_COMPRESS_STATE_DTYPE=bf16`.

The published immutable digest must repeat:

1. patch and source-label verification;
2. semantic, tool-call, strict-SSE, and clean-overload/error-contract gates;
3. low/high/max DSV4 reasoning-profile checks;
4. an 850k-token prefill with live decodes and new short arrivals;
5. sustained TP4 rank churn with NCCL desync diagnostics enabled;
6. at least 30 minutes with zero restart, malformed stream, semantic failure,
   starvation, XID, AER, or ECC growth.

The bounded-admission implementation was previously qualified on GPU31 with
reserve-0 and reserve-4096 controls across both NVLink islands. Controls reached
112.690-114.833 seconds worst short TTFT/event gap, while reserve-4096 candidates
passed at 2.086-2.118 seconds. Rebase onto nightly requires repeating that gate.

## gpu30 canary and rollback

After publication and qualification, the deployment PR should change only the
gpu30 DS4F service in `prod/glm53-flash-dsv4-flash.yaml`:

- pin the new derivative digest;
- retain TP4, target-only decode, the current sparse-prefill setting, 8192-token
  chunks, the 4096-token admission reserve, and existing queue/watchdog limits;
- add `SGLANG_DSV4_COMPRESS_STATE_DTYPE=bf16`;
- update `sglang_revision` and `engine_image` telemetry labels;
- do not carry `SGLANG_FORCE_COARSE_WAR_BARRIER=1`.

Drain and start the gpu30 backend without registration, run the direct gates,
then register it and observe for 30-60 minutes before considering another host.
Abort and drain on any worker restart, malformed or unterminated SSE, in-stream
error, semantic/tool failure, active-stream or short-arrival gap above 10
seconds, NCCL/desync signal, XID, AER, or ECC growth.

Rollback is the currently pinned production image
`docker.io/nearaidev/sglang@sha256:ec518148762ea02c23aa8615f69ca79b0c18bcd59b3c21c10229db3df323c615`
with the existing gpu30 compose settings. Do not rebuild or retag rollback.
