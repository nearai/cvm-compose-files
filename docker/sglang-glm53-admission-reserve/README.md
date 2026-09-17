# GLM-5.3 Flash admission-reserve engine

This build carries the `admission-reserve-v10` chunked-prefill scheduling patch
on the exact `fc91d24` production SGLang runtime, unchanged otherwise. Today
the scheduler spends the entire chunk budget on an in-flight long prefill, so
short requests that arrive behind it cannot enter the same prefill batch and
queue behind the whole prefill instead of just their own turn. The patch
reserves part of each chunk, page-aligned and sized on the waiting requests'
uncached (extend) length, capped at 75% of the chunk so the long prefill
always keeps at least a quarter of its budget and is never starved by a
steady stream of short arrivals; a completed pass with no waiter admitted
refunds the unused reserve back to the in-flight request. Measured on ≤2K
token prompts behind a long prefill, p95 time-to-first-token fell from 19.6 s
to 0.6 s, at a cost of raising inter-token latency (ITL) p90 from about 30 ms
to about 240 ms; median ITL is unchanged.

## Required settings

The patch is inert unless explicitly enabled. Three settings activate and
bound it:

| Setting | Purpose |
| --- | --- |
| `SGLANG_CHUNKED_PREFILL_ADMISSION_RESERVE=4096` | Opts in and caps the reserve in tokens; `0` (unset) preserves upstream behavior exactly. |
| `SGLANG_ADMISSION_RESERVE_MAX_FRACTION=0.75` | Fairness bound: the in-flight prefill always keeps at least `1 - MAX_FRACTION` of the chunk budget. |
| `--prefill-decode-interval 1` | Server flag required alongside the reserve so admitted waiters are scheduled promptly. |

## Forbidden variable

`SGLANG_ADMISSION_RESERVE_MIN_WAIT_S` must never be set; the patched
`_admission_reserve_need` raises `RuntimeError` if it is. An earlier revision
(v8/v9) gated waiter eligibility on wall-clock age via `perf_counter`. Every
TP rank runs the same scheduler loop over the same waiting queue, but
wall-clock reads drift independently per rank, so the four ranks disagreed on
which waiters qualified and built different batches. That desync surfaced as
an NCCL collective mismatch, and the watchdog killed the engine after about
2 hours. v10 replaces it with a rank-deterministic iteration counter
(`SGLANG_ADMISSION_RESERVE_MIN_ITERS`), which must stay unset (`0`, meaning
every waiter qualifies immediately) so all ranks see the same scheduling-pass
count for the same request.

## Rollback

Unset `SGLANG_CHUNKED_PREFILL_ADMISSION_RESERVE` (or leave it at `0`) and
recreate the engine. The patched files are inert with the env unset, so no
image rollback or rebuild is required — only a compose/service change.

## Release order

The repository publishes signed production images only from an exact commit
already merged into `main`. This preparation change leaves the active
production compose untouched. After review and merge:

1. Dispatch `.github/workflows/publish-glm53-admission-reserve.yaml` on
   `main` with `variant: base`, the exact merged `source_revision`, and a
   fresh publishing tag. It builds, runs CPU regressions, scans, attests and
   signs the immutable image. A resumed run verifies that the tag still
   matches the requested digest.
2. Verify the resulting signature and attestation:

   ```bash
   # Set IMAGE to docker.io/nearaidev/sglang@sha256:<published digest>.
   cosign verify \
     --certificate-identity \
     'https://github.com/nearai/cvm-compose-files/.github/workflows/publish-glm53-admission-reserve.yaml@refs/heads/main' \
     --certificate-oidc-issuer 'https://token.actions.githubusercontent.com' "$IMAGE"
   gh attestation verify "oci://$IMAGE" --repo nearai/cvm-compose-files
   docker buildx imagetools inspect --format '{{json .Provenance.SLSA}}' "$IMAGE"
   docker buildx imagetools inspect --format '{{json .SBOM.SPDX}}' "$IMAGE"
   ```

3. A separate activation PR pins the published digest and adds
   `SGLANG_CHUNKED_PREFILL_ADMISSION_RESERVE=4096`,
   `SGLANG_ADMISSION_RESERVE_MAX_FRACTION=0.75`, and the
   `--prefill-decode-interval 1` server flag to the intended canary replica.
   Publishing this image does not deploy it.

## Canary signals

Watch ITL p90 (the expected regression, bounded by the fairness fraction)
alongside time-to-first-token for the 10K-30K token prompt bucket (the
expected improvement for requests queued behind a long prefill). A canary
that improves TTFT in that bucket without pushing ITL p90 past the bound
implied by `SGLANG_ADMISSION_RESERVE_MAX_FRACTION` is healthy; rollback per
above otherwise.
