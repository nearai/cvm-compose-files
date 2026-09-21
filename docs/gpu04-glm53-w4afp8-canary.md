# gpu04 GLM-5.3 Flash W4AFP8 canary

This runbook tests `graphistry/GLM-5.3-Flash-W4AFP8` on gpu04 replica 2 while replica 1 remains the same-host production FP8 control. The candidate keeps the stable served model name `z-ai/glm-5.3-flash`, uses TP4/EP4, BF16 KV, adaptive EAGLE 5/1/6, `--chunked-prefill-size 16384`, and `--max-prefill-tokens 32768`.

The current compose file contains the reviewed W4AFP8 loader overlay, but it still inherits the admission-reserve-only engine image and therefore does **not** contain the mandatory chunked-prefill pool clamp. Its entrypoint exits 78 before patching or starting SGLang. The release-ready path is a signed `docker/sglang-glm53-w4afp8` image containing both patches, pinned by immutable registry digest. The runtime loader overlay verifies exact source bytes and remains only as a fail-closed compatibility check until that digest replaces the inherited image.

## Benchmark provenance

- gpu31 independent same-host control and candidate runs: `/data/inference-optimizer-20260913/results/w4afp8`.
- gpu32 quantization, AWQ, and W4AFP8 runs: `/data/inference-optimizer-gpu32/results/{quant,quant-awq,w4afp8}`.
- gpu31 production-envelope ship check: candidate `/data/inference-optimizer-20260913/results/w4afp8/g31-shipcheck-r1.20/run/summary.json`; control `/data/inference-optimizer-20260913/results/w4afp8/g31-shipcontrol-r1.20/run/summary.json`.
- Both hosts produced the same patched loader source with SHA-256 `039316192fb40a2aefe425102734d821c98e4c6c22a32ee51df21e47c315603d`.
- The replicated throughput runs measured about 8–12% lower E2E latency than FP8, depending on host and request rate.
- On 2026-09-19, the local two-patch image completed a matched 1,082-request gpu31 lab run at 1.20 requests/s with production's eight-request queue, priority scheduling, `--prefill-decode-interval 1`, and admission reserve enabled. W4AFP8 completed 98.61% versus 97.32% for FP8, had 15 silent aborts versus 29, reduced steady TTFT from 0.70 s to 0.63 s, reduced steady E2E from 3.02 s to 2.67 s, achieved 1.205 requests/s versus 1.157, and had no crash or OOM. Steady TPOT regressed from 9.73 ms to 10.06 ms (about 3.4%), so the scheduler result is not a win on every latency component. This resolves the scheduler-envelope experiment for the local image, but it does not qualify a future registry digest.

## Hard gates

- Deploy only on gpu04. Do not use this file on gpu02's long-context tier or on another fleet host.
- Every operational service in the canary file is gated by the `w4afp8-canary` Compose profile. Explicitly targeting a service activates its profile; an unscoped default apply has no enabled service and cannot change the stack. Still never send an empty `services` list.
- Merge the pool-clamp fix tracked by PR #278 before or with this canary. A 16,384-token chunk without that clamp can restore the full chunk after the pool budget goes negative and reproduce the gpu02 prefill-pool failure with a larger allocation request.
- Land and publish `docker/sglang-glm53-w4afp8`, built from the deployed engine with both the W4AFP8 loader fix and the byte-identical pool-clamp patch. Verify its cosign signature and GitHub attestation identify `nearai/cvm-compose-files` at the merged recipe commit. Verify the provenance pins base digest `e9d29a1cb1cd65284392c4d62d5f2a36669628057e15c60fe93ea40cfe4fc7e7`, loader patch SHA-256 `29764baa3e464d2272ea85f2e254392c2a61a3fc61a51f8d33b5910ce0cd8d00`, and pool-clamp patch SHA-256 `ba911be688556df0c0b2c9a26cde4c9f38b410a5ba51020d7754fa2e8cd010c3`. Pin the immutable image digest in the candidate service, remove this file's blocked header and executable exit, and regenerate the canary before use. A local tag such as `glm53-w4afp8:test` is not a deployable artifact.
- Replay the matched production-envelope workload on gpu31 against that exact published digest before gpu04. Require no crash or OOM and no regression in completion rate or silent aborts versus its same-host FP8 control; record the raw result path and digest in this runbook.
- Obtain product approval for the measured quality trade before exposing customer traffic: W4AFP8 was about 8–12% faster in the replicated benchmarks, with about 2.02 percentage points lower top-1 agreement and a perplexity ratio of 1.0334 versus the FP8 reference.
- Use a merged, backdated repo tag accepted by compose-manager. Fetch the complete gpu04 environment map from the dashboard and reuse it unchanged for every scoped operation.
- Confirm at least 250 GiB free on the CVM model-cache volume. The pinned checkpoint is about 165 GiB.
- Record the current tag, file, image digests, running containers, model-proxy registry entries, and a successful customer-path completion before touching the canary lane.

## Candidate identity

| Field | Value |
|---|---|
| Target | gpu04 replica 2, GPUs 4–7 |
| Control | `model-sg-glm53-fp8-tp4-r1`, GPUs 0–3 |
| Candidate | `model-sg-glm53-w4afp8-tp4-r2`, GPUs 4–7 |
| Checkpoint | `graphistry/GLM-5.3-Flash-W4AFP8` |
| Revision | `99f1fa70408c52b007d4fd69e02e5a522422e755` |
| Engine image | Blocked pending a signed, attested `docker/sglang-glm53-w4afp8` registry digest containing both patches |
| Compose file | `prod/GLM-5.3-Flash-SGL-TP4-W4AFP8-Canary.yaml` |
| Direct verification ports | r1 `8008`, r2 `8009`, via the opt-in soak relay |

## Staged deployment

Use compose-manager's scoped `compose/up` and `compose/down` operations. Every canary or customer-routing service is opt-in under the `w4afp8-canary` profile. Never send an empty `services` list for this canary.

1. Render the canary and prove the candidate service resolves to the reviewed signed W4AFP8 digest, not the inherited admission-reserve-only digest or a mutable tag. Verify the image signature, attestation, two-patch provenance, and successful gpu31 published-digest replay before running any service. Then run `model-downloader` by itself and confirm the pinned Graphistry snapshot completes while the cache volume retains the existing FP8 snapshot and chat template.
2. Stop `model-proxy-registrar` using the currently deployed canonical tag. Read back both production model-proxy registries and confirm gpu04 is withdrawn before changing replica 2.
3. Stop only `model-sg-glm53-fp8-tp4-r2` using the canonical file. Confirm replica 1 remains up and GPUs 4–7 are released.
4. Start only `model-sg-glm53-w4afp8-tp4-r2` from the canary file. Do not recreate the proxy, nginx, registrar, replica 1, DCGM, OTel, or monitoring stack at this stage.
5. Inspect the candidate log. Require the loader compatibility checks to pass, the W4AFP8 checkpoint to resolve through `W4AFp8MoEMethod` / the CUTLASS W4A8 MoE path, weight memory near the qualified 41.4 GiB/GPU, and a KV pool near the qualified 3.56 million tokens. Stop on any fallback to W4A16/Marlin, shape-validation error, restart, NCCL error, Xid, or pool-allocation error.
6. Recreate only `otelcol-contrib` from the canary file so its static r2 scrape target and labels follow `model-sg-glm53-w4afp8-tp4-r2`; require the r2 target to be up in Grafana. Then start `glm53-soak-relay` from the canary file with the verification profile and test r2 through port 8009. A successful `/health` is insufficient: send a real deterministic chat completion, verify non-empty output and the stable served model name, then run `glm53-perception-check` and require all fixtures to pass.
7. Soak r2 for at least 30 minutes before routing customer traffic. Require zero container restarts, no `Prefill out of memory`, no pool-clamp or admission-reserve crash, and an active speculative decoder with acceptance greater than 1.00. Compare matched r1/r2 completion rate, silent aborts, TTFT, TPOT, E2E, achieved request rate, queue depth, and token counts in Grafana. The candidate must not regress completion rate or silent aborts relative to the same-host control; treat unmatched output lengths as a confound.
8. After the hard gates and explicit product approval, recreate `proxy-glm53` and `nginx` from the canary file together. Recreate nginx because it resolves the proxy container address at startup. Verify direct TLS and attestation before restarting `model-proxy-registrar` from the canary file.
9. Read back both model-proxy registries, then send a real customer-path completion. Confirm logs and metrics identify r2 as `graphistry/GLM-5.3-Flash-W4AFP8` with `precision=int4-weights-fp8-activations-bf16-kv` and the W4AFP8 canary config variant.

## Rollback

Rollback on any failed gate, output/quality anomaly, restart, GPU error, sustained queue regression, worse completion or silent-abort rate than the same-host control, or unexplained 4xx/5xx increase.

1. Stop `model-proxy-registrar` and confirm gpu04 is withdrawn from both registries.
2. Stop `glm53-soak-relay` and `model-sg-glm53-w4afp8-tp4-r2` using the canary file.
3. Start `model-sg-glm53-fp8-tp4-r2` from the prior canonical tag and file. Wait for a real successful generation, not only `/health`.
4. Recreate `otelcol-contrib`, `proxy-glm53`, and `nginx` from the prior canonical tag and file. Recreate proxy and nginx together; confirm the canonical r2 scrape target is up again.
5. Restart `model-proxy-registrar` from the canonical file, read back both registries, and verify a customer-path completion plus attestation.
6. Preserve candidate logs and the exact tag for analysis. Removing the downloaded checkpoint is a separate, explicit maintenance action.
