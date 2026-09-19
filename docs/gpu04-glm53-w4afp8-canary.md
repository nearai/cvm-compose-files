# gpu04 GLM-5.3 Flash W4AFP8 canary

This runbook tests `graphistry/GLM-5.3-Flash-W4AFP8` on gpu04 replica 2 while replica 1 remains the same-host production FP8 control. The candidate keeps the stable served model name `z-ai/glm-5.3-flash`, uses TP4/EP4, BF16 KV, adaptive EAGLE 5/1/6, `--chunked-prefill-size 16384`, and `--max-prefill-tokens 32768`.

The compose file applies the reviewed W4AFP8 loader diff at container startup. It verifies the patch bytes, accepts only the exact unpatched or already-patched engine source, and verifies the resulting source bytes before starting SGLang. Any other drift fails closed without an automatic restart loop. This runtime overlay is for the canary only; a fleet rollout still requires the fix in a signed engine image or upstream SGLang.

## Benchmark provenance

- gpu31 independent same-host control and candidate runs: `/data/inference-optimizer-20260913/results/w4afp8`.
- gpu32 quantization, AWQ, and W4AFP8 runs: `/data/inference-optimizer-gpu32/results/{quant,quant-awq,w4afp8}`.
- Both hosts produced the same patched loader source with SHA-256 `039316192fb40a2aefe425102734d821c98e4c6c22a32ee51df21e47c315603d`.
- The reported gpu31 treatment was 0.96/1.49 s TTFT, 5.97/8.33 ms TPOT, and 2.23/3.30 s E2E at 0.70/1.20 request rates, with zero errors. Treat these as lab evidence until this runbook reproduces them with admission-reserve v10 on gpu04.

## Hard gates

- Deploy only on gpu04. Do not use this file on gpu02's long-context tier or on another fleet host.
- Every operational service in the canary file is gated by the `w4afp8-canary` Compose profile. Explicitly targeting a service activates its profile; an unscoped default apply has no enabled service and cannot change the stack. Still never send an empty `services` list.
- First merge, publish, GPU-qualify, and activate the pool-clamp fix tracked by PR #278. Regenerate this canary from the resulting canonical file and require `python3 scripts/prepare_glm53_w4afp8_canary.py --check` to pass. The currently deployed admission-reserve v10 image caused a prefill-pool crash on gpu02 under saturation.
- Obtain product approval for the measured quality trade before exposing customer traffic: W4AFP8 was about 10–12% faster in the replicated benchmark, with about 2.02 percentage points lower top-1 agreement and a perplexity ratio of 1.0334 versus the FP8 reference.
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
| Compose file | `prod/GLM-5.3-Flash-SGL-TP4-W4AFP8-Canary.yaml` |
| Direct verification ports | r1 `8008`, r2 `8009`, via the opt-in soak relay |

## Staged deployment

Use compose-manager's scoped `compose/up` and `compose/down` operations. Every canary or customer-routing service is opt-in under the `w4afp8-canary` profile. Never send an empty `services` list for this canary.

1. Run `model-downloader` from the canary file by itself. Confirm the pinned Graphistry snapshot completes and the cache volume retains the existing FP8 snapshot and chat template.
2. Stop `model-proxy-registrar` using the currently deployed canonical tag. Read back both production model-proxy registries and confirm gpu04 is withdrawn before changing replica 2.
3. Stop only `model-sg-glm53-fp8-tp4-r2` using the canonical file. Confirm replica 1 remains up and GPUs 4–7 are released.
4. Start only `model-sg-glm53-w4afp8-tp4-r2` from the canary file. Do not recreate the proxy, nginx, registrar, replica 1, DCGM, OTel, or monitoring stack at this stage.
5. Inspect the candidate log. Require all three SHA-256 checks to pass, the W4AFP8 checkpoint to resolve through `W4AFp8MoEMethod` / the CUTLASS W4A8 MoE path, weight memory near the qualified 41.4 GiB/GPU, and a KV pool near the qualified 3.56 million tokens. Stop on any fallback to W4A16/Marlin, shape-validation error, restart, NCCL error, Xid, or pool-allocation error.
6. Start `glm53-soak-relay` from the canary file with the verification profile. Test r2 through port 8009. A successful `/health` is insufficient: send a real deterministic chat completion, verify non-empty output and the stable served model name, then run `glm53-perception-check` and require all fixtures to pass.
7. Soak r2 for at least 30 minutes before routing customer traffic. Require zero request failures, zero container restarts, no `Prefill out of memory`, no admission-reserve fallback crash, and an active speculative decoder with acceptance greater than 1.00. Compare r1 and r2 TTFT, TPOT, E2E, queue depth, token usage, and completion-token counts in Grafana; treat unmatched output lengths as a confound.
8. After the hard gates and explicit product approval, recreate `proxy-glm53` and `nginx` from the canary file together. Recreate nginx because it resolves the proxy container address at startup. Verify direct TLS and attestation before restarting `model-proxy-registrar` from the canary file.
9. Read back both model-proxy registries, then send a real customer-path completion. Confirm logs and metrics identify r2 as `graphistry/GLM-5.3-Flash-W4AFP8` with `precision=int4-weights-fp8-activations-bf16-kv` and the W4AFP8 canary config variant.

## Rollback

Rollback on any failed gate, output/quality anomaly, restart, GPU error, sustained queue regression, or unexplained 4xx/5xx increase.

1. Stop `model-proxy-registrar` and confirm gpu04 is withdrawn from both registries.
2. Stop `glm53-soak-relay` and `model-sg-glm53-w4afp8-tp4-r2` using the canary file.
3. Start `model-sg-glm53-fp8-tp4-r2` from the prior canonical tag and file. Wait for a real successful generation, not only `/health`.
4. Recreate `proxy-glm53` and `nginx` together from the prior canonical tag and file.
5. Restart `model-proxy-registrar` from the canonical file, read back both registries, and verify a customer-path completion plus attestation.
6. Preserve candidate logs and the exact tag for analysis. Removing the downloaded checkpoint is a separate, explicit maintenance action.
