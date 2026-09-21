# gpu02 GLM-5.3 Flash W4AFP8 long-context r1

This runbook replaces only gpu02 replica 1 with `graphistry/GLM-5.3-Flash-W4AFP8`. Replica 2 stays on the deployed FP8 HiCache definition. The candidate keeps the stable served model name `z-ai/glm-5.3-flash`, TP4/EP4, BF16 KV, adaptive EAGLE 5/1/6, and the one-million-token context limit while changing r1 to `--chunked-prefill-size 16384` plus `--max-prefill-tokens 32768`.

The committed candidate is intentionally not runnable. It inherits the base engine image, which lacks the mandatory chunked-prefill pool clamp, and its entrypoint exits 78 before patching or starting SGLang. The release path is a signed `docker/sglang-glm53-w4afp8` image containing both reviewed patches, pinned by immutable registry digest.

## Why gpu02 r1

gpu02 already serves the dedicated `glm-5-3-flash-long.completions.near.ai` tier. Replacing r1 tests W4AFP8 against the traffic shape it would actually serve while keeping r2's deployed FP8 HiCache engine, GPU assignment, proxy contract, conversation affinity, long-domain registrar, and rollback path intact.

This is not a matched A/B benchmark. r2 has HiCache and r1 does not, so r2 can preserve service capacity and provide a safety reference, but the performance decision still comes from the paired gpu31/gpu32 results. The gpu02 gate is long-context safety and customer-path behavior.

## Benchmark provenance

- gpu31 independent same-host control and candidate runs: `/data/inference-optimizer-20260913/results/w4afp8`.
- gpu32 quantization, AWQ, and W4AFP8 runs: `/data/inference-optimizer-gpu32/results/{quant,quant-awq,w4afp8}`.
- gpu31 production-envelope ship check: candidate `/data/inference-optimizer-20260913/results/w4afp8/g31-shipcheck-r1.20/run/summary.json`; control `/data/inference-optimizer-20260913/results/w4afp8/g31-shipcontrol-r1.20/run/summary.json`.
- The 2026-09-19 local two-patch run completed 98.61% versus 97.32% for FP8, with 15 versus 29 silent aborts, TTFT 0.63 versus 0.70 seconds, E2E 2.67 versus 3.02 seconds, and no crash or OOM. Steady TPOT regressed about 3.4%, from 9.73 to 10.06 milliseconds.
- Both lab hosts produced the patched loader source SHA-256 `039316192fb40a2aefe425102734d821c98e4c6c22a32ee51df21e47c315603d`.

## Hard gates

1. Merge the pool-clamp fix tracked by PR #278 before or with this change. A 16,384-token chunk without the clamp can restore a phantom full chunk after the available pool goes negative.
2. Publish and pin a signed `docker/sglang-glm53-w4afp8` digest built from base digest `e9d29a1cb1cd65284392c4d62d5f2a36669628057e15c60fe93ea40cfe4fc7e7`, loader patch SHA-256 `29764baa3e464d2272ea85f2e254392c2a61a3fc61a51f8d33b5910ce0cd8d00`, and pool-clamp patch SHA-256 `ba911be688556df0c0b2c9a26cde4c9f38b410a5ba51020d7754fa2e8cd010c3`. Verify cosign and GitHub provenance identify `nearai/cvm-compose-files` at the merged recipe commit.
3. Replay the matched production-envelope workload on gpu31 against that exact published digest. Require no crash or OOM and no regression in completion rate or silent aborts versus its same-host FP8 control.
4. Qualify the 16K chunk against the real long-context envelope before customer routing. The deployed FP8 arm previously OOMed at an 8K chunk under concurrent 400K-plus-token contexts. Exercise fresh and cache-hit prompts across 100K, 250K, 400K, and near-one-million tokens, including concurrent long prefills, and require no DSA indexer, K-pool, `alloc_extend`, or CUDA OOM failure.
5. Obtain product approval for the measured quality trade: about 8–12% lower E2E latency in the replicated workload versus about 2.02 percentage points lower top-1 agreement and a 1.0334 perplexity ratio.
6. Use a merged, backdated repository tag accepted by compose-manager. Fetch gpu02's complete dashboard environment map and reuse it unchanged for every scoped operation. Never send an empty service list.
7. Confirm at least 250 GiB free on the CVM model-cache volume and record the deployed tag, file, image digests, running containers, registry entries, and a successful long-domain completion before the change.

PR #278 was approved but still open on 2026-09-21. No signed combined W4AFP8 digest was available at that check.

## Candidate identity

| Field | Value |
|---|---|
| Target | gpu02 replica 1, GPUs 0–3 |
| Preserved arm | `model-sg-glm53-fp8-tp4-r2`, GPUs 4–7, FP8 HiCache |
| Candidate | `model-sg-glm53-w4afp8-tp4-r1`, GPUs 0–3 |
| Checkpoint | `graphistry/GLM-5.3-Flash-W4AFP8` |
| Revision | `99f1fa70408c52b007d4fd69e02e5a522422e755` |
| Engine image | Blocked pending a signed, attested two-patch registry digest |
| Compose file | `prod/GLM-5.3-Flash-SGL-TP4-W4AFP8-Canary.yaml` |
| Compose profile | `w4afp8-long-context` |
| Direct verification ports | r1 `8008`, r2 `8009`, through the opt-in soak relay |

## Staged qualification

1. Render the candidate and prove the r1 image resolves to the reviewed signed digest, not the inherited base image or a mutable tag. Verify the signature, attestation, two-patch provenance, and gpu31 replay. Run `model-downloader` alone and confirm the pinned Graphistry snapshot completes without removing the existing FP8 snapshot or chat template.
2. Stop `model-proxy-registrar` from the deployed long-context tag, explicitly unregister gpu02's long endpoint, and read back both production model-proxy registries. Do not rely on SIGTERM alone to unregister the host. Keep customer routing withdrawn throughout direct qualification.
3. Stop only `model-sg-glm53-fp8-tp4-r1` from the deployed long-context file. Confirm r2 stays running, its container identity is unchanged, and GPUs 0–3 are released.
4. Start only `model-sg-glm53-w4afp8-tp4-r1` from the candidate file. Do not recreate proxy, nginx, registrar, r2, DCGM, OTel, or the monitoring stack yet.
5. Require the loader compatibility checks to pass and the model to resolve through `W4AFp8MoEMethod` and the CUTLASS W4A8 MoE path. Expect weight memory near 41.4 GiB/GPU and a KV pool near 3.56 million tokens. Stop on W4A16/Marlin fallback, shape validation, restart, NCCL, Xid, pool-allocation, or CUDA errors.
6. Start `glm53-soak-relay` with the verification profile and test r1 through port 8008. A successful `/health` is insufficient: send a deterministic generation, verify non-empty output and the stable served model id, then run the perception check against both replicas.
7. Run the long-context matrix from hard gate 4. Verify prefix-cache reuse on a second turn, speculative acceptance greater than 1.00, zero restarts, and no silent abort increase. Compare r1/r2 completion rate, TTFT, TPOT, E2E, queue depth, KV usage, and output lengths, while treating r2's HiCache as a known confound.
8. After every hard gate and product approval passes, recreate `otelcol-contrib` so its static r1 target and labels follow the candidate. Recreate `proxy-glm53` so customer routing names the candidate r1 and preserved r2. Nginx resolves the proxy dynamically and need not be recreated for this service-name-only swap.
9. Restart `model-proxy-registrar`, read back both registries, and send a real long-domain customer-path completion plus a cache-hit follow-up. Confirm metrics identify r1 as `graphistry/GLM-5.3-Flash-W4AFP8`, precision `int4-weights-fp8-activations-bf16-kv`, and the long-context W4AFP8 config variant.

## Rollback

Rollback on any failed gate, output anomaly, restart, GPU error, long-context allocation failure, sustained queue regression, silent-abort increase, or unexplained 4xx/5xx increase.

1. Stop `model-proxy-registrar`, explicitly unregister gpu02's long endpoint, and confirm withdrawal in both registries.
2. Stop `glm53-soak-relay` and `model-sg-glm53-w4afp8-tp4-r1` using the candidate file.
3. Start `model-sg-glm53-fp8-tp4-r1` from the prior long-context tag and file. Require a real successful generation, not only `/health`; verify r2's container identity never changed.
4. Recreate only `otelcol-contrib` and `proxy-glm53` from the prior long-context tag so their r1 target returns to `model-sg-glm53-fp8-tp4-r1`.
5. Restart `model-proxy-registrar`, read back both registries, and verify a long-domain completion, cache-hit follow-up, and attestation.
6. Preserve candidate logs and the exact tag for analysis. Removing the downloaded checkpoint is a separate maintenance action.
