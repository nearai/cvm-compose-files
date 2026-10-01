# gpu13 GLM-5.3 Flash migration

## 2026-10-01 DSA indexer query-split mitigation (source only)

The gpu13 GLM engine crashed during uncached 8192-token prefill in
`dsa_indexer_kpool._get_topk_ragged_kpool_plan -> deep_gemm.fp8_mqa_logits`:
13.31–13.37 GiB transient GPU allocations exceeded available VRAM. The v3
image (`47aff7910900`) was already live when this happened. This source change
sets `SGLANG_DSA_INDEXER_QSPLIT=1` only on `model-sg-glm53-fp8-tp4`, dividing
the DSA logits scratch across TP4 ranks. It reduces the per-rank allocation but
does not bound it at every context length.

Any live rollout requires separate approval. Before a future service-scoped
rollout, capture a fresh deployed tag, full dashboard environment, engine image,
container IDs and `CreatedAt` values, and prove the surviving long-tier capacity
can generate. Dry-run an up of only `model-sg-glm53-fp8-tp4` and verify that no
other service is targeted; then roll out only that engine. Verify its new image,
argv and environment, readiness and a real generation, and unchanged IDs and
`CreatedAt` for every other container. Watch for restarts, DSA/CUDA OOMs, and
queue and latency regressions. If verification fails, roll back only the engine
to the freshly recorded prior tag with the complete environment, then repeat
identity, readiness, generation, and OOM checks.

This procedure replaces the two retired DeepSeek-V4-Flash TP2 services in
`prod/small-models.yaml` with one GLM-5.3-Flash TP4/EP4 service. The final GPU
layout is:

| GPU | Workload |
| --- | --- |
| 0 | privacy-filter and the temporary Qwen3.8 service |
| 1-2 | Qwen3.6 replicas |
| 3 | FLUX, Qwen3-VL, embedding, reranker, and Whisper |
| 4-7 | GLM-5.3-Flash |

## Historical: engine-only v3 image upgrade (deployed before 2026-10-01)

The following is the earlier upgrade record; its deployment status and baseline
values are historical.

This candidate changes only `model-sg-glm53-fp8-tp4` from
`docker.io/nearaidev/sglang@sha256:fde25985aea3ebabf1eb581ae21d53be8540e32933eef942ee8b962a1bfbea20`
to the already-published original PR #308 v3 image
`docker.io/nearaidev/sglang@sha256:47aff791090003a37f893e998c44794c410d3f7bdfc7fdd2dfab5eb5592b30bb`.
The image change itself does not alter flags, devices, ports, model revisions,
dependencies, or configured serving parameters. This branch also changes the
gpu13 GLM service's default total HiCache budget to 80% of RAM available inside
the CVM at engine startup, after the co-located models have allocated memory;
`GLM53_HICACHE_RAM_BUDGET` in the dashboard environment overrides that default.
The old fixed 406 GiB budget is no longer the expected default. Keep the
CUDA-owned host-memory configuration.

Before any live `compose/down`, require a merged candidate release tag with green
repository CI and a passed tag-age gate. Refresh the full live baseline and prove
that a surviving GLM backend can generate; do not rely on historical health alone,
and do not recreate or mutate the survivor.

1. Record all 32 container IDs and `CreatedAt` values, the deployed v0.0.454
   source, and the complete dashboard `env_vars`. Pass that complete map as the
   Compose Manager request's `env`; do not rename, omit, or reconstruct values.
2. Run an exact `compose/up` dry-run for only
   `model-sg-glm53-fp8-tp4`, with `force_recreate: false`. Inspect the complete
   raw action stream and continue only if the singleton engine is the sole
   create/recreate target and no other service is removed or changed.
3. Use a service-scoped `compose/down` for the singleton at v0.0.454, wait until
   that container is absent, and then use a service-scoped `compose/up` for the
   same singleton at the candidate tag with the same complete `env` and
   `force_recreate: false`. `compose/down` is live and has no dry-run; never use
   an empty services list. Require the real streamed operation to finish with
   `done`, `success: true`, and exit code 0.
4. Verify the running digest and the v3 image source labels. Both
   `org.opencontainers.image.source` and `nearai.build.repository` must equal
   `https://github.com/nearai/cvm-compose-files`, while
   `org.opencontainers.image.revision` and `nearai.build.source_revision` must
   both equal
   `aff61fca1798512dcaec8cc88756ee0f83bb78be`, and
   `nearai.sglang.event_loop_stall_dump=v1-on-30s`. Confirm the startup log says
   the stall detector is armed, without treating that detector as proof that
   worker hangs are resolved. Record the effective budget and each rank's
   `available_bytes`; confirm all four TP ranks report the same
   `rank_budget_bytes`, consistent with the effective percentage or explicit
   override divided across the four ranks. The expected amount depends on
   startup-available RAM and any dashboard override. Then require readiness and
   a real completion from `z-ai/glm-5.3-flash`.
5. Re-read all 32 container IDs and `CreatedAt` values. The target must have a
   new identity and every one of the other 31 containers must match the
   pre-upgrade snapshot. Monitor the ready engine for 30 minutes for restarts,
   OOM/CUDA/Xid errors, event-loop stall dumps, queue depth, TTFT, and HiCache
   errors before accepting the upgrade.

Rollback is singleton-only: scoped down/up of
`model-sg-glm53-fp8-tp4` at v0.0.454 restores the old immutable pin above, using
the fresh complete dashboard environment and `force_recreate: false`, followed
by the same identity, readiness, completion, and monitoring checks.

The collector's inline static `engine_image` label remains stale until the
collector is separately recreated. This rollout does **not** authorize
recreating the collector, proxy, nginx, registrar, or any other service, and it
does not authorize changing the gateway. Attribute rollout evidence using the
target container's actual digest, ID, and deployment-time boundary rather than
claiming that the unchanged collector reloaded the new label.

The handoff must be staged. Compose Manager always passes `--remove-orphans`,
but orphan cleanup is not a GPU-release barrier. A full-project update may try
to start the new GLM service while an old engine or a shared GPU-7 service is
still stopping.

## Preconditions

1. Record the deployed tag, commit, file, and SHA256 from `/version`.
2. Record `/docker/ps`, container IDs and creation times, the model-proxy
   registry entries for gpu13, and the current GPU process inventory.
3. Render the candidate tag with the complete dashboard environment and verify
   that its service graph matches this runbook. Do not deploy a raw commit.
4. Confirm the candidate release tag has passed repository CI and the tag-age
   gate.
5. Keep OpenRouter routing unchanged. gpu13 is added to the OpenRouter gateway
   only after the base GLM route has passed the post-deploy checks below.

## Forward migration

Use the currently deployed tag and `prod/small-models.yaml` for the down phase.
Every request must include the complete dashboard environment map, and every
streamed response must end with `done`, `success: true`, and exit code 0.

1. Stop `model-proxy-registrar` first. Verify gpu13 entries are withdrawn from
   the model-proxy registries before changing any model container.
2. Stop `proxy-dsv4-flash`, `model-sg-dsv4-flash-fp4-tp2-r1`,
   `model-sg-dsv4-flash-fp4-tp2-r2`, and `dcgm-dsv4-flash`.
3. Stop the services moving from GPU 7 to GPU 3:
   `model-sg-flux2-klein-4b-tp1`,
   `model-vllm-qwen3vl-30b-a3b-fp8-tp1`,
   `model-vllm-qwen3-embedding-0.6b-tp1`,
   `model-vllm-qwen3-reranker-0.6b-tp1`,
   `model-vllm-whisper-large-v3-tp1`, and `dcgm-shared-gpu7`.
4. Read back `/docker/ps` and the GPU process inventory. Do not continue until
   the stopped containers are absent and GPUs 3-7 have no processes from the
   old topology.
5. Switch to the candidate release tag. Start the downloader, GLM engine,
   `dcgm-glm53`, `proxy-glm53`, and the shared GPU-3 FLUX, embedding, reranker,
   and Whisper engines. Start shared engines sequentially and wait for each to
   become ready. Do not start the registrar yet.
6. Start `model-vllm-qwen3vl-30b-a3b-fp8-tp1` last, after every other model has
   finished loading, then start `dcgm-shared-gpu3`. Qwen3-VL has a transient
   startup VRAM spike and must not load concurrently with the other GPU-3
   models.
7. Verify the new engine and proxy containers are healthy from Docker state and
   their logs. Keep the registrar stopped so the candidate cannot receive
   model-proxy traffic.
8. Recreate `nginx` with the candidate tag after `proxy-glm53` is healthy.
   Verify the old DSV4 SNI fails and the GLM SNI reaches the new proxy. Confirm
   unrelated proxy and model container IDs did not change.
9. On port 8009, run a non-streaming completion, a streaming completion, a
   tool-call request, and a reasoning request. Confirm the response model is
   `z-ai/glm-5.3-flash`. Treat 40 running requests and 8 queued requests as a
   canary operating point.
   Before registration, exercise bounded concurrency while watching startup,
   OOM/restart count, queue depth, TTFT, ITL, and GPU memory. Roll back if the
   instance is unstable or materially regresses the qualified baseline.
10. Start `model-proxy-registrar`. Read back both production model-proxy
   registries: gpu13 port 8009 must appear under `z-ai/glm-5.3-flash`, no gpu13
   DSV4 entry may remain, and every unrelated gpu13 endpoint must be restored.
11. Validate the public base GLM route with ordinary, streaming, tool-call, and
    reasoning requests. Check logs and metrics before declaring the migration
    complete.

## Rollback

Use the recorded pre-migration release tag and the same complete environment
map.

1. Stop the candidate `model-proxy-registrar` and verify gpu13 registry entries
   are withdrawn.
2. Stop `proxy-glm53`, `model-sg-glm53-fp8-tp4`, `dcgm-glm53`, the shared
   GPU-3 engines listed above, and `dcgm-shared-gpu3`.
3. Verify the candidate containers are absent and GPUs 3-7 are released.
4. Switch back to the recorded release tag. Start the two DSV4 engines and
   their proxy/DCGM exporter. Restore the shared GPU-7 FLUX, embedding,
   reranker, and Whisper engines sequentially, waiting for each to become
   ready.
5. Start `model-vllm-qwen3vl-30b-a3b-fp8-tp1` last, after every other model has
   finished loading, then start `dcgm-shared-gpu7`.
6. Recreate `nginx`, verify direct port 8009 and the DSV4 SNI, then start the
   old registrar.
7. Read back both registries and verify the old gpu13 DSV4 entry and every
   unrelated gpu13 endpoint are restored. Confirm that no gpu13 GLM entry
   remains and that adjacent container IDs match the pre-migration record.

Do not update the OpenRouter gateway during rollback. If a later gateway change
has already added gpu13, remove or disable that backend before beginning this
procedure.
