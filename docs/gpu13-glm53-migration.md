# gpu13 GLM-5.3 Flash migration

This procedure replaces the two retired DeepSeek-V4-Flash TP2 services in
`prod/small-models.yaml` with one GLM-5.3-Flash TP4/EP4 service. The final GPU
layout is:

| GPU | Workload |
| --- | --- |
| 0 | privacy-filter and the temporary Qwen3.8 service |
| 1-2 | Qwen3.6 replicas |
| 3 | FLUX, Qwen3-VL, embedding, reranker, and Whisper |
| 4-7 | GLM-5.3-Flash |

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
