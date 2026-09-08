# Qwen consolidation / one additional GLM-5.3-Flash replica

Proposed configuration, not an executed rollout. Production changes require separate authorization.

| Host | GPUs | Before | After |
|---|---|---|---|
| gpu02 | 0–1 | DS4F r1 | Unchanged |
| gpu02 | 2–3 | Two Qwen3.8 replicas | DS4F r2, moved from 6–7 |
| gpu02 | 4–7 | Two Qwen3.6 replicas + DS4F r2 | One GLM-5.3-Flash TP4 replica |
| gpu13 | 1 | Qwen3.6 r1 | Unchanged |
| gpu13 | 2 | Qwen3.6 r2 | One Qwen3.8 replica |

- GLM must occupy a complete NVLink island: **0–3 or 4–7**, never 2–5. This plan uses 4–7.
- Qwen3.6: 4 → 1 GPUs. Qwen3.8: 2 → 1. GLM-5.3-Flash: 8 → 12. DS4F remains 4.
- Four GPUs are reassigned, not deprovisioned; fleet allocation and GPU-hour burn do not decrease.
- Both Qwens lose replica redundancy and share one host. Memory fit does not establish peak throughput/latency capacity; a one-replica load gate is required before removing fallback capacity.
- The legacy `prod/dsv4-qwen38-glm51.yaml` filename remains to avoid changing the Compose project/file identity during migration. `prod/small-models.yaml` is the gpu13 pack.
- GLM registration defaults off (`REGISTER_GLM53=false`) so the registrar can keep DS4F available while GLM is being qualified. Enable it explicitly only after the direct gate passes.

## Preflight and staging gates

- Snapshot the actual deployed tags, file hashes, container IDs, GPU claims, routing endpoints, and the complete environment map from gpu-manager. Use those exact snapshots for rollback, not whatever is currently on main.
- Keep the current Compose project identity and service-scoped applies. Do not use a whole-stack `up`, broad `down`, `--remove-orphans`, GPU reset, or volume deletion. Keep all existing model weights/caches for rollback.
- Build GLM's source-pinned inline recipe and verify its immutable image ID and OCI source revision (`fc91d2403cc210a86bd5fc715c6609f58c932e5a`). The `:local` name is a build output, not a pullable registry artifact. This reuses the current canary recipe, including CPU image preprocessing and the 64-image limit; it is not a new engine upgrade.
- Pre-download with `model-downloader-glm53` on gpu02 and `model-downloader-qwen38` on gpu13. Confirm disk headroom and the pinned GLM weights/chat-template and Qwen checkpoint revisions before draining anything.
- Validate on a real staging CVM for at least 30 minutes with zero failures: representative prompt lengths/concurrency, text, strict streaming completion, tool calls, vision, cached-token reporting, proxy auth/signatures and attestation. Reject regressions in latency, queue growth, CUDA/XID errors, or unrelated workloads. Static Compose/CI validation does not satisfy this gate.
- Verify the real GPU UUID → index → NVLink-island mapping before placement. Require the moved Qwen runtime/attestation contract to work on gpu13; do not infer that from gpu02 qualification alone.

## Ordered production migration, after authorization

1. **Prepare gpu13.** Apply only `proxy-qwen36-35b-a3b` with the new one-backend configuration, refreshing nginx to resolve the proxy's new address. Verify r1 remains healthy. Wait for r2 running, queued, and proxy-active requests to reach zero before explicitly stopping/removing `model-sg-qwen36-35b-a3b-fp8-tp1-r2` using the old deployed configuration. Recreate only `dcgm-qwen36-35b-a3b` with its GPU1-only claim. Confirm GPU2 has no remaining engine/exporter claim.
2. **Bring up the relocated Qwen3.8.** Start `model-downloader-qwen38`, `model-sg-qwen38-27b-fp8-tp1`, `proxy-qwen38-27b`, and `dcgm-qwen38-27b`; update nginx for HTTP 8010 and the existing Qwen3.8 SNI on TLS 8444. Verify directly before starting the updated registrar. Keep both gpu02 Qwen3.8 replicas until the new route and single-replica load gate pass. Refresh gpu13 `otelcol-contrib` and confirm Qwen3.6/Qwen3.8 each count exactly one distinct GPU.
3. **Drain the gpu02 Qwens.** Stop the old registrar so it cannot re-register retired endpoints. Unregister only the old Qwen endpoints (`gpu02:8002` and `gpu02:8006`) and confirm their removal. Start the updated registrar with `REGISTER_GLM53=false` to re-establish DS4F independently of GLM. Allow in-flight requests and existing connections to drain; verify surviving Qwen routes and queue/latency health. Explicitly remove the nine old services listed below with the old deployed configuration. Confirm GPUs 2–5 have no old engine/exporter claims. Keep their cache volumes.
4. **Move DS4F r2.** Set `DSV4_BACKEND_URLS=http://model-sg-dsv4-flash-fp4-tp2-r1:8000` in the full environment map and apply only `proxy-dsv4-flash`, then refresh nginx. Verify r1; wait for r2 to drain before stopping/removing r2 and `dcgm-dsv4-flash-r2`. Start those two services with GPU IDs 2–3. Verify direct text/streaming/tool calls and attestation; then unset the temporary override, restore the normal two-backend proxy pool, and refresh nginx. Never restart r1. Proxy/nginx transitions can interrupt connections; schedule and monitor them explicitly.
5. **Add GLM.** After confirming all of GPUs 4–7 are free of old engine/exporter claims, start the prebuilt `model-sg-glm53-fp8-tp4`, `proxy-glm53`, and `dcgm-glm53`. Add nginx HTTP 8000/TLS 8008. Verify direct functional, streaming, vision, cache, and attestation checks, then set `REGISTER_GLM53=true` in the full environment map and apply only the registrar to admit traffic. Existing GLM replicas on other hosts remain unchanged.
6. **Refresh and reconcile.** Recreate gpu02 `otelcol-contrib` using the new scrape list. Verify only current targets are up, eight unique GPU UUIDs on each affected host, DS4F 4, Qwen3.6 1, Qwen3.8 1, and GLM-5.3-Flash 12 fleet-wide. Confirm the admin allocation/usage-value/burn table refreshes without double-counting or stale Qwen exporters. Verify unrelated gpu13 model container IDs remain unchanged.

gpu02 services removed in step 3:

- `model-sg-qwen38-27b-fp8-tp1-r1`
- `model-sg-qwen38-27b-fp8-tp1-r2`
- `model-sg-qwen36-35b-a3b-fp8-tp1`
- `model-sg-qwen36-35b-a3b-fp8-tp1-r2`
- `proxy-qwen38-27b`
- `proxy-qwen36-35b-a3b`
- `dcgm-qwen38-27b`
- `dcgm-qwen36-35b-a3b`
- `dcgm-qwen36-35b-a3b-r2`

## Stop / rollback

- Stop before retiring fallback capacity if either consolidated Qwen fails its load/latency gate. Do not increase queue limits to conceal insufficient capacity.
- If GLM fails qualification, leave it unregistered and keep the existing GLM hosts serving. Its allocation is not available for rollback until its engine and exporter are drained/stopped and GPU release is confirmed.
- Restore in reverse allocation order: free GLM's 4–7; drain DS4F r2 from 2–3 and restore it to 6–7 while r1 serves; restore the old gpu02 Qwen services/routes/telemetry; verify them before removing gpu13 Qwen3.8 and restoring gpu13 Qwen3.6 r2.
- Restore proxy pools, registrars, nginx, and collectors from the recorded per-host snapshot. Verify exact GPU ownership and route functionality at each phase. No blind whole-stack rollback: old and new placements overlap.
