# GLM-5.3 Flash long-context tier on W4AFP8 + HiCache (gpu02)

`prod/GLM-5.3-Flash-SGL-TP4-W4AFP8-LongContext.yaml` moves both gpu02 replicas to the gpu31 campaign-2 arm L2: `graphistry/GLM-5.3-Flash-W4AFP8` at `99f1fa7` and the signed `docker/sglang-glm53-hicache-w4afp8` image `docker.io/nearaidev/sglang@sha256:fde25985aea3ebabf1eb581ae21d53be8540e32933eef942ee8b962a1bfbea20`. Each replica gets 8192-token prefill chunks with `--max-prefill-tokens 32768`, HiCache with CUDA-owned host memory at a 40% startup budget, and no admission reserve. The file is generated from `prod/GLM-5.3-Flash-SGL-TP4-LongContext.yaml` by `scripts/prepare_glm53_w4afp8_long_context.py`. It supersedes the W4AFP8 canary on gpu02. gpu13 stays on its own configuration (#290).

gpu02 is the first CVM run of this image's HCC/PPCIe host-memory path (`cudaMallocHost`) and of the 8192-token chunk's device headroom. Treat both replica moves as a soak, one replica at a time. The drain, direct-verification and rollback mechanics are the ones in [the gpu02 W4AFP8 runbook](gpu02-glm53-w4afp8-long-context.md); this document lists what differs.

## What changes

| | r1 `model-sg-glm53-w4afp8-tp4-r1` (GPUs 0–3) | r2 `model-sg-glm53-w4afp8-tp4-r2` (GPUs 4–7) |
|---|---|---|
| Replaces | W4AFP8 canary r1: 4096 chunk, no HiCache (same service name) | `model-sg-glm53-fp8-tp4-r2`: FP8 + HiCache at 80% |
| Image | `fde25985…` (hicache-w4afp8) | `fde25985…` (hicache-w4afp8) |
| Prefill | chunk 8192, `--max-prefill-tokens 32768` | same |
| HiCache | write_through, direct IO, page_first_direct, pooled transfers, 64 staging pages | same |
| Host budget | `${GLM53_HICACHE_RAM_BUDGET:-40%}` | same variable (it sets both) |
| Admission reserve | none | none |
| `--dist-init-addr` | `127.0.0.1:29510` | `127.0.0.1:29511` |

Nothing else changes. The domains, the `:8001` discovery stub, `LONG_TIER_ONLY` (default `true`), the registrar, the proxy pooling both replicas with conversation affinity, nginx, and the verification services stay as they are. `scripts/validate_glm53_prod_config.rb` enforces equality with the long-context file outside the two engines and their telemetry.

**The OpenRouter gateway and cloud-api need no change.** The host, ports (`:8000` probe, `:8444` TLS), domains and model-proxy handle all stay the same.

## Evidence (gpu31, 2026-09-23, CC off)

- L2 against FP8 + HiCache (L0), cold requests of 128K tokens and more: TTFT p50/p95 32.6/50.3 s vs 41.9/76.2 s (0.78×/0.66×).
- L2 completed 8 concurrent 647K–756K-token cold prefills; minimum free device memory was 3.7 GiB.
- Chunk 16384 fell to 0.04 GiB free and was rejected.
- Across the campaign's 12 engine configs: 0 restarts, 0 OOM kills.
- Two HiCache arms shared one host during the runs, at 406 GiB each.
- Argv parity with the L2 launch script: only `--model-path`/`--chat-template` (paths) and `--host`/`--port`/`--dist-init-addr` differ.
- Engine-env parity: the host budget is `40%` here vs `406GiB` absolute on gpu31. `SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0` is production-only and affects only the health endpoint.
- Not covered: the HCC/PPCIe allocator path and CVM memory-encryption overhead.

## Gates before starting

1. **Tag and env.** Use a merged tag that clears compose-manager's commit-age gate (`backdate-tag`). Pass the host's full gpu-manager env map on every call. Always set `force_recreate: false` and never send an empty `services` list. Run every `compose/up` with `dry_run: true` first, and apply only if the plan creates or recreates exactly the listed services and removes nothing.
2. **HiCache budget.** `GLM53_HICACHE_RAM_BUDGET` keeps its name, the same contract as gpu13 (#290) and the other GLM-5.3 files, but here it sets both replicas.
   - Before step 1, read gpu02's gpu-manager env map and make sure `GLM53_HICACHE_RAM_BUDGET` is unset or `40%`. A leftover single-replica value such as 80% would book 160% of host RAM once both replicas run HiCache.
   - The r2-first order cannot double-book, because the old r1 has no HiCache. r2's startup log (step 3) shows the resolved `rank_budget_bytes`; stop and fix the env if it is above ~40% of the CVM's RAM across four ranks.
   - The r1 step's mandatory startup-log check (step 8) catches a wrong value again, before r1 serves.
   - gpu02's current r2 logs `rank_budget_bytes=262552089395` (244.5 GiB per rank) at 80%. At 40%, each replica gets about 489 GiB, close to the 406 GiB qualification point, and the two together equal today's single 80% budget.
   - Other CVMs have different RAM (gpu13 runs one replica at 20% under #290). Recompute before using this file on any other host.
3. **Snapshot.** gpu02 already holds the W4AFP8 snapshot, and the downloader re-run is a no-op. Any other host must pre-stage it first from the file it currently runs (#293), with at least 250 GiB free.
4. **Capacity.** gpu02 is the only long host until #290 lands. While each replica is swapped, one replica carries the whole long tier. Pick a low-traffic window and confirm the live long-domain queue and pending prefill fit one replica. An operator watches queue, TTFT and aborts throughout and rolls back on saturation. Do not unregister gpu02 to drain it (hard gate 8 of the gpu02 runbook).
5. **Record the before-state.** Capture the deployed tag and file per service (attested action log), `docker/ps`, `/backends/list` for both GLM-5.3 domains on both model-proxy peers, and one successful long-domain completion.

## The orphan rule

compose-manager always runs `docker compose up -d --remove-orphans`. Any running service the applied file does not define is removed with a plain stop, roughly 10 s before SIGKILL, instead of the service's 5 m grace. So, before the first `compose/up` of this file:

1. Stop each engine it replaces with a scoped `compose/down`, using the file that defines that engine, so the 5 m grace applies.
2. Then run `up` for this file, scoped to the new engine plus any service whose definition changed.

Never let orphan removal perform the switch. On gpu02, the only orphan relative to this file is `model-sg-glm53-fp8-tp4-r2`.

## Procedure on gpu02: r2 first, then r1

### r2: FP8 + HiCache → W4AFP8 + HiCache (r1 keeps serving)

1. `compose/down` with the file and tag that deployed `model-sg-glm53-fp8-tp4-r2` (the W4AFP8 canary file, per the action log) and `services: ["model-sg-glm53-fp8-tp4-r2"]`. Poll `docker/ps` until it is gone, retrying `compose/down` on a docker zombie. The proxy marks r2 unhealthy and sends the tier to r1.
2. `compose/up` with this file and `services: ["model-sg-glm53-w4afp8-tp4-r2"]`. Compose re-runs `model-downloader` first; it is a no-op here.
3. Cold start: up to ~50 min, plus the pinned host allocation. In r2's log, check that:
   - `HiCache startup RAM: … rank_budget_bytes=…` reports about 40% of the CVM's RAM across four ranks;
   - `HiCache startup RAM allocated` follows;
   - no `CUDA error 801` appears (that would mean the `cudaHostRegister` path, i.e. `SGLANG_HICACHE_CUDA_HOST_MEMORY` is not `1`);
   - no OOM or restart occurs.
4. Verify r2 directly through `glm53-soak-relay` (`verification` profile; `:8009` → r2): `/health`, `/v1/models`, a deterministic generation, then the perception check (steps 6–7 of the gpu02 runbook).
5. `compose/up` with this file and `services: ["proxy-glm53", "otelcol-contrib", "dcgm-glm53"]`. The proxy now pools both W4AFP8 engines. nginx resolves the proxy per request, so it needs no restart. Requests in flight on the old proxy are cut at recreate and retried by cloud-api, which is why this runs in the low-traffic window.
6. Run the verification below, then soak r2 for the agreed period before touching r1. Watch for restarts, Xid or CUDA errors, DCGM framebuffer-free and SGLang free-memory warnings under long-context load, and queue-full rejections. `--mem-fraction-static 0.78` is an untested fallback, not a live lever.

### r1: W4AFP8 without HiCache → W4AFP8 + HiCache (r2 keeps serving)

7. `compose/down` with this file and `services: ["model-sg-glm53-w4afp8-tp4-r1"]`. Poll until it is gone.
8. `compose/up` with this file and the same `services`. Run the same startup-log checks (mandatory): `rank_budget_bytes` must again be about 40% of the CVM's RAM across four ranks. Then verify through `:8008`.
9. Reconcile: `dry_run` a `compose/up` of this file for `["model-downloader", "nginx", "model-proxy-registrar", "proxy-glm53", "dcgm-glm53", "otelcol-contrib"]` and apply it only for services still on an older definition. Recreating `model-proxy-registrar` unregisters `:8001` on SIGTERM, so gpu02 leaves the long domain for a discovery cycle or two. If the plan includes the registrar, apply it in the low-traffic window.

## Verification (after each replica)

- **Completions.** A real completion through the customer path and on `glm-5-3-flash-long.completions.near.ai`, then a cache-hit follow-up whose second turn reports cached prompt tokens.
- **Host restore.** The replica's HiCache host-hit counters (engine `/metrics`, also reachable through the soak relay) increase for a follow-up whose prefix had left the device pool, with no `write_through_unbacked_eviction` drops.
- **Registries.** Both model-proxy peers list gpu02 healthy under `glm-5-3-flash-long.completions.near.ai`, and not under the base domain (`LONG_TIER_ONLY=true`).
- **Telemetry.** Both engines report precision `int4-weights-fp8-activations-bf16-kv`, the L2 `config_variant` and `engine_image` `fde25985aea3`.

## Rollback

Redeploy the previous tag and file, scoped to the same services, one replica at a time and under the same orphan rule. Keep at least one replica serving throughout.

- **r2:** `compose/down` this file `["model-sg-glm53-w4afp8-tp4-r2"]`, then `compose/up` the previous file and tag `["model-sg-glm53-fp8-tp4-r2"]`. Once it is ready, `compose/up` the previous file for `["proxy-glm53", "otelcol-contrib", "dcgm-glm53"]`.
- **r1:** `compose/down` this file `["model-sg-glm53-w4afp8-tp4-r1"]`, then `compose/up` the previous file and tag `["model-sg-glm53-w4afp8-tp4-r1"]`.
- Routing levers are unchanged: `LONG_TIER_ONLY=false` plus a registrar restart, or removing the cloud-api `long_context` block.
