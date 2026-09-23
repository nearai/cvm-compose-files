# GLM-5.3 Flash base tier on W4AFP8 (gpu03, gpu04, gpu23)

`prod/GLM-5.3-Flash-SGL-TP4-W4AFP8.yaml` moves both replicas of a base-tier host to the gpu31 campaign-2 arm B5: `graphistry/GLM-5.3-Flash-W4AFP8` at `99f1fa7` on the signed `docker/sglang-glm53-w4afp8` image `docker.io/nearaidev/sglang@sha256:8bce6a7cc872a80faded3bd1ef0a64873a1d7abae34c94e5358775ca21f133cc`. The argv is exactly the one gpu02's W4AFP8 r1 runs today (4096-token chunks, `--max-prefill-tokens 32768`, no `--revision`, no `--moe-runner-backend`), plus the canonical admission reserve (`4096`, max fraction `0.75`). There is no HiCache. The file is generated from `prod/GLM-5.3-Flash-SGL-TP4.yaml` by `scripts/prepare_glm53_w4afp8_base.py`, and that canonical FP8 file is the rollback.

## What changes

Only the two engines change: service names `model-sg-glm53-w4afp8-tp4-r1/-r2`, image, checkpoint, argv, and their telemetry (precision `int4-weights-fp8-activations-bf16-kv`, `config_variant`, `engine_image`, `model_path`).

Everything else, including the model-downloader, stays byte-identical to the canonical file: domains, nginx, the registrar, the proxy (it pools the two W4AFP8 replicas with conversation affinity), DCGM and the OTel pipeline. `scripts/validate_glm53_prod_config.rb` enforces this.

**The OpenRouter gateway and cloud-api need no change.** The host, ports (`:8000` probe, `:8444` TLS), domains and model-proxy handle all stay the same.

## Evidence (gpu31, 2026-09-23, CC off, same host and seed as the FP8 production engine B0)

| Prompt bucket | TTFT p50 / p95, B5 vs FP8 |
|---|---|
| under 1K | 1.02× / 1.11× |
| 1–4K | 1.11× / 1.15× |
| 4–16K | 1.01× / 1.29× |
| 16–64K | 1.01× / 1.13× |
| 64K and up | 1.09× / 1.13× |

- Queue-full rejections: 13 vs 10 out of 877 requests. TPOT: 10.8 vs 10.2 ms.
- Larger W4AFP8 chunks (8192–16384) made short-prompt p95 1.5–3.2× worse, which is why the base tier keeps 4096.
- The W4AFP8 device KV pool is about 2.4× the FP8 one (3.56M vs 1.45M tokens per replica).
- 0 restarts and 0 OOMs across the campaign.
- Argv parity with the B5 launch script: only `--model-path`/`--chat-template` (paths) and `--host`/`--port`/`--dist-init-addr` differ. `SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0` is production-only and affects only the health endpoint.
- gpu02's W4AFP8 r1 has served this exact argv, without the reserve, in a CVM since #288 (2026-09-22).

## Gates

1. **gpu02 first, then gpu03.** The long-context W4AFP8 soak on gpu02 (`docs/glm53-w4afp8-long-context-rollout.md`) is accepted before the first base host moves. Then go one host at a time, gpu03 first, then gpu04 and gpu23, each only after the previous one has soaked cleanly.
   - gpu03 is first because it is the first CVM run of W4AFP8 **with** the admission reserve. gpu02's r1 serves W4AFP8 without it.
   - The gpu31 production-envelope ship check (2026-09-19) ran W4AFP8 with the reserve at the production scheduler configuration: 1,082 requests, 98.6% completed.
2. **Snapshot pre-staged (#293).** The host's current canonical file must already have fetched the W4AFP8 snapshot:
   - confirm at least 250 GiB free on the model-cache volume;
   - `compose/up` with `file: prod/GLM-5.3-Flash-SGL-TP4.yaml` and `services: ["model-downloader"]`;
   - wait for `Download complete.`.

   Only then is the switch's dark window the engine cold start alone.
3. **Tag and env.** Use a merged tag that clears compose-manager's commit-age gate (`backdate-tag`). Pass the host's full gpu-manager env map on every call. Always set `force_recreate: false` and never send an empty `services` list. Run every `compose/up` with `dry_run: true` first, and apply only if the plan creates or recreates exactly the listed services and removes nothing.
4. **Capacity.** The host leaves the base pool for the cold start, roughly 50 min per TP4 replica with both replicas starting together. The other two base hosts must carry the base domain meanwhile, so pick a low-traffic window.
5. **Record the before-state.** Capture the deployed tag and file (attested action log), `docker/ps`, `/backends/list` for `glm-5-3-flash.completions.near.ai` on both model-proxy peers, and one successful base completion.

## The orphan rule

compose-manager always runs `docker compose up -d --remove-orphans`. Any running service the applied file does not define is removed with a plain stop, roughly 10 s before SIGKILL, instead of the service's 5 m grace.

Neither `model-sg-glm53-fp8-tp4-r1` nor `-r2` exists in the W4AFP8 file, so the first `compose/up` of it removes whichever FP8 engine is still running. A base host therefore cannot move one replica at a time:

1. Stop both FP8 engines with a scoped `compose/down` of the canonical file, so the 5 m grace applies.
2. Then run `up` for the new file, scoped to the new engines plus the services whose definition changed.

Never let orphan removal perform the switch.

## Procedure (per host)

1. **Drain.** `compose/down` the canonical file with `services: ["model-proxy-registrar"]`. Its SIGTERM trap unregisters `:8000` from model-proxy, and the host leaves the base domain.
   - Watch the engines' running-request gauges and wait for them to reach about 0, up to 5 minutes.
   - Bucket-pinned H2 connections can keep sending for up to an hour, so a literal zero-drop is not achievable; cloud-api retries whatever is cut.
2. `compose/down` the canonical file with `services: ["model-sg-glm53-fp8-tp4-r1", "model-sg-glm53-fp8-tp4-r2"]`. Poll `docker/ps` until both are gone, retrying `compose/down` on a docker zombie.
3. `compose/up` the W4AFP8 file with `services: ["model-sg-glm53-w4afp8-tp4-r1", "model-sg-glm53-w4afp8-tp4-r2", "proxy-glm53", "otelcol-contrib", "dcgm-glm53"]`.
   - The dry-run plan must create the two engines, recreate the proxy, the collector and the exporter, and leave `model-downloader` and `nginx` untouched.
   - The proxy points at the new engine names. nginx resolves the proxy per request.
   - If one engine fails rank initialization (`DistStoreError`), restart that engine alone once the other is up.
4. **Cold start.** Expect about 50 min, since the W4AFP8 kernel caches are cold on a first start. Then verify each replica directly through `glm53-soak-relay` (`verification` profile; `:8008` r1, `:8009` r2) and run `glm53-perception-check`.
   - The logs must show the W4AFP8 loader and the CUTLASS W4A8 MoE path. There must be no W4A16/Marlin fallback and no CUDA, NCCL or Xid errors.
5. **Rejoin.** `compose/up` the W4AFP8 file with `services: ["model-proxy-registrar"]`. Its definition is the canonical one; it re-registers `:8000` once its 1-token probe passes.
6. **Verify.**
   - A real base completion through cloud-api and on the base domain, then a cache-hit follow-up that reports cached prompt tokens.
   - The host listed healthy under the base domain on both model-proxy peers.
   - The OpenRouter gateway's base handle for the host back to healthy.
   - Engine metrics labelled with precision `int4-weights-fp8-activations-bf16-kv`, the B5 `config_variant` and `engine_image` `8bce6a7cc872`.
7. **Soak** for the agreed period before the next host. Compare short-prompt TTFT p95 (expect +11–15% under 4K tokens), queue-full rejections, restarts and device memory against the remaining FP8 hosts.

## Rollback (per host)

Redeploy the canonical file, scoped to the same services and under the same orphan rule:

1. `compose/down` the W4AFP8 file with `services: ["model-proxy-registrar"]`.
2. `compose/down` the W4AFP8 file with `services: ["model-sg-glm53-w4afp8-tp4-r1", "model-sg-glm53-w4afp8-tp4-r2"]`.
3. `compose/up` the canonical file with `services: ["model-sg-glm53-fp8-tp4-r1", "model-sg-glm53-fp8-tp4-r2", "proxy-glm53", "otelcol-contrib", "dcgm-glm53"]`.
4. Once both engines are ready, `compose/up` the canonical file with `services: ["model-proxy-registrar"]`.

The FP8 snapshot stays cached, because every dedicated GLM-5.3 Flash TP4 file's downloader keeps fetching it. Do not evict `zai-org/GLM-5.3-Flash` with `hf-cleanup` while W4AFP8 engines run: they read the chat template from its `3f1971b7…` snapshot.
