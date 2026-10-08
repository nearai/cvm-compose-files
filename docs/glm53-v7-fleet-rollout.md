# GLM-5.3 Flash v7 fleet rollout: one base config, one long config

Status: **prepared, not deployed.** Merging changes no running service. A host changes only when an operator deploys the merged tag to it, scoped to the services below, through the already-running compose-manager.

The **v7 image** is `docker.io/nearaidev/sglang@sha256:fa730e6e62b2ae8058114ce540487ade33ab93bc42b1179ae78edc92bd563fc5` (tag `glm53-hicache-w4afp8-v7`, #345, publish run 37693106399, main `29a7db9`). **Rollback image: v6, `sha256:9c6ddd4319c4ab00e351d8650459e68b8830e36ffcc029d67fa5e19d0ac3ed17`.**

## What every host of a tier deploys

| Tier | File (generated, do not hand-edit) | Hosts | Engines |
|---|---|---|---|
| Base | `prod/GLM-5.3-Flash-SGL-TP2x4-W4AFP8.yaml` (`scripts/prepare_glm53_w4afp8_tp2x4.py`) | gpu03, gpu04 | `model-sg-glm53-w4afp8-tp2-r1` .. `-r4` |
| Long | `prod/GLM-5.3-Flash-SGL-TP4-W4AFP8-LongContext.yaml` (`scripts/prepare_glm53_w4afp8_long_context.py`) | gpu02, gpu23 | `model-sg-glm53-w4afp8-tp2-r1a`, `model-sg-glm53-w4afp8-tp2-r1b`, `model-sg-glm53-w4afp8-tp2-r2a`, `model-sg-glm53-w4afp8-tp2-r2b` |
| Long (shared host) | `prod/small-models.yaml` (hand-maintained; its GLM replicas are tested equal to the long file's) | gpu13 | `model-sg-glm53-w4afp8-tp2-r1a`, `-r1b` |

| | Base replica (r1-r4) | Long replica (r1a/r1b/r2a/r2b, and gpu13 r1a/r1b) |
|---|---|---|
| image | v7 | v7 |
| KV / DSA backends | `--kv-cache-dtype fp8_e4m3`, `--dsa-prefill-backend flashmla_kv`, `--dsa-decode-backend flashmla_kv` | same |
| running / queued / decode graphs | 64 / 8 / 64 | 16 / 4 / 16 |
| mamba slots | 380 (5 per running request) | 330 |
| overlap scheduler | off (`--disable-overlap-schedule`) | off |
| HiCache | **OFF**: the four flags `--enable-hierarchical-cache --hicache-write-policy --hicache-io-backend --hicache-mem-layout` are not in the argv | **ON** (`write_through`, 325 GiB per replica as today) |
| environment | `SGLANG_PREPROCESS_WORKERS=4`, `SGLANG_PREPROCESS_TIMEOUT_S=60`, `SGLANG_PREPROCESS_LOG_SLOW_S=5`, `SGLANG_TOOL_SCHEMA_MAX_DEPTH=32`, `SGLANG_TOOL_SCHEMA_MAX_NODES=25000` | same |
| profiling | **off**: no `NEAR_SELF_PROFILE*` anywhere | off |
| `precision` / `engine_image` | `int4-weights-fp8-activations-fp8-kv` / `fa730e6e62b2` | same |
| `config_variant` | `w4afp8-qsplit-hicacheoff-mamba380-fp8kv-memopt086-mr64-...-obs-v1-v7` | `...-memopt-fp8kv-mamba330-...-mr16q4-strict-budget8192-obs-v1-v7` |

**How HiCache is off on the base tier (copied from gpu03 r3, arm B' of the 2026-10-08 online test, `pranavraja99/glm53-kvshare-ab`):** only the four HiCache flags are removed from the replica argv. Nothing else moves: the HiCache environment (`SGLANG_HICACHE_RAM_BUDGET`, `..._CUDA_HOST_MEMORY`, `..._POOLED_TRANSFERS`, `..._STAGING_PAGES`) stays and is unread without `--enable-hierarchical-cache`; mem fraction, mamba slots and labels are the v7 base values above. The generated base argv is asserted equal to r3's live argv in `scripts/test_glm53_v7_fleet.py`.

**Long file structure:** the TP4 `r1`/`r2` services, the proxy pool (`${GLM53_BACKEND_URLS:-...}`), sidecars, nginx, registrar, the ghost aggregator and otelcol are unchanged except: the four TP2 services (v7 argv/env/labels) and the ghost aggregator image (now v7, so it follows its engines). The TP4 services stay on v6 (kept only so a host not on TP2 still deploys; the fleet does not run them). Each host's `GLM53_BACKEND_URLS` is already set in its compose-manager env map for its stage; **do not change it** (verify it names the four TP2 replicas before the host's first step).

**Dashboard labels (Grafana `glm53-flash-prod`) are unchanged:** `deployment` (`glm53-flash-sgl-tp2x4` base, `glm53-flash-sgl-tp4` long), `host_machine`, `host`, `service`, `server_address` (the scrape target), `model`, `instance`, `service=dcgm-exporter`, `service=vllm-proxy`, and the DCGM, vllm-proxy, ghost-aggregator and otelcol scrape jobs. Only `precision`, `engine_image` and `config_variant` change. `scripts/test_glm53_v7_fleet.py` compares every service and scrape job with the pre-rollout files.

## Deploy (compose-manager only, no KMS step, no env-map changes)

Every step is `POST http://<host>:8080/compose/up` with `tag` = the merged tag `T` (past `MIN_TAG_AGE_HOURS`), `file` (with the `prod/` prefix), an **explicit `services` list** (never empty), the host's existing full `env` map, `force_recreate: false`, **`dry_run: true` first**. The plan must show exactly the listed services recreated and nothing else (no `model-downloader`, proxy, nginx, registrar, other engine, no `--remove-orphans` removals); if it shows more, stop. Then repeat with `dry_run: false`. Cold start is about 22 minutes under CC. Pre-check before each step: `docker/ps` and `/backends/list` healthy on the host, and the dump of the host's env map kept as the rollback reference.

### Limits (user rules)

- **Long-context hosts one at a time:** finish a long host (back serving and healthy, verified) before touching the next.
- **Never more than 2 base replicas down fleet-wide:** one base pair at a time across gpu03 and gpu04; the next pair starts only after the previous pair is back and healthy. Before each step check `docker/ps` and `/backends/list` on both hosts; start only when every other base replica is up.
- **`compose/down` takes effect immediately, ignores `dry_run` and cannot be recalled.** This runbook never needs one (`compose/up` of the service recreates it). If one is ever sent, scope it to one service and verify the logged services list in the host's compose-manager action log.

### Order

Base tier (services `model-sg-glm53-w4afp8-tp2-r1`, `model-sg-glm53-w4afp8-tp2-r2`, `model-sg-glm53-w4afp8-tp2-r3`, `model-sg-glm53-w4afp8-tp2-r4`; below, `r1` etc. abbreviate these):

1. gpu04: `r1`, `r2` (one call, two services). Verify. Then gpu04 `r3`, `r4`. Verify.
2. gpu03: `r1`, `r2`. Verify. Then gpu03 `r3`, `r4` (they are already v7-like, r3 HiCache off and r4 with profiling; redeploy them to the fleet file so labels, environment and profiling match). Verify.
3. Per base host after its four replicas: `services: ["glm53-ghost-aggregator"]` (image now v7), then `services: ["otelcol-contrib"]` with `force_recreate: true` (dry run first; plan: collector only). Verify `config_variant` ends `-v7` for all four.

Long tier (services `model-sg-glm53-w4afp8-tp2-r1a`, `-r1b`, `-r2a`, `-r2b`), `file: prod/GLM-5.3-Flash-SGL-TP4-W4AFP8-LongContext.yaml`:

4. gpu23: `r1a`, `r1b`; verify; then `r2a`, `r2b`; verify. Then `glm53-ghost-aggregator`, then `otelcol-contrib` (`force_recreate: true`).
5. gpu02 (starts only after gpu23 is verified): `r1a`, `r1b`; verify; then `r2a`, `r2b` (r2a currently runs the canary); verify. Then `glm53-ghost-aggregator`, `otelcol-contrib`.
6. gpu13 (`file: prod/small-models.yaml`; starts only after gpu02 is verified): **this ends Pranav's shared-KV experiment on gpu13, and its owner must be told before this step.** gpu13's other models (Qwen, FLUX, Qwen3-VL, whisper and the rest) must stay up, so the services list contains only the GLM engines: `r1a`, verify, then `r1b`, verify (one replica at a time: the host has only two), then `glm53-ghost-aggregator`, then `otelcol-contrib` with `force_recreate: true`. Never send a list that names another model, `nginx` or `model-downloader`. Note: `proxy-glm53` and `dcgm-glm53` carry the new `config_variant` in their container labels; they are deliberately not recreated (cosmetic until their next natural recreate; the collector's scrape jobs carry the new value).
7. **Raise gateway caps (cvm-ansible-playbooks #817 or a follow-up), only after every host above is on v7.** Until then base replicas are bounded by the current base host bound (about 40 in flight per replica, 160 per host) and long hosts by 48 per host, so the throughput gain is partly capped. Recommended values are in "Gateway caps".

## Verify after every step

- Dashboard **GLM-5.3 Flash production** (`glm53-flash-prod`) shows the replica under its host and `server_address` within 5 minutes, with the sglang, DCGM and proxy panels populated. If missing, a label changed: stop and diff against the previous tag's file.
- Engine logs (Loki `{host="<host>", container_name="<service>"}`): `preprocess pool started: workers=4 timeout=60s`; `kv_cache_dtype=fp8_e4m3` and both `flashmla_kv` backends; `enable_hierarchical_cache=False` (base) / `True` (long and gpu13); `disable_overlap_schedule=True`; `max_running_requests` 64 (base) / 16 (long); **no `NEAR_PROFILE` lines**.
- `docker inspect` image is the v7 digest; `/backends/list` healthy; one real completion with a cache-hit follow-up on long; a tool-call request with a very deep schema is rejected cleanly (no pool crash); no CUDA 801, NCCL, Xid or OOM; free GPU memory at ready not under 10 GiB.
- Neighbours were not recreated (container IDs and uptime unchanged for every service not in the list).

## Abort and rollback

Abort a step on any engine exit, Xid, OOM, free memory under 2 GiB for 5 minutes, a preprocess-pool timeout storm, or TTFT p95 / ITL p95 clearly worse than the untouched same-tier hosts (the pre-registered criteria of the canary, tee-bench exp 32 `evidence/v7-canary-prod`, apply: any engine exit, Xid or OOM; TTFT or ITL p95 more than 20% worse than the untouched hosts). **Rollback = redeploy the previous tag's copy of the same file for the same services** (`compose/up`, `dry_run` first, the v6 digest and the previous argv), then `otelcol-contrib` with `force_recreate: true`. Gateway caps: revert the three variables. Do not roll back faster than the cold start allows; the 2-base-replicas-down and one-long-host-at-a-time limits apply to rollbacks too.

## Gateway caps (recommendation; not applied here)

File: `vars/openrouter_gateway.yaml` in `nearai/cvm-ansible-playbooks`. Topology: 2 base hosts (gpu03, gpu04) and 3 long hosts (gpu02, gpu23, gpu13). Today: budget 448, long reserve 128, long per-host ceiling 48 (52 in #817), base host bound `(448 - 128) / 2 = 160`, about 40 per base replica, below v7's 64 running.

| Variable | Now (main) | Fleet value | Why |
|---|---|---|---|
| `openrouter_gateway_concurrency` (feeds `VLLM_PROXY_ADMISSION_MAX_INFLIGHT` and `..._START_INFLIGHT`) | 448 | **672** | 2 base hosts x 256 + long reserve 160 |
| `VLLM_PROXY_ADMISSION_LONG_MAX_INFLIGHT_PER_HOST` | 48 (52 in #817) | **64** | 4 replicas x 16 running on gpu02 and gpu23; one ceiling for every long host (gpu13's two replicas hold 32 and stay below it) |
| `VLLM_PROXY_ADMISSION_LONG_RESERVED_INFLIGHT` | 128 | **160** | what the long engines hold: gpu02 64 + gpu23 64 + gpu13 32; below the tier's combined bound 3 x 64 = 192, so it is usable |

The base host bound then derives to `(672 - 160) / 2 = 256` = 4 replicas x 64 running. The gateway's RPM (500) is advertised only and unchanged. These need the lane's three hosts healthy; the saturation steering (`VLLM_PROXY_ADMISSION_QUEUE_SATURATED_AT=4`) still applies. If a long host leaves, lower the reserve by 64 (gpu13: 32).

## Open decisions for a human

- **gpu13 (owner decision)** is outside "1 long config" because its GLM replicas live in `prod/small-models.yaml` beside other models. This PR gives its two GLM replicas exactly the long file's v7 argv/env (tested), HiCache on, with gpu13's own 325 GiB budgets, ports and GPU ids. It ends the shared-KV arm there: its owner (Pranav) must agree before step 6.
- The gateway budget change (448 to 672, reserve 160, ceiling 64) is a capacity decision for the gateway owner.

## Evidence (nearai/tee-bench)

- **exp 26, exp 27 and exp 29 (lab, gpu32):** FP8 KV on stock v6 fails; the v7 patch works (KV pool x1.467, quality equal: GSM8K 0.99, passkey 3/3). Base FP8 at 64 running / 380 slots: +18% tok/s against bf16 prod in the same session. Long FP8 16/6 against bf16 12/4: +10-11% tok/s, +12-13% served, TTFT p95 +18-21%; 1M-token prefill OK. 16/4 (this config) was not measured in the lab and was read in exp 32.
- **exp 30 (wedge RCA):** the long-tier preprocessor wedge (7 events since 2026-09-24); pool plus deadline fix: 962 against 117 tok/s under injected stalls, TTFT p95 1.6 against 119 s.
- **exp 31 (prod cache ceiling):** long tier 84% cached against an 84.1% infinite-cache ceiling, so HiCache stays on for the long tier (and gpu13); base 60% against 66%, so the base tier runs without HiCache.
- **exp 32 (canary):** long r2a overnight: +17% prompt tok/s, +14% generation tok/s, ITL p50 -20%, TTFT p50 -27%. Base r3 (HiCache off): ITL -23%, +14% prompt tok/s, +26-32% prefill work. Base r4: +24% prompt tok/s.
- **Projection:** fleet +11-21% tokens/day if the gateway caps rise (partly capped before that).

## What this replaces

This PR deletes the v7 canary machinery (the two `prod/*V7Canary.yaml` files, their generator and tests, `docs/glm53-v7-canary.md` and the validator rules); the fleet files supersede it. The canary record lives in tee-bench exp 32 (`evidence/v7-canary-prod`). Deleting files does not touch running containers, but note what runs from where today, because each one's next deploy uses the fleet files (redeploy them as part of the rollout, as the order above does):

- gpu03 `r4` and gpu02 `r2a` run from the deleted canary files.
- gpu03 `r3` (HiCache off) and gpu13 `r1a`/`r1b` (shared-KV arm) run from files on Pranav's `pranavraja99/glm53-kvshare-ab` branch, which the fleet files also replace.

`scripts/fixtures/pre-v7/` holds the three files as they were on main before this PR; `scripts/test_glm53_v7_fleet.py` compares the fleet files with them to prove nothing outside the engines, the three allowed telemetry labels and the gpu13 GLM blocks moved.
