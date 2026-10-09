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
| running / queued / decode graphs | 64 / 32 / 64 | 16 / 8 / 16 |
| mamba slots | 380 (5 per running request) | 330 |
| overlap scheduler | off (`--disable-overlap-schedule`) | off |
| HiCache | **OFF**: the four flags `--enable-hierarchical-cache --hicache-write-policy --hicache-io-backend --hicache-mem-layout` are not in the argv | **ON** (`write_through`, 325 GiB per replica as today) |
| environment | `SGLANG_PREPROCESS_WORKERS=4`, `SGLANG_PREPROCESS_TIMEOUT_S=60`, `SGLANG_PREPROCESS_LOG_SLOW_S=5`, `SGLANG_TOOL_SCHEMA_MAX_DEPTH=32`, `SGLANG_TOOL_SCHEMA_MAX_NODES=25000` | same |
| profiling | **off**: no `NEAR_SELF_PROFILE*` anywhere | off |
| `precision` / `engine_image` | `int4-weights-fp8-activations-fp8-kv` / `fa730e6e62b2` | same |
| `config_variant` | `w4afp8-qsplit-hicacheoff-mamba380-fp8kv-memopt086-mr64-...-obs-v1-v7` | `...-memopt-fp8kv-mamba330-...-mr16q8-strict-budget8192-obs-v1-v7` |

**How HiCache is off on the base tier (copied from gpu03 r3, arm B' of the 2026-10-08 online test, `pranavraja99/glm53-kvshare-ab`):** only the four HiCache flags are removed from the replica argv. Nothing else moves: the HiCache environment (`SGLANG_HICACHE_RAM_BUDGET`, `..._CUDA_HOST_MEMORY`, `..._POOLED_TRANSFERS`, `..._STAGING_PAGES`) stays and is unread without `--enable-hierarchical-cache`; mem fraction, mamba slots and labels are the v7 base values above. The generated base argv is asserted equal to r3's live argv in `scripts/test_glm53_v7_fleet.py`.

**Long file structure:** the TP4 `r1`/`r2` services, the proxy pool (`${GLM53_BACKEND_URLS:-...}`), sidecars, nginx, registrar, the ghost aggregator and otelcol are unchanged except: the four TP2 services (v7 argv/env/labels) and the ghost aggregator image (now v7, so it follows its engines). The TP4 services stay on v6 (kept only so a host not on TP2 still deploys; the fleet does not run them). Each host's `GLM53_BACKEND_URLS` is already set in its compose-manager env map for its stage; **do not change it** (verify it names the four TP2 replicas before the host's first step).

**Dashboard labels (Grafana `glm53-flash-prod`) are unchanged:** `deployment` (`glm53-flash-sgl-tp2x4` base, `glm53-flash-sgl-tp4` long), `host_machine`, `host`, `service`, `server_address` (the scrape target), `model`, `instance`, `service=dcgm-exporter`, `service=vllm-proxy`, and the DCGM, vllm-proxy, ghost-aggregator and otelcol scrape jobs. Only `precision`, `engine_image` and `config_variant` change. `scripts/test_glm53_v7_fleet.py` compares every service and scrape job with the pre-rollout files.

## Deploy (compose-manager only, no KMS step, no env-map changes)

**Pace: one replica per lane per step, two lanes in parallel, each step followed by a 5-minute bake.** At most one base replica and one long replica are being recreated at any time (the base lane and the long lane may run concurrently: one base plus one long in flight). After a replica is recreated and ready, bake it for 5 minutes against the checklist below; only a pass releases the next replica of that lane. The lanes do not wait for each other.

Every step is `POST http://<host>:8080/compose/up` with `tag` = the merged tag `T` (past `MIN_TAG_AGE_HOURS`), `file` (with the `prod/` prefix), an **explicit `services` list holding exactly that one replica** (never empty, never two replicas), the host's existing full `env` map, `force_recreate: false`, **`dry_run: true` first**. The plan must show exactly that one service recreated and nothing else (no `model-downloader`, proxy, nginx, registrar, other engine, no `--remove-orphans` removals); if it shows more, stop. Then repeat with `dry_run: false`: exactly one recreate. Cold start is about 22 minutes under CC; the 5-minute bake starts when the replica is ready (`/health` 200), not when the call is sent. Pre-check before each step: `docker/ps` and `/backends/list` healthy on the host, every other replica of the tier up, and the dump of the host's env map kept as the rollback reference.

### Limits (user rules)

- **Never more than one long replica down at a time** (this is why the long lane is strictly serial, replica by replica, and finishes a host before the next host starts).
- **At most one base replica down fleet-wide** (tighter than the earlier "2 base replicas down"): the base lane is strictly serial across gpu04 and gpu03.
- **One base replica and one long replica may be in flight together**, never two of the same tier.
- **`compose/down` is never used.** It takes effect immediately, ignores `dry_run` and cannot be recalled; `compose/up` of the one service recreates it. If one is ever sent anyway, scope it to one service and verify the logged services list in the host's compose-manager action log.
- Explicit services list on every call; no KMS step; no env-map changes (including `GLM53_BACKEND_URLS`).

### Order: two lanes

Service names: base `model-sg-glm53-w4afp8-tp2-r1`, `model-sg-glm53-w4afp8-tp2-r2`, `model-sg-glm53-w4afp8-tp2-r3`, `model-sg-glm53-w4afp8-tp2-r4` (below `r1`..`r4`); long `model-sg-glm53-w4afp8-tp2-r1a`, `-r1b`, `-r2a`, `-r2b`.

**Base lane** (`file: prod/GLM-5.3-Flash-SGL-TP2x4-W4AFP8.yaml`), one replica, then bake, then the next:

`gpu04 r1 -> gpu04 r2 -> gpu04 r3 -> gpu04 r4 -> gpu03 r1 -> gpu03 r2 -> gpu03 r3 -> gpu03 r4`

(gpu03 r3 and r4 are already v7-like, r3 HiCache off and r4 with profiling; they are redeployed to the fleet file so labels, environment and profiling match, and are baked like the others.)

**Long lane** (`file: prod/GLM-5.3-Flash-SGL-TP4-W4AFP8-LongContext.yaml`, except gpu13), one replica, then bake, then the next:

`gpu23 r1a -> gpu23 r1b -> gpu23 r2a -> gpu23 r2b -> gpu02 r1a -> gpu02 r1b -> gpu02 r2a -> gpu02 r2b -> gpu13 r1a -> gpu13 r1b`

(gpu02 r2a currently runs the canary and is redeployed to the fleet file.) gpu13 uses `file: prod/small-models.yaml` and the services list contains **only that one GLM engine**, so its other models (Qwen, FLUX, Qwen3-VL, whisper and the rest) stay up; never send a list that names another model, `nginx` or `model-downloader`. **gpu13 ends Pranav's shared-KV experiment; its owner must be told before gpu13 r1a.** `proxy-glm53` and `dcgm-glm53` carry the new `config_variant` in their container labels; they are deliberately not recreated (cosmetic until their next natural recreate).

**Per-host collector and aggregator, once per host after its last replica has baked green:** `services: ["glm53-ghost-aggregator"]` (image now v7), then `services: ["otelcol-contrib"]` with `force_recreate: true` (dry run first; plan: that service only). In the long lane these run between the host's last replica and the next host's first; in the base lane after gpu04 r4 and after gpu03 r4. They are not replicas and do not count as a lane step, but they never overlap another recreate on the same host. **Before this recreate the metrics are already visible:** the dashboard labels are unchanged (see above), so the running collector keeps scraping each recreated replica by its `server_address` and the sglang, DCGM and proxy panels populate during the bake; only the `precision`, `engine_image` and `config_variant` label values (and the ghost aggregator's series) stay on their old values until the collector and aggregator are recreated. Do not wait for them to bake a replica, and verify `config_variant` ends `-v7` for the host's replicas after the recreate.

**Gateway caps (see "Gateway caps") are raised only after both lanes have finished.** Until then base replicas are bounded by the current base host bound (about 40 in flight per replica, 160 per host) and long hosts by 48 per host, so the throughput gain is partly capped.

## 5-minute bake (after every replica; copy-pasteable where possible)

Run on the replica `<svc>` of host `<host>` with `<addr>` = its scrape target (`server_address`, `<host-ip>:<port>`). Baseline = the same tier's **not-yet-migrated** replicas over the same 5 minutes (canary-era replicas gpu03 r3/r4 and gpu02 r2a do not count as baseline); for the last replica of a tier, where no unmigrated peer remains, use the tier's pre-rollout figures captured over the hour before the first step. Check the metric names against the `glm53-flash-prod` panel queries if they differ.

1. **Ready and routed.** `curl -s -o /dev/null -w '%{http_code}\n' http://<addr>/health` is `200`; `/backends/list` on the host shows `<svc>` ready (the proxy marked it ready); `docker ps --filter name=<svc> --format '{{.Image}} {{.Status}}'` shows the v7 digest and an uptime growing through the bake (no restart loop: `docker inspect -f '{{.RestartCount}}' <svc>` stays 0 and `Status` never resets).
2. **Dashboard.** Grafana `glm53-flash-prod` shows `<svc>` under its `host` and `server_address` within the bake, with the sglang, dcgm and proxy panels populated. Missing means a label changed: stop and diff against the previous tag's file.
3. **Engine startup lines** (Loki `{host="<host>", container_name="<svc>"}`, or `docker logs <svc> 2>&1 | grep -E '...'`): `preprocess pool started: workers=4 timeout=60s`; `kv_cache_dtype=fp8_e4m3` with both `flashmla_kv` backends; `enable_hierarchical_cache=False` for base / `True` for long and gpu13; `disable_overlap_schedule=True`; `max_running_requests` 64 (base) / 16 (long); no `NEAR_PROFILE` lines: `docker logs <svc> 2>&1 | grep -c NEAR_PROFILE` is `0`.
4. **It serves production traffic.** Over the 5 minutes, with `sel='server_address="<addr>"'`: `increase(sglang_num_requests_total{$sel}[5m]) > 0` and requests complete (`increase(sglang_e2e_request_latency_seconds_count{$sel}[5m]) > 0`); `max_over_time(sglang_num_running_reqs{$sel}[5m]) > 0` at some point; `increase(sglang_generation_tokens_total{$sel}[5m]) > 0` and prompt tokens increasing too.
5. **Latency vs peers.** TTFT p95 and ITL p95 not more than 20% worse than the baseline: `histogram_quantile(0.95, sum by (le) (rate(sglang_time_to_first_token_seconds_bucket{$sel}[5m])))` against the same expression over the unmigrated replicas of the tier (`server_address=~"<peer1>|<peer2>|..."`), and the same with `sglang_inter_token_latency_seconds_bucket`. Fail at more than 1.2 x baseline.
6. **Clean logs.** `docker logs --since 5m <svc> 2>&1 | grep -Ei 'Xid|CUDA error|CUDA 801|NCCL|out of memory|Traceback|preprocess.*timeout|worker died'` prints nothing; free GPU memory at ready is not under 10 GiB.
7. **Neighbours untouched.** Container IDs and uptime are unchanged for every service not in the list; on long, one real completion with a cache-hit follow-up, and a tool-call request with a very deep schema is rejected cleanly (no pool crash).

**Pass -> next replica of that lane. Fail -> roll that one replica back** (`compose/up` of that service from the previous tag, `dry_run` first; see below) **and stop the lane**; the other lane may continue unless the cause is shared (image, gateway, collector), in which case stop both and escalate.

## Abort and rollback

Abort a step on any bake-checklist failure, and immediately on an engine exit, Xid, OOM, free memory under 2 GiB for 5 minutes, or a preprocess-pool timeout storm. TTFT p95 / ITL p95 more than 20% worse than the unmigrated same-tier replicas is the same pre-registered criterion as the canary (tee-bench exp 32 `evidence/v7-canary-prod`). **Rollback = redeploy the previous tag's copy of the same file for that one service** (`compose/up`, `dry_run` first, the v6 digest and the previous argv), then, if the host's collector was already recreated, `otelcol-contrib` with `force_recreate: true` again. Gateway caps: revert the three variables. Do not roll back faster than the cold start allows; the one-long-replica and one-base-replica limits apply to rollbacks too.

## Base queue cap 8 -> 32 (follow-up change)

The base replicas (r1-r4 of `prod/GLM-5.3-Flash-SGL-TP2x4-W4AFP8.yaml`) now run `--max-queued-requests 32` instead of 8; every other flag, the image and the long tier (16 running; its queue cap is changed in the next section) are unchanged. Reason: with a queue of 8, SGLang answered "The request queue is full." (503 sent inside an HTTP 200 SSE stream) on 4-10% of base requests while engines ran only ~20-26 of 64 slots, and the gateway's first-event peek turned each into a 503 and marked the host backpressured for 10 s. 32 is half the 64 running slots; queued requests hold no KV until scheduled.

Rollout: compose-manager `compose/up` with the merged tag, one base replica at a time with the same 5-minute bake (`dry_run` first, `services` list of one replica). Rollback is the previous tag. No KMS step and no env-map change. Watch TTFT p95 against the +20% abort line above and expect queue-full rejections and `backend_queue` gateway rejections to drop. The gateway's `VLLM_PROXY_ADMISSION_QUEUE_SATURATED_AT=8` (cvm-ansible-playbooks) is untouched: with a 32-deep queue it now marks a base host only when the queue is a quarter full.

## Long queue cap 4 -> 8 (follow-up change)

The four long TP2 replicas (r1a, r1b, r2a, r2b of `prod/GLM-5.3-Flash-SGL-TP4-W4AFP8-LongContext.yaml`, gpu02 and gpu23) and gpu13's two GLM replicas (`model-sg-glm53-w4afp8-tp2-r1a/r1b` in `prod/small-models.yaml`) now run `--max-queued-requests 8` instead of 4; running stays 16 and every other flag, the image and the TP4 r1/r2 services are unchanged. The `mr16q4` component of the long `config_variant` and gpu13's `max_queued_requests` labels become `mr16q8` / `8`; the dashboard selector labels are unchanged. Reason: the gateway logged 808 long-tier "The request queue is full." rejections in 5 h (gpu13 352, gpu23 230, gpu02 226; about 5% of long requests) while replicas were not out of slots (running p90 5.7, max 13 of 16; KV p90 35%): the queue holds requests waiting behind 8K-token chunked prefill, and per-host conversation affinity concentrates bursts on one replica.

Rollout: compose-manager `compose/up` with the merged tag, one long replica at a time with the same 5-minute bake (`dry_run` first): gpu23 r1a, r1b, r2a, r2b, then gpu02, then gpu13 r1a, r1b (`services` = that one GLM replica, so gpu13's other models stay up). Rollback is the previous tag. No KMS step and no env-map change. Watch long TTFT p95 against the same hour on the previous day (+20% line) and long queue-full errors; expect the rejections to fall to near zero and the queue-time / TTFT tail to rise somewhat (lab: about 18-21% TTFT p95 at saturation; less expected with spare long capacity). The gateway's `VLLM_PROXY_ADMISSION_QUEUE_SATURATED_AT=8` (cvm-ansible-playbooks) is untouched: at queue 8 it can now mark a long host saturated when its sampled replica's queue is full, which never fired for long while the cap was 4.

## Gateway caps (recommendation; not applied here)

File: `vars/openrouter_gateway.yaml` in `nearai/cvm-ansible-playbooks`. Topology: 2 base hosts (gpu03, gpu04) and 3 long hosts (gpu02, gpu23, gpu13). Today: budget 448, long reserve 128, long per-host ceiling 48 (52 in #817), base host bound `(448 - 128) / 2 = 160`, about 40 per base replica, below v7's 64 running.

| Variable | Now (main) | Fleet value | Why |
|---|---|---|---|
| `openrouter_gateway_concurrency` (feeds `VLLM_PROXY_ADMISSION_MAX_INFLIGHT` and `..._START_INFLIGHT`) | 448 | **672** | 2 base hosts x 256 + long reserve 160 |
| `VLLM_PROXY_ADMISSION_LONG_MAX_INFLIGHT_PER_HOST` | 48 (52 in #817) | **64** | 4 replicas x 16 running on gpu02 and gpu23; one ceiling for every long host (gpu13's two replicas hold 32 and stay below it) |
| `VLLM_PROXY_ADMISSION_LONG_RESERVED_INFLIGHT` | 128 | **160** | what the long engines hold: gpu02 64 + gpu23 64 + gpu13 32; below the tier's combined bound 3 x 64 = 192, so it is usable |

The base host bound then derives to `(672 - 160) / 2 = 256` = 4 replicas x 64 running. The gateway's RPM (500) is advertised only and unchanged. These need all five hosts healthy; the saturation steering (`VLLM_PROXY_ADMISSION_QUEUE_SATURATED_AT=4`) still applies. If a long host leaves, lower the reserve by 64 (gpu13: 32).

## Open decisions for a human

- **gpu13 (owner decision)** is outside "1 long config" because its GLM replicas live in `prod/small-models.yaml` beside other models. This PR gives its two GLM replicas exactly the long file's v7 argv/env (tested), HiCache on, with gpu13's own 325 GiB budgets, ports and GPU ids. It ends the shared-KV arm there: its owner (Pranav) must agree before gpu13 r1a.
- The gateway budget change (448 to 672, reserve 160, ceiling 64) is a capacity decision for the gateway owner.

## Evidence (nearai/tee-bench)

- **exp 26, exp 27 and exp 29 (lab, gpu32):** FP8 KV on stock v6 fails; the v7 patch works (KV pool x1.467, quality equal: GSM8K 0.99, passkey 3/3). Base FP8 at 64 running / 380 slots: +18% tok/s against bf16 prod in the same session. Long FP8 16/6 against bf16 12/4: +10-11% tok/s, +12-13% served, TTFT p95 +18-21%; 1M-token prefill OK. 16/4 (this config) was not measured in the lab and was read in exp 32.
- **exp 30 (wedge RCA):** the long-tier preprocessor wedge (7 events since 2026-09-24); pool plus deadline fix: 962 against 117 tok/s under injected stalls, TTFT p95 1.6 against 119 s.
- **exp 31 (prod cache ceiling):** long tier 84% cached against an 84.1% infinite-cache ceiling, so HiCache stays on for the long tier (and gpu13); base 60% against 66%, so the base tier runs without HiCache.
- **exp 32 (canary):** long r2a overnight: +17% prompt tok/s, +14% generation tok/s, ITL p50 -20%, TTFT p50 -27%. Base r3 (HiCache off): ITL -23%, +14% prompt tok/s, +26-32% prefill work. Base r4: +24% prompt tok/s.
- **Projection:** fleet +11-21% tokens/day if the gateway caps rise (partly capped before that).

## What this replaces

This PR deletes the v7 canary machinery (the two `prod/*V7Canary.yaml` files, their generator and tests, `docs/glm53-v7-canary.md` and the validator rules); the fleet files supersede it. The canary record lives in tee-bench exp 32 (`evidence/v7-canary-prod`). Deleting files does not touch running containers, but note what runs from where today, because each one's next deploy uses the fleet files (redeploy them as part of the rollout, as the lanes above do):

- gpu03 `r4` and gpu02 `r2a` run from the deleted canary files.
- gpu03 `r3` (HiCache off) and gpu13 `r1a`/`r1b` (shared-KV arm) run from files on Pranav's `pranavraja99/glm53-kvshare-ab` branch, which the fleet files also replace.

`scripts/fixtures/pre-v7/` holds the three files as they were on main before this PR; `scripts/test_glm53_v7_fleet.py` compares the fleet files with them to prove nothing outside the engines, the three allowed telemetry labels and the gpu13 GLM blocks moved.
