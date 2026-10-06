# GLM-5.3 Flash long tier: 2xTP2 memory-optimized rollout to every remaining TP4 replica

Status: PREPARED, nothing deployed by this change. Every stage below needs its own explicit GO, a merged tag that clears compose-manager's commit-age gate, a registered KMS compose hash, and the abort criteria. Builds on the gpu02 canary (#332, `docs/gpu02-glm53-2xtp2-memopt-canary.md`), whose r2 is already two TP2 replicas.

## Which hosts run what (verify before every stage)

| Host | Deploys | Replicas today | Converted by this rollout |
|---|---|---|---|
| gpu02 | `prod/GLM-5.3-Flash-SGL-TP4-W4AFP8-LongContext.yaml` | r1 TP4 (GPUs 0-3); r2 = `tp2-r2a` (4,5) + `tp2-r2b` (6,7) since #332 | r1 at stage 3 |
| gpu23 | same file, own scoped `services` list (per #332, #337 and tee-bench exp 15) | r1 TP4 (0-3), r2 TP4 (4-7) | r2 at stage 1, r1 at stage 2 |
| gpu13 | `prod/small-models.yaml` (shared CVM; Qwen and other small models on GPUs 0-3, untouched) | before: one TP4 service `model-sg-glm53-fp8-tp4` (misnamed; it serves `graphistry/GLM-5.3-Flash-W4AFP8`) on GPUs 4-7, `--disable-overlap-schedule` canary (#330), max-running 32; after: `model-sg-glm53-w4afp8-tp2-r1a` (4,5) + `-r1b` (6,7) | stage 1, with gpu23 r2 |

**gpu23.** `README.md` and `docs/glm53-w4afp8-base-rollout.md` list gpu23 as a base-tier host on `prod/GLM-5.3-Flash-SGL-TP4-W4AFP8.yaml`; that is stale. Prometheus (checked 2026-10-06) shows gpu23 running `model-sg-glm53-w4afp8-tp4-r1` and `-r2` with deployment `glm53-flash-sgl-tp4` and the long-context variants (`fc91d24-long-context-w4afp8-...-pdi1-...` and `...-pdi2-...`, TP4, adaptive EAGLE 5/1/6), i.e. this file. Grafana does not show gpu23's compose-manager scoped `services` list or env map, so **pre-deploy step 0 for any gpu23 stage** is still to read its deployed file, tag, services and env map from compose-manager and confirm there is no `GLM53_BACKEND_URLS` and no leftover HiCache budget override.

## Caps: 12 running / 4 queued on every TP2 replica (user decision)

All TP2 replicas in this PR (gpu02 r2a/r2b, r1a/r1b, gpu13 r1a/r1b) run `--max-running-requests 12 --max-queued-requests 4 --cuda-graph-max-bs-decode 12`; TP4 replicas stay at 32/8/32. The TP2 `config_variant` gains `-mr12q4` so telemetry tells it from the 24/8 that #332 deployed (a 16/4 value was considered in review and superseded). Goal: maintain throughput and cut queue time. This is a prod-evidence change, not a lab one (the lab ran 24/8).

Evidence, busy hour 2026-10-06 13:00-14:00 UTC (Grafana, via the main session):

| | r2a | r2b | gpu02 r1 TP4 | gpu23 r1 | gpu23 r2 |
|---|---|---|---|---|---|
| req/h | 591 | 607 | 590 | 588 | 748 |
| TTFT p95 | 55 s | 64 s | 36 s | 52 s | 33 s |
| queue-time p95 | 39 s | 48 s | 24 s | 37 s | 18 s |
| max running (cap) | 14 (24) | 12 (24) | 17 (32) | 23 (32) | 22 (32) |
| aborts | 26 | 38 | 5 | 35 | 18 |

- **KV-bound, not slot-bound.** While the pair had requests queued, its non-evictable KV (`sglang_kv_used_tokens / sglang_max_total_num_tokens`) was median 0.79-0.86 and max 0.99 with only 6-8 running; the TP4 replicas queued at 0.28-0.35. `max_total_num_tokens` is 1.32M per TP2 replica against 3.52M per TP4, and prompt p50 is about 130K.
- **Last 9 h on gpu02 r2a/r2b (15 s samples, about 2,000 per replica):** running >= 12 only 0.2% / 0.1% of the time and > 16 never; running while queued p90 = 9 / 8 (KV fills at about 8-9 running); queue > 0 in 22% / 20% of samples, > 2 in 10% / 8%, > 4 in 3% / 3%.
- **Why 12.** A cap of 12 almost never binds the engine, so throughput is maintained. It also sharpens the placement signal: cloud-api scores `fullness = (running + queued + pending) / max_running` (`crates/placement/src/score.rs:70`) with no KV term, so at 8 running the pair reads 0.67 instead of 0.50 at cap 16 (and 0.33 at 24), and placement routes away from a KV-full pair sooner. At 24 against TP4's 32 the pair looked about 2x roomier than it was (in the busy hour it took 1198 req/h against r1's 590); at 12 the slot ratio (12/32 = 0.375) equals its KV share (1.32M / 3.52M = 0.375) on paper, though KV actually fills at about 8-9 running, so the mismatch narrows but is not proven closed.
- **Why the queue stays 4, not 2.** A queue of 2 would roughly triple rejection events (queue > 2 in 8-10% of samples against > 4 in 3%). That trades throughput for queue time, and on gpu13 a 503 goes back to OpenRouter on `:8009`, where the request is lost. Queue 2 on gpu02 only is a one-flag follow-up test if the tail is still bad after stage 0a.
- **Host load-back.** The pair loaded back 40-50M tok/h from HiCache host against 2.9M on r1, because its small device cache evicts prefixes. Over 6 h its TTFT p95 was 31-36 s (fleet 29-39 s) and p99 73 s against r1's 46 s.
- **Expected effect (unmeasured; a prod-only placement effect).** The cap binds 0.1-0.2% of the time, so inside the engine little changes; excess beyond 4 queued becomes fast 503s that cloud-api retries elsewhere (`retryable_http_5xx`), and placement should route away earlier, shortening the pair's queue and tail while loading the TP4 replicas a bit more (they have KV headroom). The aborts (26 and 38 in the hour) are not attributed to queue pressure, so no change in them is claimed. Decode graphs at 12 scale capture memory down (the lab's mr32 cost 8% KV against mr24).
- **Follow-up, not in this PR:** KV-aware placement in cloud-api (export used-KV / max-total into the placement snapshot and add it to the score) is the proper fix; revisit the cap once it lands.

## Naming

TP2 replicas are named after the TP4 replica they split (r1 -> r1a/r1b, r2 -> r2a/r2b), so during the staged rollout one shared file can hold a TP4 replica and the TP2 halves of the other without name or instance-label collisions, and each half's GPUs stay inside its parent's GPUs (the validator checks it). gpu13's two TP2 services use the same scheme: `model-sg-glm53-w4afp8-tp2-r1a` and `-r1b` are the two halves of gpu13's single GLM replica (formerly instance 1), with instance labels `1a`/`1b`. **Their GPUs are 4,5 and 6,7, not 0,1 and 2,3 as for r1a/r1b on gpu02 and gpu23**: the `host` label and `gpu_pair` (`4-5`, `6-7`) disambiguate them in dashboards. I checked that nothing keys on the bare service name across hosts: cloud-api has no references to these container or service names, the gpu13 registrar does not register GLM, and the proxy pool and scrape jobs are per host and carry the `host` label. A flat r1..r4 rename can happen in a cleanup PR once every host is TP2 and the TP4 definitions are removed.

## Stages and sequencing

Each stage is its own deploy, independently revertible with a scoped `compose/down` / `compose/up`. One PR carries the file; the stages are operator decisions.

0a. **First: gpu02 r2a/r2b redeploy at 12/4 only (slow rollout, the pair goes first), one replica at a time.** Scoped deploys of this file with no other change: `compose/down` `["model-sg-glm53-w4afp8-tp2-r2a"]`, then `compose/up` `["model-sg-glm53-w4afp8-tp2-r2a"]` (`dry_run` first; env map and `GLM53_BACKEND_URLS` unchanged, so no proxy recreate); wait until r2a is ready, healthy and back in the proxy pool (`/backends/list`) and has served a real long-context completion; only then repeat for r2b. Otherwise the whole pair, which carries about 1200 req/h, is dark at once. Mid-rollout a 24/8 and a 12/4 replica coexist with different `config_variant` labels (`-mr12q4`); read the busy hour only after both are done. Boot and graph capture at 12 with EAGLE 4/1/5 (60 verify tokens per step at full batch) are not verified by this repo, so treat the first restart as the check and hold the second replica until it passes. Startup logs per replica must show `max_running_requests` 12 (not a value reset by speculative decoding; the 2xTP4 file warns it can) and the `Capture cuda graph bs [...]` list reaching 12.

   **Read:** at least 3 weekday busy hours (the 13:00-14:00 UTC baseline above is one hour on one day), split by prompt-length bucket as in stage 0: pair TTFT p95/p99, queue p95, aborts, 503 rate; TP4 replicas' TTFT p95 and running; cloud-api retry success rate (503s from the queue of 4 are now the main new failure mode). **Abort and roll back to 24/8** (previous tag, also one replica at a time with the same scoping) if any of: the fleet long-tier 503 rate rises by more than 2 percentage points over the baseline; cloud-api retry success on those 503s falls below 95%; the pair's TTFT p95 has not fallen by at least 15% against the baseline after 3 busy hours; or TP4 TTFT p95 rises by more than the pair's falls (in seconds). These thresholds are pre-registered here, not derived from data; adjust them before the stage, not after.
0. **Gate (unchanged, now read on 12/4): the gpu02 canary read at 24 h or more, plus the lab precondition.** The only read so far is an early 25 minute, cold-cache window (see the PR); it is not enough. Fill the table (r2a/r2b against r1, spanning a weekday peak) **with TTFT p95 split by prompt-length bucket and the count of requests of 400K tokens or more that the pair has served**. In the early read the pair saw prompt p95 of 206K-245K while r1 saw 537K (affinity keeps the giant conversations on r1), so zero such requests means the largest-prompt risk below is untested, not passed. Add a time-of-day-matched baseline from the same host a week earlier to the stage record (same-host before/after history is the comparison; another host's same-day replica is only a drift check). Without both this and the lab precondition, no stage starts.

   **Lab precondition (hard, not waivable by this runbook):** on gpu32 bare metal with the mem-opt TP2 argv, cold single prefills at 1M and about 800K tokens, and two concurrent 650K. If any OOMs, do not deploy; add the cloud-api `max_context_tokens` cap or lower the chunk size / mem-fraction first, and re-run.
1. **Two independent scoped deploys, either order, each with its own 48 h soak including a weekday peak and a count of requests of 400K tokens or more served:**
   - **1a. gpu23 r2 -> r2a + r2b.** Repeats the gpu02 canary on a second CVM. gpu23 r1 (TP4) is the same-host control, gpu02 r1 the cross-host control.
   - **1b. gpu13 TP4 -> r1a + r1b** (`prod/small-models.yaml`). gpu13 has no same-host control (both halves convert; the pool has no TP4 left), so compare against gpu23 r1 and gpu02 r1 (TP4) in strata of matched per-GPU load, and against gpu13's own prior week at the same time of day. It also ends the #330 overlap-off canary (see below).
2. **gpu23 r1 -> r1a + r1b**, after stage 1a passes its 48 h read. gpu23 is then all TP2. Control is gpu02 r1 (TP4, a different host). Soak at least 48 h including a weekday peak. r1a/r1b run at the reduced budget below.
3. **gpu02 r1 -> r1a + r1b, after gpu23 stage 2 has read clean for at least 72 h.** It removes the last TP4 long-tier replica, so later regressions can only be judged against history and the other hosts' pre-change numbers; the human may hold it longer for that reason. The cost of holding is real: in the early gpu02 read the mixed pool gave the r2 pair about 1.8x r1's requests per GPU (the proxy balances per backend, not per GPU), and the pair's queue time p95 and ITL p95 were worse than r1's (37 s / 22 s vs 8.4 s; 217 / 175 ms vs 96 ms). A uniform TP2 pool removes that imbalance, which argues for converting r1 once the evidence holds, and against a long hold.

**The #330 overlap-off canary ends by user decision.** gpu13's `--disable-overlap-schedule` experiment is stopped when stage 1b converts it (the replicas run with overlap ON), and its result stays **inconclusive**: it went live 2026-10-05 21:01 and its peer set changed about 9 hours later when gpu02 r2 converted. Do not cite it as evidence for or against the overlap scheduler. Because gpu13 changes the fleet's TP4 peer set too, write down a dated timeline of every change (gpu02 r2, gpu23, gpu13) in the stage record.

Rollback for gpu13 restores overlap-off TP4 (see its rollback below), i.e. the previous canary configuration.

The mixed state (TP4 + pair on one host) is the worst case for tails, so stages 1 and 2 are kept short and are not meant to be a long end state; stage 2 starts only after the stage 1a read passes.

Why gpu23 r2 and gpu13 first rather than "gpu23 whole, then gpu02 r1": the evidence is bare metal only (gpu32, CC off) plus one TEE canary on one host. Stage 1 costs one extra soak and buys a second CVM with a live TP4 control beside it before any host loses its control.

## What changes in the file

- Added services (never started unless a scoped services list names them): `model-sg-glm53-w4afp8-tp2-r1a` (GPUs 0,1, `--dist-init-addr 127.0.0.1:29514`, `GLM53_R1A_HICACHE_RAM_BUDGET` default 325GiB) and `-r1b` (GPUs 2,3, port 29515, `GLM53_R1B_HICACHE_RAM_BUDGET`). The argv, environment, image (`47aff7910900`) and labels are the r2a/r2b ones: tp2/ep2, mem 0.86, 330 mamba slots bf16 state, EAGLE fixed 4/1/5, **12 running / 4 queued with decode graphs capped at 12** (see "Caps" below; #332 deployed 24/8), chunk 8192, `--prefill-decode-interval 2`, write_through, no admission reserve, overlap on.
- TP4 r1 and r2 are unchanged byte for byte (unit test). A host not yet converted keeps deploying them, and an unset `GLM53_BACKEND_URLS` is still r1 + r2.
- Scrape jobs `sglang-model-sg-glm53-w4afp8-tp2-r1a/-r1b` with the same `deployment` label and `instance` (`1a`, `1b`), `gpu_pair` (`0-1`, `2-3`) and `config_variant` labels.
- Dist-init ports in the file: r1 29510, r2 29511, 2a 29512, 2b 29513, 1a 29514, 1b 29515 (the validator rejects duplicates). Each TP2 replica must stay inside the GPUs of the TP4 replica it replaces; no two TP2 replicas share a GPU.
- There is one new compose hash for the shared file. Register it with the KMS contract before any deploy of this file on any host; gpu02 and gpu23 are both affected even if their running services do not change.

## Proxy pool value per host and stage

A host whose pool is not in this list must leave `GLM53_BACKEND_URLS` unset. gpu13 is not in the list on purpose: its proxy pool is fixed in `prod/small-models.yaml` (`VLLM_BACKEND_URLS=http://model-sg-glm53-w4afp8-tp2-r1a:8000,http://model-sg-glm53-w4afp8-tp2-r1b:8000`) and there is no env override.

- `gpu02` `r2-pair`: `GLM53_BACKEND_URLS=http://model-sg-glm53-w4afp8-tp4-r1:8000,http://model-sg-glm53-w4afp8-tp2-r2a:8000,http://model-sg-glm53-w4afp8-tp2-r2b:8000`
- `gpu02` `all-tp2`: `GLM53_BACKEND_URLS=http://model-sg-glm53-w4afp8-tp2-r1a:8000,http://model-sg-glm53-w4afp8-tp2-r1b:8000,http://model-sg-glm53-w4afp8-tp2-r2a:8000,http://model-sg-glm53-w4afp8-tp2-r2b:8000`
- `gpu23` `r2-pair`: `GLM53_BACKEND_URLS=http://model-sg-glm53-w4afp8-tp4-r1:8000,http://model-sg-glm53-w4afp8-tp2-r2a:8000,http://model-sg-glm53-w4afp8-tp2-r2b:8000`
- `gpu23` `all-tp2`: `GLM53_BACKEND_URLS=http://model-sg-glm53-w4afp8-tp2-r1a:8000,http://model-sg-glm53-w4afp8-tp2-r1b:8000,http://model-sg-glm53-w4afp8-tp2-r2a:8000,http://model-sg-glm53-w4afp8-tp2-r2b:8000`

(`r2-pair` is the #332 state: r1 TP4 plus the r2 pair. `all-tp2` has no TP4 replica.) The proxy balances by least connections with conversation affinity and has no size-aware routing, so the pair splits arrivals per engine, not per GPU: with `r2-pair`, the two TP2 replicas together receive about twice r1's connections on the same number of GPUs. Compare per GPU, and read latency at equal in-flight requests per GPU, not raw request counts.

## Capacity check for the largest prompts

| | TP4 replica (r1/r2) | TP2 replica (r1a..r2b) |
|---|---|---|
| GPU KV pool | about 3.52M tokens | 1,392,896 tokens (lab boot fact) |
| HiCache host tier | 406 GiB (r1) / 650 GiB (r2) | 325 GiB |
| Free GPU memory at ready | not recorded here | 17.6 GB (lab) |
| Context limit | 1,048,576 | 1,048,576 |

- A 1M-token request fits one TP2 replica (1.39M pool) but leaves about 340K tokens for everything else on it, where a TP4 replica could hold three at once. Two such requests arriving together queue on separate TP2 replicas at best, and at most 12 requests run per replica in any case (`--max-running-requests 12`). Expect more queueing for the very largest conversations, not unservability by KV size.
- **The real open risk is the DSA indexer scratch, not KV.** The generator header records that `deep_gemm.fp8_mqa_logits` asks for an fp32 buffer of chunk x total_context (11.96 GiB at about 392K context with an 8192 chunk), and that `SGLANG_DSA_INDEXER_QSPLIT=1` divides it by TP and "does not bound it". Extrapolating that arithmetic (an estimate, not a measurement): at TP4 the buffer is about 3 GiB at 392K and about 8 GiB at 1M; at TP2 it is about 6 GiB at 392K and about 16 GiB at 1M, against 17.6 GB free at ready. 17.6 GB free is about 16.4 GiB, so the headroom for a single 1M-token cold prefill is under 0.4 GiB before the chunk's activations, EAGLE verify scratch, allocator fragmentation or any concurrent prefill. Treat that request as an expected OOM on TP2, not an edge case. The 17.6 GB is also a product of mem-fraction 0.86; the crash in the header happened with 11.88 GiB free. The lab record in this repo covers passkey 128K on TP2 and 8 concurrent 647K-756K cold prefills only on TP4 (gpu31). **No TP2 run of a roughly 800K-1M cold prefill is recorded.** The lab run in step 0 is a hard precondition; if it OOMs, add a cloud-api cap (for example `max_context_tokens` on the long domain) that sends the over-limit class elsewhere, or lower chunk size / mem-fraction.
- No request class can be shown to remain servable by this repository alone, and the proxy cannot steer large prompts to a bigger replica. The abort OOM trigger below does not make up for skipping the lab run: an OOM costs a restart and a cold 325 GiB HiCache.
- A replica OOM costs more than one request: the engine restarts and its HiCache (up to 325 GiB) is cold, while the sibling pair member carries the load.
- Cold-start: every converted replica starts with an empty prefix and host cache, and conversation affinity maps for the replaced replica are lost. Expect a worse cache hit rate for about the 1 h warm-up.

## Host RAM (HiCache budgets)

Each TP2 replica asks for a fixed 325 GiB (pinned, `cudaMallocHost`). Planned host totals, assuming the other services take little:

| Host state | HiCache RAM |
|---|---|
| gpu02 today: r1 TP4 406 + r2 pair 2 x 325 | 1056 GiB |
| gpu23 today: r1 TP4 406 + r2 TP4 650 | 1056 GiB |
| stage 1a on gpu23: r1 406 + r2 pair 2 x 325 | 1056 GiB (r2 -> pair is RAM-neutral) |
| gpu13 before: one TP4 at 80% | about 830 GiB; after: 2 x 325 = 650 GiB (fits) |
| all TP2 at 4 x 325 | 1300 GiB (+244 GiB over today) |

Measured from the HiCache startup logs (Loki; an earlier note claiming stage 3 was RAM-neutral was wrong, because r1 holds 406 GiB, not 650):
- **gpu23**, 2026-09-29 17:10, r1 TP4 `config=406GiB`, `available_bytes` = 1,393,243,672,576 before r2 started, about 1297.5 GiB available to the engines in total. r2 TP4 (`config=650GiB`) then saw 951,294,345,216 after r1.
- **gpu02**, r2a/r2b startup (2026-10-06): `available_bytes` = 952,050,814,976 (about 886.7 GiB) with r1 (406 GiB) held, so about 1293 GiB in total.

gpu13 (Loki, model-sg-glm53-fp8-tp4, 2026-10-05 21:30:40 UTC): `HiCache startup RAM: config=80% available_bytes=1114550976512 total_budget_bytes=891640781209 ranks=4 rank_budget_bytes=222910195302 reserve_bytes=10737418240`, i.e. about 1038 GiB available to GLM and a 830 GiB budget; two replicas at 325 GiB take 650 GiB and leave about 380 GiB, so gpu13 is RAM-safe at the validated 325 GiB with no reduction.

All TP2 at 4 x 325 GiB + the 10 GiB reserve (gpu02 and gpu23) is 1310 GiB: about 13 GiB over on gpu23 and about 17 GiB over on gpu02. **Stage 1 (r2 -> pair) fits. Stages 2 and 3 do not fit at 325 GiB.** With the other pair holding 650 GiB, the headroom for the two new replicas is about 637 GiB on gpu23 and 633 GiB on gpu02 after the reserve, i.e. at most about 316-318 GiB each with no margin. The plan for stages 2 and 3 is therefore **r1a/r1b at a reduced budget, 2 x 305 GiB (about 25 GiB of margin), set with `GLM53_R1A_HICACHE_RAM_BUDGET` and `GLM53_R1B_HICACHE_RAM_BUDGET`**, re-derived from `MemAvailable` read at the time. This is a variable that differs from the validated 325 GiB (about 6% less host tier on those two replicas): record it as part of the stage, expect a slightly lower host-tier hit rate, and judge the stage against that config. The file's default stays 325.

Before every stage record `MemAvailable` inside the CVM with the services of the next state stopped, and compare with the planned total plus the 10 GiB reserve. gpu23 has no MemAvailable series in Prometheus, so it must be read inside the CVM. The formulas below give the budget; abort rules still apply.

Budget formulas. Read `MemAvailable` with only the services of the **next** state stopped, so it already excludes any replicas that keep running.
- Cold start of all four (`all-tp2` on a host with nothing running): `B = floor((MemAvailable - 10 - 30) / 4)` GiB.
- Stage 2 or 3 (the other pair is already running and holds its 325 GiB): size **only the two new replicas**, `B = floor((MemAvailable - 10 - 30) / 2)` GiB, capped at 325, set via their own `GLM53_R1A/R1B_HICACHE_RAM_BUDGET`. Do not change the budgets of running replicas; change them only if they are restarted anyway, because a later broader `up` would recreate them with the new value.
- If `B` comes out below 85 GiB, **abort the stage**; do not run with a smaller budget (the host tier is an inclusive copy of the 1.39M-token GPU pool, so below that it adds nothing).
A reduced budget is an untested second variable: record it as a separate experiment and do not read its result as the 325 GiB config; expect a lower cache hit rate. Also check the env map for leftover `GLM53_HICACHE_RAM_BUDGET` / `GLM53_R2_HICACHE_RAM_BUDGET` values.

## Deploy steps (one stage at a time; each compose-manager call needs the host's full env map)

Preconditions for every stage: step 0 above for gpu23 (gpu13 reads its own env map as described in stage 1b); merged tag past the commit-age gate; KMS hash registered; the host's complete env map captured with this runbook's variables; `docker/ps`, `/backends/list`, container IDs, and one successful long-domain completion plus a cache-hit follow-up recorded first; `force_recreate: false` except where stated; the lab cold-prefill precondition passed and recorded (no waiver); the env-map dump for the host attached to the stage record and checked to contain no `GLM53_BACKEND_URLS` (gpu23 stage 1) and no stale `GLM53_R1A/R1B/R2A/R2B_HICACHE_RAM_BUDGET` or `GLM53_R2_HICACHE_RAM_BUDGET` override (hard stop otherwise); the KMS hash registered before the first `compose/down` or `dry_run` that uses the new tag; no alert configured on `up == 0` for `sglang-model-sg-glm53-w4afp8-tp2-*` scrape jobs (a gate: otelcol is recreated at every stage and all four jobs exist on both hosts). Never send an unscoped or empty `services` list; whether compose-manager rejects an empty list is unconfirmed, and a `profiles:` guard on the TP2 services is a follow-up pending confirmation of its profile behaviour. A bare `compose/up` of this file would start r1 and r2 beside any pair on the same GPUs. Do not unregister the host to drain it, and pick a low-traffic window.

### Stage 1a: gpu23 r2 -> pair (and, unchanged, the gpu02 canary steps)

1. `compose/down`, previous tag and file, `services: ["model-sg-glm53-w4afp8-tp4-r2"]`. Poll `docker/ps` until gone (5 m grace; never rely on orphan removal). r1 carries the host.
2. `compose/up`, this file, `services: ["model-sg-glm53-w4afp8-tp2-r2a", "model-sg-glm53-w4afp8-tp2-r2b", "proxy-glm53", "otelcol-contrib"]`, `dry_run: true` first (low-traffic window; whether the proxy tolerates starting while pair members are not yet ready is not shown by this repo, so watch the first minutes), env map including the gpu23 `r2-pair` value above. The plan must create r2a, r2b and recreate proxy-glm53 and otelcol-contrib and remove nothing. Apply, then check startup logs: `server_args` as above, `rank_budget_bytes` 325 GiB over two ranks, KV pool about 1.39M tokens, 330 mamba slots, ready before the pool counts it healthy.
3. **Immediately after step 2**, `compose/up` `services: ["nginx"]`, `force_recreate: true` (dry_run first, nginx only). Until it completes, traffic through nginx fails because it still resolves the old proxy address, so run the dry_run before step 2 finishes and apply it as soon as the proxy is up. Then check `/backends/list` **through nginx**.
4. Verify `/backends/list` (r1, r2a, r2b healthy), a real long-domain completion and a cache-hit follow-up, labels `instance="2a"/"2b"`, `gpu_pair`, the new `config_variant`, and unchanged IDs for every other container. The `:8008`/`:8009` soak listeners live in `glm53-soak-relay` (profile `verification`, not nginx) and point at the TP4 service names, which are down after conversion. Keep that relay unstarted; check TP2 replicas with `docker exec` from inside the CVM. Retargeting it is a separate follow-up.

### Stage 1b: gpu13 TP4 -> two TP2 replicas (`prod/small-models.yaml`)

gpu13 is a shared CVM: Qwen and other small models on GPUs 0-3 and the shared `nginx`, `model-proxy-registrar` and `otelcol-contrib` stay as they are. **GLM there is a single replica (GPUs 4-7), so this stage is a full GLM outage on gpu13 from the `compose/down` until the first TP2 replica is ready** (a boot with a 325 GiB pinned HiCache allocation takes minutes). The OpenRouter lane reaches GLM on gpu13 directly on `:8009` (it is not registered with model-proxy), so coordinate a window with its owner before the down. Rollback is an outage of the same kind.

Preconditions: the general ones above, plus capture gpu13's full env map (the file now has one variable per replica, `GLM53_R1A_HICACHE_RAM_BUDGET` and `GLM53_R1B_HICACHE_RAM_BUDGET`, each defaulting to 325GiB; the old shared `GLM53_HICACHE_RAM_BUDGET` (possibly `80%`) no longer applies to GLM there, but remove it anyway. The 1038 GiB available figure was read at startup of the TP4 replica, with the TP4 service not yet holding its pool; re-read `MemAvailable` with GLM stopped), `docker/ps` and the container IDs and `CreatedAt` of every non-GLM service, which must not change.

1. `compose/down`, previous file and tag, `services: ["model-sg-glm53-fp8-tp4"]` only. Poll `docker/ps` until gone. Never rely on orphan removal; the Qwen services are not in the list.
2. `compose/up`, this file, `services: ["model-sg-glm53-w4afp8-tp2-r1a", "model-sg-glm53-w4afp8-tp2-r1b", "proxy-glm53", "otelcol-contrib", "dcgm-glm53"]`, `dry_run: true` first. The plan must create r1a and r1b, recreate proxy-glm53, otelcol-contrib and dcgm-glm53, and remove nothing; no Qwen, FLUX, Whisper, embedding, reranker, registrar or nginx service may appear. Before applying, run `compose config` or read the dry-run output and confirm `SGLANG_HICACHE_RAM_BUDGET=325GiB` appears exactly twice (once per replica). The otelcol-contrib recreate drops scrape and log collection for all gpu13 services for a few seconds (its config changes, so the recreate is required); after it, check that the Qwen scrape jobs resume. Apply. Startup checks per replica: `server_args` as in the file (tp2/ep2, 0.86, 330 mamba slots, EAGLE 4/1/5, 12 running / 4 queued, chunk 8192, pdi 2, overlap ON), `rank_budget_bytes` 325 GiB over two ranks, KV pool about 1.39M tokens, `--dist-init-addr` 29510 and 29511 distinct.
3. `nginx` resolves `proxy-glm53` when it starts, so it must re-resolve the recreated proxy. It is shared with all gpu13 vhosts (Qwen traffic included), and compose's force-recreate also recreates the `depends_on` closure unless dependencies are excluded (nginx depends on `proxy-glm53` and the Qwen, FLUX, embedding, reranker and Whisper proxies, `prod/small-models.yaml`). Prefer an `nginx -s reload` if the operator can issue one (it re-resolves names without dropping connections). Otherwise use `compose/up` `["nginx"]` with `force_recreate: true` **only with `no_deps` if compose-manager supports it**, and abort if the `dry_run` plan shows anything other than the single nginx container (any `proxy-*` in it breaks the Qwen-untouched claim). Low-traffic window; verify a Qwen request afterwards.
4. Verify: `/backends/list`-equivalent for the pool (both TP2 replicas healthy through `proxy-glm53`), a real long-context completion and a cache-hit follow-up on `:8009`, metrics carry `instance="1a"/"1b"`, `gpu_pair`, and the new `config_variant` (no `overlap-off`), and every Qwen/small-model container keeps its ID and `CreatedAt`.

### Stage 2: gpu23 r1 -> pair

1. `compose/down`, previous tag, `services: ["model-sg-glm53-w4afp8-tp4-r1"]`; poll until gone. The r2 pair carries the host.
2. `compose/up`, this file, `services: ["model-sg-glm53-w4afp8-tp2-r1a", "model-sg-glm53-w4afp8-tp2-r1b", "proxy-glm53", "otelcol-contrib"]`, `dry_run` first, env map switched to the gpu23 `all-tp2` value and `GLM53_R1A_HICACHE_RAM_BUDGET=305GiB`, `GLM53_R1B_HICACHE_RAM_BUDGET=305GiB` (re-derived from `MemAvailable`; see Host RAM). Plan: create r1a, r1b; recreate proxy-glm53 and otelcol-contrib; remove nothing. Same startup checks (dist-init ports 29514/29515, GPUs 0-3).
3. nginx recreate as in stage 1a. Verify as in stage 1a with four healthy backends.

### Stage 3 (held): gpu02 r1 -> pair

Same as stage 2 with the gpu02 `all-tp2` value and the same reduced r1a/r1b budgets, after the decision above. The same-host control disappears: judge against the pre-change history and against gpu23 pre-change numbers, and keep the replica logs.

## Abort criteria

Pre-registered before the stage, evaluated after a 1 h cache warm-up. The proxy balances per backend, not per GPU, so in a mixed stage (TP4 beside a pair) the TP4 control is under-loaded: the pair receives about 1.6-1.8x the requests per GPU in the early gpu02 read (mean running 9.0 on the pair's 4 GPUs against 5.5 on r1's 4). Latency and queue triggers are therefore **compared in strata of matched per-GPU running requests** (for example 0-2, 2-4, 4-6 running per GPU), never on window means; if too few samples fall in a stratum, extend the window rather than judge it. Do not read a mixed-pool latency gap as engine quality. The alternative, weighting the control at 2x in the proxy for the soak, needs a proxy change and is not part of this PR.

Roll back that stage on:

- any exit or restart of a converted replica, or any Xid. **Owner:** the operator running the stage reads the host's Xid log every 6 h during the soak (Grafana does not show Xid);
- any CUDA OOM in the DSA indexer (`fp8_mqa_logits`), or any request of 400K tokens or more failing with 5xx other than 503 queue-full;
- 5xx other than 503 queue-full above 0.5% of requests;
- TTFT p95 for prompts of 64K tokens or more more than **60%** above the control in the same per-GPU-load stratum over 60 minutes. The lab predicts +36% from slower 256K+ prefill on one TP2, so the trigger sits above the expected cost; a smaller miss is a finding, not an abort;
- ITL p95 more than 2x the control in the same stratum over 30 minutes. The early read already shows ITL p95 175-217 ms against 96 ms in an unmatched comparison, which differs from the lab's "same ITL p95"; the stratified read decides whether that is load or the engine.

Queue-full 503s are **not** an abort or a success criterion at current load: the engines run about 5 of their cap with about 1 queued, so there is little to reduce. The lab's -84% rejections came from saturated engines and are re-checked only if a stage sees saturation.

## Rollback (per stage, scoped; the other stages are untouched)

**gpu13 (stage 1b)**: `compose/down` this file with `services: ["model-sg-glm53-w4afp8-tp2-r1a", "model-sg-glm53-w4afp8-tp2-r1b"]` and poll until gone; `compose/up` the previous file and tag with `services: ["model-sg-glm53-fp8-tp4", "proxy-glm53", "otelcol-contrib", "dcgm-glm53"]`, `dry_run` first (this restores the TP4 overlap-off canary and a single-backend pool); nginx re-resolve as in stage 1b; then verify a long-context completion on `:8009` and a Qwen request. **Restore the captured pre-stage env map, including the HiCache budget variables** (the TP4 file defaults to 80% and would silently take a smaller budget if a 325GiB value were left set). The 5 m `stop_grace_period` can stretch both downs, so poll `docker/ps` with a timeout of at least 6 minutes. GLM on gpu13 is down for the whole swap. The other stages below do not apply to gpu13.

Stages on the shared long-context file:

1. `compose/down`, this file, `services` = exactly the converted replicas of that stage (stage 1: `["model-sg-glm53-w4afp8-tp2-r2a", "model-sg-glm53-w4afp8-tp2-r2b"]`; stage 2/3: `["model-sg-glm53-w4afp8-tp2-r1a", "model-sg-glm53-w4afp8-tp2-r1b"]`); poll until gone. The remaining replicas carry the host.
2. Set `GLM53_BACKEND_URLS` explicitly (the env map is outside the repo, so nothing here can catch a stale value, and a leftover `r2-pair` would silently exclude the recreated TP4 r2 and leave GPUs 4-7 idle):
   - gpu23 stage 1a rollback: **unset** the variable.
   - gpu23 stage 2 rollback: the gpu23 `r2-pair` value.
   - gpu02 stage 3 rollback: the gpu02 `r2-pair` value.
   - gpu02 canary rollback (#332): unset, or r1 + r2.
   Also remove any `GLM53_R1A/R1B/R2A/R2B_HICACHE_RAM_BUDGET` overrides set for the rolled-back replicas.
3. `compose/up` `services: ["model-sg-glm53-w4afp8-tp4-r2", "proxy-glm53", "otelcol-contrib"]` for stage 1 (or `tp4-r1` for stages 2 and 3), `dry_run` first; the TP4 replica re-warms its HiCache. Then `compose/up` `["nginx"]` with `force_recreate: true`.
4. Verify `/backends/list` shows **exactly** the expected URLs for the rolled-back state, a long-domain completion and a cache-hit follow-up. Keep the replica logs. Stages already done on other hosts are not affected; pause the rest of the rollout until the cause is understood.

## Monitoring notes

- gpu13's collector (`prod/small-models.yaml`) loads only its own two jobs. gpu02's and gpu23's collectors load all four TP2 jobs from the shared file, so `up == 0` appears for jobs whose services do not run on that host. Check that no alert fires on down scrape targets for `sglang-model-sg-glm53-w4afp8-tp2-*` before stage 1.
- The open PR #337 (ghost cache and `-obs-v1` variants) edits the same generator and generated file. Whichever lands second must regenerate and re-pin the variant strings (the validator pins exact argv, environment and variant), and the four TP2 services need its ghost environment; deploying this PR's hash before #337 means the TP2 replicas run without it. Preferred order: #337 first, then rebase this PR.
