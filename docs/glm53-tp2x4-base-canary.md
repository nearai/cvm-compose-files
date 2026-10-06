# GLM-5.3 Flash base tier: 4x TP2 canary (one host, gpu03 or gpu04)

`prod/GLM-5.3-Flash-SGL-TP2x4-W4AFP8.yaml` runs **four TP2/EP2 replicas** on one 8x H200 base-tier host instead of the two TP4/EP4 replicas of `prod/GLM-5.3-Flash-SGL-TP4-W4AFP8.yaml`. That TP4 file is the rollback. The new file is generated from it by `scripts/prepare_glm53_w4afp8_tp2x4.py`; do not hand-edit it. **Status: prepared, not deployed. Gates 1–4 below are open.**

## What changes

- **Engines.** Four services `model-sg-glm53-w4afp8-tp2-r1` … `-r4` on GPUs `0,1` / `2,3` / `4,5` / `6,7`. Each pair sits inside one NVLink island.
- **Image.** The HiCache + W4AFP8 image the long tier runs, `docker.io/nearaidev/sglang@sha256:47aff791090003a37f893e998c44794c410d3f7bdfc7fdd2dfab5eb5592b30bb` (`docker/sglang-glm53-hicache-w4afp8`, original #308 offloop-v3).
- **Argv.** `--tp-size 2 --ep-size 2`, `--chunked-prefill-size 8192` (the lab-qualified TP2 argv), plus `--enable-hierarchical-cache --hicache-write-policy write_through_selective --hicache-io-backend direct --hicache-mem-layout page_first_direct`. Everything else is unchanged: `--max-prefill-tokens 32768`, pdi 1, 32 running, 8 queued.
- **Environment.** The admission reserve is unchanged (4096 / 0.75). The file adds:
  - `SGLANG_HICACHE_RAM_BUDGET=${GLM53_HICACHE_RAM_BUDGET:-325GiB}` per replica (4 × 325 = 1,300 GiB);
  - `SGLANG_HICACHE_CUDA_HOST_MEMORY=1`, `SGLANG_HICACHE_POOLED_TRANSFERS=1` and `SGLANG_HICACHE_STAGING_PAGES=64`;
  - `SGLANG_DSA_INDEXER_QSPLIT=1`.
- **Fan-out.**
  - The proxy pools all four engines, with affinity unchanged.
  - The collector has one scrape job per engine.
  - The perception check loops over r1..r4.
  - The soak relay maps `:8008`–`:8011` to r1..r4.
- **Telemetry.** Prometheus scrape labels use `deployment="glm53-flash-sgl-tp4"`, so existing dashboards and alerts include the host, plus `topology="tp2x4"` to split out the canary. Container labels and log tags keep `glm53-flash-sgl-tp2x4`. `config_variant` is `hicache-w4afp8-qsplit-selective325-c8192-admission-reserve-v10-pdi1-h200-tp2-ep2-eagle-adaptive-5-1-6-strict-budget8192` and `engine_image` is `47aff7910900`.
- **Unchanged.** Domains, nginx, the registrar (except one log line), the downloader, DCGM and the OTel pipeline. `scripts/validate_glm53_prod_config.rb` enforces this.

## Evidence

The lab ran on gpu31/gpu32 with CC off, on 2026-10-01. Both arms had 1,300 GiB of DRAM per 8 GPUs. Traffic was base-tier synthetic traffic from the 2026-09-17 prod snapshot, through a least-conn + affinity router.

| Per 8 GPUs | 4x TP2 | 2x TP4 |
|---|---|---|
| Peak served throughput | 8.18 req/s, 1,871 out tok/s | 6.49 req/s, 1,545 out tok/s (+26% / +21% for TP2) |
| Unserved backlog at 8 req/s offered | 4.7% | 18% |
| Unserved backlog at 10 req/s offered | 16% | 37% |
| HTTP errors | 0 | 0 |
| Hit rate, synthetic | 84–85% | 84–85% |
| Hit rate, agent-trace replay (96 agents) | 93.1% | 93.2% |

- **TTFT.** It is lower for prompts of 10K tokens or fewer at every load. Long prompts (30K+) are slower at low load, because TP2 prefills with half the GPUs.
- **Memory per replica.** TP2 holds 81.7 GiB of weights per GPU and a 1.08M-token VRAM KV pool. Its 325 GiB DRAM tier holds 5.87M tokens.
- **Write policy.** `write_through_selective` (vs `write_through`) cut recomputed tokens by 17–25% on the replay, with no throughput or ITL cost.
- **Running cap (second canary step).** With the default 77 mamba state slots, each TP2 replica caps at 15 running requests; that is what `v0.0.462` runs on gpu04. This file now sets `--max-mamba-cache-size 165 --mamba-ssm-dtype bfloat16`: 32 running per replica, with the VRAM KV pool kept at about 1.0M tokens and a 5.4M-token DRAM tier.

  | 10 req/s offered, per 8 GPUs (lab, gpu32) | 15 running (gpu04 today) | 25 running (125 slots, FP32 state) | **32 running (165 slots, BF16 state)** |
  |---|---|---|---|
  | VRAM KV pool per replica | 1.08M | 0.50M | **1.00M** |
  | Served req/s / output tok/s | 7.74 / 1,771 | 8.64 / 2,027 | **8.83 / 2,064 (+14% / +16%)** |
  | Unserved at run end | 925 | 356 | **228** |
  | TTFT p50 / p90 | 1.60 / 5.56 s | 0.43 / 3.89 s | **0.39 / 3.20 s** |
  | TPOT p50 / p90 | 26 / 70 ms | 38 / 169 ms | 39 / **188 ms** |
  | E2E p50 / p90 | 3.80 / 20.5 s | 3.04 / 21.5 s | 2.89 / 21.1 s |
  | Hit rate | 71.5% | 75.4% | 76.8% |

  - All three saturate at about 8–8.8 req/s; beyond that they only queue. The base tier is prefill-heavy, so bigger decode batches help less than 2×.
  - **Cost: tail TPOT 70 → 188 ms** (each decode step carries a larger batch).
  - Rejected: 165 slots with FP32 state left a 67K-token pool (3.1 req/s served at 8 offered); raising `--mem-fraction-static` to 0.88 instead OOMed under load.
  - EAGLE target-verify graphs follow the running cap.

## Gates before any deploy

**For the 32-running change (on top of the live `v0.0.462` canary):**
1. **BF16 mamba-state quality: PASSED (2026-10-02).** GSM8K 97.8% vs 97.4% (FP32-state TP2) and 97.5% (TP4); perception check 7/7; agent-trace replay 93.4% cached, TTFT p90 1.50 s (FP32-state 15-running: 93.1%, 1.54 s).
2. **Product call** on the TPOT trade-off (tail TPOT about 2.7× worse, TTFT and throughput better).
3. **Rollout (both base hosts already run the 15-running TP2 file since 2026-10-02).** One host at a time, in the lowest-traffic window:
   - A TP2 host cold-starts in about 22 minutes (pinning 4 x 325 GiB under CC plus warm-up). For that whole window the other base host carries the entire base tier and saturates (seen 2026-10-02 04:50-05:15 UTC: TTFT p95 about 19 s on the remaining host). Do not start the second host until the first one is registered again (`/backends/list` shows two base handles on both model-proxy peers).
   - Drain properly: stop the registrar AND `POST /unregister/endpoint` (the registrar's SIGTERM trap does not unregister), then wait for running requests to reach about 0. Expect a trickle of traffic to continue for 10+ minutes after unregistering.
   - The gpu-manager dashboard's `compose/down` ignores `dry_run`: treat every down as real.
   - Watch on the first host for 2 hours before the second: `num_running_reqs` should exceed 15 per replica under load, queue time p95 should drop, tail TPOT will rise (expected up to about 190 ms p90), no OOM (lab peak 127.5 of 140 GB per GPU).
   - Rollback per host: the `v0.0.463` TP2 file (15 running), same orphan rule.

**For the original 4x TP2 file (done 2026-10-01; gpu04 runs it since 18:40 UTC):**

1. **TP2 quality gate (pending).** In the lab, GSM8K, MMLU and the perception check must reach parity with TP4.
2. **Prod-CVM test on a drained host, agreed with Lloyd.** Check:
   - CC-on startup of 4 × 325 GiB pinned host tiers (`cudaMallocHost`);
   - TP2 NCCL under PPCIe;
   - CVM `MemAvailable` headroom after all four tiers are allocated;
   - cold-start time.
3. **Image already pulled on the host.** The `47aff791…` digest is in `docker images`, so the dark window is the cold start alone.
4. **Go from both Pranav and Lloyd.**

Also, as for every base-host change:
- use a merged tag that clears the commit-age gate;
- pass the host's full gpu-manager env map;
- set `force_recreate: false`;
- run every call with `dry_run: true` first.

**Image check (2026-10-01, gpu32):** `47aff791…` accepts `write_through_selective`. Its engine, cache, scheduler and DSA kernel sources are byte-identical to the lab build the TP2 numbers came from (`glm53-hicache-w4afp8:qsplit-verify`). Only the HTTP/event-loop files differ (`http_server.py`, `serving_base.py`, `tokenizer_manager.py`, `event_loop_stall_dump.py`: the off-loop patches and stall dump).

## Tiered rollout

The base tier has two hosts, gpu03 and gpu04. Move one host at a time, and each stage needs its gate before the next.

| Stage | Where | What runs | Exit gate |
|---|---|---|---|
| 0 | lab (gpu31/gpu32) | TP2 quality gate (GSM8K, MMLU, perception) vs TP4 on the same image | parity with TP4 |
| 1 | **gpu04, drained** (registrar down; gpu03 carries base alone in a low-traffic window) | This file, out of rotation. Perception check on r1..r4 and a short paired replay through the soak relay. | All 4 replicas up under CC with their 325 GiB tiers, no 801/NCCL/Xid/OOM, `MemAvailable` ≥ 20 GiB, perception `ok: true`, replay TTFT/throughput in line with the lab |
| 2 | **gpu04 in rotation** (re-register) | 1 of 2 base hosts on 4x TP2; gpu03 stays on 2x TP4 as the live control | 24–48 h clean: no restarts/OOM, 503s not above gpu03, hit rate ≈ gpu03, TTFT p50/p90 by bucket ≤ gpu03 for ≤10K prompts |
| 3 | **gpu03** | Same procedure as stages 1–2 | Another 24 h clean on both hosts |
| 4 | follow-ups, lab first | Raise the TP2 running cap (`--max-mamba-cache-size`), then re-canary on one host | Separate change and PR |

At any stage, roll back that host to `prod/GLM-5.3-Flash-SGL-TP4-W4AFP8.yaml` (see Rollback). The other base host stays on TP4 until stage 3, so the base domain always keeps one known-good host.

## The orphan rule

compose-manager always runs `up -d --remove-orphans`. None of the four new engine names exists in the TP4 file, and neither of the two old names exists in the new file. Stop the old engines with a **scoped `compose/down` of the TP4 file first**, so their 5 m grace applies. **Never let `--remove-orphans` perform the switch.**

## Procedure

1. **Drain.** Run `compose/down` on the TP4 file with `services: ["model-proxy-registrar"]`. The host unregisters. Wait for `num_running_reqs` to reach about 0, up to 5 min.
2. **Stop the old engines.** Run `compose/down` on the TP4 file with `services: ["model-sg-glm53-w4afp8-tp4-r1", "model-sg-glm53-w4afp8-tp4-r2"]`. Poll `docker/ps` until both are gone.
3. **Start the new engines.** Run `compose/up` on this file with `services: [the four tp2-r engines, "proxy-glm53", "otelcol-contrib", "dcgm-glm53", "nginx"]`.
   - `nginx` is included because its deployment label changed.
   - Run with `dry_run: true` first. Apply only if the plan creates the four engines, recreates the listed services, and removes nothing.
4. **Cold start.**
   - Logs show four HiCache host pools of 325 GiB each. There must be no CUDA error 801, NCCL errors, Xid errors or OOM.
   - Bring up `glm53-soak-relay` (`:8008`–`:8011`). Check `/health`, `/v1/models` and one generation per replica.
   - Run `glm53-perception-check` and expect `qualification_finished` with `ok: true`.
   - `compose/down` both verification services.
5. **Re-register.** Run `compose/up` on this file with `services: ["model-proxy-registrar"]`. Then send a real base completion and a cache-hit follow-up.

For a collector-only config change, dry-run `compose/up` with `services: ["otelcol-contrib"]` and `force_recreate: true`: inline `configs:` are not hashed. Apply only when the dry-run plan recreates `otelcol-contrib` alone.

## Watch list (canary host vs the other base hosts)

- `cached_tokens_total` by source: device vs host tier. Hit rate should hold at about 85%.
- `num_running_reqs` per replica. It plateaus at 15, the TP2 cap.
- TTFT by prompt bucket (expect better at 10K or fewer, worse at 30K+ under low load) and ITL.
- Queue-full 503s / 429s.
- Engine restarts and OOMs (device and host).
- CVM `MemAvailable` (four pinned 325 GiB tiers).

## Rollback

Redeploy `prod/GLM-5.3-Flash-SGL-TP4-W4AFP8.yaml` under the same orphan rule:

1. `compose/down` this file with `services: ["model-proxy-registrar"]`.
2. `compose/down` this file with the four `tp2-r` engines.
3. `compose/up` the TP4 file with `services: ["model-sg-glm53-w4afp8-tp4-r1", "model-sg-glm53-w4afp8-tp4-r2", "proxy-glm53", "otelcol-contrib", "dcgm-glm53", "nginx"]`.
4. Once both engines are ready, `compose/up` the TP4 file with `services: ["model-proxy-registrar"]`.

The W4AFP8 and FP8 snapshots stay cached, because both files run the same downloader.
