# gpu13 GLM-5.3 Flash: 2x TP2 canary

**Status: prepared, not deployed.** `prod/small-models.yaml` replaces gpu13's single GLM-5.3 Flash TP4/EP4 engine (`model-sg-glm53-fp8-tp4`, GPUs 4-7) with two TP2/EP2 engines behind the same `proxy-glm53`. It follows the 4x TP2 base-tier canary (`docs/glm53-tp2x4-base-canary.md`) and the TP4 overlap-off canary (#330).

## What changes

| | Before | After |
|---|---|---|
| Engines | `model-sg-glm53-fp8-tp4`, TP4/EP4, GPUs 4-7 | `model-sg-glm53-w4afp8-tp2-r1` (GPUs 4,5, rendezvous `127.0.0.1:29510`) and `-r2` (GPUs 6,7, `127.0.0.1:29511`), TP2/EP2 |
| Image | `nearaidev/sglang@sha256:47aff791...` | Same digest, already cached on gpu13 |
| Mamba | image default | `--max-mamba-cache-size 165 --mamba-ssm-dtype bfloat16` (the TP2x4 file's setting) |
| HiCache RAM | `GLM53_HICACHE_RAM_BUDGET`, default 80% | `GLM53_R1_HICACHE_RAM_BUDGET` and `GLM53_R2_HICACHE_RAM_BUDGET`, default 40% each |
| Proxy | `VLLM_BACKEND_URLS` has one backend | Both replicas, `VLLM_PROXY_POOL_IDLE_TIMEOUT_SECS=3` added (as in the TP2x4 file), affinity stays on |
| Telemetry | one engine scrape job | One scrape job and label set per replica; `dcgm-glm53` still covers GPUs 4-7 |

Kept from gpu13 today: `write_through` HiCache policy, no admission-reserve env (`scripts/validate_ds4f_migration.rb` forbids it on the long tier after the gpu02 long r2 crash), context length 1,048,576, chunked prefill 8192, 32 running / 8 queued, EAGLE 5/1/6 adaptive, strict thinking, `SGLANG_DSA_INDEXER_QSPLIT=1`, `--disable-overlap-schedule`. Gateway slots stay at 32 for the canary; raising them is a later gateway change.

Telemetry choices:
- `deployment` stays `glm53-flash-sgl-tp4`, the label gpu02 and gpu23 carry, so existing dashboards and the per-bucket peer comparison in the abort criteria keep matching. (The TP2x4 base hosts use `glm53-flash-sgl-tp2x4`; gpu13 is a long-tier host, so it is compared with its long-tier peers.)
- Replicas are told apart by `container_name`, `instance` (1, 2) and `gpu_pair` (`4-5`, `6-7`). The proxy and DCGM exporter keep `gpu_pair` `4-7`.
- `config_variant` changes to the new `...-overlap-off-mamba165-bf16state-h200-tp2x2-ep2-...` string on every GLM service, so the canary is separable from the TP4 arm.

## Evidence (gpu32 bare metal, CC off, one run each)

Source: `tee-bench/evidence/gpu32-2xtp2-selective/README.md`. The TP2 pair is one TP2 replica at half rate, doubled.

| Workload | 1x TP4 (today) | 2x TP2 |
|---|---|---|
| prod mix x2 | 269 ok / 34 err, 325 tok/s, TTFT p50/p95 2.10 / 31 s | 326 / 0, 472 tok/s (+45%), 0.78 / 10.5 s |
| long mix x1 | 41 / 44, 88 tok/s, TTFT p50 73 s | 62 / 30, 166 tok/s, 92 s |
| returning sessions | 847 / 21, 381 tok/s, TTFT p50/p95 2.34 / 9.44 s, ITL >100 ms 14.7% | 908 / 0, 404 tok/s, 1.45 / 10.7 s, 7.6% |
| single prompt TTFT 35K / 140K | 2.02 / 7.74 s | 2.85 / 11.25 s (+41%) |

GSM8K-20 0.95 and passkey 32K 5/5 on both. Prior production evidence: 4x TP2 vs 2x TP4 per 8 GPUs gave 8.18 vs 6.49 req/s (+26%) on gpu03/gpu04.

Risks:
- Single long prompts are about 40% slower at low load.
- The per-replica KV pool is about 1.0M tokens vs 3.56M; a 1M-context request barely fits.
- Prod 7-day cache share on TP2 replicas is 74-76% vs 85-86% for TP4 on gpu03/gpu04.
- The lab test replicas included the admission reserve and `write_through_selective`; this config excludes both. The lab numbers are therefore not an exact measurement of this config.
- Bare metal, not TEE.

Expected effect on gpu13: about +20-40% long-tier capacity, better TTFT under load, slower isolated huge prompts, ITL about flat.

## Deploy

Needs separate approval. Record the deployed tag and full dashboard `env_vars` first, and pass the complete map as `env` on every request. Confirm the image is cached on gpu13 and that gpu02/gpu23 can carry the long tier while gpu13 is down.

1. Record container IDs and `CreatedAt` for the whole host. Scoped `compose/down` of `model-sg-glm53-fp8-tp4` only; wait until it is gone and GPUs 4-7 are free. Never use an empty services list and never rely on `--remove-orphans` for this switch: the old and new engine names do not overlap.
2. `compose/up` with `dry_run` first, services `[model-sg-glm53-w4afp8-tp2-r1, model-sg-glm53-w4afp8-tp2-r2]`, `force_recreate: false`. Continue only if those two are the only create targets.
3. Run the real up with the same scoped list. Starting both in one up lets their percentage budgets resolve against the same available RAM. In each startup log check `rank_budget_bytes` and `available_bytes`; the two replicas should agree. If r2 got visibly less, set `GLM53_R2_HICACHE_RAM_BUDGET` to an explicit GiB value and redo r2 only. Wait for both to be ready and run a real completion against each.
4. **Proxy and nginx must be recreated.** `proxy-glm53` has new `VLLM_BACKEND_URLS` and `VLLM_PROXY_POOL_IDLE_TIMEOUT_SECS` env, and compose does not apply env changes without a recreate. `compose/up` with services `[proxy-glm53]` and `force_recreate: true` (dry run first), then `compose/up` with services `[nginx]` and `force_recreate: true` so nginx re-resolves the new proxy address (the same step the 2026-09-18 migration used).
5. Telemetry: `dcgm-glm53` (new `config_variant` label) and the OTel collector (two engine scrape jobs, new variant) also changed. Recreating them is a separate step that needs its own approval, scoped the same way. Until the collector is recreated, its old `sglang-model-sg-glm53-fp8-tp4` job is the only one running, so gpu13 engine metrics are missing; do not start the 60-minute abort window before it is done.
6. Re-read all container IDs: only the two new engines, `proxy-glm53` and `nginx` (and the collector, if approved) should have changed.
7. Warm the caches, then compare gpu13 with gpu02/gpu23 per prompt-length bucket (see below). Test ordinary, streaming, tool-call and reasoning requests on port 8009 first.

## Abort criteria

Measured against gpu02/gpu23 per bucket, after caches warm. Roll back on any of:
- any container exit or Xid;
- 5xx other than 503 queue-full above 0.5%;
- TTFT p95 for prompts of 64K tokens or more more than 50% above peers over 60 minutes;
- ITL mean more than 10% above peers for 30 minutes.

## Rollback

1. Scoped `compose/down` of `model-sg-glm53-w4afp8-tp2-r1` and `-r2`; wait until both are gone and GPUs 4-7 are free.
2. Redeploy the previous release tag, which still has `model-sg-glm53-fp8-tp4`, with the same complete env: dry-run then scoped `compose/up` of `[model-sg-glm53-fp8-tp4]`, then recreate `proxy-glm53` and `nginx` (both with `force_recreate: true`) so the proxy points back at the single backend.
3. Verify readiness and a real completion, and that unrelated container IDs are unchanged. `GLM53_R1_*`/`GLM53_R2_*` env values are ignored by the old file and can stay set.
