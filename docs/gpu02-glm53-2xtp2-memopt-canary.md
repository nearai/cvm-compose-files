# gpu02 GLM-5.3 Flash long tier: r2 as two memory-optimized TP2 replicas

Status: DRAFT canary, not deployed. Nothing here is approved to run; it needs an explicit GO, a merged tag that clears compose-manager's commit-age gate, and the abort criteria below.

## What changes, and on which host

`prod/GLM-5.3-Flash-SGL-TP4-W4AFP8-LongContext.yaml` is **one shared file**: gpu02 and gpu23 both deploy it, each with its own compose-manager `services` list and env map. The change is therefore built so that it is gpu02-only in effect:

| Piece | gpu02 (canary) | gpu23 (unchanged) |
|---|---|---|
| r1 `model-sg-glm53-w4afp8-tp4-r1`, GPUs 0-3 | unchanged | unchanged |
| r2 `model-sg-glm53-w4afp8-tp4-r2`, GPUs 4-7 (TP4) | stopped (scoped `compose/down`), definition kept in the file | keeps deploying it |
| `model-sg-glm53-w4afp8-tp2-r2a`, GPUs 4,5 | started | never named, never started |
| `model-sg-glm53-w4afp8-tp2-r2b`, GPUs 6,7 | started | never named, never started |
| `proxy-glm53` pool | `GLM53_BACKEND_URLS` = r1 + r2a + r2b | variable unset, default = r1 + r2 (byte-identical to before) |

- The TP4 r2 definition is untouched (the unit tests assert its rendered text is identical with and without the canary).
- The proxy line is now `VLLM_BACKEND_URLS=${GLM53_BACKEND_URLS:-http://model-sg-glm53-w4afp8-tp4-r1:8000,http://model-sg-glm53-w4afp8-tp4-r2:8000}`. Conversation affinity (`VLLM_BACKEND_CONVERSATION_AFFINITY=1`) is unchanged.
- **gpu23 must NOT get `GLM53_BACKEND_URLS`** in its compose-manager env map, and must keep deploying r2 with its scoped services list. Setting it there would point its proxy at services that do not run there.
- `GLM53_R2_HICACHE_RAM_BUDGET` (r2's 650 GiB override) does not affect the pair. The pair has its own per-replica variables, each defaulting to half of r2's budget: `GLM53_R2A_HICACHE_RAM_BUDGET` and `GLM53_R2B_HICACHE_RAM_BUDGET`, default `325GiB` (2 x 325 = 650 GiB, so total host RAM use is unchanged). Before starting, check gpu02's env map for a leftover `GLM53_R2_HICACHE_RAM_BUDGET` or `GLM53_HICACHE_RAM_BUDGET` and make sure the 406 + 325 + 325 GiB plan fits available RAM minus the 10 GiB reserve.
- A bare unscoped `compose/up` of this file on gpu02 would start r2 (TP4, GPUs 4-7) beside the pair and collide on GPUs. Always send a scoped `services` list and never an empty one. The pair and r2 are mutually exclusive on gpu02; the validator enforces that the pair stays inside r2's GPUs and does not touch r1's.
- otelcol: `sglang-model-sg-glm53-w4afp8-tp2-r2a` and `-r2b` scrape jobs carry the same `deployment` label as r1/r2 so dashboards match, and are distinguished by `instance` (`2a`, `2b`), `gpu_pair` (`4-5`, `6-7`) and `config_variant`. gpu23's collector also loads these two jobs from the shared file; their targets do not exist there, so expect `up == 0` for them on gpu23 (see risks in the PR).
- dcgm is unchanged (it already covers all eight GPUs).

## Why

Lab (tee-bench, bare metal): two TP2 replicas beat one TP4 replica on the same four GPUs for the tier's mixed load (exp 18), and the memory-optimized settings recover KV and cache hit rate versus the production TP2 argv (exp 19). The settings per TP2 replica, all lab-validated together:

- `--tp-size 2 --ep-size 2`, `--mem-fraction-static 0.86`
- `--max-mamba-cache-size 330 --mamba-ssm-dtype bfloat16` (mamba/bf16-state settings from `prod/GLM-5.3-Flash-SGL-TP2x4-W4AFP8.yaml`, slots doubled for the memory headroom)
- EAGLE fixed 4/1/5 (`--speculative-num-steps 4 --speculative-eagle-topk 1 --speculative-num-draft-tokens 5`, no `--speculative-adaptive`)
- `--max-running-requests 24 --max-queued-requests 8`
- chunked prefill stays 8192 (32K chunks are under test and conflict with 0.86 at TP2)
- HiCache `write_through`, `direct` IO, `page_first_direct` layout, as r2 today
- no admission-reserve environment (forbidden on the long tier), no `--disable-overlap-schedule` (gpu13 is the separate overlap-off canary)
- `SGLANG_DSA_INDEXER_QSPLIT=1` kept, same v3 image as r2 (`47aff7910900`), unique `--dist-init-addr` ports (`29512`, `29513`; r1 uses `29510`, r2 `29511`)

Boot facts at TP2 in the lab: KV pool 1,392,896 tokens per replica, 330 mamba slots, 17.6 GB free at ready.

## KMS compose hash

The compose file is shared, so its content changes for both hosts and there is **one** new compose hash. That hash must be registered with the KMS contract before any deploy of this file, and gpu23's next redeploy from this file will also be under the new hash even though its running services do not change. Do not deploy before the hash is registered.

## Deploy steps (gpu02 only; each call needs the full gpu02 env map)

Preconditions: merged tag past the commit-age gate; gpu02's complete gpu-manager env map captured (this runbook's variables included); `force_recreate: false` on every call except where stated; record `docker/ps`, `/backends/list`, container IDs and one successful long-domain completion first. r1 carries the whole long tier while r2 is down and the pair boots, so pick a low-traffic window and watch queue, TTFT and aborts. Do not unregister gpu02 to drain it.

1. `compose/down` for the previous file and tag with `services: ["model-sg-glm53-w4afp8-tp4-r2"]`. Poll `docker/ps` until it is gone (the 5 m grace applies; never let orphan removal do this). The proxy marks r2 unhealthy and sends traffic to r1.
2. `compose/up` for this file with `services: ["model-sg-glm53-w4afp8-tp2-r2a", "model-sg-glm53-w4afp8-tp2-r2b", "proxy-glm53", "otelcol-contrib"]`, `dry_run: true` first, with gpu02's env map including `GLM53_BACKEND_URLS=http://model-sg-glm53-w4afp8-tp4-r1:8000,http://model-sg-glm53-w4afp8-tp2-r2a:8000,http://model-sg-glm53-w4afp8-tp2-r2b:8000`. The plan must create r2a, r2b and recreate proxy-glm53 and otelcol-contrib, and remove nothing. Apply only then. Check each replica's startup log: `server_args` match the settings above, HiCache `rank_budget_bytes` is 325 GiB across two ranks, KV pool about 1.39M tokens, 330 mamba slots, and ready before the proxy pool counts it healthy.
3. `compose/up` with `services: ["nginx"]` and `force_recreate: true` after the proxy recreate, so nginx re-resolves the recreated proxy. Run its `dry_run` first; it must plan nginx only.
4. Verify: `/backends/list` shows r1, r2a, r2b healthy; a real long-domain completion plus a cache-hit follow-up succeed; metrics carry `instance="2a"/"2b"`, `gpu_pair`, and the new `config_variant`; r1 and every other container keep their IDs.

## Abort criteria

Compare against the same-host control r1 and against gpu23 r1/r2, per input-length bucket, after a **1 h cache warm-up**. Roll back on:

- any container exit of r2a/r2b, or any Xid;
- 5xx other than 503 queue-full above 0.5% of requests;
- TTFT p95 for prompts of 64K tokens or more more than 30% above r1 over 60 minutes;
- ITL mean more than 15% above r1 at similar concurrency over 30 minutes.

Isolated huge prompts are expected to be about 40% slower (half the GPUs per prefill); that alone is not an abort, the 30% criterion is judged on p95 under live load.

## Rollback

1. `compose/down` for this file with `services: ["model-sg-glm53-w4afp8-tp2-r2a", "model-sg-glm53-w4afp8-tp2-r2b"]`; poll until both are gone. r1 carries the tier meanwhile.
2. Remove `GLM53_BACKEND_URLS` from gpu02's env map (or set it back to r1 + r2).
3. `compose/up` with `services: ["model-sg-glm53-w4afp8-tp4-r2", "proxy-glm53", "otelcol-contrib"]`, `dry_run` first; r2 re-warms its HiCache. Then `compose/up` `["nginx"]` with `force_recreate: true`.
4. Verify `/backends/list`, a long-domain completion and a cache-hit follow-up. gpu23 never changed. Keep the replica logs for analysis.
