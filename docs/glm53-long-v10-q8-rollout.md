# GLM-5.3 long tier: v10 Q8 prefill + 16K chunks, fleet rollout

Status: **prepared, not deployed** (except gpu02 r2b, which has run the same engine values as a canary since 2026-10-18 05:44 UTC, #364).
Merging changes no running service. Each replica changes only when an operator deploys the merged tag to it with a scoped `compose/up`.

## What changes

Only the long tier's TP2 replicas change. That is gpu02 and gpu23 (`prod/GLM-5.3-Flash-SGL-TP4-W4AFP8-LongContext.yaml`, r1a/r1b/r2a/r2b) and gpu13 (`prod/small-models.yaml`, r1a/r1b). Everything else stays as it is, including the base tier (v7), the TP4 r1/r2 definitions (v6, not deployed), proxies and registrars.

| | before (v7 fleet) | after (v10 fleet) |
|---|---|---|
| image | `sha256:fa730e6e…` (glm53-hicache-w4afp8-v7, #345) | `sha256:c15a3367…` (glm53-hicache-w4afp8-v10 = #363 v9 recipe + #365 CVE fix, publish run 39094781691, signed) |
| `--dsa-prefill-backend` | `flashmla_kv` | `flashmla_sparse_q8` (Q8 DSA prefill on the same FP8 NoPE KV layout) |
| `--chunked-prefill-size` | 8192 | 16384 (memory-safe at 0.86 with `SGLANG_DSA_INDEXER_QSPLIT=1`, already set on every replica) |
| unchanged | FP8 KV, `flashmla_kv` decode, 16 running / 8 queued, graphs 16, mem 0.86, HiCache write_through 325 GiB, admission reserve off, EAGLE 4/1/5, overlap off, every env var, GPUs | same |
| labels | `engine_image` `fa730e6e62b2`, variant `…-c8192-…-v7` | `c15a33672ade`, `…-c16384-…-v10-q8` |

The `glm53-ghost-aggregator` sidecar follows the TP2 image, so it moves to v10 too.

## Evidence

- **Lab, tee-bench exp 45b** (bare gpu31, TP2, prod long argv, reserve off, queue 8; lane-swapped, 2 runs vs 3 v7 controls):
  - +43% completed, +25% total tok/s;
  - TTFT p50/p95 −66%/−28%;
  - >64K TTFT p95 −22%;
  - TPOT p95 +44-58%.
- **30-minute soak:** +24% completed, TTFT p95 −39%; the candidate won all six 5-minute windows.
- **Lab, exp 46** (the exact published v10 bytes vs v7, 600 s):

  | | v10 | v7 |
  |---|---|---|
  | ok / err | 221 / 3 | 163 / 10 |
  | total tok/s | +25% | baseline |
  | TTFT p95 | 39 s | 63 s |
  | >64K TTFT p95 | 45 s | 70 s |
  | GSM8K-100 | 0.99 | 0.99 |

  The smoke gate showed no new failures.
- **Prod canary, gpu02 r2b**, first 68 minutes (light overnight load, ~1.4 running per replica), vs the gpu02 controls:
  - TTFT p95 14.0 s vs 13.0-29.9 s (−30% vs the control mean);
  - e2e p95 54 s vs 65-97 s;
  - ITL p95 72 ms vs 73-75 ms;
  - no restarts or engine errors; accept length unchanged.

  Owner decision (2026-10-18): good enough to prepare the fleet rollout.

**Expected cost:** a heavier ITL/TPOT tail under prefill-heavy load (lab +44-58% TPOT p95). It was not visible at the canary's light load.

## Rules

- **Long tier: one replica at a time.** Wait until the replica is back serving and baked before taking the next one out (owner rule: never take several long-context replicas out at once, even on different hosts).
- **Never kill a running `compose/up`.** The gpu-manager proxy cuts the stream after 5 minutes, which is shorter than a GLM drain plus recreate. If the call returns without `Started`, check `docker/ps`. If the replica is absent (created but not started), re-send the identical scoped `compose/up`; it starts in about 15 s. This happened on the r2b canary and was recovered this way.
- `compose/down` is never used.
- **Abort and roll back only on a crash, engine errors or not serving.** Watch for Traceback, CUDA/NCCL/Xid/OOM, a restart loop, `/backends/list` unhealthy, or no traffic. Latency comparisons are advisory: never roll back a working replica for latency alone.
- Tell the team before starting, and post progress per host.

## Order

r2b is already on v10, so it gets redeployed from the fleet file only to align its labels. That recreate is optional and goes last on gpu02.

1. gpu02 r1a, then r1b, then r2a; then r2b (labels only), then gpu02's `glm53-ghost-aggregator`.
2. gpu23 r1a, r1b, r2a, r2b, then its `glm53-ghost-aggregator`.
3. gpu13 r1a, r1b (`file: prod/small-models.yaml`), then its `glm53-ghost-aggregator`.
4. Per host, after its engines: `otelcol-contrib` with `force_recreate: true`, so the scrape labels show the v10 image and variant. Dry run first; the plan must recreate the collector only.

Each replica takes a ~5-minute drain plus a ~24-minute cold start under CC (r2b: down 05:20, serving 05:44). Ten replicas therefore take about 5-6 hours, done one at a time.

## Per-replica procedure (compose-manager via gpu-manager)

`POST /api/instances/<host id>/compose/up` with:
- `tag`: the merged tag;
- `file`: `prod/GLM-5.3-Flash-SGL-TP4-W4AFP8-LongContext.yaml` (gpu02, gpu23) or `prod/small-models.yaml` (gpu13);
- `services`: `["model-sg-glm53-w4afp8-tp2-<r>"]`;
- `env`: the host's full env map from `/api/instances`, never a hand-picked subset;
- `force_recreate: false`.

Steps:
1. **Pre-check.** On every long host, all long replicas are `running` and none is recreating. Base hosts are untouched.
2. **Dry run** (`dry_run: true`): the plan must show exactly one recreate, that replica. The dry run reports a "No such image" error for the not-yet-pulled image; that is expected.
3. **Apply** (`dry_run: false`). Then follow the stream rule above.
4. **Ready:** the container is `Up` on the v10 image, and its server args show `flashmla_sparse_q8` and `chunked_prefill_size=16384`. The log shows proxy `/v1/loads` polls answered 200, then real `Prefill batch` / `Decode batch` lines, with no Traceback, CUDA, NCCL, Xid or OOM.
5. **Bake:** about 10 minutes of real traffic with no errors, and the replica serving.
6. Then move to the next replica.

## Rollback

Per replica: the same scoped `compose/up` with the previous tag (the v7 fleet values) and the same file, dry run first. The plan must recreate only that replica, back on `sha256:fa730e6e…` with `flashmla_kv` and 8192.

## Reads

Compare converted replicas with not-yet-converted ones over the same window while the rollout is in progress, and with the pre-rollout week afterwards:
- TTFT p50/p95 by prompt bucket (64-128K, ≥128K);
- ITL p95 at matched running;
- requests served and queue-full 503s;
- prompt and generation tok/s.

Note: before `otelcol-contrib` is recreated on a host, Prometheus labels for converted replicas still show the old `engine_image`/`config_variant`. Filter by host and job, not by `engine_image`.
