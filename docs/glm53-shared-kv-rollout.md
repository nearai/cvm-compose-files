# GLM-5.3 Flash long tier: shared KV cache rollout

What: gpu02's two TP4 replicas share one CPU-RAM KV store inside the CVM, so a prefix one replica
computed restores on the other instead of re-prefilling. Image recipe:
`docker/sglang-glm53-v0520-shared-kv` (SGLang v0.5.20 + the NEAR AI port + the shared-KV patch).
Compose: `experiments/GLM-5.3-Flash-SGL-TP4-W4AFP8-LongContext-SharedKV.yaml`, generated from the
live long-context file by `scripts/prepare_glm53_shared_kv.py`. Write-up and data:
https://app.notion.com/p/3e629a6526bf81878e3ac044c650de63

## What to expect (honest estimate)

gpu02 recomputes 0.47B of 4.24B prompt tokens a day (11%). Most of that is each turn's new
content, which no cache can serve. Sharing recovers an estimated **10–20% of the uncached prefill
tokens** (≈1–3% of replica GPU time), concentrated in the windows when a replica is down or has just
restarted: failover restores instead of recomputing (5–10%), and the store survives an engine restart
(3–8%). Per affected request the gain is large (131K: 1.9 s instead of 6.1 s; 262K: 3.1 s instead of
12.3 s). Whether least-load routing without affinity then also cuts the tail is unmeasured.

## Prerequisites

1. **The v0.5.20 port is qualified.** This image carries `nearai-v0520@4fb82813` as `v0520-port.diff`;
   that port is still in its own stability A/B. If the port gets its own recipe first, rebase this
   recipe onto it and keep only `shared-kv.diff`.
2. **The image is published** from a merged commit: dispatch `publish-glm53-v0520-shared-kv.yaml`
   with the merge commit as `source_revision`. It rebuilds, checks every patch and source digest,
   runs `test-cpu.sh`, scans, signs and attests.
3. **Record the digest:** write the digest-pinned reference to
   `docker/sglang-glm53-v0520-shared-kv/RELEASED_IMAGE`, run
   `uv run scripts/prepare_glm53_shared_kv.py --write`, and commit both (the CI step then enforces
   `--check`).

## Soak on an HCC/PPCIe CVM before gpu02 (gpu32 had CC off)

Deploy the generated file to a TEE host with the same two-island layout. Gates:

| gate | pass |
|---|---|
| startup | both replicas log `HiCache shared store reserve: budget_bytes=… reserved_bytes=…` and allocate their 406 GiB host tier; `SGLANG_HICACHE_CUDA_HOST_MEMORY=1` path works under CC |
| store keeps pace | `sglang:backuped_tokens_total` tracks `sglang:hicache_host_used_tokens`; store directory size grows with host use |
| cross-replica restore | a long prompt sent to r1 and then to r2 reports `cached_tokens_details.storage` > 0 on r2 |
| failover | kill r1 for ~5 min under long-tier load: r2's hit fraction stays ≥0.85 (private baseline 0.70) |
| no harm | TPOT ≤5% worse, TTFT not worse in steady state; quality at parity |
| memory | CVM MemAvailable never drops below 20 GiB with the store at its budget |

## Deploy on gpu02

Sharing needs both replicas on the new image, so switch them together during a low-traffic window
(the r1/r2 engine flags are unchanged from the live file; only the image and the store change).
Restart r1 first, wait for health, then r2; the store starts empty and warms from write-through.

**Watch:**
- `sglang:storage_prefetch_hit_tokens_total` (tokens served from the shared store);
- `sglang:storage_prefetch_unfulfilled_tokens_total{reason=…}` (why lookups missed);
- `sglang:backuped_tokens_total` vs `sglang:hicache_host_used_tokens`;
- the `shared_kv` tmpfs usage (≤ `GLM53_SHARED_KV_STORE_BUDGET`, default 400GiB).

Note: `sglang:hicache_backup_tokens_total` overcounts 4× on this layout.

**Rollback:** redeploy `prod/GLM-5.3-Flash-SGL-TP4-W4AFP8-LongContext.yaml`. The tmpfs store
disappears with the volume; nothing persistent is left behind.

## Sizing

| | per replica | total |
|---|---|---|
| private host tier (`SGLANG_HICACHE_RAM_BUDGET`) | 406 GiB | 812 GiB |
| shared store budget (`GLM53_SHARED_KV_STORE_BUDGET`) | — | 400 GiB |
| rank-0 evictor cap (`GLM53_SHARED_KV_REPLICA_CAP`) | 110 GiB (+3 × 27.5 GiB mamba sidecars) | 385 GiB ≤ budget |
| tmpfs size (`GLM53_SHARED_KV_TMPFS_SIZE`) | — | 440 GB |

gpu02's CVM had ~1.43 TB available at start-up. The first replica's budget reserves the whole store
budget; the second reserves only the store's remaining growth (tmpfs pages already count against
MemAvailable).

**Rules:**
- Size the cap so `replicas × cap × (1 + (TP−1) × 0.25) ≤ store budget`; each evictor only counts its own writes.
- Never NUMA-pin engine memory: a 406 GiB host tier plus store pages exceed one socket.
- The private host tier must stay ≥ the GPU KV pool; a smaller one starves the store.

## Known gaps

- The last prefill chunk (16K tokens) of a restored prompt is recomputed; the final mamba checkpoint
  does not reach the store yet (~0.9 s per restore).
- Two evictors share one directory without coordination. The per-replica cap keeps the total within
  budget. A page evicted by one replica turns the other's lookup into a recompute, not an error.
- Buffer host-memory mode and runtime attach of a store are refused for compressed DSA.
