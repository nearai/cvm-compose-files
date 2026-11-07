# GLM-5.3 long tier: HiCache OFF (2026-11-06)

Turns the HiCache host tier off on all five long-context replicas: gpu23 `r1a`/`r1b`/`r2a`/`r2b`
and gpu04 `l4`. This is a **mitigation for a crash, not a performance change** — it costs the long
tier its host-cache hits, and we take that cost on purpose.

## The crash

Since 2026-11-02 the long replicas die with a CUDA "illegal memory access", always the same stack:

```
eagle_worker_v2.verify -> run_eagle_verify -> eagle_prepare_for_verify
  -> prepare_mamba_track_for_verify -> set_mamba_track_indices_from_reqs
  -> schedule_batch.py .to(device, non_blocking=True)   # torch.AcceleratorError
```

Eight crashes so far (UTC): 11-02 11:12 r2b · 11-03 13:31 r2a · 11-03 23:02 r1b · 11-06 08:09 r2a ·
11-06 11:56 r2b · 11-06 17:32 and 19:30 gpu04 l4. Each one costs a replica restart.

**Only the long tier crashes.** No base replica has, on the same v12 image `da8d4e4b5ec6`. HiCache is
the difference between the two tiers.

## Why HiCache, and why not the crash line

The line the traceback names is **not** the bug. In this image:

- `mamba_ping_pong_track_buffer_size = 2 if enable_overlap_schedule else 1`, and the long argv sets
  `--disable-overlap-schedule`, so the ping-pong buffer is width 1.
- The strategy is `extra_buffer`, not `extra_buffer_lazy`, so `mamba_lazy_spec_prepare` — which
  computes `other_idx = 1 - next_idx` with no width check, and would be a genuine out-of-bounds
  gather — is dead code here.
- The non-lazy advance goes through `get_mamba_ping_pong_other_idx`, which returns the same index
  when the width is 1, so `mamba_next_track_idx` stays 0 and the gather is in bounds.

So the fault is **asynchronous, from an earlier kernel**, merely reported at the next host→device
copy. The leading candidate is CUDA graph replay racing a HiCache host→device load: a graph replayed
while the cache controller is still writing device KV pages holds a stale page table. This image
still carries the open upstream TODO in `managers/scheduler.py`:

```
# todo (zhiqiang): disable cuda graph execution if hicache loading triggered
```

which upstream **sgl-project/sglang#19177** closes. Related: #16771 (HiCache IMA that disappears with
HiCache off), #19002 (mamba ping-pong slot exhaustion — asserts, a different failure).

**This is a leading hypothesis, not a proven root cause.** A patched-engine repro naming the faulting
kernel is still outstanding. HiCache off removes the suspected mechanism now, without new engine code.

## Why it started on 11-02, and why a restart helps

Nothing changed on 11-02. gpu23's long replicas were last redeployed on **10-28** and then left
alone; before that they were restarted every 1-4 days, and the longest untouched run was 3.5 days.
The first crash came at **+5.5 days** of uptime. Per-replica intervals since: r2b 4.0 d, r2a 2.8 d,
r1b 6.9 d. gpu04 l4 is the outlier at 1.6 d and carries a whole long lane with `--max-queued-requests 32`.

Frequent deploys had been masking a latent, accumulation-dependent bug. A restart buys roughly 2-6
days, which is why the fleet looked healthy until now — and why **the deploy of this change will itself
reset every replica's clock. Surviving a week proves nothing on its own; judge it against the 2-6 day
per-replica MTBF above, so give it two weeks before calling it fixed.**

## What changes

Per long replica, the four argv switches are dropped and the host-RAM budget variable is removed:

| | before | after |
|---|---|---|
| argv | `--enable-hierarchical-cache --hicache-write-policy write_through --hicache-io-backend direct --hicache-mem-layout page_first_direct` | (none) |
| env | `SGLANG_HICACHE_RAM_BUDGET=${GLM53_*_HICACHE_RAM_BUDGET:-325GiB}` | (absent) |
| `config_variant` | `…-offloop-v3-hicache-cuda-host-pooled-v1-host325g-memopt-…` | `…-offloop-v3-hicacheoff-memopt-…` |

The remaining `SGLANG_HICACHE_*` variables stay and are inert with the host tier off.

Files: `prod/GLM-5.3-Flash-SGL-TP4-W4AFP8-LongContext.yaml` (gpu23 r1a/r1b/r2a/r2b) and
`prod/GLM-5.3-Flash-SGL-TP2x4-W4AFP8-V12Q8-gpu04-3B1L.yaml` (gpu04 l4, derived from gpu23 r2b).
Both are generated; the sources are `scripts/prepare_glm53_w4afp8_long_context.py` and
`scripts/prepare_glm53_gpu04_3b1l.py`. `scripts/validate_glm53_prod_config.rb` now **asserts HiCache
stays off** on a TP2 long replica, so it cannot come back by accident.

The TP4 `r1`/`r2` services in the long file are untouched: they are historical and not deployed.

## The cost, and what to watch

gpu23 served **33-347M host-source cached tokens per 6 h**; gpu04 13-131M. All of that becomes
recompute. Expect long-prompt TTFT and prefill cost to rise, possibly a lot.

Watch, against the same weekday before the change:

- `sglang_cached_tokens_total{cache_source="host"}` on the long tier → should go to zero.
- Long-tier TTFT p50/p95 and tok/s.
- **Long-tier 429s (`long_reserve`, `host_share`).** These feed OpenRouter's learned capacity cap:
  the GLM traffic drop diagnosed on 11-04 came from exactly this loop. If 429s climb, lower the
  gateway's long budget rather than letting OR back off.
- No new `illegal memory access` in the long replicas.

If the capacity loss is worse than the crashes, roll back and take the crashes until #19177 is
validated.

## Deploy

Prod deploys come from a merged main tag. **One replica at a time, drained first, with Pranav's
explicit go for each step**, and a team notice before the first one.

1. gpu23 `r2a` first, alone. Drain it (gateway PR removing the host, or a model-proxy
   `POST /unregister/endpoint` for its `ip:8000` with the host's `MODEL_PROXY_TOKEN` via stdin),
   wait for running = 0, redeploy that one service, verify, then **hold for at least a few hours**
   and compare its TTFT and tok/s against r1a/r1b/r2b, which are still on HiCache. That side-by-side
   is the only clean read of the cost we will get.
2. If the cost is acceptable, roll r2b, r1a, r1b the same way, then gpu04 l4.
3. Per replica, verify: TLS chat returns 391 on the long SNI; the signing address is unchanged;
   `/backends/list` shows it healthy in the long domain; its Prometheus target is up; its
   `config_variant` label now reads `hicacheoff`; boot logs show no host-tier allocation.

Note l4's cold boot was ~15 min **including 325 GiB of HiCache**; without it, boot should be faster.

## Rollback

Revert this commit, regenerate (`--write` on both scripts), redeploy the affected replicas one at a
time. Or, to roll back a single replica without a revert, restore its four argv switches and its
`SGLANG_HICACHE_RAM_BUDGET` line — but the validator will then fail, so the revert is the clean path.

## Follow-up

The real fix is the #19177 port, which is staged on gpu31 under `pranav-claude-glm-iam`
(`~/glm-iam-20261106`: `port19177.py`, `overlay/`, `p19177-near.patch`, `arm.sh`, `trigger.py`).
It deviates from upstream deliberately — upstream overloads `hicache_consumer_index` as the
graph gate and sets a synthetic `0`, but that index is also fed to `layer_done_counter.set_consumer()`
via `tp_worker`, so it can make the worker wait on the wrong event; the port uses a separate
`hicache_loading` flag. Once that patch has a measured performance cost, HiCache can come back on
with it.
