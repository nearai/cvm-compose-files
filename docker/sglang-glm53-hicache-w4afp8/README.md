# GLM-5.3 Flash — HiCache + W4AFP8 engine

This is `docker/sglang-glm53-w4afp8` rebased onto the **HCC-safe HiCache** image instead of the
plain admission-reserve image, so a w4afp8 checkpoint and hierarchical caching can run together.
It exists because the two capabilities previously lived in different images: the deployed W4AFP8
image is built from `e9d29a1c` (admission reserve, **no** HiCache patches), so turning on
`--enable-hierarchical-cache` there on a TEE host takes the `cudaHostRegister` path and fails with
CUDA error 801.

## What it layers

Base `sha256:3eccc307…` (HiCache + admission reserve v10) plus two correctness patches, both
byte-identical copies of the reviewed originals, one opt-in performance patch, two patches that move
blocking work off the HTTP event loop, one diagnostic, two opt-in measurements and an opt-in self-profiler:

| patch | origin | scope |
| --- | --- | --- |
| `chunked-prefill-pool-clamp.diff` | `docker/sglang-glm53-pool-clamp` | `PrefillAdder.add_chunked_req` only |
| `modules-to-not-convert.diff` | `docker/sglang-glm53-w4afp8` | `W4AFp8Config.from_config` only |
| `dsa-indexer-qsplit.diff` | new here (opt-in, `SGLANG_DSA_INDEXER_QSPLIT=1`) | `IndexerKPool._get_topk_ragged_kpool_plan` only |
| `sglang-pr30771.diff` | upstream sglang PR #30771 (open), byte-identical to its diff at head `60a56aa` | `OpenAIServingBase.handle_request`, `TokenizerManager._tokenize_texts`, plus the PR's three CPU tests |
| `shm-off-loop.diff` | new here, applied after `sglang-pr30771.diff` | `TokenizerManager._send_one_request`, requests with multimodal inputs only |
| `event-loop-stall-dump.diff` | new here (diagnostic, on by default at 30 s) | new `utils/event_loop_stall_dump.py` plus a hook in the HTTP server lifespan |
| `ghost-prefix-cache.diff` | new here (opt-in, `SGLANG_GHOST_CACHE=1`) | new `observability/ghost_cache.py` plus one call at the top of `mem_cache/common.py` `release_kv_cache` |
| `kv-tier-metrics.diff` | new here (opt-in, `SGLANG_KV_TIER_METRICS=1`) | new `observability/kv_tier_metrics.py` plus recorder hooks in `mem_cache/unified_cache/unified_tree_core.py` eviction and load-back paths |
| `near-self-profile.diff` | new here (opt-in, `NEAR_SELF_PROFILE=1`) | new `utils/near_self_profile.py` plus a creation call in `Scheduler.init_profiler`, a `tick` call in `Scheduler.run_batch` and one barrier guard in `SchedulerProfilerManager._stop_profile` |

## DSA indexer query split (opt-in)

GLM-5.3 Flash's DSA indexer is `ReplicatedLinear`, so every attention-TP rank computes
`fp8_mqa_logits` and the K-pool top-k for **all** prefill query rows. At long context that is the
only prefill cost that grows with context (about 78 ms of a 4096-token chunk at 500K on H200 TP4).
With `SGLANG_DSA_INDEXER_QSPLIT=1` each rank scores a quarter of the rows against the same gathered
keys and the int32 top-k rows are all-gathered (about 32 MB per layer per 8K chunk). The logits
scratch per rank shrinks by the TP size too, which is what lets 16384-token chunks fit. It is only
used for prefill batches of at least `SGLANG_DSA_INDEXER_QSPLIT_MIN_ROWS` rows (default 1024).

Evidence (gpu31/gpu32, 2026-09-24, lab form of this patch; every A/B in both island orders):

- Stored long-tier traffic, paired TTFT split ÷ base: 0.83 / 0.84 (burst of 8 × ~700K),
  0.83 / 0.85 (long-a2), 0.81 (burst at chunk 16384). 100% completion on every arm.
- Burst at chunk 16384 (the gpu02 memory cliff): minimum free device memory 9.5 GB with the split,
  0.65 GB without.
- Correctness: logits bit-identical; top-k sets differ only on exact ties that the unsplit kernel also
  flips run to run. GSM8K 97.57% vs 97.57%; passkey retrieval 12/12 vs 12/12 (58K–449K tokens);
  teacher-forced top-1 agreement on 116K / 359K real text 97.75% / 96.73% vs a base-vs-base noise
  floor of 97.41% / 96.97%.

## Blocking work moved off the HTTP event loop

The `sglang serve` main process (the tokenizer manager) answers `/health`, `/metrics` and
`/v1/chat/completions` from one uvicorn event loop, so any synchronous work on that loop also stops
`/health`. The CVM proxy marks a replica unhealthy after three consecutive missed probes (3 s timeout
each). On the gpu02 long tier this showed up as replicas flapping under image-heavy traffic.

- `sglang-pr30771.diff` runs request conversion (chat template rendering and `tokenizer.encode` in
  `_convert_to_internal_request`) and the regular-tokenizer path of `_tokenize_texts` in a
  single-worker request-preprocessor thread. Requests are still preprocessed one at a time, now in
  that thread.
- `shm-off-loop.diff` moves the dispatch-time copy of multimodal features into `/dev/shm`
  (`wrap_shm_features`: `posix_fallocate` plus `copy_`, about 141 MiB per large image) to its own
  single-thread executor, `sglang-mm-shm`. A multi-GiB copy then neither blocks the loop nor waits
  behind other requests' preprocessing. If the request is cancelled during the copy, its segments are
  discarded once the copy finishes. It extends the executor setup PR #30771 adds, so it is applied
  after it.

Evidence (gpu32, 2026-09-25, CC off): the same source bytes bind-mounted over the v1 image
`fde25985…` on an arm that also ran `--enable-dynamic-batch-tokenizer`, one request at a time,
`/health` probed every 200 ms. Worst `/health` probe per request, in two sessions, each patched run
next to a stock arm in the same half hour:

| request | stock | PR #30771 alone | PR #30771 + shm-off-loop (+ stall dump at 30 s) |
| --- | --- | --- | --- |
| 4.7 MB text chat (653,502 prompt tokens) | 3,444 / 3,033 ms | 38 ms | 30 ms |
| 64 images of 9 MP | 3,047 / 3,098 ms | 3,665 ms | 163 ms |
| 64 images of 48 MP | 3,046 / 2,976 ms | 3,903 ms | 551 ms |
| 64 images of 169 MP | 4,021 / 4,062 ms | ≥ 5,007 ms (probe timeout) | 1,258 ms |

- PR #30771 removes the text stall. With it applied, py-spy puts each remaining image stall in
  `ShmPointerMMData.__init__` on the event loop: 64 segments of ~141 MiB, 8.8-8.9 GiB per request.
  That copy is what `shm-off-loop.diff` moves.
- With both patches no probe went over 3 s: 0 of 3,211, against one over 3 s on three of the four
  requests on the stock arm. py-spy taken during the copies shows the loop thread idle while
  `sglang-mm-shm_0` runs `ShmPointerMMData.__init__`.
- The image stalls with PR #30771 alone ran 0.6-1.0 s longer than on the stock arm. That code is not
  touched by the patch, and two unpatched arms had differed by up to 0.3 s in either direction the
  same day, so the difference is not attributed to it.
- `prompt_tokens` stayed identical on 11 correctness and small-image cases and on the four large
  requests. The PR's own tests pass 5/5 on the patched files; on the stock files 3 fail and 2 are
  skipped.
- A 15-minute flood of screenshot conversations through the pinned proxy (450 requests, GPUs about
  90 % busy) caused no unhealthy transitions on either the patched or the stock arm.
- The shm move was tested CPU-only in the v1 image with real `ShmPointerMMData` segments
  (16 × 141 MiB per request, 32 CPUs). The stock inline copy blocked the loop for 897 ms; with the
  patch the longest loop gap was 8.1 ms, the copy ran on `sglang-mm-shm_0`, and the features arrived
  byte-identical. A request cancelled mid-copy left 0 segments behind, while a plain
  `run_in_executor` without the cleanup leaked all 16.
- `test-cpu.sh` step 6 checks the same mechanism without touching `/dev/shm`, which the publish
  workflow's test container keeps at Docker's 64 MB default: the copy runs on `sglang-mm-shm` while
  the loop keeps running, and a cancelled copy is discarded. It fails if the copy is put back on the
  loop or the discard is dropped.
- What remains with both patches (163 ms, 551 ms and 1,258 ms above) comes just before the copy: C
  code holding the GIL on the loop where `process_mm_data_async` returns, the same line on patched
  and stock engines. It grows with the source image size, from 9 MP to 169 MP (13000x13000). Neither
  patch touches it.
- `prompt_tokens` were identical across all runs, every request returned HTTP 200, `/dev/shm` was
  back to the same used bytes and segment names 13 s after every request, and the engine log had no
  `Traceback`, no `Already borrowed` and, at the default 30 s, no stall dump. End-to-end time matched
  the stock arm on three requests. The 9 MP request took 256 s against 81 s, all of it before
  dispatch in image processing, during host memory pressure; the patches do not run before dispatch.
- Not covered: the minutes-long hang seen on gpu02 has not been reproduced in the lab, so neither
  patch is known to fix it. The stall dump below is there to capture it.
- Also not covered on GPU: the production engines do not set `--enable-dynamic-batch-tokenizer`,
  which on the lab arm ran the second tokenization pass in its own thread. In production that pass
  goes through PR #30771's `_tokenize_texts` change instead, which only the PR's unit test covers.

## Event-loop stall dump (on by default)

`event-loop-stall-dump.diff` adds `python/sglang/srt/utils/event_loop_stall_dump.py` and arms it
from the HTTP server's lifespan, before the warmup thread starts. A task on the loop records a
heartbeat every second and a daemon thread (`event-loop-stall-watch`) checks it.

- After `SGLANG_EVENT_LOOP_STALL_DUMP_SECS` (default 30) without a tick, it writes a header and every
  thread's Python stack to stderr, one line per write, each line prefixed `[event-loop-stall]`. The
  header has the pid, the stall age and how much CPU the loop thread used since its last tick.
- It repeats every `SGLANG_EVENT_LOOP_STALL_DUMP_REPEAT_SECS` (default 60) while the stall lasts, and
  logs `event loop recovered after N s` when the loop ticks again.
- If the loop thread holds the GIL in one long C call, the Python watcher cannot run. faulthandler
  then dumps every thread without the GIL 3 s later, as an unprefixed `Timeout (0:00:33)!` block.
- `0` disables it. An invalid value logs one `event-loop stall dump not armed` warning and the server
  starts normally. At startup it logs
  `event-loop stall dump armed: stall=30s repeat=60s (pid N, loop thread 0x…)`.
- Request handling is unchanged. The cost is one timer re-arm per second on the loop (median 198 µs
  in the image on gpu32) and one watcher wakeup per second. Stalls shorter than the threshold are not
  dumped.
- Validated CPU-only: `test_stall_dump.py` (58 checks, shipped here) runs as step 7 of `test-cpu.sh`.
- Validated live on a GPU engine (gpu32, threshold lowered to 2 s): armed at startup, silent through
  startup, warmup and 2.5 minutes idle. One 64-image request produced two dumps, both naming
  `ShmPointerMMData.__init__` (`dst.copy_`) on the event-loop thread with the loop thread using 2.0 s
  of CPU in the first 2.8 s, then `event loop recovered after 5.3s`. Each dump was about 16 KB for 11
  threads.

To read it, filter the engine log on `[event-loop-stall]`. Loop-thread CPU near 0 means the loop is
waiting (a lock, I/O, the GIL); CPU close to the stall age means it is computing.

## Ghost prefix cache (opt-in)

The engine reports the prefix hits it got, but not the hits it missed because the KV had been
evicted. So "would a bigger KV cache, write_back, or a disk tier help?" could only be answered by
deploying each change and waiting. `ghost-prefix-cache.diff` measures it directly, without storing KV
or text.

With `SGLANG_GHOST_CACHE=1`, TP rank 0 takes each finished request (not aborted) as its KV is
released and hands its token ids to a background thread (`sglang-ghost-cache`), which:

1. cuts prompt + output into 64-token pages and hashes them as a chain, so each page hash covers
   every token before it. The hash is BLAKE2b keyed with 32 random bytes drawn at process start.
2. keeps an LRU of page hashes only, and for each prompt page records whether it was never seen
   (a compulsory miss no cache can save) or seen before at LRU stack distance d: the number of
   distinct tokens touched since that page was last used. A page hits in an LRU cache of C tokens
   exactly when d < C, so one measurement gives the hit rate at every cache size. The chain keeps
   the radix-tree prefix property: an ancestor page is touched whenever a descendant is.
3. tracks only pages whose hash falls in a 1/`SGLANG_GHOST_CACHE_SAMPLE` slice (default 16) and
   scales their counts, which bounds memory and CPU (SHARDS sampling). Hashing still covers every
   page, because the chain needs it.

**Confidentiality.** The key never leaves process memory and is never logged. Only 16-byte digests
are kept, in RAM, and they are gone on restart. Nothing per request is logged or exported. What leaves
the process is aggregate Prometheus counters, the same kind of numbers as `sglang:cached_tokens_total`.

| metric | meaning |
| --- | --- |
| `sglang:ghost_prompt_tokens_total` | prompt tokens of the accounted requests |
| `sglang:ghost_actual_cached_tokens_total` | tokens the engine really served from cache, same requests |
| `sglang:ghost_lookup_tokens_total` | full-page prompt tokens looked up (sampled, scaled) |
| `sglang:ghost_reused_tokens_total{within="5M"}` | of those, seen before at LRU distance under 5M tokens (cumulative buckets 0.25M-1024M, and `inf` = seen at any distance) |
| `sglang:ghost_reused_age_tokens_total{within_s="600"}` | seen before, last used under 600 s earlier (cumulative, 10 s-24 h, `inf`) |
| `sglang:ghost_requests_total`, `sglang:ghost_dropped_requests_total` | accounted, and skipped because the queue was full |
| `sglang:ghost_tracked_pages` | sampled hashes held (bounded by `SGLANG_GHOST_CACHE_MAX_PAGES`, default 1,000,000) |

How to read it, per replica over a window (each counter's `increase`):

- **predicted hit rate with a C-token LRU cache** = `reused{within=C}` / `lookup`
- **compulsory misses** = (`lookup` - `reused{within="inf"}`) / `lookup`: the share no cache can save
- **avoidable misses** = `reused{within="inf"}` / `lookup` - `actual_cached` / `prompt`
- the long tier's device pool is about 3.5M tokens; a 406 GiB write_through host tier holds about 5M
  in total (it is an inclusive copy), 650 GiB about 8M; write_back adds host to device.

Cost when enabled: 0.06 / 0.56 / 2.8 / 14 ms of background-thread CPU per 1.5K / 20K / 100K / 500K-
token request (hashing dominates), a list copy of the token ids on the scheduler thread, and about
150 bytes per tracked page (about 150 MB at the default cap, which covers about 1B tokens of history
at 1/16). The engine's scheduling and caching are unchanged.

Validated CPU-only by `test_ghost_cache.py` (step 8 of `test-cpu.sh`): with sampling off the predicted
hit tokens equal a brute-force LRU page cache at six sizes; at 1/16 they stay within 0.5 points;
compulsory misses equal never-seen pages; only keyed digests are retained; the tracked set is
bounded; the hook is a no-op when disabled.

Model limits: it predicts an LRU cache at page granularity. The engine rounds hits to its 256-token
tree page, and a restored host hit only pays when loading it is faster than recomputing (true for
the host tier). Without shared mode each replica has its own key, so cross-replica reuse is not
visible.

### Shared mode: one pooled ghost cache per CVM

Two replicas in one CVM keep separate KV caches, and conversation affinity pins each conversation to
one of them. Shared mode measures what that costs, for #304 (shared KV) and routing decisions:

- `SGLANG_GHOST_CACHE_KEY_FILE` on every replica points at the same file on an in-memory volume that
  only this CVM's replicas mount. The first replica to start creates it (32 random bytes, mode 0600,
  atomic hard-link, so racing replicas agree); the others read it. The same prefix then has the same
  digest on every replica.
- `SGLANG_GHOST_CACHE_SOCKET` makes each engine also send its sampled digests (1/16 of pages, never
  tokens) as unix datagrams to the aggregator; `SGLANG_GHOST_CACHE_REPLICA` names it in the metrics.
  Sends never block: if the aggregator is down or behind, the message is dropped and counted in
  `sglang:ghost_forward_dropped_total`.
- The aggregator is a sidecar running the same image,
  `python3 -m sglang.srt.observability.ghost_aggregator --socket /ghost/aggregator.sock --port 9464 --model-name z-ai/glm-5.3-flash`.
  It keeps one LRU across replicas (digests in RAM only) and exports, per replica:

| metric | meaning |
| --- | --- |
| `sglang:ghost_pool_lookup_tokens_total{replica}` | prompt tokens looked up (sampled, scaled) |
| `sglang:ghost_pool_reused_tokens_total{replica,within}` | seen before on any replica, pooled LRU distance under `within` (cumulative, `inf` = any) |
| `sglang:ghost_pool_other_replica_only_tokens_total{replica}` | seen before, but only on other replicas: what sharing KV or routing there could have served |
| `sglang:ghost_pool_messages_total`, `sglang:ghost_pool_bad_messages_total`, `sglang:ghost_pool_tracked_pages` | health |

Read `other_replica_only / lookup` as the upper bound on what cross-replica sharing recovers, and
`ghost_pool_reused{within=2C}` against the engines' own `ghost_reused{within=C}` as the value of
one cache of combined size C+C over two caches of size C.

Compose sketch (not in any prod file yet; needs the published v4 digest):

```yaml
volumes:
  ghost: {driver: local, driver_opts: {type: tmpfs, device: tmpfs, o: "size=1m,mode=0700"}}
services:
  model-sg-glm53-w4afp8-tp4-r1:
    environment:
      - SGLANG_GHOST_CACHE=1
      - SGLANG_GHOST_CACHE_KEY_FILE=/ghost/key
      - SGLANG_GHOST_CACHE_SOCKET=/ghost/aggregator.sock
      - SGLANG_GHOST_CACHE_REPLICA=r1
    volumes: [ghost:/ghost]
  # r2: the same with SGLANG_GHOST_CACHE_REPLICA=r2
  ghost-aggregator:
    image: <v4 digest>
    command: python3 -m sglang.srt.observability.ghost_aggregator --socket /ghost/aggregator.sock --port 9464 --model-name z-ai/glm-5.3-flash
    volumes: [ghost:/ghost]
```

Validated CPU-only (step 8): 9 racing loaders get one key; the wire format round-trips and splits
large requests into datagrams under 64 KB; the pooled counts equal a brute-force two-replica
simulation with a failover halfway through (cross-replica tokens before it from shared system
prompts, and after it from the failed-over conversations); and two recorders feeding a running
aggregator over a real socket produce /metrics equal to the direct computation.

## KV tier metrics (opt-in)

The stock cache counters (`sglang:evicted_tokens_total`, `load_back_tokens_total`,
`hicache_backup_tokens_total`, `hicache_dropped_tokens_total`) are created on every TP rank without a
rank label, so multiprocess Prometheus sums them to TP x the real volume. They also cannot tell a VRAM
eviction that demotes a prefix to DRAM from one that deletes it, or a DRAM eviction that frees a
duplicate of VRAM-resident KV from one that throws away the only copy, which is what decides whether
the HiCache host tier is used well. `kv-tier-metrics.diff` records those events.

With `SGLANG_KV_TIER_METRICS=1`, TP rank 0 (PP rank 0) only:

| metric | meaning |
| --- | --- |
| `sglang:kv_tier_vram_evicted_tokens_total{outcome="demoted"\|"deleted"}` | Full-KV tokens evicted from VRAM, kept on DRAM or gone |
| `sglang:kv_tier_dram_evicted_tokens_total{kind="duplicate"\|"dram_only"}` | Full-KV tokens evicted from DRAM, a copy of VRAM data or the only copy |
| `sglang:kv_tier_load_back_tokens_total` | tokens restored from DRAM to VRAM |
| `sglang:kv_tier_{vram_evicted,dram_evicted,load_back}_idle_tokens_total{...,within_s}` | the same tokens by how long the prefix had been unused (cumulative, 10 s-7200 s, `inf` = all) |
| `sglang:kv_tier_vram_cached_tokens` | gauge: tokens the radix tree holds in VRAM |
| `sglang:kv_tier_dram_duplicate_tokens` | gauge: DRAM tokens that are also in VRAM (write_through copies) |
| `sglang:kv_tier_dram_only_tokens` | gauge: DRAM-only tokens, the part that can produce DRAM hits |
| `sglang:kv_tier_dram_only_matchable_tokens` | gauge: DRAM-only tokens with a mamba checkpoint at or below them; a hybrid-model match must end at one, so the rest cannot hit |

The gauges come from an exact walk of the tree, at most every `SGLANG_KV_TIER_METRICS_RESYNC_S`
seconds (default 30), triggered by an eviction or load-back. Idle time is wall-clock: the patch turns
`UnifiedTreeNode.last_access_time` into a property whose setter also stamps `last_access_wall`
(`time.monotonic()`). Every hook runs through `kv_tier_metrics.safe`, which logs and swallows errors,
so accounting cannot break eviction. Eviction and caching decisions are unchanged.

Validated CPU-only by `test_kv_tier_metrics.py` (step 9 of `test-cpu.sh`): inert by default, 15
counters and gauges exact against a fake tree, all hooks present in the installed tree core, `safe()`
logs unless strict. Step 9 also runs the upstream `test_unified_radix_cache_unittest.py` on CPU with
the recorder forced on and hook errors raised (`SGLANG_KV_TIER_METRICS_FORCE=1`,
`SGLANG_KV_TIER_METRICS_STRICT=1`, test-only switches): 489 tests pass, the tests that need a CUDA
device are skipped, and `kvtm_check_plugin.py` fails the run unless the recorder existed and recorded
VRAM evictions and load-backs. It deselects three tests that replace the tree's node arena with a
Mock, which the strict-mode gauge walk cannot iterate (outside strict mode the error is logged and
they pass); on the unpatched file the same run passes 492.

## Self-profiling hook (opt-in)

Inside a TEE (NVIDIA CC, PPCIe) nobody can shell in or fetch files, but engine stdout reaches Loki.
With `NEAR_SELF_PROFILE=1` the engine profiles a few scheduler steps once, prints a compact summary of
where decode time goes (host syncs, D2H copies, launches, GPU idle) prefixed `NEAR_PROFILE `, and
deletes the trace. It exists to target the host-overhead work: on the production base tier the GPU is
only about half busy in the TEE and the scheduler spends most of its time inside `run_batch` (lab
measurements, see the PR that added this), and a profile taken in the TEE says which calls that is.

**Inert unless `NEAR_SELF_PROFILE=1`.** Unset, `0` or anything else: `maybe_create` returns `None`,
`Scheduler.near_self_profile` is `None`, the `run_batch` call is skipped and the barrier guard is
true. The hook runs on tp_rank 0 only; other ranks never create it.

| var | default | meaning |
| --- | --- | --- |
| `NEAR_SELF_PROFILE` | unset | `1` enables the hook |
| `NEAR_SELF_PROFILE_AFTER_S` | 600 | seconds after scheduler init before the first eligible decode batch |
| `NEAR_SELF_PROFILE_STEPS` | 50 | scheduler steps to record |
| `NEAR_SELF_PROFILE_ACTIVITIES` | unset | `cpu` forces the CPU-only profile (the fallback path) |

Behaviour:

- Once per process, at the first decode batch after `NEAR_SELF_PROFILE_AFTER_S`. It drives the
  scheduler's existing `SchedulerProfilerManager` (CPU+CUDA, `with_stack=True`, output
  `/tmp/near_selfprof`) for `NEAR_SELF_PROFILE_STEPS` steps. No new tracer.
- If starting with CUDA activity raises, or the trace has no kernel events (CUPTI restricted), it
  retries once CPU-only. CPU-only still shows time blocked inside `item()`, `synchronize()` and copies
  by code path. `SGLANG_PROFILE_V2` is not supported; the hook then disables itself.
- The trace is parsed by a niced child process (stdlib only, runs `near_self_profile.py` directly), so
  the scheduler's GIL is not held by the parse. The child prints at most about 30 lines and removes the
  trace dir.
- Any error prints one `NEAR_PROFILE error ...` line and disables the hook.
- `_stop_profile` skips its all-rank `torch.distributed.barrier` only while the hook's own profile runs
  (`near_selfprof` is set at start and cleared as soon as the profile stops or the hook errors);
  otherwise rank 0 would wait for ranks that never profile. Profiles started through the HTTP profiler
  API keep the barrier, before and after the hook has run.
- The hook does not start while another profile is in progress; it disables itself instead of touching
  it. It profiles tp_rank 0 only, so use it with PP=1 and attention-DP layouts (the production ones).
- The parse child sets `oom_score_adj` 1000 so the kernel kills it rather than the engine, and the
  hook removes `/tmp/near_selfprof` if the child dies.

Summary lines: `mode`, steps captured, per-iteration wall time (from `Scheduler.run_batch` spans), GPU
busy and idle % with an idle-gap histogram, kernel count, graph vs eager launches, blocking calls
(`cudaStreamSynchronize`, `cudaEventSynchronize`, `item()` ...) with count and blocked ms, the top 8
sglang code paths that block (`file:line` is the function definition line), memcpy counts and bytes by
direction, and the top 5 GPU idle gaps with the CPU frame running at that moment.

Cost when on, once per process: the torch profiler's overhead for 50 steps, then a synchronous trace
export in the scheduler thread, which stalls that replica for a few seconds and up to about 20 s. The
parse child needs a few GB of RAM for a 50-step trace with stacks; lower `NEAR_SELF_PROFILE_STEPS` if
the container is tight. Enable it on one replica, not a fleet.

**Open risk: CUPTI under CC.** Whether CUDA activity tracing works on a confidential-computing GPU has
not been tested; the lab runs had CC off. The CPU-only fallback is the mitigation, and
`NEAR_SELF_PROFILE_ACTIVITIES=cpu` forces it. It covers CUPTI failing with an exception or returning no
kernel events, not a CUPTI fault that crashes or hangs the process, so enable it on one replica that
can be restarted. The first TEE run answers this.

**Rollback:** unset `NEAR_SELF_PROFILE` (the hook is then inert), or redeploy the v6 digest; v7 changes
no v6 patch. The hook is per process and persists nothing, so a restart clears it.

The patch is applied last. `scheduler.py` and `profiler_manager.py` are touched by no other patch here
and are pinned to the base image's bytes (`scheduler.py` `6ffd1584…`, `profiler_manager.py`
`8e4a2992…`). Validated CPU-only by `test_self_profile.py` (step 10 of `test-cpu.sh`): inert by default,
summariser on synthetic traces, hook state machine against a stub profiler manager, hook sites present.

## Why the composition is safe

The HiCache patches touch `mem_cache/*`, `disaggregation/*` and `schedule_batch.py`. None of the
added correctness, performance or event-loop patches touches any of those files, so those patch sets
are disjoint. The two measurement patches do edit `mem_cache` files (`common.py`, and
`unified_cache/unified_tree_core.py`, see below) with observation-only hooks, against the bytes the
base ships with its HiCache patches applied. `source-manifest.json`
is the one from `docker/sglang-glm53-w4afp8` plus the `dsa_indexer_kpool.py` entry and the
event-loop entries: `http_server.py`, `serving_base.py`, `tokenizer_manager.py`, the new stall-dump
module and the three new test files (`"before": null` asserts that a file is new).
`tokenizer_manager.py` is the only file two patches touch (PR #30771, then shm-off-loop). Its
manifest entry spans both, and `PROVENANCE` records the hash between them.

`kv-tier-metrics.diff` is the one added patch inside the HiCache code: it edits
`mem_cache/unified_cache/unified_tree_core.py`, whose before-hash is the file as the base image ships
it (HiCache patches included). That file is byte-identical in the base `3eccc307…`, the published v3
image `47aff791…` and the lab build the patch was written against (checked 2026-10-05), and no
other patch in this recipe touches it:

```
e2f1884ddd55721786cb9b7365b5d0a619a9d9a6ebf88c9c09b690ee39b45b94  mem_cache/unified_cache/unified_tree_core.py
```

That disjointness was verified against the real base image, not assumed — all three patch targets (base bytes, before the self-profiling patch)
hash identically in `3eccc307…` and in `e9d29a1c…`:

```
21e9c527c9b83e350cdc35ce2bc62891cda1550934b2a5d302f0f807f752f125  layers/quantization/w4afp8.py
02def839ec597d2b141f9d97aace6dc376c047cf670e115e851c86cb875362f0  managers/schedule_policy.py
6ffd1584f6bcbbd3384f1ec45e3313a7b4b010fd3f900b8f19345d4b02f5d0ba  managers/scheduler.py
```

The event-loop targets were checked the same way in `3eccc307…` (2026-09-25). They are
byte-identical to fc91d24, and the four new paths do not exist there:

```
3d0613b92abae51e8566a11ae80a2369a46422b49bd63c4cd6aa593d8a4bdbc2  entrypoints/openai/serving_base.py
cc4d633d8c609300e095a16ef7fe460e2c3c0e0ad52fcc76e9134d0aee4dff51  managers/tokenizer_manager.py
553e5d1108dfded05e1693f7a26efd733e297b9ef7b238905e09e52a5e848652  entrypoints/http_server.py
```

`apply-patches.py` re-checks every before-hash at build time, then checks each patch with
`git apply --check` against the tree the previous patches left and applies it, in `PATCHES` order,
because `shm-off-loop.diff` edits lines that `sglang-pr30771.diff` adds. It then re-checks every
after-hash and `ast.parse`s each patched file, so a base drift fails the build rather than
producing a silently different image.

## Runtime

Added here: the DSA indexer query split is opt-in (`SGLANG_DSA_INDEXER_QSPLIT=1`), the two
event-loop patches have no switch, and the stall dump is on at 30 s
(`SGLANG_EVENT_LOOP_STALL_DUMP_SECS=0` turns it off). The ghost prefix cache (`SGLANG_GHOST_CACHE=1`),
the KV tier metrics (`SGLANG_KV_TIER_METRICS=1`) and the self-profiler (`NEAR_SELF_PROFILE=1`) are
opt-in. Inherited
opt-ins:

- admission reserve — `SGLANG_CHUNKED_PREFILL_ADMISSION_RESERVE`
  (never set `SGLANG_ADMISSION_RESERVE_MIN_WAIT_S`: the wall-clock gate desynchronises TP ranks and
  trips the NCCL watchdog)
- HiCache — `--enable-hierarchical-cache` plus `SGLANG_HICACHE_*`

**On an HCC/PPCIe (TEE) host `SGLANG_HICACHE_CUDA_HOST_MEMORY` must stay `1`.** It selects the
`cudaMallocHost` allocator; `0` falls back to anonymous mmap plus `cudaHostRegister` and fails with
CUDA error 801.

## Status

Qualified on gpu31 (2026-09-23): loads and serves, HiCache genuinely attaches
(`hicache_attached=True`, host pool 2,460,992 tokens, device pool unchanged at 3,558,464), and a
30-minute soak ran 420 requests per arm with **0 errors**.

**This does not qualify the TEE path.** gpu31 reports `CC status: OFF`, so the `cudaMallocHost`
vs `cudaHostRegister` allocator — the entire reason the HCC-safe base exists — was never
exercised. A soak on the intended HCC/PPCIe topology is still required before deployment.

Two interactions to measure rather than assume:

1. **HiCache's benefit should shrink.** W4AFP8 frees ~35 GB/GPU of weight memory into the device KV
   pool (3.56M tokens vs 1.45M). A device pool that large already holds much of what HiCache's host
   tier would have served, so the hit-rate lift may be small.
2. **Host RAM budget.** `SGLANG_HICACHE_RAM_BUDGET` defaults to `80%` and was sized against the fp8
   device pool. It is not obviously right for a 2.43× larger one.

**v3** (the event-loop patches and the stall dump) was run as a recipe against its base in a
CPU-only container: `apply-patches.py`, then `test-cpu.sh` steps 1-7. On GPU the lab bind-mounted
the same source bytes over the v1 image, all three patches together with the stall dump at its 30 s
default. No v3 build has run on GPU, so the published image first does so as the long-context r2
canary.

**v4** adds the ghost prefix cache. The recipe was built and `test-cpu.sh` steps 1-8 run in a
CPU-only container on gpu31 (2026-09-26). The GPU check compares its predictions with the hit rates
measured at 406 GiB, 650 GiB and with HiCache off on the same replay (gpu31 HiCache write-policy A/B).

**v5** adds the KV tier metrics on top of v4. The recipe was built and `test-cpu.sh` steps 1-9 run in
a CPU-only container on gpu31 (2026-10-05, tag `glm53-hicache-w4afp8:v5-obs`). On GPU the lab
bind-mounted the same module and patched tree core over the v4 lab image for the HiCache policy A/B
(gpu31, 2026-09-30); this build has not run on GPU.

**v7** adds the self-profiling hook on top of v6, with no change to any v6 patch. `scheduler.py` is
the only source file that was pinned unchanged (before = after) and now has a new after-hash. The hook
patch was applied to the base image's real `scheduler.py` and `profiler_manager.py` bytes and
reproduces the manifest hashes; `test-cpu.sh` step 10 was run against the patched sources outside the
image. A GPU run of the hook on gpu32 (TP4, CC off) completed in the lab, but its output was not
retained, so no GPU result is claimed, and that run predates the review fixes to the hook (no start
while another profile runs, barrier-skip flag cleared after the hook's profile, parse-child cleanup and
OOM priority), which are covered by `test_self_profile.py` only. Nothing has run in a TEE.

## Security remediation

The base image ships PyJWT 2.13.0, which has CVE-2026-102268 (critical, fixed in 2.14.0). PyJWT is only a
dependency of `msal`; SGLang does not import it. The Dockerfile installs the fixed 2.14.0 wheel by exact
hash (`security-requirements.txt`, `--require-hashes --no-deps`) and asserts the version. Published as
`glm53-hicache-w4afp8-v6`; the v5 publish run stopped at the fixable-critical gate and was never signed.
