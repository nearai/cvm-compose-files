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
blocking work off the HTTP event loop, one diagnostic, two opt-in measurements, an opt-in self-profiler, FP8 KV cache support for GLM's NoPE DSA (active only
with `--kv-cache-dtype fp8_e4m3`), an opt-in preprocess process pool and an opt-in tool-schema size cap:

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
| `fp8kv-flashmla.diff` | new here (flag-gated: `--kv-cache-dtype fp8_e4m3` with `flashmla_kv`) | 656 B fp8 row in `kv_cache_configurator.py`, q pad and kpool index pad in `dsa_backend.py`, `flashmla_kv` allowed for `index_kpool > 1` in `dsa_backend_kpool.py`, `k_pe=None` and hybrid-wrapper metadata in `forward_mha.py` |
| `preprocess-pool.diff` | new here (opt-in, `SGLANG_PREPROCESS_WORKERS=N`), applied after `sglang-pr30771.diff` and `shm-off-loop.diff` | new `managers/preprocess_pool.py`, a call in the HTTP lifespan and a pool attempt in `OpenAIServingBase.handle_request` |
| `tool-schema-depth-cap.diff` | new here (opt-in, `SGLANG_TOOL_SCHEMA_MAX_DEPTH` / `_MAX_NODES`) | new `utils/tool_schema_guard.py` plus one call in `OpenAIServingChat._validate_request` |

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

## FP8 KV cache for GLM's NoPE DSA (flag-gated, v8)

`fp8kv-flashmla.diff` lets GLM-5.3 Flash (NoPE MLA: `qk_rope_head_dim == 0`) run
`--kv-cache-dtype fp8_e4m3` on Hopper. On the stock v6 image that flag does not boot with any DSA
backend combination (tilelang fp8 KV is ROCm-only on CUDA; `fa3` cannot read the packed fp8 pool;
`flashmla_kv` rejects `index_kpool > 1`; tee-bench exp 26). The patch routes fp8 lanes through
`flashmla_kv` for prefill and decode, using the DeepSeek-V3.2 packed row that kernel hard-asserts:

- `mem_cache/kv_cache_configurator.py`: with a rope-less model and `flashmla_kv` selected, the fp8 pool
  row is 656 B (512 fp8 + 16 B of fp32 tile scales + 64 never-written zero bf16 rope slots) instead of
  1024 B for bf16. This sizes the pool and the cell-size estimate.
- `layers/attention/dsa_backend.py`: `_forward_flashmla_kv` zero-pads q from 512 to 576 so the kernel
  sees `d_qk == 576`, pads the `index_topk + index_kpool - 1`-wide (2051) index table with `-1` to a
  multiple of 128 (2176), and sizes the tile-scheduler metadata with the padded width.
- `layers/attention/dsa/dsa_backend_kpool.py`: `flashmla_kv` joins the backends allowed for
  `index_kpool > 1`.
- `models/deepseek_common/attention_forward_methods/forward_mha.py`: after the one-shot fp8 prefix
  dequant, `k_pe = None` for rope-less models, and the hybrid linear-attention wrapper's
  `full_attn_backend` is resolved before `forward_metadata` is read (the run-1 crash under load).

**It is active only with `--kv-cache-dtype fp8_e4m3` and the `flashmla_kv` DSA backends**
(`--dsa-prefill-backend flashmla_kv --dsa-decode-backend flashmla_kv`). With the default bf16 KV cache
(tilelang) the new q-pad width is 0, the padded topk equals `index_topk`, the pool row code is not
reached, the fp8 dequant helper is not called, and the `index_kpool > 1` whitelist only relaxes a check
that previously raised for `flashmla_kv`. Note that `flashmla_kv` with a bf16 cache is not meaningful
(it requantises the whole cache per call); fp8 KV needs both flags together.

Evidence (bare metal, CC off, lab image = v6 + this patch; never run in a TEE):

- Boot: KV pool x1.467 (base 1,062,848 -> 1,559,168 tokens per rank; long 1,557,952 -> 2,285,568). The
  row ratio is 1.56x; mamba state, scratch and the indexer cache are fixed costs. A 528 B row (1.94x)
  needs a new kernel and is not in this patch. (exp 27, `evidence/gpu32-fp8-kv-fix`)
- Quality: GSM8K n=100 0.99 fp8 vs 0.97 bf16 (base) and 0.979 vs 0.979 (long, answered); passkey 32K
  and 128K 3/3 both; greedy 20-prompt compare 16/20 (base) and 11/20 (long) identical, mean abs
  logprob delta about 0.010 (ordinary fp8 drift, no garbage). (exp 27)
- Latency: single-prompt TTFT equal or better (2K 0.24 vs 0.26 s, 128K 9.96 vs 10.32 s). (exp 27)
- Throughput, long tier at 12/4 caps: +17% tok/s with fp8. Base tier with fp8 + max-running 64 + 380
  mamba slots: +14-17% tok/s over prod bf16 at W2 3.0 / 4.0, two rotated rounds. That figure includes
  the max-running 64 effect (+8-13% alone, exp 26), so it is not fp8 alone. (exp 28)
- A cold prefill of about 1M tokens works on TP2: 88 s with fp8 vs 94 s with bf16, peak 129.8 GB.
  (exp 28)
- Negative result to keep in mind: at W2 2.0 on a cold fp8 lane, exp 27 saw -13% tok/s in one run
  (JIT-cold, not seen at 3.0 or in the long tier, unresolved by a single run). HiCache with 656 B rows
  ran in the lanes but its hit and transfer behaviour was not separately validated.

FP8 needs `flashmla_kv` for both prefill and decode (a mixed config would size a 656 B pool for a
backend that does not expect it, and `flashmla_kv` with a bf16 cache and `index_kpool > 1` is now
accepted but meaningless); the patch does not assert this, the canary flags must.

Rollout is a flag on a canary replica, plus the DSA backend flags, not an image switch: do not pass
`--kv-cache-dtype fp8_e4m3` and nothing in the image changes. The 4 files' bytes are pinned to the base
image's (`source-manifest.json`), and the patch is byte-identical to the one validated in exp 27.

## Preprocess process pool with a per-request deadline (opt-in, v8)

Root cause of inference-proxy #287 (tee-bench `evidence/long-tp2-wedge-rca`): patch 4 (PR #30771)
renders the chat template and tokenizes in `ThreadPoolExecutor(max_workers=1)`. One request that keeps
that thread busy for minutes parks every other request of the replica behind it, while the scheduler
idles, KV is empty and `/health` stays 200. A thread cannot be killed, so there was no deadline.

`preprocess-pool.diff` adds `managers/preprocess_pool.py`: N worker processes forked from a zygote
taken at HTTP-lifespan start (the zygote is one fork of the threaded server, made once before traffic
from the event-loop thread; replacing a worker later forks only the zygote), one request per worker at
a time, results pickled back. Every first worker must answer a ping at startup, otherwise (a child
that inherited a held lock) the pool is disabled and the thread path is used. A request that exceeds
`SGLANG_PREPROCESS_TIMEOUT_S` fails alone with **HTTP 422** `RequestPreprocessingTimeout` (the cause is
the payload; a 5xx makes gateways retry the same body, which turned one request into hours), its
worker is SIGKILLed and replaced; the write, the wait and the read all sit under that deadline. A
worker that dies while running a request (OOM kill, crash) fails that request with HTTP 422
`RequestPreprocessingWorkerDied` and is replaced; the request is not retried on the unguarded thread
path, since it may be what killed the worker. Multimodal `/dev/shm` hand-off and the `_tokenize_texts`
closures stay on the thread path; if the pool itself fails (not started, spawn failure, request not
picklable) the request falls back to the thread path, and a pool that breaks wakes the requests queued
for a worker.

**Default OFF.** `SGLANG_PREPROCESS_WORKERS` unset, `0`, negative or not an integer: `install()`
returns before forking anything, no attribute is set, and `handle_request` takes the patch-4 thread
path unchanged. The lab image defaulted to 4; here it is opt-in per replica, like every other
behaviour change in this recipe, because a pool forks processes that hold a copy-on-write tokenizer
inside a memory-constrained CVM and has not run in one.

| var | default | meaning |
| --- | --- | --- |
| `SGLANG_PREPROCESS_WORKERS` | 0 (off) | number of worker processes; the canary uses 4 |
| `SGLANG_PREPROCESS_TIMEOUT_S` | 60 | per-request deadline, then 422 for that request only |
| `SGLANG_PREPROCESS_LOG_SLOW_S` | 5 | INFO line (rid, message and tool counts, chars, images, seconds) when a worker takes longer |
| `SGLANG_PREPROCESS_TEST_HOOK` | unset | test only: `__PP_SLOW_<s>__` / `__PP_CRASH__` markers in a message |

Risks and how they are bounded:

- Fork safety: the one fork of the threaded server happens at lifespan start (the startup ping above
  catches a child stuck on an inherited lock); workers are forked from the single-threaded zygote.
  The zygote and workers drop the parent's `set_wakeup_fd` and restore default SIGTERM/SIGQUIT, so a
  signal delivered to a child cannot be reported to the parent's asyncio loop (tested).
- Lifetime: workers exit on EOF of their socket (the other end belongs to the server) and the zygote on
  EOF of its control socket, so a dead server leaves no orphans. Nothing calls `shutdown()` explicitly.
- Memory: each worker is a copy-on-write copy of the tokenizer manager at lifespan start and its RSS
  grows toward a full copy as refcounts touch pages. Size the CVM for N extra tokenizer copies in the
  worst case; start with `SGLANG_PREPROCESS_WORKERS=4` on one replica and watch RSS. The server runs
  under granian: with `tokenizer_worker_num > 1` every granian worker runs its own lifespan, so you
  would get N workers per server worker. The canary uses a single worker process.
- The zygote and workers inherit the server's open descriptors (listening socket, ZMQ ipc). They hold
  no state on them, but they keep them open until they exit. `pickle` of the request and result runs
  on the event loop (0.2 s for a 5M-id result).
- A client disconnect does not interrupt a running worker; a shielded task returns the worker to the
  pool or kills it at the deadline.
- The 60 s default deadline is far above the measured cost of any normal request (a 640K-token
  agent trace preprocesses in about 2 s). Raise it for known huge prompts.

Evidence: output through the pool is identical to the in-process path (prompt ids, sampling params,
routing key) at 13 to 3003 messages and up to 640K tokens, IPC overhead not measurable (RCA section 9,
`tests/real_equiv.py`). GPU (exp 28, lab image with the pool at 4 workers): three injected 120 s
blocking requests at W2 3.0 caused no stall, TTFT stayed 0.6 / 1.6 s. Bare metal only. CPU tests:
`test_preprocess_pool.py` (step 12).

## Tool-schema size cap (opt-in, v8)

`OpenAIServingChat._validate_request` runs `jsonschema` `Draft202012Validator.check_schema` on every
tool's `parameters` on the HTTP event loop, before the request reaches the preprocessor. A client
controls the size of that schema. `tool-schema-depth-cap.diff` adds `utils/tool_schema_guard.py` and one
call in `_validate_request`: a tool whose schema is over a limit gets the usual **HTTP 400** ("Tool N
function 'parameters' schema is too large: ...") before `check_schema` runs.

| var | default | meaning |
| --- | --- | --- |
| `SGLANG_TOOL_SCHEMA_MAX_DEPTH` | 0 (off) | max nesting of JSON objects/arrays inside one tool's `parameters` |
| `SGLANG_TOOL_SCHEMA_MAX_NODES` | 0 (off) | max number of JSON values (objects, arrays and every value inside them, scalars included) summed over all tools of one request |

Both default to off (the call returns immediately), so the image behaves as v7 until a replica sets
them. Suggested canary values: depth 32, nodes 25000. A schema level costs two containers (the schema
object and its `properties` / `anyOf` / `items` container), so depth 32 is about 16 schema levels, far
beyond real tool definitions. The RCA saw requests with 40 to 160 tools, so the node budget is shared by
the whole request and set high enough not to reject those (25000 values is about 1.3 s of `check_schema`
in the worst case); tighten it from the first week of canary logs (a rejection names the tool index
and the variable).

Why two limits, and a correction to the RCA's premise: the RCA measured nested `anyOf` at x2 per level
(depth 12 = 2.9 s) and extrapolated "depth 20 = 12 min". Measured again with `jsonschema` 4.26.0 (the
version in the base image), `check_schema` is **linear in the number of nodes**, about 50 us per node:
a chain of any shape up to depth 80 takes under 15 ms (no exponential in depth), scalar entries cost
the same as containers (`properties` mapped to `true`, a long `type` list: about 65 us per entry, so
they are counted too), and the doubling per
level in the RCA is the schema tree itself doubling when each `anyOf` level has two recursive branches
(depth 12 = 12288 nodes, 135 KB, 0.65 s here; depth 14 = 540 KB, 2.7 s). The cost is therefore bounded
by the body size, not by depth: a depth cap alone would not stop a wide bomb, so the node cap is the
one that bounds the stall, and the depth cap (requested) bounds recursion and pathological chains.
The walk is iterative and stops at the first violation. `test_tool_schema_guard.py` (step 13) covers
the boundaries, a 200k-deep and a cyclic structure, bad env values, and runs the shipped
`_validate_request` on stubs.

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
opt-in, FP8 KV needs `--kv-cache-dtype fp8_e4m3` with the `flashmla_kv` DSA backends, the preprocess pool
needs `SGLANG_PREPROCESS_WORKERS=N` and the tool-schema cap needs `SGLANG_TOOL_SCHEMA_MAX_DEPTH` /
`SGLANG_TOOL_SCHEMA_MAX_NODES`. Inherited
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

**v8** adds FP8 KV cache support, the opt-in preprocess pool and the opt-in tool-schema cap on top of
v7, with no change to any v6 or v7 patch. Applied in this order after `near-self-profile.diff`:
`fp8kv-flashmla.diff`, `preprocess-pool.diff` (stacked on the two event-loop patches: `http_server.py`
and `serving_base.py` change their after-hashes), `tool-schema-depth-cap.diff`. The four FP8 files and
`serving_chat.py` are pinned to the base image's bytes. All twelve patches were applied in order to the
real base image sources (registry layers of `3eccc307`) with `apply-patches.py`, which verified every
before and after hash, and the manifest hashes of the FP8 files equal the ones validated in exp 27.
`test-cpu.sh` steps 11-13 cover the new patches; steps 1-10 were not re-run end to end here (they need
the full image). The FP8 and pool GPU numbers are bare metal with CC off from lab images (v6 + FP8,
and + pool at 4 workers); nothing has run in a TEE.

## Security remediation

The base image ships PyJWT 2.13.0, which has CVE-2026-102268 (critical, fixed in 2.14.0). PyJWT is only a
dependency of `msal`; SGLang does not import it. The Dockerfile installs the fixed 2.14.0 wheel by exact
hash (`security-requirements.txt`, `--require-hashes --no-deps`) and asserts the version. Published as
`glm53-hicache-w4afp8-v6`; the v5 publish run stopped at the fixable-critical gate and was never signed.
