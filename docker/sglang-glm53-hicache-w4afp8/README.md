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
blocking work off the HTTP event loop, and one diagnostic:

| patch | origin | scope |
| --- | --- | --- |
| `chunked-prefill-pool-clamp.diff` | `docker/sglang-glm53-pool-clamp` | `PrefillAdder.add_chunked_req` only |
| `modules-to-not-convert.diff` | `docker/sglang-glm53-w4afp8` | `W4AFp8Config.from_config` only |
| `dsa-indexer-qsplit.diff` | new here (opt-in, `SGLANG_DSA_INDEXER_QSPLIT=1`) | `IndexerKPool._get_topk_ragged_kpool_plan` only |
| `sglang-pr30771.diff` | upstream sglang PR #30771 (open), byte-identical to its diff at head `60a56aa` | `OpenAIServingBase.handle_request`, `TokenizerManager._tokenize_texts`, plus the PR's three CPU tests |
| `shm-off-loop.diff` | new here, applied after `sglang-pr30771.diff` | `TokenizerManager._send_one_request`, requests with multimodal inputs only |
| `event-loop-stall-dump.diff` | new here (diagnostic, on by default at 30 s) | new `utils/event_loop_stall_dump.py` plus a hook in the HTTP server lifespan |

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

## Why the composition is safe

The HiCache patches touch `mem_cache/*`, `disaggregation/*` and `schedule_batch.py`. None of the
added patches touches any of those files, so the patch sets are disjoint. `source-manifest.json`
is the one from `docker/sglang-glm53-w4afp8` plus the `dsa_indexer_kpool.py` entry and the
event-loop entries: `http_server.py`, `serving_base.py`, `tokenizer_manager.py`, the new stall-dump
module and the three new test files (`"before": null` asserts that a file is new).
`tokenizer_manager.py` is the only file two patches touch (PR #30771, then shm-off-loop). Its
manifest entry spans both, and `PROVENANCE` records the hash between them.

That disjointness was verified against the real base image, not assumed — all three patch targets
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
(`SGLANG_EVENT_LOOP_STALL_DUMP_SECS=0` turns it off). Inherited opt-ins:

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
