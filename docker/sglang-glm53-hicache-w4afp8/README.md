# GLM-5.3 Flash — HiCache + W4AFP8 engine

This is `docker/sglang-glm53-w4afp8` rebased onto the **HCC-safe HiCache** image instead of the
plain admission-reserve image, so a w4afp8 checkpoint and hierarchical caching can run together.
It exists because the two capabilities previously lived in different images: the deployed W4AFP8
image is built from `e9d29a1c` (admission reserve, **no** HiCache patches), so turning on
`--enable-hierarchical-cache` there on a TEE host takes the `cudaHostRegister` path and fails with
CUDA error 801.

## What it layers

Base `sha256:3eccc307…` (HiCache + admission reserve v10) plus two correctness patches, both
byte-identical copies of the reviewed originals:

| patch | origin | scope |
| --- | --- | --- |
| `chunked-prefill-pool-clamp.diff` | `docker/sglang-glm53-pool-clamp` | `PrefillAdder.add_chunked_req` only |
| `modules-to-not-convert.diff` | `docker/sglang-glm53-w4afp8` | `W4AFp8Config.from_config` only |

## Why the composition is safe

The HiCache patches touch `mem_cache/*`, `disaggregation/*` and `schedule_batch.py`. Neither added
patch touches any of those files, so the two patch sets are disjoint and `source-manifest.json` is
reused unchanged from `docker/sglang-glm53-w4afp8`.

That disjointness was verified against the real base image, not assumed — all three patch targets
hash identically in `3eccc307…` and in `e9d29a1c…`:

```
21e9c527c9b83e350cdc35ce2bc62891cda1550934b2a5d302f0f807f752f125  layers/quantization/w4afp8.py
02def839ec597d2b141f9d97aace6dc376c047cf670e115e851c86cb875362f0  managers/schedule_policy.py
6ffd1584f6bcbbd3384f1ec45e3313a7b4b010fd3f900b8f19345d4b02f5d0ba  managers/scheduler.py
```

`apply-patches.py` re-checks every before-hash at build time, runs `git apply --check` first,
re-checks every after-hash, and `ast.parse`s each patched file, so a base drift fails the build
rather than producing a silently different image.

## Runtime

Nothing added here is switched on by default. Inherited opt-ins:

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
