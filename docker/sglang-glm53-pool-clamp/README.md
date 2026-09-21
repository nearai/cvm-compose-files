# GLM-5.3 Flash engine: chunked-prefill pool clamp

Layers one upstream correctness fix onto the deployed engine `sha256:e9d29a1c…`
(recipe `docker/sglang-glm53-admission-reserve`). Nothing else changes: same SGLang pin, same
admission-reserve v10 patch, same flags, no new environment variable.

## The defect

`PrefillAdder.add_chunked_req` clamps an in-flight chunked prefill with

```python
_rem_tokens = min(self.rem_chunk_tokens, int(self.rem_total_tokens))
```

and then, if that lands at or below zero, restores the **full** chunk budget:

```python
if _rem_tokens <= 0:
    if self.is_hybrid_swa:
        return req            # SWA parks the chunk and retries next pass
    _rem_tokens = self.rem_chunk_tokens   # <- no reference to the pool
```

`rem_total_tokens` is `available + evictable - rem_total_token_offset`, and that offset is a
*projection* of decode across the whole running batch (about `CLIP_MAX_NEW_TOKENS` per running
request, independent of prompt size). It therefore goes negative while the pool still holds a
little free space. The fallback then asks `alloc_extend` for a full `--chunked-prefill-size`
worth of tokens that do not exist, which raises `Prefill out of memory` and kills the engine.

Hybrid-SWA models get a park-and-retry escape one line above. Hybrid-SSM models — GLM-5.3 Flash
is one, which is why the crash log carries an `Available mamba:` line — get none.

Enabling the admission reserve on the long-context tier is what caused the outage. The tier ran
clean for 14 days, crashed 36 minutes after the build landed, and has been clean since the
rollback, so the reserve is the but-for cause and rolling it back was the right call.

The defective line itself is pre-existing upstream code — the admission-reserve v10 diff does not
modify it — but that is a statement about where the bug lives, not about what triggered it. The
reserve makes the precondition reachable: admitting short requests behind a long prefill is its
entire purpose, and every admitted request enlarges the decode projection that drives
`rem_total_tokens` negative. In the lab A/B under identical load, only the reserve arm ever
reached that state; the control arm logged zero. Both need fixing, and this recipe fixes the
latent defect so the reserve can be re-enabled safely.

## What it looked like in production

GLM-5.3 Flash long-context tier, gpu02 r2, 2026-09-17: four crash-restarts in about four hours,
32 `Prefill out of memory` events, every one of them identical.

```
Prefill out of memory. Try to lower your batch size.
Try to allocate 4096 tokens.
Available full tokens: 3136 (full_available_size=3136 + full_evictable_size_=0)
```

Zero such events on gpu02/03/04/23 in the 14 days before the build landed, and none since the
rollback. The other hosts run the same image with sub-100K traffic, where the prefix cache keeps
`evictable` large and the fallback is never reached.

## The fix

Clamp the fallback to what the allocator can actually serve, page-aligned and keeping one page of
margin exactly as the hybrid-SWA branch above does, and park the chunk if not even one page is
free. The caller already handles a parked chunk (see the "A parked chunk (add_chunked_req
hybrid-SWA early-return)" comment in `scheduler.py`), so this reuses an existing supported path
rather than inventing one. The healthy path is untouched: the new code only runs inside the
`_rem_tokens <= 0` branch.

## Validation

`test-cpu.sh` runs in the published image with no GPU and asserts the invariant "the extend
length never exceeds `available + evictable`" against every pool state observed in production
and in the lab reproduction, plus a healthy-cache case that must still get a full chunk. It fails
on the unpatched base image (7 violations) and passes here.

Reproduction and evidence: `gpu31:/data/inference-optimizer-20260913/results/oom6`
(`REPRO.md`, `FINDINGS.md`). The lab A/B crashed the admission-reserve arm twice and left the
control arm — identical image and mounted scheduler files, reserve unset — alive under the same
load.

## Build

Published by `.github/workflows/publish-glm53-admission-reserve.yaml` (`variant=base-clamp`).
`apply-patches.py` verifies both
scheduler files against `source-manifest.json` before and after applying the diff and refuses a
drifted base, so this recipe can only be built on top of the exact deployed engine.
