# GLM-5.3 Flash HiCache W4AFP8 engine with a self-profiling hook

The production engine image (`docker/sglang-glm53-hicache-w4afp8`, published digest
`sha256:47aff791…`) plus one patch: an env-gated, one-shot self-profiler. Inside a TEE (NVIDIA CC,
PPCIe) nobody can shell in or fetch files, but engine stdout reaches Loki. With `NEAR_SELF_PROFILE=1`
the engine profiles a few scheduler steps once, prints a compact summary of where decode time goes
(host syncs, D2H copies, launches, GPU idle) prefixed `NEAR_PROFILE `, and deletes the trace.

## Behaviour

- Inert unless `NEAR_SELF_PROFILE=1`. `assert-inert.py` checks that at build time.
- Only tp_rank 0 profiles, once per process, at the first decode batch after
  `NEAR_SELF_PROFILE_AFTER_S` seconds (default 600).
- It drives the scheduler's existing `SchedulerProfilerManager` (CPU+CUDA, `with_stack=True`, output
  `/tmp/near_selfprof`) for `NEAR_SELF_PROFILE_STEPS` steps (default 50). No new tracer.
- If starting with CUDA activity raises, or the trace has no kernel events (CUPTI restricted), it
  retries once CPU-only. CPU-only still shows time blocked inside `item()`, `synchronize()` and copies
  by code path. `NEAR_SELF_PROFILE_ACTIVITIES=cpu` forces that path for testing.
- The trace is parsed by a niced child process (stdlib only, runs `near_self_profile.py` directly), so
  the scheduler's GIL and memory are not used by the parse. The child prints at most about 30 lines and
  removes the trace dir. The scheduler thread only pays the profiler's own start and export cost.
- Any error prints one `NEAR_PROFILE error ...` line and disables the hook.

## Summary lines

`mode`, steps captured, per-iteration wall time (from `Scheduler.run_batch` spans), GPU busy/idle % and
an idle-gap histogram, kernel count, graph vs eager launches, blocking calls (`cudaStreamSynchronize`,
`cudaEventSynchronize`, `item()` ...) with count and blocked ms, the top 8 sglang code paths that
block (`file:line` is the function definition line), memcpy counts and bytes by direction, and the top
5 GPU idle gaps with the CPU frame running at that moment.

## Patch

`near-self-profile.diff` adds `python/sglang/srt/utils/near_self_profile.py`, creates the hook in
`Scheduler.init_profiler`, calls `tick(batch)` from `Scheduler.run_batch` just before the existing
profiler predicate, and makes `SchedulerProfilerManager._stop_profile` skip its all-rank barrier only
when the hook drove the profile (the other TP ranks never enter it). `apply-patches.py` refuses to run
unless the patch and the three source files match the hashes in `source-manifest.json`.

## Open risk

CUPTI under CC has not been tested; changing GPU CC mode was out of scope for the lab. The CPU-only
fallback is the mitigation. Memory: parsing a 50-step trace with stacks needs a few GB in the child
process; lower `NEAR_SELF_PROFILE_STEPS` if the container is tight.
