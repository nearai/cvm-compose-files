# GLM-5.3 Flash on SGLang v0.5.23 (NEAR AI patch series)

This recipe replaces the `fc91d24` fork stack: `sglang-glm53-fc91d24` → `-hicache` → `-hicache-admission-reserve` → `-hicache-w4afp8` (v12/v13). It builds upstream SGLang **v0.5.23** (tag 297e54fd, released 2026-10-28) with our patch series.

Upstream v0.5.23 closes these CVEs, which are open on the fork:
- CVE-2026-15976: `update_weights_from_disk` pickle RCE, fixed by #42760;
- CVE-2026-86793: `update_weights_from_tensor` unpickler bypass, fixed by #39858 and #40259;
- CVE-2026-15969: LoRA-from-tensors unpickler bypass, fixed by #41322 and #40259.

The series adds the fixes v0.5.23 still lacks:
- credential redaction (CVE-2026-15977; upstream #48518 landed after the release);
- exact-rid aborts (CVE-2026-112266, a customer-reachable abort of other users' requests);
- customer-data log redaction (media URLs, base64, JSON schemas, tool-argument values).

The base is `lmsysorg/sglang@sha256:f832b5c3…`, the `v0.5.23` / `v0.5.23-cu130` tag (amd64 `f7876bef…`, upstream build run 40978390987). That image ships `/sgl-workspace/sglang` as a git checkout of 297e54fd with an editable install. `apply-patches.py` refuses any other tree, checks every patch's sha256 against `SERIES`, and applies them in order. The image also gets the same hardening as `sglang-glm53-v0519`: hash-pinned nltk 3.10.3 + defusedxml 0.7.1 (the base has nltk 3.9.1), and removal of the two unused Nsight EFA sampler binaries.

## Series

How the series was built: the fork stack was rebuilt as commits on fc91d24, then each patch was ported to v0.5.23 by intent (3245 upstream commits apart), not re-applied as text. Patches that upstream already covers were dropped, with the upstream commit as evidence.

| # | patch | origin | verdict vs fork | prod flag / env |
|---|---|---|---|---|
| 01 | admission-reserve-v10 | `sglang-glm53-hicache-admission-reserve` | adapted. The reserve follows upstream's `PrefillBudget` refactor; the in-flight check is in `_select_prefill_admission`; it reads `prefix_len`/`extend_end`, because the fork's `prefix_indices`/`extend_range` no longer exist (the old code would have silently fallen back) | `SGLANG_CHUNKED_PREFILL_ADMISSION_RESERVE` (opt-in) |
| 02 | chunked-prefill-pool-clamp | `sglang-glm53-pool-clamp` | adapted. The OOM bug is still in v0.5.23 (`available_chunk_tokens` returns the full chunk at `remaining_total <= 0`); the clamp now lives in `mem_cache/prefill_budget.py` | unconditional |
| 03 | modules-to-not-convert | `sglang-glm53-w4afp8` | clean. Still needed: stock v0.5.23 cannot load `graphistry/GLM-5.3-Flash-W4AFP8` | unconditional |
| 04 | near-self-profile | v6 | adapted. Rank comes from `runtime_context.get_parallel()` | `NEAR_SELF_PROFILE` (opt-in) |
| 05 | rid-abort-exact-match | **new** | CVE-2026-112266. Aborts match the exact rid; `<rid>_<n>` children match only for an rid the worker does not own (`AbortReq.match_children`, last field, wire-compatible) | unconditional |
| 06 | dsa-indexer-qsplit | v1 | adapted. The row slice uses upstream's logits-budget row chunking (#40854) | `SGLANG_DSA_INDEXER_QSPLIT=1` (opt-in) |
| 07 | fp8kv-flashmla | v7 | adapted. Keeps our 656 B NoPE row; upstream's own FP8 row is 528 B and refuses `flashmla_kv` with kpool > 1 | `--kv-cache-dtype fp8_e4m3` + `flashmla_kv` |
| 08 | dsa-q8-flashmla-sparse | v9 | adapted. Upstream ships `flashmla_sparse_q8` but rejects FP8 + NoPE. Ours is accepted only with the 656 B row; upstream's own top-k pad is used | `--dsa-prefill-backend flashmla_sparse_q8` |
| 09 | sglang-pr30771 | upstream PR #30771 (still open) | adapted. v0.5.23 moves conversion off the loop on the default executor (#39771); we keep the single request-preprocessor thread and put tokenization on it | unconditional |
| 10 | shm-off-loop | v3 | clean, plus a lazy-init guard | unconditional |
| 11 | event-loop-stall-dump | v3 | clean | on, 30 s |
| 12 | preprocess-pool | v7 | adapted. A worker result that does not pickle falls back to the thread path instead of returning a 500 | `SGLANG_PREPROCESS_WORKERS` (opt-in) |
| 13 | tool-schema-depth-cap | v7 | clean. Runs before v0.5.23's `normalize_json_schema_types`/`check_schema` | `SGLANG_TOOL_SCHEMA_MAX_DEPTH`/`_MAX_NODES` |
| 14 | admin-auth-redaction | v13 | auth half dropped (upstream #41322 in v0.5.23). Redaction rebuilt from upstream #48518 and adapted: `build_server_info` (HTTP + gRPC + rust launch info), scheduler `internal_states`, dumps always a redacted dict, crash-dump `launch_command` | unconditional |
| 15 | customer-data-log-redaction | **new** | media URL → `<url host=H>`, `data:` → `<base64 len=N>`, schemas/values → `<len=N sha256=…>`, redacted tracebacks. Keeps the `NNN Client Error … for url:` wording that inference-proxy maps to a 400 | unconditional |
| 16 | runtime (HCC-safe HiCache) | `sglang-glm53-hicache` | adapted, partly dropped. Kept: `SGLANG_HICACHE_RAM_BUDGET` (central `host_memory_budget_bytes`), pooled transfers re-targeted to the new MLA host pool. Dropped (native in v0.5.23): DSA index side cache, compressed-index ownership, MLA geometry checks | `SGLANG_HICACHE_*` |
| 17 | hcc-managed-memory | `sglang-glm53-hicache` | adapted. The `cudaMallocHost` path runs before upstream's ROCm branch | `SGLANG_HICACHE_CUDA_HOST_MEMORY=1` (must stay 1 on TEE hosts) |
| 18-19 | tests / hcc-tests | `sglang-glm53-hicache` | cut down to the kept features | – |
| 20 | ghost-prefix-cache | v6 | adapted. Model label from `get_serving()` | `SGLANG_GHOST_CACHE` (opt-in) |
| 21 | kv-tier-metrics | v6 | clean, plus the same label fix | `SGLANG_KV_TIER_METRICS` (opt-in) |
| 22 | fork-thinking-token-budget | fork 1fea50d9fc | **needed**. v0.5.23 has no public `thinking_token_budget`: pydantic drops it silently and the GLM-5.3 default of 8192 no longer applies. Restores the mapping to `custom_params.thinking_budget` (8192 default for `glm5_next`; `null` opts out; an explicit native budget wins) | request fields |
| 23 | fork-final-prefill-abort-race | fork 4a7929c6ad (upstream PR #34153, open) | adapted (`reqs_to_exclude`, `discard_output_reason`). The scheduler half is only live with the overlap scheduler; prod runs `--disable-overlap-schedule` | unconditional |
| 24 | fork-glm5-perception-gate | fork 0cf7426be5 / fc91d2403c | tests only (`test/manual/test_glm5_next_perception.py` for GPU runs) | – |
| 25 | fork-glm53-review-findings | fork e4d313ef03 (remainder) | adapted. GLM video dispatch in the EPD encoder and hybrid-DSA draft state in P/D; prod uses neither | – |

Fork commits not carried, because v0.5.23 has them:
- the GLM-5.3-Flash branch (squash-merged upstream as #36507);
- the GLM5 image processor fallback and geometry (transformers 5.19.0 ships `Glm5NextProcessor` / `Glm5NextImageProcessor{,Pil}`; `--disable-fast-image-processor` maps to the PIL backend);
- the deferred DSA imports;
- NEXTN OOB (#37548), the KDA padding sentinel (#36885), MHC reduce-scatterv (#36884), KDA split-grid (#37250), the modelopt_fp4 swiglu gate, and the NEXTN alias.

The fork-era Mamba checkpoint grid was not carried either. Upstream reverted it for non-DCP in #40907, and v0.5.23's unified-radix tests pin the current accounting: free the unaligned tail, keep the aligned prefix matchable.

**Dropped as upstream:**
- `kda-zero-sync`: v0.5.23 computes the KDA token count on the host (#39688).
- `dsa-indexer-kpool-logits-budget`: same formula upstream (#40854).

**Not fixed upstream:** the CUDA-graph + HiCache-load IMA (sglang#19177 was closed unmerged; the TODO is still at `scheduler.py:4352`). The long tier keeps HiCache off.

## v0.5.23 behaviour changes that affect prod
- `SGLANG_EXPERIMENTAL_DSA_KPOOL_METADATA_FUSION` now defaults to true (the fork had false). Set it to 0 to keep the fork's behaviour; qualify both ways.
- Breakable prefill CUDA graphs are on by default for GLM-5.3-Flash (#42845; off in the fork), and they now capture the KDA extend. `--cuda-graph-backend-prefill disabled` restores the fork's behaviour; `SGLANG_DISABLE_KDA_PREFILL_GRAPH_EXTEND=1` drops only the KDA capture. Check prefill-graph memory and boot time.
- All 19 flags of the prod long-tier argv exist with the same meaning (field definitions moved to `arg_groups/fields/`). `--disable-fast-image-processor` still logs a deprecation warning, as on the fork.
- Upstream's logits-budget chunking calls `mem_get_info` per large call, not once per device. This could add per-step cost under TDX+PPCIe; measure it.

## Tests
`test-cpu.sh` (CPU only) runs the recipe checks plus the upstream unit tests that the series adds or edits. The recipe tests were ported to the v0.5.23 APIs. The redaction, log-redaction and rid tests fail on stock v0.5.23 and pass on this tree.
