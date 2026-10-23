# gpu13 GPU 0: serving Qwen3.8 beside the production privacy filter

## Why and what is being tested

From 2026-10-02 through 2026-10-05, gpu13 GPU 0 carried the privacy filter and Qwen3.8 at `--mem-fraction-static 0.60 --max-mamba-cache-size 128`. Qwen's non-static use grew from roughly 88 to 122 GiB over four weeks. The filter could no longer allocate a full batch: CUDA OOM, `os._exit(1)`, roughly 40 restarts/day, and classify HTTP 500s. The user then required a dedicated GPU for the filter.

On 2026-10-23 the user chose a bounded production co-location trial on **gpu13 GPU 0**. The hardened filter serves real classify traffic through `model-privacy-filter`, `proxy-privacy-filter`, and the existing route. `model-sg-qwen38-27b-fp8-tp1-coloc` serves real private Qwen3.8 traffic through `proxy-qwen38-27b`, pooled with r2 on GPU 2. There is no synthetic load. This trial relaxes the dedicated-card rule and frees gpu04 GPU 7 for its fourth GLM replica.

**Update (2026-10-23):** the small models now run on gpu23, and Levanto Sage takes GPU 2 of the small-models host. Qwen3.8 r2 and its `dcgm-qwen38-27b` exporter are removed. `proxy-qwen38-27b` pools only the coloc engine, which is now the only private Qwen3.8 replica on that host. Criterion 7 below compares against r2's last 14 days before its removal. The abort rule now takes private Qwen3.8 off that host until it is placed again.

**Update (2026-10-23, later):** on gpu23, the privacy filter and Qwen3.8 coloc pair moved together from GPU 0 to GPU 2 with the same images, caps, and Qwen static fraction 0.50. Levanto Sage now holds GPU 0. The trial's `config_variant` label remains unchanged. GPU memory criteria now read GPU 2 through `dcgm-privacy-filter`.

The dedicated Qwen3.8 engine on gpu13 GPU 2 grew from 128 to 139.2 GiB in five days at static fraction 0.85, mamba cache 320, and 64 running requests on a roughly 139.4 GiB card. SGLang image `lmsysorg/sglang@sha256:febfb971…` has no hard cap for non-static memory. The filter idles at 5–7 GiB and peaked at 20–33 GiB over 12-hour windows with batch 1, a 131072-token window, and SDPA. Qwen private serving averaged 2.3 running requests (p95 9, maximum about 25), and the filter received about 4,000 classify requests/day.

## Design and GPU 2 memory budget

`prod/small-models.yaml` retains the production filter image, model revision, API, `transformers==5.16.1`, batch 1, 131072-token window, stride 256, SDPA, and GPU 2 assignment. The filter now applies `torch.cuda.set_per_process_memory_fraction(0.25)`, about 35 GiB of the 139.4 GiB H200. The per-process cap is the active limit. Its existing expandable-segments allocator setting remains. The watchdog emits `filter_mem` with reserved, maximum reserved, and allocated GiB every interval.

On request CUDA OOM, the filter emits `filter_oom`, clears cached blocks, and retries with the same model weights and a deep-copied tokenizer. The retry window is the smaller of 16384 tokens and half the primary window; there is no retry if that would be under 8 tokens. If memory remains short, it returns HTTP 503 with `Retry-After: 2` and `{"error":"gpu_memory_pressure"}`. The retry and the 503 response run after their `except` blocks, once the failed forward's memory has been released with `gc.collect()` and `empty_cache()`. After `FILTER_OOM_EXIT_AFTER` (3) consecutive memory-pressure 503s with no successful classify in between, it logs `filter_oom_exit`, sends the 503, and exits so Docker restarts it, as production did. The watchdog's 36 GiB limit sits above the 0.25 cap and is not the restart path. The request path uses no classify lock and does not mutate the primary tokenizer, preserving production concurrency and its real memory peaks.

The coloc engine runs on GPU 2 with static fraction **0.50**, about 70 GiB, plus non-static allocations that must be measured. Relative to the dedicated GPU 2 recipe, mamba cache **320 → 128** reduces persistent memory; running requests **64 → 32** and queued requests **128 → 64** bound admission; chunked prefill and maximum prefill **8192 → 4096** reduce activation spikes; CUDA graph max batch **32** bounds graph workspace to the running-request ceiling. Its model, revision, parsers, mixed-chunk and metrics flags match the dedicated recipe. It does not set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: SGLang hands tensors between its processes over CUDA IPC, which under expandable segments needs `pidfd_getfd`, and the container may not call it. With the setting, the engine loaded on gpu23 on 2026-10-23 and then crashed on its first request (`pidfd_getfd: Operation not permitted`). It uses its own `qwen38_coloc_kernel_cache`, separate from r2's kernel cache. The downloader completes before it starts. The existing `otelcol-contrib` collector scrapes the coloc engine and proxy; Docker logs use the existing pipeline.

## Pass criteria and abort rule

Observe for **at least 14 days, ideally 28**. All criteria must hold:

1. Zero filter restarts after its initial start: Loki `{container_name="model-privacy-filter"} |= "Started server process"`.
2. Zero classify 5xx at `proxy-privacy-filter` (`http_requests_total{container_name="proxy-privacy-filter",endpoint="/v1/privacy/classify",status=~"5.."}`) and zero `filter_oom` or `filter_oom_exit` logs.
3. Zero coloc engine restarts after its initial start: Loki `{container_name="model-sg-qwen38-27b-fp8-tp1-coloc"} |= "Started server process"`.
4. Zero Qwen3.8 5xx at `proxy-qwen38-27b`.
5. GPU 2 framebuffer use from `dcgm-privacy-filter` (`DCGM_FI_DEV_FB_USED`) never exceeds **134 GiB**.
6. Estimated coloc memory, GPU 2 used minus the filter's `reserved_gb` in `filter_mem`, sets no new rolling 24-hour high during week two.
7. Coloc TTFT and ITL p95 stay within **20% of r2** over the same windows (since r2's removal, of r2's last 14 days).
8. Classify latency p95 is no worse than in the prior 14 days.

On **any** `filter_oom` or `filter_oom_exit`, classify 5xx, coloc OOM or crash, or GPU 2 above 134 GiB, `compose/down` only `model-sg-qwen38-27b-fp8-tp1-coloc`. The proxy health check marks it unhealthy within approximately 15 seconds (checks every 5 seconds, 3-second timeout, 3 failures), and GPU 2 is the filter's alone again. With r2 removed, private Qwen3.8 then has no replica on that host until it is placed again; follow up with a PR that does so. Record the timeline, metrics, logs, tag, and container identities.

## Deploy after merge

These are operator steps, not actions performed by this source change. Use only a **merged `v0.0.N` tag** and `prod/small-models.yaml`. Read gpu23's **full** dashboard environment map first and submit that complete map on every call. For each `compose/up` below, send the same scoped payload with `dry_run:true` first and review its plan, then send it with `dry_run:false`:

1. `services: ["model-privacy-filter"]`: recreate the hardened filter. Classify is briefly unavailable for about 1–3 minutes while it loads. Confirm classify returns HTTP 200.
2. `services: ["model-sg-qwen38-27b-fp8-tp1-coloc"]`: start the engine. It is not pooled until step 3. Confirm `/health`, GPU 2 memory, and classify HTTP 200.
3. `services: ["proxy-qwen38-27b"]`, then `services: ["nginx-qwen38"]` with `force_recreate:true`, then `services: ["qwen38-registrar"]` with `force_recreate:true`. gpu23's dual-ingress rule requires both nginx and the registrar after a proxy change: nginx caches proxy IPs, and skipping the recreation de-registered models on 2026-09-09. Expect an approximately one-minute Qwen3.8 blip, as on 2026-10-06.
4. `services: ["otelcol-contrib"]` with `force_recreate:true`, because Compose does not hash its inline collector config.

After each step, and again at the end, confirm every other container retained its previous `CreatedAt`. Confirm the coloc engine receives requests through `sglang_num_running_reqs`, `qwen3-8-27b` is healthy in `https://completions.near.ai/registry`, and the synthetics are green before starting the observation clock. A merged source tag alone does not perform a deployment.

## Interaction with PR #382

PR #382 changes the `proxy-qwen38-27b` image and also edits `prod/small-models.yaml` and `scripts/test_glm53_v7_fleet.py`. Whichever PR merges second rebases on the first. If both merge before deployment, one `proxy-qwen38-27b` recreation covers both changes.

## Teardown and rollback

If the abort rule fires, `compose/down` only `model-sg-qwen38-27b-fp8-tp1-coloc`, then follow up with a PR removing it from the pool. Before `compose/down`, dry-run `compose/up` on the same merged tag and file so Compose Manager has that checkout; `compose/down` does not check out a tag and ignores `dry_run`. Confirm the engine is gone and the filter continues to classify; with r2 removed, Qwen3.8 is unserved on that host until it is placed again. The hardened filter can stay.

## Brave

gpu13's small-models file is already not approvable for Brave, and gpu13 is not sampled. This trial has no Brave impact.

## Scope of the result

Qwen3.6 sharing is not tested. A pass qualifies only this exact gpu23 GPU 2 configuration, images, real classify and Qwen3.8 traffic, and observation window.
