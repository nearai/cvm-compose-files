# gpu04 GPU 7: privacy filter + Qwen3.8 soak

## Why

This is a **synthetic soak**, not a serving route. It asks whether the privacy filter can share one H200 with Qwen3.8 without the GPU memory exhaustion seen on gpu13. gpu04 GPU 7 is the only free TEE GPU in the fleet at the planning snapshot: UUID `GPU-85a29d2c-5a46-97a9-f08d-5f916e6509db`, 0 GiB used. Recheck that state before deployment.

From 2026-10-02 through 2026-10-05, gpu13 GPU 0 carried both services with Qwen3.8 at `--mem-fraction-static 0.60 --max-mamba-cache-size 128`. The engine's non-static use (activations, CUDA graphs, and the PyTorch allocator) grew from roughly 88 to 122 GiB over four weeks. The filter could no longer allocate a full batch: CUDA OOM, `os._exit(1)`, roughly 40 restarts/day, and classify HTTP 500s. The user then required a dedicated GPU for the filter; this experiment tests whether a bounded sharing configuration earns a reversal of that requirement.

The dedicated Qwen3.8 engine on gpu13 GPU 2 has also grown: at static fraction 0.85, mamba cache 320, and 64 running requests, framebuffer usage increased from 128 to 139.2 GiB in five days on a roughly 139.4 GiB card. SGLang image `lmsysorg/sglang@sha256:febfb971…` (nightly `c4271c3`, 2026-08-14) has no hard cap for non-static memory. Its activation footprint scales with prefill chunk size; CUDA graph buffers scale with decode batch size. The filter idles at 5–7 GiB on its own GPU and peaked at 20–33 GiB over 12-hour windows with batch 1, `PRIVACY_MAX_LENGTH=131072`, and SDPA. Over three days, Qwen3.8 private serving averaged 2.3 running requests (p95 9, maximum about 25), with KV use p95 at 28% of a 680K-token pool. The filter received about 4,000 classify requests/day.

## Design

`prod/coloc-soak-privacy-filter-qwen38.yaml` is project `coloc-soak` inside gpu04's existing CVM. Its containers, default Compose network, and volumes are project scoped. It has no host ports, nginx, registrar, inference proxy, Docker socket, or dstack socket. Qwen and the filter alone pin GPU 7 by UUID. The downloader pins both model revisions and must complete before either GPU process starts. No DCGM exporter is added: gpu04's existing `dcgm-glm53` exporter already covers GPU 7.

On a 139.4 GiB H200, Qwen's static fraction 0.50 is about 70 GiB: approximately 26.5 GiB of weights, 17 GiB for 128 mamba slots, and 26 GiB KV. Its *non-static* allocation is additional and unbounded by that fraction. The filter's `torch.cuda.set_per_process_memory_fraction(0.25)` caps PyTorch allocations at about 35 GiB. The 36 GiB watchdog remains a last-resort clean restart. Headroom between those budgets and the 134 GiB pass ceiling is deliberately tested under load, not assumed safe.

Relative to the gpu13 dedicated Qwen recipe, static fraction 0.85 → 0.50 and mamba cache 320 → 128 free persistent space. Running requests 64 → 32 and queued requests 128 → 64 bound admission. Chunked prefill and maximum prefill 8192 → 4096 reduce activation spikes. CUDA graph max batch size 32 bounds graph workspace to the running-request ceiling. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` reduces allocator fragmentation. The image digest, model and revision, reasoning/tool parsers, and metrics flags match the dedicated recipe.

The filter keeps the production base image, `transformers==5.16.1`, server API, model revision, batch 1, 131072-token window, stride 256, and SDPA. Classify requests retain production's FastAPI threadpool concurrency, with no lock, so concurrent memory peaks remain visible. The extra PyTorch cap defaults to 0.25. After the final primary pipeline is built, a retry pipeline is built once with a deep-copied tokenizer whose window is the smaller of the effective primary window and `FILTER_OOM_RETRY_MAX_LENGTH=16384` (the effective window is 4096 after an eager fallback); both pipelines share one model object and GPU weights. On request CUDA OOM the filter emits one JSON `filter_oom` line, clears cached blocks, and uses the retry pipeline with stride capped below its window. A second OOM clears cached blocks and returns `503`, `Retry-After: 2`, and `{"error":"gpu_memory_pressure"}` without exiting. Neither request path mutates a shared tokenizer. Every watchdog interval emits one JSON `filter_mem` line containing `reserved_gb`, `max_reserved_gb`, and `allocated_gb`; exceeding `GPU_MEM_LIMIT_GB=36` still triggers a clean restart. Those controls are environment tunable. The cap applies to the PyTorch allocator; GPU telemetry remains the authority for whole-card use.

The CPU-only load generator uses seeded built-in words and fake names, emails, and phone numbers. It does not ingest customer data or fetch external data at runtime. It continuously streams Qwen chat completions at concurrency 4 with approximately 1–8K-token prompts and 256–1024 output tokens. Each 60-minute cycle contains a 15-minute concurrency-30 burst, above the observed p95 and close to the 32-request cap. Every two hours it adds three concurrent 32K–120K prompts and four 8192-token output requests. Filter arrivals are Poisson at 0.05/s (about 4,320/day), with log-uniform 1K–131K-token synthetic documents at approximately four characters per token. A 131K-token filter document begins with each Qwen burst. All profile sizes, concurrency, rates, intervals, and the RNG seed are environment tunable. This concentrates prefill, decode, and filter peaks into days to expose growth that previously took weeks.

The load generator logs one JSON line per request with timestamp, target, HTTP status, latency, Qwen TTFT, approximate prompt tokens, and error. Its `:9100/metrics` exposes request totals by target/status, latency and TTFT histograms, filter 503 count, and phase. The project's `otelcol-contrib` service retains the unique `coloc-soak-otelcol` container name and sends Qwen and load-generator Prometheus metrics to `telemetry.infra.near.ai` using the existing `otlphttp/gateway` pattern and `MONITORING_INGEST_TOKEN`; it attaches `host=gpu04` and `deployment=coloc-soak`. It does **not** open a second Docker log tailer against the shared `/var/lib/docker/containers` directory. gpu04's existing CVM collector already tails Docker JSON logs and selects labeled application containers, including this project's `deployment:coloc-soak` logs, into Loki. Verify those logs appear before starting the acceptance clock.

## Pass criteria

Observe for **at least 14 days, ideally 28**. All five criteria must hold:

1. Zero filter process restarts after its first start. Count Loki `{container_name="coloc-soak-privacy-filter"} |= "Started server process"` and compare with the initial start.
2. Zero filter HTTP 503 responses and zero `filter_oom` log events. Check load-generator `coloc_soak_filter_oom_503_total` and Loki `{container_name="coloc-soak-privacy-filter"} |= "filter_oom"`.
3. GPU 7 framebuffer used, `DCGM_FI_DEV_FB_USED{host="gpu04",UUID="GPU-85a29d2c-5a46-97a9-f08d-5f916e6509db"}`, never exceeds 134 GiB.
4. Estimated Qwen memory (GPU 7 framebuffer used minus the filter's `reserved_gb` from `filter_mem`) sets no new rolling 24-hour high during week two.
5. At load up to concurrency 25, Qwen3.8 TTFT p95 **and** ITL p95 remain within 20% of gpu13 r2 measured over the same windows. Use engine metrics and load-generator phase/request metrics to align windows.

If any criterion fails, stop the soak and record the timeline, metrics, logs, image/file tag, and container identities. Do not tune flags silently. Every changed flag starts a new PR and a new observation period.

## Deploy after merge

These are operator steps, not actions performed by this change. Use only a **merged `v0.0.N` tag**, never an experiment tag. Reconfirm gpu04 GPU 7 is idle and the `work`/`sage` project identities, current serving file/tag, and full dashboard environment map before the first call.

1. Run `brave_check.py compose prod/coloc-soak-privacy-filter-qwen38.yaml --project coloc-soak` after its required setup/live baseline. Expect the filter to be flagged for GPU access without a recognized server. gpu04 already fails Brave because of Sage. The verifier is an impact prediction, not production approval.
2. At `http://160.72.54.198:8080`, send `POST /compose/up` with `{"tag":"v0.0.N","file":"prod/coloc-soak-privacy-filter-qwen38.yaml","project":"coloc-soak","env":<gpu04 full dashboard env map>,"dry_run":true}`. Review the returned plan: it must create **only** `coloc-soak-*` containers and leave `work` and `sage` untouched. Repeat with `dry_run:false` only after that review.
3. Read `/docker/ps`; every pre-existing `work` and `sage` container must retain its prior `CreatedAt`. Confirm the new containers, GPU UUID, logs in Loki, and metrics in Prometheus before starting the 14-day clock.
4. **Brave follow-up is required immediately.** Brave treats the latest `compose_up` in the CVM actions log as the serving compose. Dry-run a no-op `compose_up` using gpu04's *current* serving file and live tag (at writing, `prod/GLM-5.3-Flash-SGL-TP2x3-W4AFP8-gpu04.yaml`), `project:"work"`, `services:["proxy-glm53"]`, and the full dashboard env map. Its plan must be empty. Then execute that same no-op serving up. This restores the serving file as the latest action (lesson from 2026-09-30).

GPU 7 is the second GPU for gpu04's possible fourth GLM replica if Sage leaves GPU 6. While this soak runs, gpu04 cannot add that replica.

## Teardown

Call `compose/down` for project `coloc-soak`; **`compose/down` ignores `dry_run`**. Verify `work` and `sage` container `CreatedAt` values again. Immediately perform the same dry-run-empty then no-op serving `compose_up` described above, so Brave sees the production GLM file last. The project's named volumes, holding roughly 30 GiB of model weights, may be pruned after evidence is retained.

## What is not tested

This does not test co-location with Qwen3.6: it is the busiest model, has one replica, and has a 133–139 GiB footprint. It also does not reproduce production traffic shape; all requests and PII are synthetic. A pass qualifies only this exact configuration, host, GPU, images, and observation window for a separate production decision.
