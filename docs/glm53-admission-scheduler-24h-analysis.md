# GLM-5.3 Flash Admission Scheduler: 24-Hour Analysis

Admission-reserve v10 improved admission and first-token latency, but made streaming token cadence worse. It did not demonstrate more usable capacity during this observation.

## Observed measurements

The observation window was `2026-09-17T22:26:04Z` through `2026-09-18T22:26:04Z` for `z-ai/glm-5.3-flash`. The base group was gpu23 r1+r2 on the upstream scheduler; the treatment group was gpu04 r1+r2 on admission-reserve v10. Both groups served concurrently with `max-running-requests=32` and `max-queued-requests=8`.

TTFT is time to first token, ITL is inter-token latency, and E2E is end-to-end request latency.

| Metric | Base gpu23 | Admission v10 gpu04 | Delta |
|---|---:|---:|---:|
| Streaming requests | 69,728 | 69,195 | -0.8% |
| TTFT p50 | 1.657 s | 1.406 s | -15.1% |
| TTFT p90 | 6.153 s | 5.598 s | -9.0% |
| TTFT p95 | 8.514 s | 8.268 s | -2.9% |
| Queue time p90 | 2.671 s | 1.374 s | -48.5% |
| Queue time p95 | 4.900 s | 4.018 s | -18.0% |
| ITL p50 | 13.9 ms | 14.3 ms | +2.2% |
| ITL p90 | 42.5 ms | 56.3 ms | +32.4% |
| ITL p95 | 59.7 ms | 154.2 ms | +158.3% |
| E2E latency p50 | 11.534 s | 11.933 s | +3.5% |
| E2E latency p95 | 95.301 s | 94.966 s | -0.4% |
| Generation throughput | 414.5 tok/s | 394.5 tok/s | -4.8% |
| Running requests average | 10.44 | 10.41 | flat |
| Running requests p95 / max | 29 / 32 | 28 / 32 | same maximum |
| Queue depth average | 0.349 | 0.379 | +8.8% |
| Queue depth p95 / max | 2 / 8 | 2 / 8 | flat |
| Token usage p95 / max | 60% / 74% | 58% / 85% | higher treatment peak |
| Engine aborts per 1,000 streaming completions | 34.9 | 38.8 | +11.4% |
| Chat-completion HTTP 5xx | none observed | none observed | healthy |

### Workload shape

| Metric | Base gpu23 | Admission v10 gpu04 | Comparison |
|---|---:|---:|---:|
| Prompt tokens per streaming request | ~13,515 | ~11,931 | 11.7% fewer on treatment |
| Generation tokens per streaming request | ~777 | ~773 | effectively equal |
| Prompt tokens p50 / p95 | 4,570 / 54,954 | 3,814 / 52,017 | lower on treatment |
| Uncached prompt tokens p50 / p95 | 1,257 / 27,502 | 1,283 / 28,323 | similar |
| Output tokens p50 / p95 | 256 / 2,764 | 263 / 2,721 | similar |

### Priority-stratified evidence

The TTFT improvement and ITL regression remain after separating request priorities:

| Metric | Priority | Base gpu23 | Admission v10 gpu04 | Treatment direction |
|---|---:|---:|---:|---:|
| TTFT p50 | -2 | 1.998 s | 1.852 s | 7.3% better |
| TTFT p50 | -1 | 1.649 s | 1.395 s | 15.4% better |
| TTFT p50 | 0 | 1.482 s | 1.052 s | 29.0% better |
| TTFT p95 | -2 | 17.217 s | 17.280 s | effectively flat |
| TTFT p95 | -1 | 7.850 s | 7.616 s | 3.0% better |
| TTFT p95 | 0 | 9.491 s | 7.241 s | 23.7% better |
| ITL p90 | -2 | 45.7 ms | 58.2 ms | 27.3% worse |
| ITL p90 | -1 | 40.8 ms | 55.2 ms | 35.5% worse |
| ITL p90 | 0 | 47.3 ms | 59.8 ms | 26.4% worse |

## Methodology

- Runtime identity came from live compose-manager `/docker/ps` container labels and configuration. This was used to classify gpu04 as treatment because its Prometheus `config_variant` label was stale.
- Performance metrics came from the production Grafana Prometheus datasource. Histogram buckets were aggregated across request priorities; metrics duplicated on every tensor-parallel rank were filtered to `tp_rank="0"` before aggregation.
- Engine aborts are engine telemetry and can include client disconnects. They are not equivalent to HTTP 5xx responses.
- This was a concurrent host-level natural A/B comparison, not a randomized same-host benchmark.
- gpu02 was excluded per operator direction. gpu03 was not combined with the control because its r2 uses HiCache and is not configuration-equivalent.

## Interpretation

Admission-reserve v10 reduced TTFT and queue-time percentiles while materially worsening ITL tails. The priority-stratified measurements show the same TTFT/ITL tradeoff within priorities, making priority-mix bias a less likely sole explanation.

The treatment did not demonstrate more usable capacity. Average running concurrency was flat, both groups reached the same maximum of 32 running requests, generation throughput was 4.8% lower on treatment, and E2E p95 was effectively unchanged. The 4.8% throughput difference is not definitively causal: treatment received 11.7% fewer prompt tokens per streaming request, and the comparison was not randomized.

## Limitations

The groups ran on different hosts with different observed workload shapes, so unmeasured host and traffic differences can confound the comparison. The data does not establish that raising `--max-running-requests` from 32 to 40 is safe.

This document records observations only. It does not prove current deployment state or authorize a production rollout.

## Recommendation

Test 40 running requests with an 8-request queue (40/8) in the operator's separate canary environment. During that canary, watch ITL tails, generation throughput, engine aborts, token-usage peaks, queue saturation, container restarts, prefill out-of-memory events, and HTTP 5xx responses before making any production decision.
