# DeepSeek V4 Flash EAGLE prefill fairness image

This image applies one default-off scheduler guard to the exact SGLang image
currently used by the DeepSeek V4 Flash deployment. Set
`SGLANG_EAGLE_PREFILL_FAIRNESS=1` to force one decode turn between consecutive
EAGLE chunked-prefill batches when active decode requests exist.

Published image:

```text
nearaidev/sglang@sha256:1e335c485bfe064e1b9cdfdcb2765e327235a59fbb65df91be9b429d23e1db08
```

Source base:

```text
lmsysorg/sglang@sha256:6bb5fee34b6c4537c09a4775e2292ac40350d5ad1218fcc835b2692142f443b1
SGLang revision 7de33ce806c12664b647604d61cf1403d2d18013
```

Build:

```bash
docker build \
  -t nearaidev/sglang:dsv4-eagle-prefill-fairness-7de33ce8-v1 \
  docker/sglang-dsv4-eagle-prefill-fairness
```

The candidate passed three independent 10/10 mixed-load runs, a GPU-pair swap,
and a 15-cycle, 150/150-stream bare-metal H200 soak. The strict detector
required HTTP 200, valid SSE, no inline error, finish reason, and terminal
`[DONE]`. The unpatched EAGLE control failed 9/10 streams twice on different
GPU pairs.

This guard addresses scheduler starvation only. It does not include the
separate open DSV4/EAGLE metadata fixes.

## Production A/B shape

The compose canary is stacked on `cvm-compose-files#150`, which gives gpu30's
two DeepSeek replicas independent proxies and routing ports:

- r1 / `instance:1` / port `8001`: unchanged production image, control.
- r2 / `instance:2` / port `8002`: patched image with the fairness flag,
  candidate.

Only the r2 engine and the OTel collector need to change for the canary. The
r1 engine, both inference proxies, nginx, Qwen, registrar, and other model
services remain unchanged. Collector-side labels add the full A/B dimensions
to both engine and proxy metrics without recreating the control containers.

Required gates:

1. Confirm `#150` is merged, deployed, and all three DeepSeek backends are
   healthy before starting this canary.
2. Drain only r2, replace only
   `model-sg-dsv4-flash-fp4-tp2-r2`, then start/reconfigure
   `otelcol-contrib`.
3. Verify the r2 startup warning says EAGLE prefill fairness is enabled and
   the runtime image matches the candidate digest.
4. Run the strict 10-stream detector against both indexed endpoints and require
   HTTP 200, valid SSE, finish reason, terminal `[DONE]`, and no idle gap above
   20 seconds.
5. Compare request-matched prompt buckets, TTFT, ITL, E2E, queue depth,
   running requests, aborts, retractions, throughput, and proxy incomplete
   streams by `experiment_arm`.
6. Roll back r2 to the exact control image and remove the fairness environment
   variable on any outcome regression.
