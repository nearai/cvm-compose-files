# gpu13 GLM-5.3 prefill/decode (PD) canary

Status: proposed. Not deployed.

## What changes

`prod/small-models.yaml` replaces the gpu13 GLM-5.3 Flash TP4 replica
(`model-sg-glm53-fp8-tp4`, GPUs 4-7) with prefill/decode disaggregation on the same GPUs:

| Service | GPUs | Role |
|---|---|---|
| `model-sg-glm53-w4afp8-tp2-prefill` | 4,5 | TP2/EP2 prefill, HiCache (325 GiB host pool, `cudaMallocHost`) |
| `model-sg-glm53-w4afp8-tp2-decode` | 6,7 | TP2/EP2 decode, EAGLE 5/1/6 adaptive, radix cache off |
| `model-sg-glm53-w4afp8-pd-router` | none | SGLang router `--pd-disaggregation`, internal port 8000 |

`proxy-glm53` points at the router (`VLLM_BACKEND_URLS=http://model-sg-glm53-w4afp8-pd-router:8000`)
and health-checks the router's `/readiness` (`VLLM_BACKEND_HEALTH_PATH=/readiness`). The router's
`/health` returns 200 before its workers are ready (measured on gpu32: `/health` 200, `/readiness`
503, chat 503 "No prefill workers available"). With `/readiness`, the proxy's `/healthz`, and through
it the OpenRouter gateway (probes `/healthz` every 5 s, ejects after 3 failures), follows the real
worker state. The gateway keeps gpu13 out while the engines start and ejects it within about 15 s if
an engine fails.
The served model name stays `z-ai/glm-5.3-flash`.

KV transfer uses Mooncake over TCP (`--disaggregation-transfer-backend ${GLM53_PD_BACKEND:-mooncake}`,
`MOONCAKE_PROTOCOL=tcp`). Both engines log NCCL transport selection at startup
(`NCCL_DEBUG=INFO`, `NCCL_DEBUG_SUBSYS=INIT,P2P,ENV`) so the PPCIe transport is visible in Loki.

## Why

On the TP4 replica, prefill stalls decode:

- About 9% of tokens wait more than 100 ms, and those stalls make up about half of all decode time.
- Mean ITL tracks the prefill rate (r = 0.8).

PD runs prefill and decode on separate GPUs, so prefill no longer stalls decode. This canary runs PD
on real long-tier traffic inside the production PPCIe TEE.

Lab results (gpu32, H200 NVL, CC without PPCIe):

- PD over Mooncake TCP works.
- KV moved at about 0.6 GB/s, adding 0.5-0.8 s per request.
- Some requests failed with `KVTransferError` under load.

The canary measures how the KV transfer behaves under PPCIe.

## Procedure

No OpenRouter gateway change is needed. The gateway probes `proxy-glm53`'s `/healthz` every 5 s and
ejects gpu13 after 3 failures. The proxy checks the router's `/readiness`. As a result:

- gpu13 leaves the pool about 15 s after the TP4 engine stops.
- It stays out while the PD engines start.
- It comes back on its own once both engines are ready.

Run `dry_run: true` first wherever compose-manager supports it. Pass the full gpu-manager env map on
every call, use `force_recreate: false` unless stated otherwise, and never send an empty `services` list.

1. **Window.** gpu13 holds 32 of the 160 long-tier slots. While it is out, `TIER_STRICT=1` returns
   429s for long requests over capacity. Choose a low-traffic window.
2. **Stop the TP4 engine.**
   - `compose/down` at the currently deployed tag with services `["model-sg-glm53-fp8-tp4"]`.
     There is no dry run for this call.
   - Requests in flight on the engine fail. Every other gpu13 service keeps running.
3. **Deploy PD.**
   - `compose/up` at the PD tag with services
     `["model-sg-glm53-w4afp8-tp2-prefill","model-sg-glm53-w4afp8-tp2-decode","model-sg-glm53-w4afp8-pd-router","proxy-glm53","otelcol-contrib"]`.
   - The dry-run plan must create the three PD services, recreate `proxy-glm53` (its env changes) and
     `otelcol-contrib` (its scrape config changes), and remove nothing else. Anything more means abort.
   - nginx resolves `proxy-glm53` only at startup. Right after `proxy-glm53` is up, run `compose/up`
     with services `["nginx"]` and `force_recreate: true`. Every model nginx fronts on gpu13 sees a
     few seconds' interruption. The 2026-09-18 gpu13 migration did the same.
   - Cold start takes about 15-30 minutes.
4. **Watch.**
   - Read the engine logs for "ready", NCCL transport lines and Mooncake init.
   - Confirm the gateway re-admits gpu13 once `/readiness` returns 200.
   - Compare TTFT, ITL, error rate, `KVTransferError` and `kv_transfer_*` metrics against the gpu02
     and gpu23 TP4 replicas, which serve the same tier.

## Abort criteria

Any one of these means rollback:

- Any `KVTransferError`.
- Error rate above 1%.
- TTFT p95 above 2x the TP4 replicas on gpu02 and gpu23.
- Any request stuck for more than 10 minutes.
- Any failed perception or soak check.

## Rollback

1. `compose/down` at the PD tag with services
   `["model-sg-glm53-w4afp8-tp2-prefill","model-sg-glm53-w4afp8-tp2-decode","model-sg-glm53-w4afp8-pd-router"]`.
   The gateway ejects gpu13 within about 15 s.
2. `compose/up` at the previous tag with services `["model-sg-glm53-fp8-tp4","proxy-glm53","otelcol-contrib"]`.
   Cold start takes about 50 minutes.
3. `compose/up` with services `["nginx"]` and `force_recreate: true`. The gateway re-admits gpu13 once
   the TP4 engine is healthy.

## Before deploy

- Decode memory: decode runs at `--mem-fraction-static 0.72`. In decode mode each rank also allocates
  `intermediate_ssm_state_cache` (26.52 GB) and `intermediate_conv_window_cache` (1.86 GB). At 0.80
  that left 7.02 GB after the pool, and the adaptive speculative CUDA graph capture ran out of memory
  on gpu32 (2/2 runs). At 0.72 the log shows 17.19 GB free after the pool. A full boot at 0.72 is not
  yet confirmed: the gpu32 run stopped on a GPU hardware fault (Xid 175/154) unrelated to the config.

- Confirm that `sglang_router` is in the prod image digest.
- Confirm that the proxy's health probe works against the router.
- Confirm that `MOONCAKE_PROTOCOL=tcp` alone avoids RDMA probing. If it does not, set
  `GLM53_PD_BACKEND=mooncake_tcp`.
- Confirm that host RAM on gpu13 fits the 325 GiB prefill pool (free RAM minus 10 GiB). If it does
  not, override with `GLM53_PD_HICACHE_RAM_BUDGET`.
- Labels keep `deployment=glm53-flash-sgl-tp4`, so existing dashboards and alert rules keep working.
  `config_variant` and `pd_role` identify the PD engines.
- The proxy's `/v1/metrics` returns 404, because the router serves Prometheus metrics on port 29000,
  not its serving port. The gateway's engine-load probe for gpu13 falls back to its own request
  counts.
- Check that the router passes `priority`, `reasoning_effort` and
  `stream_options.continuous_usage_stats` through to the engines.

## Approvals

GLM serving owners (Pranav, Lloyd), the infra owner for the gateway workflow, cvm-compose-files
reviewers, and the KMS owners for the compose hash.
