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

Run `dry_run: true` first wherever compose-manager supports it.

1. **Window and capacity.** The long tier is gpu02 (64 slots), gpu23 (64) and gpu13 (32). The gateway
   runs `TIER_STRICT=1`, so while gpu13 is out, long-tier capacity drops by 20% and requests over
   capacity get 429s. Choose a low-traffic window. See the 2026-09-21 fallback incident in
   `docs/gpu02-glm53-w4afp8-long-context.md`.
2. **Drain (optional).** When the TP4 engine stops, the gateway ejects gpu13 on its own within about
   15 s (failed `/healthz`). Requests in flight on the engine at that moment fail. To avoid that,
   drain first:
   - In `vars/openrouter_gateway.yaml` (cvm-ansible-playbooks), remove gpu13 from
     `openrouter_gateway_long_context_backend_urls` and `openrouter_gateway_long_context_probe_urls`,
     then run "Deploy OpenRouter Gateway".
   - Wait until
     `sum(rate(sglang_num_requests_total{host_machine="gpu13",model="z-ai/glm-5.3-flash"}[5m]))` is 0.
   - Re-add gpu13 after the smoke check (step 5).
3. **Stop the TP4 engine only.**
   - `compose/down` with services `["model-sg-glm53-fp8-tp4"]` (5 min grace).
   - Leave every other gpu13 service running.
4. **Deploy PD.**
   - `compose/up` with services
     `["model-sg-glm53-w4afp8-tp2-prefill","model-sg-glm53-w4afp8-tp2-decode","model-sg-glm53-w4afp8-pd-router","proxy-glm53","otelcol-contrib"]`
     and `force_recreate: false`. `proxy-glm53` is recreated because its env changes.
     `otelcol-contrib` is recreated because its scrape config changes.
   - Run with `dry_run: true` first. The only removal allowed is the stopped `model-sg-glm53-fp8-tp4`
     container. Any other removal, or a recreate of any other service, means abort.
   - nginx resolves `proxy-glm53` only at startup (AGENT.md). Right after `proxy-glm53` is up, run
     `compose/up` with services `["nginx"]` and `force_recreate: true`. This is a few seconds' blip for
     every model nginx fronts on gpu13; the 2026-09-18 gpu13 migration did the same.
   - Cold start takes about 15-30 minutes. The gateway keeps gpu13 out until `/readiness` passes.
5. **Smoke check.**
   - Run `glm53-perception-check` and the `glm53-soak-relay` checks against `proxy-glm53`.
   - Read the NCCL INFO lines from both engines in Loki.
   - Confirm `kv_transfer_*` metrics on a few requests.
6. **Serve.**
   - If step 2 drained gpu13, re-add it to both gateway lists. Otherwise the gateway re-admits it on
     its own once `/readiness` passes.
   - Watch TTFT, ITL, error rate, `KVTransferError` and the `kv_transfer_*` metrics against the gpu02
     and gpu23 TP4 replicas, which serve the same tier.

## Abort criteria

Any one of these means rollback:

- Any `KVTransferError`.
- Error rate above 1%.
- TTFT p95 above 2x the TP4 replicas on gpu02 and gpu23.
- Any request stuck for more than 10 minutes.
- Any failed perception or soak check.

## Rollback

1. `compose/down` at the PD commit with services
   `["model-sg-glm53-w4afp8-tp2-prefill","model-sg-glm53-w4afp8-tp2-decode","model-sg-glm53-w4afp8-pd-router"]`.
   The previous file does not define these services, so use the PD commit for this step. The proxy's
   `/readiness` check fails and the gateway ejects gpu13 within about 15 s.
2. `compose/up` of the previous `prod/small-models.yaml` commit with services
   `["model-sg-glm53-fp8-tp4","proxy-glm53","otelcol-contrib"]`. Cold start takes about 50 minutes.
3. `compose/up` with services `["nginx"]` and `force_recreate: true`.

## Before deploy

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
