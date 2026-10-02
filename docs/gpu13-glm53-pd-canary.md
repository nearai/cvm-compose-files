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
| `glm53-pd-t0-probe` | 4-7 | T0 transport probe, profile `pd-t0`, never started by a plain `compose/up` |

`proxy-glm53` points at the router. The served model name stays `z-ai/glm-5.3-flash`.
No other gpu13 service changes.

KV transfer uses Mooncake over TCP (`--disaggregation-transfer-backend ${GLM53_PD_BACKEND:-mooncake}`,
`MOONCAKE_PROTOCOL=tcp`). Both engines log NCCL transport selection
(`NCCL_DEBUG=INFO`, `NCCL_DEBUG_SUBSYS=INIT,P2P,ENV`).

## Why

Prefill stalls decode on the TP4 replica: about 9% of tokens wait over 100 ms, and those stalls are
about half of all decode time. Mean ITL tracks the prefill rate (r = 0.8). PD isolates decode from
prefill. This canary checks whether PD works inside a production PPCIe TEE and whether it pays off.

Lab evidence (gpu32, H200 NVL, single-GPU CC mode without PPCIe; tee-bench round 1):

- SGLang PD over Mooncake TCP works under CC.
- KV moved at 0.57-0.60 GB/s, adding 0.5-0.8 s per request.
- 36 of 293 requests failed with `KVTransferError` at the higher load.
- NIXL fails at startup (`cuMemHostRegister ... not supported`).
- Without PPCIe there is no GPU-to-GPU path; NCCL only works over sockets at about 4-8 MB/s.

PD only pays off on gpu13 if PPCIe provides a fast GPU-to-GPU path. Gate T0 checks this before any
GLM engine is replaced.

## Procedure

Every step needs the approvals listed below. Run `dry_run: true` first wherever compose-manager
supports it.

1. **Window and capacity.** The long tier is gpu02 (64 slots), gpu23 (64) and gpu13 (32). The gateway
   runs `TIER_STRICT=1`, so removing gpu13 cuts long-tier capacity by 20% and over-capacity
   requests get 429s. Choose a low-traffic window. See the 2026-09-21 fallback incident in
   `docs/gpu02-glm53-w4afp8-long-context.md`.
2. **Drain.** In `vars/openrouter_gateway.yaml` (cvm-ansible-playbooks), remove gpu13 from
   `openrouter_gateway_long_context_backend_urls` and `..._probe_urls`, then run
   "Deploy OpenRouter Gateway". Confirm zero GLM traffic on gpu13 for at least 10 minutes:
   `sum(rate(sglang_num_requests_total{host_machine="gpu13",model="z-ai/glm-5.3-flash"}[5m])) == 0`.
3. **Stop the TP4 engine only.** `compose/down` with services `["model-sg-glm53-fp8-tp4"]`
   (5 min grace). Leave every other gpu13 service running.
4. **Gate T0 (no model).** `compose/up` with services `["glm53-pd-t0-probe"]` at this commit.
   Read `compose/logs` for `T0RESULT` lines and the NCCL INFO transport lines.
   - **Go:** NCCL between the prefill and decode GPU pairs uses P2P/NVLink (`via P2P` in NCCL INFO)
     and send/recv is at least 10 GB/s at 64 MB.
   - **No-go:** socket transport only, or under 1 GB/s. Skip PD and go to Rollback to restore TP4.
5. **Deploy PD.** `compose/up` with services
   `["model-sg-glm53-w4afp8-tp2-prefill","model-sg-glm53-w4afp8-tp2-decode","model-sg-glm53-w4afp8-pd-router","proxy-glm53","dcgm-glm53"]`,
   `force_recreate: false`, `dry_run: true` first. The dry run may remove only the stopped
   `model-sg-glm53-fp8-tp4` and the finished T0 probe. Any other removal means abort.
   Cold start takes about 15-30 minutes.
6. **Private validation, gateway still drained.** Run `glm53-perception-check` and the soak relay
   against `proxy-glm53`, then the tee-bench suite load at 1x and 2x. Watch `kv_transfer_*` metrics,
   TTFT, ITL and `KVTransferError` counts.
7. **Pool admission, only if step 6 passes.** Re-add gpu13 to the gateway lists and watch live.

## Abort criteria

Any one of these during step 6 or 7 means rollback:

- Any `KVTransferError`.
- Error rate above 1%.
- TTFT p95 above 2x the TP4 baseline for the same tier.
- Any request stuck for more than 10 minutes.
- Any failed perception or soak check.

## Rollback

1. `compose/down` with services `["model-sg-glm53-w4afp8-tp2-prefill","model-sg-glm53-w4afp8-tp2-decode","model-sg-glm53-w4afp8-pd-router"]`.
2. `compose/up` of the previous `prod/small-models.yaml` commit with services
   `["model-sg-glm53-fp8-tp4","proxy-glm53","dcgm-glm53"]` (about 50 minutes cold start).
3. Re-add gpu13 to the gateway after the soak checks pass.

## Open items (VERIFY)

- Confirm that `sglang_router` is present in the prod image digest and that the proxy's `/health`
  probe works against the router.
- Confirm that `MOONCAKE_PROTOCOL=tcp` alone avoids RDMA probing. If not, set
  `GLM53_PD_BACKEND=mooncake_tcp`.
- Confirm that host RAM on gpu13 fits the 325 GiB prefill pool (free RAM minus 10 GiB). If not,
  override with `GLM53_PD_HICACHE_RAM_BUDGET`.
- Dashboards keyed on `deployment=glm53-flash-sgl-tp4` will not show the new engines.
- If T0 shows a fast NCCL path but Mooncake TCP is slow, PD needs an NCCL-based SGLang transfer
  backend, estimated at about 2-3 weeks of work.

## Approvals

GLM serving owners (Pranav, Lloyd), the infra owner for the gateway workflow, cvm-compose-files
reviewers, and the KMS owners for the compose hash.
