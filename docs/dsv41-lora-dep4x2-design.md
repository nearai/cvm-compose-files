# DeepSeek-V4.1-Flash-1017 + abliterated LoRA on one 8x H200 TDX host: design note (draft)

Status: draft for review by Henry and Pierre. Nothing here is deployed.
- Target host: probably gpu02, after its current replicas are decommissioned (decision pending).
- Owner: Pranav. Evidence: tee-bench exp 50 (PR #13) and exp 51 (PR #14).

## Goal

Two identical, optimized replicas of `deepseek-ai/DeepSeek-V4.1-Flash-1017` on one 8-GPU CVM. Each serves:
- the base model;
- the abliterated variant `deepseek-ai/DeepSeek-V4.1-Flash-1017-abliterated`, i.e. LoRA `UnderTides/DeepSeek-V4.1-Flash-1017-heretic-lora` rev `196c06e3`.

The request's `model` selects the variant. Routing is static configuration: no runtime adapter loading. More variants later means another `--lora-modules` entry plus one more proxy, see below.

## Engine: one replica = vLLM DEP4 on four GPUs

Image: `docker/vllm-dsv41-lora` (vLLM v0.32.0 + three patches). Its patched files are byte-identical to exp 51 arm l9: 5,064 agent-trace requests and 0 errors in a 1.5 h soak, plus a 3 h soak (results pending).

Per replica there are four `vllm serve` processes, one per GPU (`--data-parallel-size 4 --data-parallel-rank r --enable-expert-parallel --tensor-parallel-size 1`, external DP load balancing).

**Exp 50 best stack:**
- W4A8 (`VLLM_HUMMING_INPUT_QUANT_CONFIG` fp8)
- `--max-num-batched-tokens 8192`
- DSpark 3
- `--max-num-seqs 64`
- `--gpu-memory-utilization 0.85`

**Plus LoRA:**
- `--enable-lora --max-loras 1 --max-lora-rank 8 --lora-target-modules wo_b --lora-modules <abliterated id>=<adapter dir>`

**Replica placement:** r1 on guest GPUs 0-3, r2 on 4-7 (one NVLink island each). The 8 rank endpoints are r1 dp0-3 and r2 dp0-3.

**Measured capacity of one replica on gpu31 (bare metal) on real agent traces:**

| Load | TTFT p95 | Errors |
|---|---|---|
| 1.08 req/s | 2.8 s | 0 |
| ~1.6 req/s | 1.05 s | 0 |
| ~2.2 req/s | 25 s (past the knee) | 0 |

Two replicas: about 3.2 req/s at TTFT p95 ≤1.1 s.

## Engram tables under TDX (decided: HBM; first boot is the gpu02 deploy)

vLLM keeps DS4.1's 186 GiB Engram tables in pinned host memory and reads them through UVA (`--engram-config` `cpu_offload`, default true). A TD guest does not give the GPU direct access to its private memory, so that layout is expected to fail or degrade in the CVM.

**The gpu02 pack uses the CC-safe layout:** `--engram-config '{"cpu_offload":false,"embedding_across_dp":true}'` with `--gpu-memory-utilization 0.95`. The tables are sharded across the four DP ranks' HBM (~47 GiB each, on top of ~82 GiB of weights), with no host-memory reads at runtime.

**It has not been booted anywhere yet:**
- The bare-metal retry (l11b) was cancelled when the experiments moved to gpu02.
- The first gpu02 boot is the fit and speed test.
- If a rank cannot allocate KV at 0.95, the fallback is a follow-up tag with the host layout. Expect that to fail under TDX unless vLLM pins the tables through the TD's shared bounce buffers.
- The other fallback is a CPU-gather patch, as was built for SGLang on 10-07.

## Open blocker 2: the front proxy inside the CVM (needs an inference-proxy change)

How vllm-proxy-rs behaves in CVM mode, read at origin/main a737734:
- A request whose `model` differs from `MODEL_NAME` is **passed through unchanged** (`model_list::model_for`), so the adapter works.
- It is **billed under `MODEL_NAME`** (`proxy.rs:586`) and gets that model's discount.
- It is **not validated**: anyone who can reach the base endpoint can ask for the abliterated id. That bypasses the per-org access planned for it (cloud-api #1237).
- The multi-model list (`VLLM_PROXY_MODEL_LIST_FILE`, #289) requires `NON_TEE_DEPLOYMENT=1`, so it cannot run in a CVM.
- The affinity key hashes `MODEL_NAME` plus the conversation start, not the request's `model`.

**Proposal: one proxy per model id in the same CVM.**
- `proxy-dsv41`: `MODEL_NAME` = base.
- `proxy-dsv41-abliterated`: `MODEL_NAME` = abliterated id.
- Both list the same 8 rank endpoints and have their own model-proxy registration, attestation and billing.
- Strict mode (inference-proxy #329, `VLLM_PROXY_STRICT_REQUEST_MODEL=1`): each proxy refuses (404 `model_not_found`) a body `model` that is not its own `MODEL_NAME`. Without it, the base endpoint serves the adapter at base price.
- Affinity: both proxies hash the same conversation to the same rank. That is fine, since the prefix cache is per adapter on the engine.

Adding another variant later means one more `--lora-modules` entry on both replicas and one more proxy service and registration.

## Router

Pierre's tuned `sglang_router` (`cache_aware`, threshold 0.6, balance 16/1.25) fixed a hotspot caused by prefix matching on the shared agent system prompt. The CVM proxies' conversation affinity hashes the conversation start, not the shared prefix, so the plan is to use the proxies alone, with no sglang router in the CVM. This needs a lab check: run vllm-proxy-rs in front of l9/l10 and replay W1.

## Rollout (when approved)

Follow the existing rules:
- deploy only from a main tag;
- take one replica out of the pool at a time;
- tell the team before any prod redeploy;
- validate under CC on a CC-on host before gpu02.

## Staging only, not listed (decided 10-26)

- Both ids are reachable only through the staging zone (`deepseek-v4-1-flash[-abliterated].completions-stg.near.ai`); the pack's nginx also answers those names.
- No prod cloud-api catalog row. A cloud-stg-api row, if wanted, uses the `completions-stg` inferenceUrl.
- Caveat: any active `PATCH /v1/admin/models` on cloud-stg-api fires `validate_model_staging`, which auto-merges infra-tests PRs (staging and prod tests) and then runs `promote_model_prod`. The promote's zone guard refuses a `completions-stg` URL, so the model will not reach prod cloud-api, but the infra-tests prod PR would still land. Agree with Henry before that PATCH, or test by calling the proxies directly.

## Before prod

1. Done: engine image published from #393 (`nearaidev/vllm@sha256:0956859f…`); proxy from inference-proxy main 5faff1b with Henry's #329 `VLLM_PROXY_STRICT_REQUEST_MODEL` (`nearaidev/vllm-proxy-rs@sha256:4937ad68…`). No `VLLM_PROXY_BACKEND_LORA`: vLLM selects the adapter by the model id.
2. gpu02 staging (unlisted, from a main tag): both replicas boot with Engram in HBM under TDX+PPCIe, and the DEP NCCL all-to-all works. This replaces the cancelled gpu31 l11/l12 runs.
3. On gpu02: strict refusals per proxy, KV-per-adapter (`scripts/lab_kv_adapter_check.py` with `KV_URL_MAP`), Henry's smoke per id, the tee-bench load matrix (base:adapter 1:0 / 1:1 / 0:1), then a 24 h soak.
4. A refusal-behaviour eval of the adapter on the team's set.
5. The cloud-api catalog row for the abliterated id (#1237, unlisted + per-org).
