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

## Staging-only design (superseded)

The 10-26 staging-only routing and separate staging certificate were replaced by the single-pack design below. Its cloud-stg-api catalog migration and promotion risk remain rollout gates.

## Before prod

1. Done: engine image published from #393 (`nearaidev/vllm@sha256:0956859f…`); proxy from inference-proxy main 5faff1b with Henry's #329 `VLLM_PROXY_STRICT_REQUEST_MODEL` (`nearaidev/vllm-proxy-rs@sha256:4937ad68…`). No `VLLM_PROXY_BACKEND_LORA`: vLLM selects the adapter by the model id.
2. gpu02 staging (unlisted, from a main tag): both replicas boot with Engram in HBM under TDX+PPCIe, and the DEP NCCL all-to-all works. This replaces the cancelled gpu31 l11/l12 runs.
3. On gpu02: strict refusals per proxy, KV-per-adapter (`scripts/lab_kv_adapter_check.py` with `KV_URL_MAP`), Henry's smoke per id, the tee-bench load matrix (base:adapter 1:0 / 1:1 / 0:1), then a 24 h soak.
4. A refusal-behaviour eval of the adapter on the team's set.
5. The cloud-api catalog row for the abliterated id (#1237, unlisted + per-org).

## DEP8 in CVMs (decided 10-28, replaces 2 × DEP4 for staging)

- **Pack:** `prod/DeepSeek-V4.1-Flash-1017-vLLM-DEP8-Abliterated.yaml`, generated by `scripts/prepare_dsv41_lora_dep8.py`.
  - One DEP8 replica, Engram in HBM (~23 GiB per rank), gpu-memory 0.90, 18.0M KV tokens per rank.
- **Why DEP8:** the gpu02 dev-CVM A/B (tee-bench #16) used the same image and the prod suite at 1:1 base:abliterated. At rate ×8, DEP8 delivered +15% output tok/s, +23% input tok/s, and TTFT p50 5.3 s vs 9.4 s against 2 × DEP4.
  - 2 × DEP4 in a CVM needs the CPU-gather Engram patch (05), which forces PIECEWISE graphs, and it has 1.7× less KV.
  - It keeps one advantage: one replica can restart while the other serves.
- **TDX+PPCIe requirements (all in the pack):**
  - `VLLM_NEAR_DISABLE_UVA=1` (image patch 04, #401). The GPU reads a stale snapshot of pinned host memory, so vLLM's UVA input buffers corrupt decode (NaN logits, garbage, DSpark acceptance 0). v0.0.519 served garbage for this reason.
  - Engines on the host network, with all GPUs visible and per-rank `CUDA_VISIBLE_DEVICES`. In per-container namespaces, NCCL used TCP sockets instead of NVLink.
  - `HUMMING_COMPILER=nvcc` with a `humming_cache` volume. Humming's nvrtc helper opens a CUDA context per kernel, and under CC those queue for hours: this was the 2 h 37 min "graph capture" on v0.0.519.
  - nginx and the registrar on 8000-8003. gpu02's CVM forwards only 8000-8010, 8080 and 8444, so v0.0.519's TLS ports 8011/8013 were unreachable.
  - No host-memory Engram: `cpu_offload: true` led to the `nvidia_uvm` CC fault-decrypt guest panic on 10-27.
- **Proxies:** they stay on `dstack_default` (for compose-manager attestation) and reach the ranks at `dsv41-host:9100-9107` through `host-gateway`.

## Prod and staging from one pack (DEP8)

`scripts/prepare_dsv41_lora_dep8.py` generates one file, `prod/DeepSeek-V4.1-Flash-1017-vLLM-DEP8-Abliterated.yaml`, for both zones, following the GLM and small-models pattern. Its registrar registers only `deepseek-v4-1-flash[-abliterated].completions.near.ai` on `https://completions.near.ai`. The staging model-proxy copies the prod registry about every 30 s and rewrites those names to `*.completions-stg.near.ai`; nginx answers both zones' names using the CVM certbot certificate for `*.completions.near.ai`, which both proxies attest. Catalog rows on both cloud-api and cloud-stg-api must use the `*.completions.near.ai` URLs: cloud-api checks hostnames, so a `-stg` URL fails with `NotValidForName` against this certificate.

The pack sets `VLLM_PROXY_ALLOWED_ORG_IDS=${DSV41_ABLITERATED_ALLOWED_ORG_IDS:-__no_organizations__}` only on the abliterated proxy. It denies direct `sk-` customer keys by default unless their organization is listed; configured tokens used by cloud-api and the registrar bypass this check. cloud-api traffic is separately governed by its catalog row, which must be unlisted and restricted per organization (cloud-api #1237). `promote_model.py` does not copy `isListed`, so never promote the abliterated id with it.

infra-tests (`tests/test_model_proxy_staging.py`, nearai/infra-tests #95) calls the abliterated proxy directly with its `CLOUD_API_KEY`, which is a customer `sk-` key. Before the first deploy of the pack from a tag that contains this allow-list, set `DSV41_ABLITERATED_ALLOWED_ORG_IDS` in the gpu02 dashboard environment to every organization that must call the abliterated proxy directly, including the organization that owns that key. Otherwise those tests get 403. infra-tests also fails if a DS4.1 id appears in its production model lists, so update it in the same window as the prod switch.

Before switching, complete the gates above: 24 h soak, load matrix, refusal evaluation, an approved exposure decision, live cloud-api #1237, and DS4.1 alerts and probes. The rollout is:

1. Set `DSV41_ABLITERATED_ALLOWED_ORG_IDS` in gpu02's full dashboard environment map before the first deploy of this tag if infra-tests or another direct caller needs the abliterated proxy. Leaving it unset denies all direct customer `sk-` keys.
2. On gpu02's existing project, dry-run a compose/up of the new tag and the single pack, scoped to `nginx`, `proxy-dsv41`, `proxy-dsv41-abliterated` and `model-proxy-registrar`. Force-recreate nginx and the registrar because Compose does not hash their inline configs. Confirm that no `model-vllm-dsv41-dp*` or `model-downloader` is recreated, and use the complete dashboard environment map. The exited `stg-cert-init` container and `work_stg_certs` volume are harmless leftovers; remove them later only if wanted (`compose/down` ignores `dry_run`).
3. The old registrar unregisters gpu02 from completions-stg on shutdown. The new registrar registers the prod domains on completions.near.ai; the staging model-proxy picks them up through its rewrite within about 30 s. The prod names then become reachable directly.
4. The cloud-stg-api DS4.1 rows still use `-stg` URLs and fail with `NotValidForName` until they are PATCHed to the `*.completions.near.ai` URLs, as the GLM and Qwen rows are. Coordinate with Henry: an active PATCH on cloud-stg-api starts `validate_model_staging` → `promote_model_prod`, and promotion drops `isListed`, which would make the abliterated model public in prod. Ensure promotion cannot run for the abliterated id before PATCHing its staging row.
5. Create prod rows by admin PATCH on cloud-api. Set the abliterated row to `isListed:false` with per-org grants.
6. Verify one backend per prod domain on the prod model-proxy nodes gpu23 and cpu05, and the rewritten `-stg` domains on staging. Check strict TLS and that attestation SPKI matches nginx's certificate on the prod names. Complete one chat per id through cloud-stg-api and one per id through cloud-api.

Rollback uses the same scoped compose/up from tag `v0.0.527` with its staging-only file; add `stg-cert-init` to the services. The `work_stg_certs` volume still holds the certificate. Then PATCH the cloud-stg-api rows back to their `-stg` URLs.

## Back to 2 × DEP4 for availability (decided 10-30)

- **Decision (Pierre, 10-30):** give up DEP8's throughput (+8% at rate ×4, +15% at ×8 in the A/B above) so that one replica keeps serving while the other restarts or redeploys. With DEP8, a crashed rank or a redeploy takes the model down for ~15 min. Both replicas share one host, so a host or CVM failure still takes both down.
- **Pack:** `prod/DeepSeek-V4.1-Flash-1017-vLLM-DEP4x2-Abliterated.yaml`. The DEP8 generator renders it as a second layout, so the engine stack, proxies, queue limit and priority settings stay identical (`scripts/prepare_dsv41_lora_dep8.py`, `DEP4X2`).
  - r1 = dp0-3 on GPUs 0-3, r2 = dp4-7 on GPUs 4-7. Each replica has its own DP coordinator (`--data-parallel-rpc-port` 13345 / 13445) and master port (`VLLM_DP_MASTER_PORT` 29600 / 29700), and no rank depends on the other replica's containers.
  - **Engram tables:** host RAM through image patch 05 (`VLLM_NEAR_ENGRAM_CPU_GATHER=1`, `cpu_offload` + `dp_shared_memory`, PIECEWISE graphs). The GPU never reads host memory, so the 10-27 `nvidia_uvm` panic (UVA reads) does not apply; this is the layout the A/B ran in the dev CVM.
  - **The one shared resource:** one `/dev/shm/near_engram_*` copy for both replicas. A restarting replica's leader takes the flock, finds both `.ready` markers and maps the copy as-is ("reuse"); it never truncates or rewrites it. Only a missing marker makes a leader reload, and that rewrites the same bytes.
  - gpu-memory 0.85, as in the lab.
- **Proxies:** both list all eight ranks. A stopped replica refuses connections, and `VLLM_BACKEND_CONNECT_FAILOVER=1` retries those on another rank. The health checker (`/health` every 5 s, 3 failures) then drops the replica.
- **To verify on gpu02 after the deploy** (no other CC host is free):
  1. Under steady load, `docker stop` all of r1's ranks: r2's TTFT and throughput hold, with no client errors beyond requests in flight on r1.
  2. Start r1 again: r2 holds during r1's weight load, kernel builds and graph capture (CPU, page cache and PCIe contention).
  3. Kill one rank of r1: check whether r1's other ranks exit or hang with `/health` still 200. A hang would leave the proxy sending traffic to the stuck ranks.
  4. Both model ids on both replicas (17*23), and the abliterated id still answers a prompt the base id refuses.
