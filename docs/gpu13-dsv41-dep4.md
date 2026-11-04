# gpu13 DeepSeek-V4.1 DEP4 rollout

GPUs 4-7 move from two GLM-5.3 Flash TP2 long-tier replicas to one DeepSeek-V4.1-Flash-1017 DEP4 replica. The four ranks are gpu02's r2 replica, with `CUDA_DEVICE_ORDER=PCI_BUS_ID` added on gpu13 so CUDA GPU indices agree with the `device_ids` indices used by the other services. The base model and abliterated LoRA share the ranks. Host port 8009 has an auth-gated `/v1/models` stub for both ids; host port 8444 selects the two proxies by SNI.

On 2026-11-02, `/dstack/persistent` had 1,204 GiB free of 1,969 GiB. DS4.1 needs about 475 GiB, so no eviction is needed before deployment. The CVM has 2,048 GiB RAM; the two GLM engines used about 761 GiB, and DS4.1 keeps one 186 GiB Engram copy in `/dev/shm`.

## Rollout

1. Check at least ~500 GiB free on the CVM's `/dstack/persistent` ZFS cache; use `hf-cleanup` only if needed. In cvm-ansible-playbooks, remove gpu13's long backend handle `glm-5-3-flash-long-b52008698b596` from `vars/openrouter_gateway.yaml`. Silence `Glm53BackendScrapeIncomplete`, `Glm53DcgmScrapeMissing`, and `Glm53GpuTelemetryIncomplete` for the window; the last two page. Drain gpu13 by stopping `proxy-glm53`: the :8009 stub's `auth_request` then fails, model-proxy drops gpu13 from `glm-5-3-flash-long`, and the registrar stops refreshing :8009. A dead registered backend also makes the `model_registry` synthetic red. Keep the shared `model-proxy-registrar` running during the drain: SIGTERM unregisters every gpu13 small-model port. Cloud-api's long tier then has gpu23 only.
2. Deploy from a main tag. Compose/down the GLM services. Compose/up `model-downloader`, then `model-vllm-dsv41-dp4` through `dp7`. Cold start takes roughly 15-30 minutes for weights, Engram tables, humming nvcc builds, and PIECEWISE graphs. Then up `proxy-dsv41`, `proxy-dsv41-abliterated`, and `dcgm-dsv41`. Force-recreate `nginx` and `otelcol-contrib`, then recreate the shared `model-proxy-registrar` once at the end because inline configs are not hashed. Its restart briefly drops all gpu13 small models until re-registration. Scope every up to these services; leave other model containers running. `DSV41_ABLITERATED_ALLOWED_ORG_IDS` is unset on gpu02 and gpu13, so both deny direct `sk-` keys; keep the hosts aligned.
3. Check `curl http://<HOST_IP>:8009/v1/models` lists both ids. Send one chat per id using `--resolve` on port 8444 for both SNIs. Check the model-proxy registries on gpu23 and cpu05 show gpu13 in both DS4.1 domains. Verify the attestation signing addresses match gpu02: base `0xfe0071895500a9755c18044948ab4f4e1a7f337e`, abliterated `0xcc1a41717db31b871950559a41201561716d0c22`. The hosts share compose-manager app id `2c0a0c96…`, and proxies derive keys from its app key plus `MODEL_NAME`; a different address means E2EE key divergence, so stop. Confirm the four rank, two proxy, and one DCGM Prometheus targets are up. Remove the gpu13 Grafana overrides after the window.

Rollback by redeploying the previous tag's GLM services. A GLM cold start can take more than one hour.

## combo4 (2026-11-04): gpu13 diverges from gpu02's r2 ranks on purpose

gpu13 is the first host to run the combo4 settings from the gpu31 lever lab (`/data/dsv41-levers-20261104`, prod gpu13 flags, prod-mix load). All four ranks add:
- `OMP_NUM_THREADS=8`. vLLM otherwise gives each worker one torch thread, which made patch 05's Engram CPU gather and weight loading single-threaded (weight load 330 s → 45 s).
- `VLLM_NEAR_ENGRAM_FAST_GATHER=1`: image patch 06. It is a no-op until the pinned image carries patch 06.
- `--speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":false}'`. Accepted length goes from 2.7 to 3.5.
- `config_variant` `...-dspark5pb-mr64-engram-host-cpugather-omp8-fastgather`, so Grafana can compare gpu13 with gpu02.

`scripts/test_gpu13_dsv41_dep4.py` reverts exactly these changes before its byte-for-byte comparison with r2, so any other drift still fails.

Lab results against the prod config, at rate x2 / x3 / x4:

| Metric | x2 | x3 | x4 |
|---|---|---|---|
| Output tok/s | +12% | +15% | +11% |
| TTFT p50 | −26% | −26% | −49% |
| Per-request TPOT p50 | −44% | −42% | −36% |
| GSM8K-1319 (prod 0.970) | 0.967 | | |

**Deploy.** Deploy from the main tag, one host. First drain: stop `proxy-dsv41` and `proxy-dsv41-abliterated` so the model-proxy and the OpenRouter gateway (budget 320, which gpu02 can carry alone) move traffic to gpu02, and wait for the ranks' running count to reach 0. Then recreate the four ranks, then the two proxies.

Verify:
- both ids return 391 over TLS;
- signing addresses match gpu02;
- the ranks log `Loading weights took ~45 seconds`;
- in prod `vllm_spec_decode_*`, accepted length is about 3.5.

Watch about 24 h against gpu02 before changing gpu02. gpu02 keeps its Engram in HBM, so it would take only the speculative config and `OMP_NUM_THREADS=8`.

**Rollback.** Revert this section's settings, which is the previous main tag.
