# Qwen3.8 27B abliterated LoRA candidate

The standalone staging recipe is
[`experiments/Qwen3.8-27B-abliterated-LoRA-test.yaml`](../experiments/Qwen3.8-27B-abliterated-LoRA-test.yaml).
It loads the pinned Qwen3.8 27B FP8 checkpoint and the rank-one adapter in
[`adapters/qwen38-27b-abliterated-r1`](../adapters/qwen38-27b-abliterated-r1/README.md).
The downloader verifies both adapter files against SHA-256 hashes before the
SGLang service starts. SGLang serves the base as `Qwen/Qwen3.8-27B` and the
adapter as the proposed public ID `Qwen/Qwen3.8-27B-abliterated`.

The existing production Qwen3.8 replicas live in
[`prod/qwen36-qwen38.yaml`](../prod/qwen36-qwen38.yaml). They serve the base
model on a separate engine. This candidate is a dedicated stack so the
abliterated adapter cannot be selected through the existing base-model route.

This file is **not deployable to production**. The model-proxy registrar is
behind the `register-abliterated` profile and must remain off during staging.
The adapter was exercised in an isolated direct-SGLang lab, but the full
proxy, attestation, billing, streaming, multimodal, load, and public routing
path has not been qualified.

The candidate proxy sets `VLLM_PROXY_ALLOWED_ORG_IDS` to a nonmatching value
unless `QWEN38_ABLITERATED_ALLOWED_ORG_IDS` is supplied. Customer API keys are
therefore denied by default during staging. The operator must supply only
approved organization IDs and a dedicated internal proxy token for a scoped
test. Provision `QWEN38_ABLITERATED_PROXY_TOKEN` separately from the existing
Qwen deployment and keep it internal. This static list is a temporary backstop;
it must be reconciled with Cloud API grants before every rollout or grant
change.

## Promotion gates

1. Deploy [Cloud API's unlisted-model grants](https://github.com/nearai/cloud-api/pull/1237).
   Create the distinct catalog row as active and unlisted, configure its
   pricing, and grant access only to the intended organizations. Keep the
   abliterated route unregistered until those grants are effective.
2. Land and pin the inference-proxy
   [model-access enforcement](https://github.com/nearai/inference-proxy/pull/297)
   for this stack, then verify its policy snapshot includes the exact model ID
   and is fresh before registration. A missing or stale snapshot must not
   silently widen access. Confirm SGLang lists the exact adapter ID, then run
   authenticated base and adapter completions through the proxy. Verify
   request and usage model IDs match and pin proxy-side request-model
   validation before registration.
3. Complete the repo's 30-minute staging soak with zero failures, including
   mixed base/adapter requests, streaming, tools, vision, queueing, restarts,
   and quality checks. A single behavioral example is insufficient.
4. Verify usage reporting and billing, then coordinate the SNI/domain with
   model-proxy. Confirm the TLS path denies public metrics and admin routes.
5. Promote a reviewed, digest-pinned stack into `prod/` in a separate change.
   Only then plan a scoped deployment and enable the registrar.

No deployment or catalog change is part of this PR.
