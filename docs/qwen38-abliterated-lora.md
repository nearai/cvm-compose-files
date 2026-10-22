# Qwen3.8 27B abliterated LoRA candidate

The standalone staging recipe is
[`experiments/Qwen3.8-27B-abliterated-LoRA-test.yaml`](../experiments/Qwen3.8-27B-abliterated-LoRA-test.yaml).
It loads the pinned Qwen3.8 27B FP8 checkpoint and the rank-one adapter in
[`adapters/qwen38-27b-abliterated-r1`](../adapters/qwen38-27b-abliterated-r1/README.md).
The downloader verifies both adapter files against SHA-256 hashes before the
SGLang service starts. SGLang serves the base as `Qwen/Qwen3.8-27B` and the
adapter as the proposed public ID `Qwen/Qwen3.8-27B-abliterated`.

This file is **not deployable to production**. The model-proxy registrar is
behind the `register-abliterated` profile and must remain off during staging.
The adapter was exercised in an isolated direct-SGLang lab, but the full
proxy, attestation, billing, streaming, multimodal, load, and public routing
path has not been qualified.

## Promotion gates

1. Confirm SGLang lists the exact adapter ID, then run authenticated base and
   adapter completions through the inference proxy. Verify request and usage
   model IDs match, and that only the approved model ID can select the
   adapter through the public route. Pin a proxy build with the required
   model-identity validation before registration.
2. Complete the repo's 30-minute staging soak with zero failures, including
   mixed base/adapter requests, streaming, tools, vision, queueing, restarts,
   and quality checks. A single behavioral example is insufficient.
3. Add the distinct model ID and pricing to Cloud API and its public catalog,
   verify usage reporting and billing, then coordinate the SNI/domain with
   model-proxy. Confirm the TLS path denies public metrics and admin routes.
4. Promote a reviewed, digest-pinned stack into `prod/` in a separate change.
   Only then plan a scoped deployment and enable the registrar.

No deployment or catalog change is part of this PR.
