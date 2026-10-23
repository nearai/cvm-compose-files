# GLM-5.3 Flash eMudhra proxy rollout

The `proxy-glm53` service moves from inference-proxy `0f37728` (`sha256:d61357da…`) to `f18298f4` (`sha256:e4a42b32…`, #310 eMudhra CA distrust + #311 web_context_search count). The latter retains priority normalization and replica-state publishing. No engine or other proxy service is part of this rollout.

| Host | Live compose file | Previous tag |
| --- | --- | --- |
| gpu03 | `prod/GLM-5.3-Flash-SGL-TP2x4-W4AFP8-V8KdaCanary.yaml` | `v0.0.496` |
| gpu04 | `prod/GLM-5.3-Flash-SGL-TP2x3-W4AFP8-gpu04.yaml` | `v0.0.500` |
| gpu02, gpu23 | `prod/GLM-5.3-Flash-SGL-TP4-W4AFP8-LongContext.yaml` | `v0.0.492` |
| gpu13 | `prod/small-models.yaml` | `v0.0.496` |

Brave's verifier currently pins proxy `0f37728`. Brave must pin `f18298f4` before any host deploys. Each changed compose file has a new SHA, so coordinate a Brave verifier run after **each host's** deploy to create that host's serving-compose marker; check Brave verification before moving to the next host. A compose file SHA without its marker can fail Brave's relay even when the proxy is healthy.

For each host, read its current **full dashboard environment map** and submit that entire map with `POST /compose/up`, the merged tag, the table's live `file`, `services: ["proxy-glm53"]`, and `force_recreate: false`. Send `dry_run: true` first and confirm that only `proxy-glm53` changes. Then send the same request with `dry_run: false`. Verify the proxy's new image, readiness, and traffic, then run Brave's verifier and confirm its result before proceeding to the next host. On gpu13, the other proxies and the shared anchor remained on `sha256:d61357da…` during this GLM-only rollout; the follow-up below moves them.

For rollback, use the same scoped `compose/up` and full environment map with `dry_run: true` first, then `dry_run: false`, selecting that host's **previous tag** from the table and `services: ["proxy-glm53"]`. Verify the old `sha256:d61357da…` image and traffic; coordinate Brave verification of the restored compose SHA.

## gpu13 small-models: all proxies on f18298f4

Deploy the merged tag with `prod/small-models.yaml` using `POST /compose/up`, the full dashboard environment map, `services: ["proxy-qwen36-35b-a3b","proxy-qwen38-27b","proxy-glm53","proxy-flux2-klein-4b","proxy-qwen3vl-30b-a3b","proxy-qwen3-embedding-0.6b","proxy-qwen3-reranker-0.6b","proxy-whisper-large-v3","proxy-privacy-filter"]`, and `force_recreate: false`. Run `dry_run: true` first, confirm only those nine proxies change, then run `dry_run: false`. Next, `compose/up` `services: ["nginx","nginx-qwen38"]` with `force_recreate: true`, followed by `compose/up` `services: ["model-proxy-registrar","qwen38-registrar"]` with `force_recreate: true`. Dry-run each scoped step before applying it. Both nginx services resolve proxy IPs once at load; skipping their recreation can silently deregister models. Verify all nine gpu13 domains in `https://completions.near.ai/registry` and coordinate Brave verification of the new compose SHA.

For rollback, repeat those proxy, nginx, and registrar steps with tag `v0.0.502` and the full dashboard environment map. Its eight non-GLM proxies use `sha256:d61357da…`; `proxy-glm53` already uses `sha256:e4a42b32…`.
