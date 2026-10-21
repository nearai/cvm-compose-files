# GLM-5.3 Flash eMudhra proxy rollout

The `proxy-glm53` service moves from inference-proxy `0f37728` (`sha256:d61357da…`) to `1f0cfb71` (`sha256:295f16f8…`, eMudhra CA distrust). The latter retains priority normalization and replica-state publishing. No engine or other proxy service is part of this rollout.

| Host | Live compose file | Previous tag |
| --- | --- | --- |
| gpu03 | `prod/GLM-5.3-Flash-SGL-TP2x4-W4AFP8-V8KdaCanary.yaml` | `v0.0.496` |
| gpu04 | `prod/GLM-5.3-Flash-SGL-TP2x3-W4AFP8-gpu04.yaml` | `v0.0.500` |
| gpu02, gpu23 | `prod/GLM-5.3-Flash-SGL-TP4-W4AFP8-LongContext.yaml` | `v0.0.492` |
| gpu13 | `prod/small-models.yaml` | `v0.0.496` |

Brave's verifier currently pins proxy `0f37728`. Brave must pin `1f0cfb71` before any host deploys. Each changed compose file has a new SHA, so coordinate a Brave verifier run after **each host's** deploy to create that host's serving-compose marker; check Brave verification before moving to the next host. A compose file SHA without its marker can fail Brave's relay even when the proxy is healthy.

For each host, read its current **full dashboard environment map** and submit that entire map with `POST /compose/up`, the merged tag, the table's live `file`, `services: ["proxy-glm53"]`, and `force_recreate: false`. Send `dry_run: true` first and confirm that only `proxy-glm53` changes. Then send the same request with `dry_run: false`. Verify the proxy's new image, readiness, and traffic, then run Brave's verifier and confirm its result before proceeding to the next host. On gpu13, the other proxies and the shared anchor must keep `sha256:d61357da…`.

For rollback, use the same scoped `compose/up` and full environment map with `dry_run: true` first, then `dry_run: false`, selecting that host's **previous tag** from the table and `services: ["proxy-glm53"]`. Verify the old `sha256:d61357da…` image and traffic; coordinate Brave verification of the restored compose SHA.
