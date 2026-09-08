# gpu02 upgrade: evacuate before shutdown

**Shutdown is allowed only when gpu02 is no longer needed to serve any model.**
This is the controlling order for #234 and #235. Neither PR is a whole-stack
migration command. No model is removed from the catalog or redirected to a different
model. Existing source capacity stays available until replacement capacity passes.

## Required placement before shutdown

| Model currently on gpu02 | Serving outside gpu02 before shutdown |
|---|---|
| DeepSeek-V4-Flash | gpu03: two TP2 replicas, GPUs 4–5 and 6–7 (#235) |
| Qwen3.6-35B-A3B-FP8 | gpu13: one TP1 replica, GPU 1 (#234) |
| Qwen3.8-27B | gpu13: one TP1 replica, GPU 2 (#234) |

GLM5.1 remains on gpu03 GPUs 0–3 and gpu13 GPUs 3–6. Other models and existing
GLM5.3 capacity remain in place. During the upgrade both Qwens share gpu13 and
DS4F's replicas share gpu03; this removes the shutdown dependency, not host-failure
redundancy. Reconcile the live catalog, registry and containers: an extra model or
dependency absent from this table blocks shutdown until it is accounted for.

## 1. Prepare destinations; leave gpu02 serving

- Snapshot the running revisions, container IDs, complete environment maps, GPU
  claims and routing on every live model-proxy peer. Store credentials only in the
  existing control plane. Pin the reviewed revisions used for each operation.
- Apply the **gpu13 portion only** of #234: retain Qwen3.6 on GPU 1, safely drain
  its GPU 2 replica, then qualify Qwen3.8 on GPU 2 while both source Qwen3.8 replicas
  remain available. Require the single-replica load/latency gate for both Qwens.
- Prepare #235 on gpu03: keep GLM5.1 r1 on 0–3, drain/remove GLM r2, then qualify
  both DS4F replicas on 4–5 and 6–7 while gpu02 DS4F continues serving.
- Shared-ingress changes are separate guarded operations. In particular, gpu13's
  nginx also serves utility models that have no spare host. Do not blindly recreate
  it or restart its multi-model registrar. Use a validated connection-preserving
  handover; otherwise stop preparation. Keeping model engines running alone does
  not prove their routes or active streams survive an ingress replacement.
- Inspect exact dry-run plans before each apply. Compose-manager uses
  `--remove-orphans` even with explicit services. A final file that omits an old
  replica must not be used while that replica is still serving. Drain/remove it
  against the actual old materialized configuration first; if a safe drain-only
  transition is not available, prepare it before proceeding. Never force past this.
- Do not apply #234's **gpu02 final stack** yet: that file is for the replacement
  CVM after upgrade, not for moving DS4F between slots inside the old CVM.

## 2. Prove all replacement paths before withdrawing gpu02

- Qualify every destination replica on its real CVM: semantic completions with an
  adequate token budget, complete streams, tools, existing model IDs, TLS/SNI,
  authentication and attestation. One successful pooled request is insufficient.
- Verify direct destination paths and routed Cloud API paths, including the private
  inference path where enabled, using fresh and reused client connections. Check
  every current model-proxy peer, not only the public load-balanced registry.
- Confirm request traffic reaches the expected destination engines, with fresh
  metrics, bounded queues/latency, and no new errors, aborts, CUDA/OOM/XID events.
  Carry over #234's representative single-replica capacity gate; if it fails, keep
  source replicas and do not shut down the CVM.
- Finish destination proxy/ingress/registrar changes before entering withdrawal.
  At that point all three models must already have independently working off-host
  routes. Do not treat model-list, readiness, CI or registration alone as proof.

## 3. Evacuate gpu02 as one unit

- Only after **all three models** pass the destination gates, stop gpu02's old
  multi-model registrar once. It withdraws DS4F and both Qwens; at this point that
  is intentional and must leave each model served by its off-host destination.
- Explicitly reconcile withdrawal on every discovered production model-proxy peer.
  Confirm the old HTTP probe endpoints and TLS routing backends are absent for
  every gpu02 model and do not return over at least two registrar/peer-sync cycles.
  Use actual configured intervals, and allow any provider-discovery caches to settle.
- **Do not restart a Qwen-only or DS4F-only registrar on gpu02.** That would put
  the CVM back into the serving path and invalidate the shutdown gate.
- Keep old ingress, proxies and engines alive while existing traffic drains.
  Unregistration affects new routing; it does not close existing HTTP/2 sessions.
  Require three fresh zero-work samples for every source engine (running, queued
  and active inference POSTs), and independently account for remaining client
  connections/streams on all source inference ports. A removed registry entry or
  a zero response-header-lifetime proxy gauge is not evidence of completed streams.
- Long-lived connections must finish or be gracefully retired through a validated
  protocol-aware drain. A forced close, fixed sleep, shortened timeout or SIGKILL
  does not meet the no-downtime requirement. If they cannot be drained, **no-go**.

## 4. Shutdown go/no-go

The operator must record all of the following immediately before shutdown:

- [ ] Current gpu02 model/dependency inventory reconciles to the evacuation list.
- [ ] Both DS4F replicas and each retained Qwen replica pass direct and routed tests.
- [ ] All peers and provider discovery use only off-host destinations for these models.
- [ ] Source registration is stopped; withdrawn endpoints do not reappear.
- [ ] Source engines have zero running, queued and active inference work in three
      fresh samples; no client session or stream remains dependent on the old CVM.
- [ ] Off-host capacity passes representative load/latency checks without source help.
- [ ] A controlled graceful stop of the evacuated model-serving stack has completed
      without forced termination; public probes and telemetry remain healthy while
      gpu02 inference is unavailable. Keep the CVM/control plane available for rollback.
- [ ] Exact CVM ID and VMM/KMS configuration reviewed; no unrelated VM is targeted.

Only then may the upgrade operator stop/rebuild the production GPU CVM. Missing or
stale evidence is **no-go**, not assumed success. Continue client probes and fleet
monitoring during the CVM shutdown and throughout the host upgrade.

## 5. Upgrade and rebuild

- Coordinate a single operator; preserve the separately running CPU-only staging
  CVM and assess any host-wide VMM restart effect before proceeding.
- Follow the internally reviewed host-runtime upgrade and KMS compatibility
  procedure. Verify current implementation and live versions; keep KMS identity
  unchanged. Host-specific settings and credentials belong in the private ops repo.
- One PPCIe GPU-enabled CVM owns all eight cards. Treat old guest state as lost.
  Recreate certificates, deployment environment, pinned model weights/caches and
  telemetry from the preserved control-plane sources before admitting new capacity.
- Apply #234's separate `prod/DSV4-GLM53-After-Upgrade.yaml` only in the new CVM:
  DS4F on 0–3 and GLM5.3 on the complete 4–7 island. Its registrar is profile-gated.
  Keep returning/new endpoints unregistered until their
  real-CVM gates pass; keep gpu03 DS4F and existing GLM hosts serving meanwhile.
- Returning DS4F to gpu02 and restoring gpu03 GLM r2 are later guarded cutovers,
  not prerequisites for keeping users served during this shutdown.

## Rollback before destructive rebuild

On any failed gate, stop the sequence and retain the running old CVM. Requalify and
restore the old source routes from the recorded revisions if needed; verify peer
convergence and client success before withdrawing destination capacity. Do not delete
volumes, weights, keys or the old CVM to recover from an incomplete migration.
