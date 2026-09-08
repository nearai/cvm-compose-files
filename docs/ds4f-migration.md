# DS4F temporary relocation

This is the prerequisite for #234, not the host-upgrade or Qwen-consolidation rollout.
One GPU-enabled PPCIe CVM per host. No host reset, VMM/KMS change or dev-host use.

Destination: `prod/GLM-5.1-DSV4-Migration.yaml`:

- GLM5.1 r1 retains GPUs 0–3; drain and remove r2 before applying this file.
- One DS4F TP2 replica uses GPUs 4–5; 6–7 remain unused.
- Preserve the destination GLM engine container identity. Its runtime is unchanged;
  only obsolete Datadog check labels differ from the older deployed file.
- DS4F image, checkpoint revision and inference arguments match the source pack.
- `ds4f-migration-registrar` has a profile and must be explicitly started **after**
  local functional qualification. Shared nginx exposes DS4F HTTP 8001 and TLS 8444.

## Execution gates

1. Capture `/version`, `/status`, `/docker/ps`, `/host/gpu`, registry on every peer,
   current queues/running/active requests, source and destination health. Preserve
   the current compose references and full authoritative environment in the control
   plane; never copy credentials into this repository or evidence files.
2. Use a fixed, validated commit. Every compose call must include the full environment
   map and explicit `services`. Dry-run the exact call first. Reject unexpected
   creates, recreates or removals and concurrent operations. No whole-stack apply:
   compose-manager uses `--remove-orphans` even with service scoping.
3. Drain destination GLM routing before changing its proxy pool. Set
   `GLM51_BACKEND_URLS=http://model-sg-glm51-awq-tp4-r1:8000` and apply **only** the
   proxy in the original GLM file. Refresh shared nginx only after its active work
   has drained, if needed. Verify the other-host GLM replica throughout and the
   destination r1 container remains unchanged. Restore destination GLM registration
   only after a semantic completion succeeds.
4. Require three fresh samples with r2 running, queued and active inference all zero.
   Selectively down only GLM r2 using the original file, `volumes=false`. Verify GPU
   release and both retained GLM replicas. Never evict caches or prune volumes.
5. Apply the mixed file in stages: destination DS4F downloader, engine/proxy/exporters,
   then collector/shared ingress. Do not select the retained GLM engine. Inspect
   dry-run dependencies and orphans before each step. DS4F remains unregistered.
6. On the real destination CVM validate exact model listing, adequate-budget semantic
   completions, streaming to a terminal event, tool calling, TLS/SNI, and attestation.
   Check fresh engine/proxy/DCGM labels, GPU claims, queues, errors, and CUDA/OOM/XID
   signals. HTTP 200/readiness/one-token probes alone do not qualify the replica.
7. Explicitly start the destination DS4F registrar. Verify registration on every peer
   and routed client completions before withdrawing the source.
8. Set source `REGISTER_DSV4=false` in its full environment and apply **only** its
   registrar in `prod/dsv4-qwen38-glm51.yaml`. The old registrar's shutdown unregisters
   all three endpoints: immediately restore/verify both source Qwen registrations,
   then verify the new registrar renews only Qwens for at least two cycles. Do not
   stop or recreate any Qwen engine, proxy or ingress. Explicitly confirm source
   DS4F is absent from every peer before draining its engines.
9. Unregistration does not terminate existing HTTP/2 connections. Wait for three
   fresh zero-work samples on both source DS4F engines and active proxy requests;
   verify destination capacity and failures throughout. Selectively down only DS4F
   engines and exporter, preserving source proxy/nginx while Qwens share the pack.
   Confirm source DS4F GPU claims disappear and the destination remains healthy.
10. Record final topology and runtime identities. Host upgrade is a separate operation;
    the source Qwens still need an approved migration before its CVM can be reset.

## Rollback

- Before cutover, leave source DS4F serving; withdraw the destination DS4F registrar
  before stopping its engine. Use mixed-file scoped operations; preserve GLM r1.
- After cutover, start the original source DS4F services from the recorded immutable
  reference and qualify them before restoring `REGISTER_DSV4=true` and routing.
- To restore destination GLM r2, first withdraw and fully drain destination DS4F,
  remove its engine/exporter, verify GPUs 4–7 are free, then restore GLM r2 and its
  original two-backend pool. Never whole-apply the old file over live DS4F services.
- A failed guard requires investigation, not an automatic retry or forced recreation.
