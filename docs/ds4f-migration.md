# DS4F temporary relocation

This is the DS4F portion of the [evacuate-before-shutdown plan](gpu02-upgrade.md),
coordinated with the gpu13 Qwen portion of #234. Neither PR's complete old runbook
should be executed independently for this upgrade. All three gpu02 models must be
serving off-host before source registration is stopped or the old CVM is shut down.
One GPU-enabled PPCIe CVM per host. No host reset, VMM/KMS change or dev-host use.

Destination: `prod/GLM-5.1-DSV4-Migration.yaml`:

- GLM5.1 r1 retains GPUs 0–3; drain and remove r2 before applying this file.
- Two DS4F TP2 replicas use GPUs 4–5 and 6–7. All eight GPUs remain assigned;
  DS4F retains its existing two-replica/four-GPU capacity after the move.
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
5. Apply the mixed file in stages: destination DS4F downloader, both engines/proxy/exporters,
   then collector/shared ingress. Do not select the retained GLM engine. Inspect
   dry-run dependencies and orphans before each step. DS4F remains unregistered.
   Before downloading weights, explicitly run `migration-disk-preflight` (profile
   `migration-preflight`) and inspect its JSON output through compose-manager logs.
   Compare guest cache free space with the required checkpoint/image download and
   retain working headroom. Host filesystem capacity is not guest free space. This
   check mounts only the existing cache read-only, with no network, credentials or GPU.
6. Qualify **both destination DS4F replicas**, not just a successful pooled request.
   On the real destination CVM validate exact model listing, adequate-budget semantic
   completions, streaming to a terminal event, tool calling, TLS/SNI, and attestation.
   Check fresh engine/proxy/DCGM labels, GPU claims, queues, errors, and CUDA/OOM/XID
   signals for each replica. HTTP 200/readiness/one-token probes alone do not qualify
   the replicas. Do not withdraw source capacity until both destination replicas pass.
7. Explicitly start the destination DS4F registrar. Verify registration on every peer
   and routed client completions before withdrawing the source.
8. **Do not withdraw source DS4F in isolation for this upgrade.** First complete
   and qualify both off-host Qwen paths from #234. Then follow the controlling
   runbook to stop gpu02's old registrar once and withdraw all three source model
   routes together. Do not restart the registrar with `REGISTER_DSV4=false`; the
   Qwen routes must remain off gpu02 too. The switch remains a backward-compatible
   standalone migration option, not the full-host evacuation procedure.
9. Drain all source engines and client sessions as specified in the controlling
   runbook. Registry removal and zero queued/running gauges alone do not establish
   that HTTP/2 streams are drained. Keep the old CVM available for rollback.
10. Record the shutdown go/no-go evidence and validate public serving with the old
    inference stack stopped before the upgrade operator shuts down the CVM.

## Rollback

- Before cutover, leave source DS4F serving; withdraw the destination DS4F registrar
  before stopping its engines. Use mixed-file scoped operations; preserve GLM r1.
- After cutover, start the original source DS4F services from the recorded immutable
  reference and qualify them before restoring `REGISTER_DSV4=true` and routing.
- To restore destination GLM r2, first withdraw and fully drain destination DS4F,
  remove both DS4F engines/exporter, verify GPUs 4–7 are free, then restore GLM r2 and its
  original two-backend pool. Never whole-apply the old file over live DS4F services.
- A failed guard requires investigation, not an automatic retry or forced recreation.
