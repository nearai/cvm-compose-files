# Qwen consolidation and post-upgrade GLM5.3 capacity

**The old gpu02 CVM must be fully evacuated before shutdown.** Follow the
[controlling upgrade runbook](gpu02-upgrade.md).
The former in-place DS4F slot-move procedure is superseded; do not run it.

## Two different stages, two different files

| Stage | File | Scope |
|---|---|---|
| Before shutdown | `prod/small-models.yaml` | gpu13: Qwen3.6 on GPU 1, Qwen3.8 on GPU 2 |
| During evacuation | `prod/dsv4-qwen38-glm51.yaml` | Existing gpu02 stack retained for serving and rollback; unchanged by this PR |
| After upgrade | `prod/DSV4-GLM53-After-Upgrade.yaml` | New gpu02 CVM only: DS4F on 0–3, GLM5.3 on 4–7 |

A separate final file prevents the replacement-CVM configuration from overwriting
the old host's serving/rollback recipe. The Compose project can remain `work`;
the old and new guests are never run concurrently with split PPCIe GPU ownership.

## Before shutdown: prepare only gpu13

Use the [connection-preserving handover](gpu13-qwen-handover.md) for this stage.
The shared nginx and registrar are unchanged; Qwen3.8 gets its own HTTP8000 /
TLS8010 listener and registrar. Do not execute the final file as a whole-stack up.

1. Snapshot actual deployed revisions, container IDs, full environment maps, GPU
   UUID/slot mapping and routing on every live model-proxy peer.
2. Prepare the Qwen3.6 one-backend drain while retaining its current r2 until idle.
   Never apply a final file that omits a serving replica: compose-manager passes
   `--remove-orphans` even on service-scoped calls. Inspect the exact dry-run and
   prepare a safe drain-only transition if needed. Preserve the r1 engine.
3. After r2 is no longer selected and has three fresh zero-work samples, remove it
   and release GPU 2 using the actual old materialized configuration; retain caches.
4. Bring up the pinned Qwen3.8 engine on GPU 2. Verify both consolidated Qwens on
   the real destination CVM before retiring source capacity: semantic completions,
   full streams, tools, auth/attestation and at least 30 minutes of representative
   single-replica load without failures, growing queues or latency regression.
5. Establish and verify both gpu13 routes on every peer and through Cloud API.
   Shared nginx/registrar changes must preserve existing routes and streams for
   utility models and GLM5.1 too. A blind container restart is not a safe handover.
   If a connection-preserving transition is unavailable or unverified, stop.
6. Reconcile Qwen3.6 GPU 1 and Qwen3.8 GPU 2 engine/DCGM/OTel claims. Keep all gpu02
   engines serving until gpu03's two DS4F replicas from #235 are also qualified.

## Evacuate and shut down gpu02

- This is coordinated by #235's controlling runbook, not a separate Qwen-only
  withdrawal. All three models must already serve off-host before stopping the old
  gpu02 registrar; do not briefly withdraw the only working route for another model.
- Stop old source registration once, remove all source routes on every peer, allow
  discovery caches to settle, and drain existing inference work and HTTP/2 sessions.
- Prove public model serving with the evacuated inference stack gracefully stopped
  while the CVM is still recoverable. Only after the full shutdown checklist passes
  may the operator stop/rebuild the production GPU CVM.
- Do not move DS4F from 6–7 to 2–3 in the old guest. Do not pre-build GLM there and
  assume its local tag or weights survive the destructive guest replacement.

## After the upgraded CVM is ready

- Verify the internally reviewed VMM/guest and KMS-compatible host configuration, one
  GPU-enabled CVM, certificates/secrets, metrics, GPU health and both NVLink islands.
- On the new CVM, use `prod/DSV4-GLM53-After-Upgrade.yaml`: DS4F r1 on 0–1, r2 on
  2–3, GLM5.3 TP4 on the complete 4–7 island. Never allocate GLM across 2–5.
- Rebuild and prove the source-pinned GLM image by immutable image ID/OCI revision
  `fc91d2403cc210a86bd5fc715c6609f58c932e5a`. Download the pinned weights into the
  new guest. Keep gpu03 DS4F and existing GLM hosts serving throughout qualification.
- The new registrar is profile-gated. Start it explicitly only after both returning
  DS4F replicas pass; keep `REGISTER_GLM53=false` until GLM also passes its text,
  streaming, tools, vision, cache, auth/attestation and stability gates.
- Return DS4F routing to gpu02 only after qualification; drain/remove gpu03 DS4F and
  restore gpu03 GLM r2 in a later guarded cutover. Never whole-apply overlapping
  old/new GPU allocations as rollback.
- Final intended counts remain DS4F 4 GPUs, Qwen3.6 1, Qwen3.8 1, GLM5.3 12.
  Qwens lose replica and host redundancy; capacity qualification is mandatory.

## Stop / rollback

Before destructive rebuild, any failed gate means retain or requalify the old CVM
and restore its recorded source routes; verify client success before withdrawing
destinations. After rebuild, keep users on the verified off-host replicas until the
new host passes. No forced connection termination, queue-limit inflation, volume
deletion or blind whole-stack rollback to get past a failed gate.
