# Read-only GPU migration preflight

Use `prod/migration-gpu-preflight.yaml` through compose-manager with an explicit
non-serving project such as `migration-preflight`, immutable reviewed revision,
and only service `migration-gpu-preflight`. Inspect the exact dry-run first.
Never apply this standalone file to the serving `work` project: orphan removal
could remove unrelated containers. No model or registrar depends on this helper.

For both `POST /compose/up` (including the dry-run) and `POST /compose/down`,
set the supported API field `"project": "migration-preflight"` explicitly.
The YAML also has `name: migration-preflight` as a standalone Compose default;
that does not protect a request which explicitly selects the serving project.
Example up payload (replace the revision with the reviewed immutable commit):

```json
{
  "tag": "<reviewed-40-character-commit>",
  "file": "prod/migration-gpu-preflight.yaml",
  "project": "migration-preflight",
  "services": ["migration-gpu-preflight"],
  "env": {},
  "dry_run": true
}
```

Require no removals or changes to existing serving containers and only the
requested diagnostic service under the `migration-preflight-` prefix. Then send
the identical payload with `dry_run: false`. Use that same project/file/service
for scoped `POST /compose/logs`, and for cleanup with `volumes: false`.
If the deployed manager cannot select this project, stop; do not use `work`.
Project-scoped log reads require nearai/compose-manager#60. Older managers accept
the field but silently ignore it and can return empty logs from `work`; an empty
result is not a successful diagnostic. Validate log retrieval on staging before
using this helper in production.

The one-shot container has no network, credentials, host filesystem mounts or
Docker socket. NVIDIA's utility-only runtime exposes `nvidia-smi`; the only added
Linux capability is `SYSLOG` for kernel-ring **read-all** and size operations.
It does not clear logs/counters, change clocks/ECC settings or reset GPUs.
There is no privileged mode, host PID namespace or serving restart.

Read its single JSON result through scoped compose logs. Only selected GPU-health
fields and kernel error counts/timestamps are emitted, never raw kernel messages.
Unknown/missing fields remain unknown, failed reads report errors, and
`full_window_available=false` explicitly marks incomplete retained kernel history.
Window selection uses guest uptime, not the operator machine's clock. Kernel
timestamps ahead of uptime are conservatively counted, never silently discarded.
`clock_alignment_ok=false` and `future_records` expose clock uncertainty; resolve
that uncertainty before claiming a clean recent window. `full_window_available`
describes retained history coverage, not clock alignment or a health verdict.

`collection_ok` means evidence was collected, **not** that the GPUs passed.
Require the expected unique device inventory; inspect volatile uncorrectable ECC,
row-remap failures/pending retirement, recovery/reset requirements and recent
XID/ECC/AER/OOM events. Investigate nonzero or unknown results. Compare repeated
snapshots to distinguish historical/latched counters from new errors, alongside
fresh metrics and successful real-CVM inference. Never treat missing telemetry
or a successful container exit as zero errors.

This closes a diagnostic gap in the staged migration from #235; it does not change
the no-downtime or source-preservation requirements of that runbook.

Local parser/isolation checks: `ruby scripts/validate_migration_gpu_preflight.rb`.
For CPU-only staging CVMs, explicitly select `migration-kernel-preflight`: it
exercises the identical kernel collector without requesting GPUs. Its result is
clearly marked `mode=kernel-only` and is not GPU-health evidence. Validate the
full GPU variant on a development GPU host too, then collect the actual target
CVM's evidence before model admission. Verify all pre-existing container identities
remain unchanged at each step. Both services are profile-gated one-shot tools.
