# Read-only GPU migration preflight

Use `prod/migration-gpu-preflight.yaml` through compose-manager with an explicit
non-serving project such as `migration-preflight`, immutable reviewed revision,
and only service `migration-gpu-preflight`. Inspect the exact dry-run first.
Never apply this standalone file to the serving `work` project: orphan removal
could remove unrelated containers. No model or registrar depends on this helper.

The one-shot container has no network, credentials, host filesystem mounts or
Docker socket. NVIDIA's utility-only runtime exposes `nvidia-smi`; the only added
Linux capability is `SYSLOG` for kernel-ring **read-all** and size operations.
It does not clear logs/counters, change clocks/ECC settings or reset GPUs.
There is no privileged mode, host PID namespace or serving restart.

Read its single JSON result through scoped compose logs. Only selected GPU-health
fields and kernel error counts/timestamps are emitted, never raw kernel messages.
Unknown/missing fields remain unknown, failed reads report errors, and
`full_window_available=false` explicitly marks incomplete retained kernel history.
The timestamp uses the guest kernel's uptime, not the operator machine's clock.

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
Run the helper on a staging CVM before its first production use and verify all
existing serving container identities are unchanged.
