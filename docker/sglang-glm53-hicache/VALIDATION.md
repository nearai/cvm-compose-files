# Pre-publication validation

Validation on 2026-09-10 used the pinned production base image and the runtime
source bytes recorded in `source-manifest.json`. All GPU checks below used four
native H200 GPUs with CC disabled. They are **not TEE/PPCIe qualification**.

| Check | Result |
| --- | --- |
| Derivative build and all before/after source checksums | Passed, 22 source/test files |
| CPU restore regressions in the built image | 59 tests and 94 subtests passed; one explicit old-SWA-API skip |
| Pooled KV byte round trips | All four ranks passed, 57,409,536 bytes per rank/direction |
| Pooled index byte round trips | All four ranks passed, 4,316,928 bytes per rank/direction |
| Packed target/MTP restore and completion callbacks | All four ranks passed |
| Separate FP8 target/BF16 draft geometry, index live/off | All four ranks passed |
| Four-rank collective | Passed |
| GLM TP4 recurrent-state clone geometry | Passed, 73,809,920 bytes compared |
| Full-model packaged-image boot, 64 GB host cache per rank | Passed; no source mounts or lab scheduler hooks |
| Synthetic tool requests, cold and GPU-resident reuse | All nine passed, through 131,329 input tokens; exact tool name/arguments and natural completion |
| Promotion scope and invalid-config regressions | Five tests passed, including seven invalid-config mutations |
| Repository Ruby validators and streaming keepalive validator | Passed |
| Compose syntax | All 31 existing files passed |
| Generated r2-only candidate syntax and embedded collector config | Passed using a synthetic digest fixture, not a released image |
| Publishing workflow YAML and embedded shell syntax | Parsed, 14 shell steps |

The initially isolated GPU test's elastic rendezvous could not resolve its
container hostname. The completed run used the documented static loopback
rendezvous. No failed or incomplete attempt is included as a passed GPU run.
The CPU harness uses an unavailable CUDA ordinal because this pinned upstream
test utility cannot parse an empty `CUDA_VISIBLE_DEVICES` value.

The native full-model boot initially used the 32 GB-per-rank diagnostic budget.
It reported 1,179,072 host KV token slots against 1,449,280 device slots. The
64 GB-per-rank candidate then booted with 2,358,144 host KV token slots, above
the same device capacity. These are logical capacities, not multiplied by TP4.
The native container used approximately 262 GiB after the synthetic checks.
The nine serving checks cover cold/device reuse, not CPU restoration under
production eviction pressure; CPU restoration is covered separately by the
byte tests above. Exact-topology TEE memory and serving still need qualification.

The signed workflow has not run for this source. Registry publication,
signature/attestation verification, exact-topology TEE/PPCIe serving, a staging
soak of at least 30 minutes and production canary measurements are outstanding.
The preparation PR changes no production service configuration. The activation
PR must pin the real published image and retain these qualification boundaries.
