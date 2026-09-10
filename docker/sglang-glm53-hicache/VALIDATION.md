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

The signed workflow has not run for this source. Registry publication,
signature/attestation verification, exact-topology TEE/PPCIe serving, a staging
soak of at least 30 minutes and production canary measurements are outstanding.
The preparation PR changes no production service configuration. The activation
PR must pin the real published image and retain these qualification boundaries.
