# Pre-publication validation

Validation on 2026-09-10 used the pinned production base image and the runtime
source bytes recorded in `source-manifest.json`. The transfer byte checks
preceded the startup-sizing addition; its transfer kernels are unchanged.
All GPU checks below used four
native H200 GPUs with CC disabled. They are **not TEE/PPCIe qualification**.

| Check | Result |
| --- | --- |
| Derivative build and all before/after source checksums | Passed, 28 source/test files |
| CPU restore and startup RAM regressions in the built image | 71 tests and 108 subtests passed; one explicit old-SWA-API skip |
| Pooled KV byte round trips | All four ranks passed, 57,409,536 bytes per rank/direction |
| Pooled index byte round trips | All four ranks passed, 4,316,928 bytes per rank/direction |
| Packed target/MTP restore and completion callbacks | All four ranks passed |
| Separate FP8 target/BF16 draft geometry, index live/off | All four ranks passed |
| Four-rank collective | Passed |
| Startup RAM rendezvous (real CPU Gloo, four processes) | Passed: same budget on all ranks, real cgroup sampling, rank cap enforcement, peer invalid-config/CLI conflict failures |
| Deployment RAM variable interpolation | Passed: default 80%, custom 60%, 256GB and 256GiB; r1 receives no RAM variable |
| GLM TP4 recurrent-state clone geometry | Passed, 73,809,920 bytes compared |
| Prior full-model packaged-image boot, fixed 64 GB host cache per rank | Passed; no source mounts or lab scheduler hooks |
| Full-model automatic startup budget | Passed: 80% in a 320 GiB container, 177,043,164,364 total budget bytes shared by four ranks; actual payload 176,955,092,992 bytes |
| Synthetic tool requests after automatic sizing, cold and GPU-resident reuse | All nine passed, through 131,328 input tokens; exact tool name/arguments and natural completion |
| Promotion scope and invalid-config regressions | Five tests passed, including eight invalid-config mutations |
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
That earlier native container used approximately 262 GiB after its synthetic checks.

With startup sizing enabled, all four ranks measured the same available
221,303,955,456 bytes inside a 320 GiB container. Each received a
44,260,791,091-byte budget and allocated 44,238,773,248 payload bytes across
KV/MTP, recurrent state and index pools. The resulting host KV capacity was
1,630,784 logical tokens, against 1,449,280 device slots. All nine new synthetic
serving checks passed through 131,328 input tokens, with no OOM or restart.
The CPU Gloo rendezvous test makes no large allocations and sets its reserve
to zero for small CI guests; separate unit tests enforce the production 10 GiB
reserve and reject oversized explicit budgets.
The nine serving checks cover cold/device reuse, not CPU restoration under
production eviction pressure; CPU restoration is covered separately by the
byte tests above. Exact-topology TEE memory and serving still need qualification.

Registry publication, signature/attestation verification, exact-topology
TEE/PPCIe serving, a staging soak of at least 30 minutes and production canary
measurements were outstanding when the candidate was prepared.

## Production HCC/PPCIe result

The `v0.0.412` candidate at `d1a72bf`, using
`docker.io/nearaidev/sglang@sha256:67cf951972594cdbf7437faf315145556ae6f3a643355ff4d3078fb670a12b9f`,
was activated only on gpu02 replica 2 on 2026-09-11.
All four ranks reported a 9,812,352-token host-cache allocation (120.57 GB per
rank), then TP1 failed before readiness in `cudaHostRegister`:

```text
cudaHostRegister failed (rc=801)
TypeError: cudaGetErrorString(): incompatible function arguments
Invoked with: 801
```

Error 801 is `cudaErrorNotSupported`. NVIDIA's [Hopper
confidential-computing release
notes](https://docs.nvidia.com/550trd3-nvidia-trusted-computing-solutions-release-notes.pdf)
state that pinned-host-memory APIs are unsupported in HCC and that
`cudaHostRegister`/`cudaHostUnregister` return this error. The Python
`TypeError` is a secondary formatting defect caused by passing the raw integer
return code to the enum-typed binding; correcting it would not make registration
or direct HiCache transfers supported.

Replica 2 was restored to the signed control image and HiCache-disabled command.
Both replicas passed the semantic check, the machine registry advertised gpu02,
and a fresh Cloud API request returned HTTP 200. Production activation remains
blocked until the runtime uses an HCC-supported host-transfer mechanism and
passes exact-topology qualification.
