# DeepSeek-V4-Flash-0731 staged rollout image

This directory reproduces the SGLang runtime qualified on GPU31 on
2026-08-07 and its bounded admission-fairness update qualified on 2026-08-10
for a staged production rollout of
`deepseek-ai/DeepSeek-V4-Flash-0731`.

It deliberately keeps the stable public model name
`deepseek-ai/DeepSeek-V4-Flash`. The 0731 checkpoint is an implementation
change and must not change `MODEL_NAME`, `--served-model-name`, or the
`dsv4-flash.completions.near.ai` SNI domain.

This PR does not change a production compose file. The tested image currently
exists only on GPU31, so a production config must not reference its local image
ID. Build, scan, publish, and pin the resulting registry digest first; then
make the one-replica compose delta described below.

## Provenance

- Base image:
  `lmsysorg/sglang@sha256:984699c298a95b73c469b2191403ddc85fd780506e13c39c4afff3845e27bc6c`
- SGLang source:
  `fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1` (`v0.5.16`)
- Pre-stream admission change: upstream PR #28175, change
  `36613db9813b5c8213edf2c5d74796b811cec491`
- Official DSV4 reasoning-effort support:
  `059269594c5f245f77dad711631843c299d7713f`
- Combined patch SHA-256:
  `0666a644e0791a92606c73d91725b0b52908aa347880973ffa1d3e9dc47282f4`
- Bounded chunked-prefill admission patch SHA-256:
  `3f1c92b7a4655d4cd8d675a4a7cd33d78eb95156a962e53e4ac6da81273f4099`
- Original qualified local image ID:
  `sha256:dac8a4f3f9906a3ef9ac3fdb4b9499492c85ed9d822e179728b0daa2ed1d8c54`
- Fairness-qualified local image ID:
  `sha256:f418325ee720598664e54dd454fe1c815a320678297563c1e329114f47109f52`
- Reproduced Dockerfile image ID:
  `sha256:38727a8d787528b37e2a411ad1659efbf25d5811b544e07c0dd064d3a516fd6e`
- Model revision:
  `7872f01b1d1fe23eabc4c98b48bffcef5a386062`

The local image ID is evidence only. It is not a registry digest and must
never appear in `prod/*.yaml`.

The published production image removes the unused Nsight Compute and Nsight
Systems EFA `nic_sampler` profiler helpers inherited from the CUDA base. Trivy
0.70.0 reported the Compute helper's Go standard library as affected by the
fixable CRITICAL `CVE-2025-68121`. These profiler helpers are not referenced by
the SGLang serving tree and are not required for inference.

## Build and promotion

```bash
docker build \
  --pull \
  --tag ds4f-0731-sglang:v0.5.16-fdebc938 \
  docker/sglang-dsv4-0731
```

The build fails if the patch checksum differs, if it no longer applies to the
pinned base, if the resulting diff has whitespace errors, or if the modified
Python files do not compile.

Before a config PR:

1. scan the image and retain the SBOM;
2. publish it through the approved registry workflow;
3. record the immutable `repository@sha256:...` reference;
4. verify the published digest, patch label, and source labels;
5. run semantic, tool-call, strict-SSE, official low/high/max, clean-overload,
   and long-prefill gates against the published digest on one TP2 replica,
   including the 850k-prefill-plus-short-arrivals admission gate;
6. complete the repository's required 30-minute staging soak with zero
   failures before adding the image to `prod/*.yaml`.

Normal merged tags are not deployable for 48 hours. A same-day rollout must
use the repository's sanctioned `backdate-tag` skill for the deployment
commit. Do not manually tag the commit or lower compose-manager's
`MIN_TAG_AGE_HOURS` gate.

## Qualified per-replica command delta

Keep TP2, FP4 Marlin, FP8 KV, decode CUDA graphs through batch 64, memory
fraction 0.83, 64 running, 16 queued, scheduler conservativeness 1.3, an
8192-token chunk, and the 1800-second watchdog. Remove every EAGLE/DSPARK flag
and add:

```text
--model-path deepseek-ai/DeepSeek-V4-Flash-0731
--revision 7872f01b1d1fe23eabc4c98b48bffcef5a386062
--max-prefill-tokens 16384
--enable-mixed-chunk
--num-continuous-decode-steps 1
--json-model-override-args '{"dsv4_reasoning_effort_profile":"official"}'
```

Set the qualified bounded-fairness control in the engine environment:

```text
SGLANG_CHUNKED_PREFILL_ADMISSION_RESERVE=4096
```

The value is intentionally smaller than the 8192-token chunk. It leaves one
aligned slice for complete queued requests only when the queue head fits; the
active long prefill retains progress and immediately returns to full chunks
after the short burst clears.

Retain:

```text
--served-model-name deepseek-ai/DeepSeek-V4-Flash
--tool-call-parser deepseekv4
--reasoning-parser deepseek-v4
--enable-cache-report
```

The downloader must fetch the 0731 checkpoint at the exact revision above.
Update the DS4F engine's `nearai.otel.model` and Datadog `model:` tag to the
checkpoint path, while keeping the served-model contract unchanged.

## Evidence

The GPU31 release qualification used four TP2 replicas on all eight H200 GPUs
with A/B/B/A followed by B/A/A/B physical-island placement:

- 80/80 strict long-prefill streams completed;
- worst active-decode gap: 2.789 seconds while an 850k-token prefill ran
  behind nine active decodes;
- 56/56 production-shaped quality cases, 32/32 model-card cases, and 16/16
  official-profile cases passed;
- overload returned 936 valid pre-stream HTTP 503 responses with zero
  protocol failures;
- zero OOM, restart, malformed SSE, XID, ECC growth, or AER growth;
- c64: 3,608 total tokens/s/GPU, 10.765-second p99 TTFT;
- c128: 4,194 total tokens/s/GPU, 13.768-second p99 TTFT.

Peak sampled HBM was 143,136 MiB/GPU, leaving only 635 MiB of physical
headroom. Do not raise `--mem-fraction-static 0.83`, loosen admission limits,
or allow a GPU co-tenant.

The frozen 10-second c64 p99 gate missed by 0.765 seconds. This qualifies a
guarded canary, not an immediate fleet cutover.

The first production canary of the published but fairness-unpatched image
(`docker.io/nearaidev/sglang@sha256:eac0a7c825c1c29588fef2f514b7328c1a41b7ac1ed222b3afecc52e32a18525`)
was withdrawn on 2026-08-10. Five short streams admitted behind an already
active 850k chunked prefill reached about 30 seconds TTFB. The backend was
drained and restored to the original checkpoint with no OOM or hardware
failure.

The bounded-fairness validation then ran two reserve-0 controls and two
reserve-4096 candidates simultaneously across both GPU31 NVLink islands:

- both controls reproduced the failure at 112.690-114.833 seconds worst short
  TTFT/event gap;
- both candidates passed at 2.086-2.118 seconds;
- 850k prefill elapsed time was 116.472-118.388 seconds for controls and
  119.332-119.633 seconds for candidates;
- a 6,017-token queue head correctly bypassed the 4,096-token reserve, with
  candidate/control long-prefill ratios of 1.010 and 1.019;
- quality passed 14/14 with zero OOM, restart, malformed stream, engine error,
  XID, ECC growth or AER growth.

Raw artifacts are on GPU31 under
`/data/validation/ds4f-0731-admission-fairness-20260810/runs/v2-candidate-validation-v2`.

## Staged rollout

The first deployment step must override only one DS4F replica with the newly
published and requalified digest. The other four replicas stay on the original
checkpoint and current image as rollback controls. Do not merge the fleet-wide
compose expansion until that exact digest passes direct gates and a registered
canary observation.

One of five production replicas is nominally about 20% of backend capacity,
not 10%. Prefix affinity and queue state can skew actual traffic, so validate
the measured per-backend request share rather than assuming equal routing.

1. Pre-download the checkpoint and start r1 without registration.
2. Repeat the direct semantic/tool/SSE/profile/admission gates and one
   nine-decode-plus-850k-prefill cycle against the published digest.
3. Register r1 and observe it for 30-60 minutes before changing another
   replica.
4. Hold at one replica through a representative window, then continue one
   replica at a time. The complete guarded observation is capped at three
   hours; there is no 24-hour soak requirement.
5. Roll the remaining configs one replica at a time, always preserving an
   original-checkpoint control until the final step.

Abort and drain the candidate on any OOM, worker restart, malformed or
unterminated SSE, in-stream error, semantic/tool failure, active-stream or
short-arrival admission gap above 10 seconds, XID, ECC or AER growth, or two
consecutive five-minute windows above 12 seconds c64-like p99 TTFT or 15
seconds c128-like p99 TTFT.

## Rollback

For the first canary, restore r1 to:

```text
lmsysorg/sglang@sha256:6bb5fee34b6c4537c09a4775e2292ac40350d5ad1218fcc835b2692142f443b1
deepseek-ai/DeepSeek-V4-Flash@553034d7dd9e06c2eeaee68cf85a17d6d4754cf0
EAGLE 3/1/4
```

Withdraw the 0731 backend before recreating it, then re-register only after
the original checkpoint passes readiness. Do not rebuild or retag the rollback
image during the rollout.
