# GLM-5.3 Flash HiCache canary

This build carries complete hybrid-cache restoration and opt-in pooled transfers
on the exact `fc91d24` production SGLang runtime. It is intended for **one TP4
replica inside one CVM**, with the other TP4 replica retaining its existing image
and HiCache-disabled configuration. It adds no storage backend or cross-CVM
cache exchange.

## Release order

The repository publishes signed production images only from an exact commit
already merged into `main`. This preparation change therefore leaves the active
production compose untouched. After review and merge:

1. Dispatch `.github/workflows/publish-glm53-hicache.yaml` on `main`, supplying
   the exact merged `source_revision` and a fresh publishing tag. It builds,
   runs CPU regressions, scans, attests and signs the immutable image. A resumed
   run verifies that the tag still matches the requested digest.
2. Verify the resulting signature and attestation using the commands below.
3. Preview the exact canary configuration, then create its activation PR:

   ```bash
   python3 scripts/prepare_glm53_hicache_canary.py --image "$IMAGE"
   python3 scripts/prepare_glm53_hicache_canary.py --image "$IMAGE" --write
   ruby scripts/validate_glm53_prod_config.rb
   python3 scripts/test_glm53_hicache_canary.py
   ```

   The writer edits only local files; it does not push, merge or deploy. It pins
   `RELEASED_IMAGE`, overrides r2's image/command/environment, and updates its
   log and metric `config_variant`. r1, GPU allocations, routing, conversation
   affinity, model/template revisions, MTP, request limits and logging policy
   retain their existing values. The generated compose remains self-contained.
4. Qualify the signed image in staging on the intended TEE/PPCIe topology,
   including a minimum 30-minute soak, before separately authorized activation.

Do not substitute a local Docker image ID, a mutable tag, or the unchanged base
digest for the published candidate digest. Image publication does not deploy.

```bash
# Set IMAGE to docker.io/nearaidev/sglang@sha256:<published digest>.
cosign verify \
  --certificate-identity \
  'https://github.com/nearai/cvm-compose-files/.github/workflows/publish-glm53-hicache.yaml@refs/heads/main' \
  --certificate-oidc-issuer 'https://token.actions.githubusercontent.com' "$IMAGE"
gh attestation verify "oci://$IMAGE" --repo nearai/cvm-compose-files
docker buildx imagetools inspect --format '{{json .Provenance.SLSA}}' "$IMAGE"
docker buildx imagetools inspect --format '{{json .SBOM.SPDX}}' "$IMAGE"
```

## Candidate configuration

| Setting | r1 control | r2 candidate |
| --- | --- | --- |
| Image | Existing signed production digest | New signed derivative digest |
| HiCache | Disabled | Enabled |
| Host cache size | None | `--hicache-size 32` per rank, nominal 128 GB for TP4 |
| Write policy | None | `write_through` |
| I/O and host layout | None | `direct`, `page_first_direct` |
| Transfers | Existing runtime | `SGLANG_HICACHE_POOLED_TRANSFERS=1` |
| Persistent staging | None | `SGLANG_HICACHE_STAGING_PAGES=64` per pool/direction/rank |

The cache-size argument is in decimal GB and is **not a container memory cap**.
Allow additional RAM for Python/tokenizers, pinned allocator/page rounding,
draft sidecars where applicable, shared memory and the other replica. Check
available memory inside the CVM before starting the candidate. Host cache holds
KV/index pages and finite recurrent checkpoints; token capacity alone does not
determine how many conversations can resume. There is no conversation-length
admission policy in this change.

Pooling packs many small layer transfers into a persistent device buffer,
copies complete pages with CUDA DMA, then scatters them on the GPU. Triton
kernels access GPU memory only. The NVIDIA driver retains responsibility for
the confidential-computing transfer path; this patch does not change CC mode,
PPCIe, NCCL, peer-access or attestation settings. It neither establishes nor
verifies the platform's CC configuration.

Staging belongs to the existing rank-local transfer streams. The existing
completion events still govern cache lifetime. Packed target/MTP consumers
are skipped only after full unique layer coverage is restored. Duplicate
recurrent-state reads become GPU clones only for disjoint destinations with
matching owners, indices and layer maps. Unsupported layouts or sharding use
the existing transfer path. The optimization requires NVIDIA CUDA and the
`direct` backend. No lab control endpoint or scheduler interception is carried.

## Source and correctness

`runtime.patch` combines the adapted public
[complete-restore fix](https://github.com/sgl-project/sglang/pull/38212) at
`230102db838cbbe6eca7952c729518ee6d93c3a9` with pooled transfers. That fix already
includes the packed-row dependency from
[PR #37534](https://github.com/sgl-project/sglang/pull/37534); do not apply it twice.
It restores DSA index sidecars and fixes compressed-prefix ownership, recurrent
checkpoint boundaries and target/draft row geometry. The adaptation targets
the older `fc91d24` MLA constructor and CPU-test APIs.

`source-manifest.json` records exact before/after hashes, including newly added
files. Build-time verification refuses base drift, checks patch integrity and
parses every resulting Python file. `tests.patch` contains public upstream
regressions and synthetic byte-preservation tests. It contains no captured
conversations or customer-derived fixtures.

## Reproduce validation

Build locally without publishing:

```bash
docker build --build-arg BUILD_SOURCE_REVISION="$(git rev-parse HEAD)" \
  -t glm53-hicache:test docker/sglang-glm53-hicache
docker run --rm --runtime=runc --network=none --memory=16g \
  --entrypoint=bash glm53-hicache:test \
  /usr/share/nearai/glm53-hicache/test-cpu.sh
```

After acquiring four idle GPUs, test exact byte round trips on all four ranks:

```bash
docker run --rm --gpus '"device=0,1,2,3"' --shm-size=4g --memory=32g \
  --ulimit memlock=-1 -e SGLANG_HICACHE_POOLED_TRANSFERS=1 \
  --entrypoint=bash glm53-hicache:test -c \
  'set -e; torchrun --nnodes=1 --nproc-per-node=4 --master-addr=127.0.0.1 --master-port=29500 test/manual/test_pooled_pooled_tp4.py; python3 test/manual/test_pooled_kda_clone.py'
```

The CPU suite includes one explicit skip for an SWA request field absent from
the pinned base; the GLM path does not use SWA. GPU tests cover sparse pages,
incremental backups, poisoned buffers, changed destinations, staging boundaries,
packed MTP, separate draft geometry, recurrent-state clones and a collective.
Native GPU checks do not qualify TEE/PPCIe behavior or prove a speedup there.

For serving validation, distinguish CPU host hits from GPU-resident hits using
cache-tier metrics. Use synthetic multi-turn tool fixtures, force device
eviction through workload pressure, and require correct natural completion
after restoration. Keep the control and candidate's input/output limits and
concurrency identical. Warm compilation before timing. Compare TTFT, decode
latency, errors and host memory; include resident-cache traffic to measure the
write-through cost. Compare per-replica `instance`/`config_variant` metrics;
conversation affinity is sticky routing, not randomized A/B assignment.

Pooling restores a whole packed pool before marking layer consumers ready,
which can reduce layer/compute overlap. Native measurements must not be used
to promise a TEE gain. Long prompts beyond tested lengths, long soaks and
the exact four-GPU TEE/PPCIe topology remain separate qualification work.

## Rollback

Revert the activation commit (the compose overrides plus `RELEASED_IMAGE`) and
validate the canonical contract. Apply the authorized rollback to r2 and the
collector only, preserving r1. Reverting the runtime preparation is unnecessary.
Removing or recreating r2 drops its in-memory host cache; requests can prefill
again. Nothing in this change persists KV to disk or another host.
