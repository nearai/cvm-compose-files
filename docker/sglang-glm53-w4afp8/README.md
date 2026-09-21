# GLM-5.3 Flash W4AFP8 engine: loader fix plus pool clamp

This recipe layers both changes required by the GLM-5.3 W4AFP8 checkpoint onto the pinned
admission-reserve v10 engine image:

1. `modules-to-not-convert.diff` preserves the checkpoint's layer exclusions by propagating
   `modules_to_not_convert` (and the supported `ignored_layers` / `ignore` fallbacks) into
   `W4AFp8Config.ignored_layers`.
2. `chunked-prefill-pool-clamp.diff` prevents `PrefillAdder.add_chunked_req` from requesting more
   tokens than the allocator can serve when the decode projection makes `rem_total_tokens`
   non-positive.

Both patches are required. The pool clamp is an unconditional correctness fix with no opt-in
switch. The admission-reserve v10 behavior inherited from the base image is unchanged: it remains
inert unless `SGLANG_CHUNKED_PREFILL_ADMISSION_RESERVE` is set above zero.

## Reproducibility and failure behavior

The Dockerfile pins the base image by digest and records immutable labels for the build source,
SGLang source, inherited admission-reserve revision, loader patch, and pool clamp. Before changing
source, the build verifies every recipe input through `SHA256SUMS`. `apply-patches.py` then:

1. verifies both patch files against their reviewed SHA256 values;
2. verifies all three touched SGLang files against exact pre-patch hashes;
3. runs `git apply --check` for both patches before applying either one;
4. applies both patches;
5. verifies exact post-patch hashes and AST-parses every manifest file.

Any base, patch, or post-apply drift fails the image build closed.

## Validation

`test-cpu.sh` runs in the published image with `--runtime=runc --network=none`. It preserves the
complete pool-clamp regression matrix, asserts that admission reserve is inert by default, and
calls the actual `W4AFp8Config.from_config` API to prove all supported checkpoint exclusion keys
populate `ignored_layers`.

The release workflow also verifies the immutable image identity, including both
`nearai.sglang.w4afp8_loader_patch=modules-to-not-convert-v1` and
`nearai.sglang.chunked_prefill_pool_clamp=v1`, before the shared provenance, SBOM, scan,
attestation, signing, and signature-verification steps proceed.

## Promotion boundary

This infrastructure-only recipe does not change any Compose file, activate the W4AFP8 canary,
publish by itself, or deploy anything. After the workflow publishes and verifies an image, its
immutable digest must be pinned in a separate reviewed promotion PR before use.
