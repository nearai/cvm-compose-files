# GLM-5.3 SGLang v0.5.21 + W4AFP8 loader fix

Engine image for the GLM-5.3 Flash W4AFP8 prefill/decode (PD) canary. It is stock upstream
`lmsysorg/sglang:v0.5.21` plus exactly one source patch and the same security hardening used by
`docker/sglang-glm53-v0519`.

## Why

The fork-based images (`docker/sglang-glm53-w4afp8` and ancestors, SGLang fc91d24 base) cannot run
GLM-5.3 PD. Two bugs are fixed only upstream:

- sgl-project/sglang#38417: draft DSA tail pool;
- the v0.5.21 removal of the kpool top-k width assertion (2051 assert).

Stock v0.5.21 runs PD correctly (1.5K to 750K prompts, passkey correct, gpu32 rehearsal), but
cannot load `graphistry/GLM-5.3-Flash-W4AFP8`: upstream `W4AFp8Config.from_config` drops
`modules_to_not_convert`, so narrow layers hit `output_partition_size = 32 is not divisible by
block_n = 128`. Evidence: `tee-bench/docs/FINDINGS.md` sections 15-17 and
`tee-bench/evidence/gpu13-pd-island-b/` (logs and `w4afp8-ignored-layers.patch`).

## What is patched

`modules-to-not-convert.diff` (byte-identical to the one in `docker/sglang-glm53-w4afp8`) edits
only `python/sglang/srt/layers/quantization/w4afp8.py`: `from_config` reads
`modules_to_not_convert` (falling back to `ignored_layers` / `ignore`) and passes it as
`ignored_layers`. The v0.5.21 file has the same SHA256 as the fork file the patch was written
against, so the diff applies unchanged.

Nothing else differs from upstream except hardening: hash-pinned `nltk 3.10.3` plus
`defusedxml 0.7.1`, removal of unused Nsight `efa_metrics/nic_sampler` binaries
(CVE-2025-68121), and the `PROVENANCE` record.

HiCache is OFF for PD prefill, so the fork's HCC-safe HiCache patch and managed-memory
allocator are intentionally absent. Do not use this image with HiCache flags.

## Build-time verification (fail closed)

1. `sha256sum --check SHA256SUMS` over all recipe inputs;
2. `apply-patches.py`: patch checksum, exact pre-patch hash of `w4afp8.py`, `git apply --check`,
   apply, exact post-patch hash, AST parse;
3. hardened package versions and absence of `nic_sampler`;
4. an import-time assertion that `W4AFp8Config.from_config` returns the exclusion list from
   `modules_to_not_convert`.

## Publishing and verification

Only `.github/workflows/publish-glm53-v0521-w4afp8.yaml` publishes this image (default tag
`glm53-v0521-w4afp8-near-v1` on `docker.io/nearaidev/sglang`). Production must pin the digest.
Verify as for the v0.5.19 image, with certificate identity
`https://github.com/nearai/cvm-compose-files/.github/workflows/publish-glm53-v0521-w4afp8.yaml@refs/heads/main`.

## TODO: upstream

Send the loader fix to sgl-project/sglang so this image can become unpatched. Once
upstream carries it, retire this recipe and use `docker/sglang-glm53-v0519`-style hardening only.
