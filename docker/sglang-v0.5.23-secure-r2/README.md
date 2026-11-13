# SGLang v0.5.23 + NEAR AI security patches (model-agnostic)

This recipe is for every stock-SGLang model (Qwen3.6, Qwen3.8, FLUX.2 and others). It builds upstream SGLang **v0.5.23** (tag 297e54fd, released 2026-10-28) with only the four engine-agnostic security patches from the GLM-5.3 series (`docker/sglang-glm53-v0523`). It has no model-specific or performance patch. GLM-5.3 keeps its own recipe.

Upstream v0.5.23 already closes these CVEs:
- CVE-2026-15976: `update_weights_from_disk` pickle RCE, fixed by #42760;
- CVE-2026-86793: `update_weights_from_tensor` unpickler bypass, fixed by #39858 and #40259;
- CVE-2026-15969: LoRA-from-tensors unpickler bypass, fixed by #41322 and #40259.

The patches add the fixes v0.5.23 still lacks:

| patch | what | prod flag / env |
|---|---|---|
| 05-rid-abort-exact-match | CVE-2026-112266, a customer-reachable abort of other users' requests. Aborts match the exact rid; `<rid>_<n>` children match only for an rid the worker does not own (`AbortReq.match_children`, last field, wire-compatible) | unconditional |
| 14-admin-auth-redaction | CVE-2026-15977, credential redaction rebuilt from upstream #48518 (it landed after the release): `build_server_info` (HTTP + gRPC + rust launch info), scheduler `internal_states`, dumps always a redacted dict, crash-dump `launch_command`. The admin-auth half is upstream #41322, already in v0.5.23 | unconditional |
| 15-customer-data-log-redaction | media URL → `<url host=H>`, `data:` → `<base64 len=N>`, schemas/values → `<len=N sha256=…>`, redacted tracebacks. Keeps the `NNN Client Error … for url:` wording that inference-proxy maps to a 400 | unconditional |
| 26-log-format-followup | log line format follow-up to 15 | unconditional |

The patch files are byte-identical to the same-numbered patches in `docker/sglang-glm53-v0523/patches/` (the sha256 values in `SERIES` match). They keep their series numbers so the two recipes can be compared. They do not depend on the other GLM patches:
- each one passes `git apply --check` on a clean 297e54fd checkout, in this order;
- no other GLM patch adds or removes a line that uses `log_redaction`, `abort_match` or `match_children`. Patch 25 has the `log_redaction` import only as context.

So on every code path that stock v0.5.23 has, this image behaves like the GLM image. The GLM-only paths (preprocess pool, off-loop tokenization, tool-schema cap) do not exist here.

### sglang-diffusion (FLUX.2) patches 50-53
These touch only `python/sglang/multimodal_gen` and apply after 05-26:

| patch | what | prod flag / env |
|---|---|---|
| 50-diffusion-media-url-ssrf-guard | CVE-2026-15974. Client image/video URLs (`/v1/images/edits` `url`, pipeline `load_image`/`load_video`) must be http(s); every resolved address must be public (no private, loopback, link-local, CGNAT, metadata, IPv4-in-IPv6 forms); the connection is pinned to the checked IP (Host header + TLS SNI kept); redirects are followed by hand (max 5), each one checked. Edits always download remote URLs under this policy instead of passing them to the pipeline | unconditional; `SGLANG_DIFFUSION_ALLOW_PRIVATE_MEDIA_URLS=1` drops the address check (lab only) |
| 51-diffusion-operator-only-request-fields | `perf_dump_path` (arbitrary file write), `profile`, `profile_all_stages`, `upscaling_model_path`, `frame_interpolation_model_path` (HF downloads) in a request → 400; `num_profiled_timesteps` is dropped. CLI and Python API unchanged | unconditional; `SGLANG_DIFFUSION_ALLOW_REQUEST_DEBUG_FIELDS=1` allows them |
| 52-diffusion-admin-api-key | sglang#47067. Bearer-key auth for the diffusion HTTP server with the srt `ADMIN_OPTIONAL` rules (`decide_request_auth`): LoRA set/merge/unmerge, `update_weights_*`, `get_weights_checksum`, `release/resume_memory_occupation`, `/rollout/generate` are admin routes. Keys come from the environment only (not argv, logs or `/server_info`) | `SGLANG_DIFFUSION_ADMIN_API_KEY` (admin routes), `SGLANG_DIFFUSION_API_KEY` (all but `/health*`, `/metrics`); unset = stock behaviour |
| 53-diffusion-cuda-host-register-switch | sglang#46799 (startup `CUDA error: operation not supported` under TDX). A failed `cudaHostRegister` in layerwise offload left the runtime's last error set, so the next kernel launch raised it from an unrelated `module.to()`; the error is now cleared. The env switch skips `cudaHostRegister` and uses torch's pinned pool (`cudaHostAlloc`) instead. **Not verified in a TDX CVM** | `SGLANG_DIFFUSION_DISABLE_CUDA_HOST_REGISTER=1` (default off) |

Deployment notes: to close the admin routes, set `SGLANG_DIFFUSION_ADMIN_API_KEY` to a random secret (nothing calls them). Without patch 53, `--pin-cpu-memory false` alone does not avoid `cudaHostRegister`: the mapped-layer courier pins its staging slots regardless, so it also needs `SGLANG_DIFFUSION_DISABLE_MAPPED_COURIER=1`.

The base is `lmsysorg/sglang@sha256:f832b5c3…`, the `v0.5.23` / `v0.5.23-cu130` tag (CUDA 13.0; amd64 `f7876bef…`, upstream build run 40978390987). That image ships `/sgl-workspace/sglang` as a git checkout of 297e54fd with an editable install. `apply-patches.py` refuses any other tree, checks every patch's sha256 against `SERIES`, and applies them in order. The image also gets the same hardening as the GLM recipes: hash-pinned nltk 3.10.3 + defusedxml 0.7.1 (the base has nltk 3.9.1), and removal of the two unused Nsight EFA sampler binaries.

## Tests
`test-cpu.sh` (CPU only) runs:
1. the credential-redaction checks (`tests/test_admin_auth_redaction.py`, with `--live` against the installed package);
2. the customer-data log-redaction checks (`tests/test_customer_data_log_redaction.py`);
3. the upstream unit tests that the patches add or edit, including `test_abort_rid_match.py`, which patch 05 adds;
4. the upstream mem_cache / managers / entrypoints / constrained / multimodal / function_call suites. A failure that is not in `tests/stock-v0523-known-failures.txt`, the stock v0.5.23 CPU baseline shared with the GLM recipe, fails the script.
5. the diffusion patch checks (`tests/test_diffusion_security.py`: SSRF address/redirect/pinning cases with stubbed DNS and HTTP, operator-only fields, admin/API key gating, the cudaHostRegister switch) plus the upstream multimodal_gen unit files that patch 50 edits or that cover the same code (no GPU or network needed). Diffusion result (CPU only, 2026-11-12): all 8 patches pass `git apply --check` in order on 297e54fd; step 5 297 passed, 1 skipped; steps 1-4 unchanged (427 passed; 556 failed / 19 errors, all known); 19 multimodal_gen unit files around the touched modules have the same failure set as the image without 50-53 (13, all in `test_layerwise_offload.py`); `test_diffusion_security.py` cannot import on the image without 50-53.

The redaction, log-redaction and rid tests fail on stock v0.5.23 and pass on this tree.

Run it with network access. About 50 upstream tests in step 4 fetch Hugging Face configs, and under `--network none` they fail on stock v0.5.23 too: the stock and patched failure sets were identical.

Result (CPU only, local build of this recipe, 2026-11-11):
- the three patches apply standalone with `git apply --check` on 297e54fd;
- steps 1-3 pass (425 passed, 188 subtests);
- step 4: 556 failed / 19 errors, all in the stock known-failures list, so no regression;
- the redaction checks fail on stock and pass here.

```
docker build --build-arg BUILD_SOURCE_REVISION=$(git rev-parse HEAD) -t sglang-v0.5.23-secure-r2 docker/sglang-v0.5.23-secure-r2
docker run --rm --runtime runc --entrypoint bash sglang-v0.5.23-secure-r2 /usr/share/nearai/sglang-v0.5.23-secure-r2/test-cpu.sh
```

Publish with `.github/workflows/publish-sglang-v0.5.23-secure-r2.yaml` from a merged commit on main. Production must use the resulting digest.
