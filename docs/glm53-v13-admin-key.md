# GLM-5.3 engine v13: admin key on the SGLang engines

Status: **image recipe prepared, not published, not deployed.** Compose files stay pinned to v12
(`sha256:da8d4e4b5ec6…`). A follow-up PR pins the v13 digest and adds the launch change below.

## Why

In the 11-06..11-08 incident, the attacker reached an engine through the vllm-proxy OHTTP SSRF and called
SGLang's unauthenticated `/update_weights_from_tensor` (pickle → code execution). Our engines run with no
`--api-key` and no `--admin-api-key`, so every admin endpoint is open to anything that can reach `:8000`.

SGLang (fc91d24, our base) supports `--admin-api-key`: admin endpoints (weight updates, `/flush_cache`,
`/abort_request`, profiling, …) then need the key, while chat, `/v1/loads`, `/v1/models`, `/health` and
`/metrics` stay open. On v12 that flag is **not safe on its own**:

| v12 gap | effect with `--admin-api-key` set | upstream fix |
| --- | --- | --- |
| `/server_info` and `/get_server_info` (open) return every server arg | the admin key is returned in clear to anyone who can reach the engine | #48518 (2026-11-07) |
| the engine logs `server_args={…}` at INFO | the key lands in Loki | #48518 |
| crash/request dumps pickle `ServerArgs` and `sys.argv` | the key lands on disk | #48518 |
| `/load_lora_adapter_from_tensors`, `/set_trace_level` have no auth level | stay open even with the key | #41322 (2026-09-30) |

## What changes (this PR)

`docker/sglang-glm53-hicache-w4afp8` gains `admin-auth-redaction.diff` (patch 16, applied last), the v12
recipe otherwise unchanged:

- the two endpoints get `@auth_level(AuthLevel.ADMIN_OPTIONAL)` (byte-identical to #41322);
- `server_args.py`: `redact_server_args_secrets`, `redact_secret_cli_args` and a redacting `ServerArgs.__repr__`;
- every whole-config readback goes through the redaction: `/server_info` (record + `internal_states`),
  `Engine.get_server_info`, both `server_args=` log lines, gRPC `get_server_info`, the request and crash
  dumps (a redacted dict instead of the `ServerArgs` object; `launch_command` redacted).

Test: `test_admin_auth_redaction.py`, test-cpu.sh step 18. Static checks run the changed code on stubs and
fail 12 ways on the v12 sources; `--live` (in the image) checks the real FastAPI app and a real
`/server_info` call. Publish: `publish-glm53-admission-reserve.yaml`, variant `hicache-w4afp8`, default tag
now `glm53-hicache-w4afp8-v13`, with the new recipe hashes and a label check for
`nearai.sglang.admin_auth_redaction`.

Nothing in our stack calls a gated endpoint: inference-proxy uses chat, `/v1/models`, `/v1/loads` and
`/health`. The perception check and the soak relay use `/health`, `/metrics`, `/v1/models` and the chat
and completion routes. Without `--admin-api-key`, v13 behaves like v12 for every caller.

## Launch change (follow-up PR, not applied here)

We never call the admin endpoints, so the key is a **random value per container start**. It is never
stored, put in an env var or logged. The image's entrypoint is `/opt/nvidia/nvidia_entrypoint.sh`, which
runs `exec "$@"`, and the engine's `command: >` argv stays byte-identical. A shell between the two appends
the flag:

```yaml
x-sg-glm53-flash-common: &sg-glm53-flash-common
  image: docker.io/nearaidev/sglang@sha256:<v13 digest>
  # sh -c: $0 = "sglang", $@ = the rest of `command`; $$ escapes compose interpolation.
  entrypoint:
    - /opt/nvidia/nvidia_entrypoint.sh
    - /bin/sh
    - -c
    - 'exec "$$0" "$$@" --admin-api-key "$$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")"'
```

Checked locally on 2026-11-09 with docker compose, using `python:3.12-slim` as a stand-in and `/usr/bin/env`
in place of the nvidia exec-entrypoint. A `command: >` argv that includes the
`--limit-mm-data-per-request '{"image": 64}'` JSON argument reaches the process unchanged, followed by
`--admin-api-key <43 chars>`.

Why this form:
- A string `command` is split into argv by compose, without a shell. So `$(…)` in `command` itself would
  never run, and compose would also try to interpolate it.
- Keeping `command` untouched means the generators' argv tests and `validate_glm53_prod_config.rb`
  (`argv_of`) still compare the same argv. The follow-up PR adds an assertion on `entrypoint` instead.

Exposure that is left:
- The key sits in the engine process's argv inside the CVM. Root in the CVM can read `/proc/<pid>/cmdline`.
- `docker inspect` shows only the unexpanded `$$(…)`.
- With v13's redaction, no HTTP readback, log or dump returns it.

Where it applies (10 GLM replicas, all on the v12 image today):

| host | file | replicas |
| --- | --- | --- |
| gpu03 | `prod/GLM-5.3-Flash-SGL-TP2x4-W4AFP8-V12Q8.yaml` | r1-r4 (base) |
| gpu13 | `prod/small-models.yaml` (GPUs 4-7) | 2 base |
| gpu23 | `prod/GLM-5.3-Flash-SGL-TP4-W4AFP8-LongContext.yaml` | r1a, r1b, r2a, r2b (long) |

Every generator that emits these anchors gets the same change: `prepare_glm53_base_v12_q8_fleet.py`,
`prepare_glm53_w4afp8_long_context.py` and `prepare_gpu13_glm53_base.py`. Each gets a test that every
GLM engine has the wrapper and the v13 digest.

## Rollout

1. Merge this PR, then run the publish workflow (`variant: hicache-w4afp8`, `source_revision` = the merge
   commit). Gates: Trivy, test-cpu.sh including step 18 `--live`, attestation.
2. Lab: run one replica on gpu31 with the wrapper.
   - `/server_info` must show `admin_api_key: "<redacted>"`.
   - `POST /flush_cache` and `POST /set_trace_level?level=0` without the key must return 401.
   - `/v1/loads`, `/health` and chat must answer 200.
   - The `server_args=` log line must be redacted.
3. Open the digest + launch-change PR and merge it.
4. Prod: **one replica at a time, fleet-wide.**
   - Order: gpu03 r1 → r4, then gpu13 ×2, then gpu23 ×4.
   - Each replica is a scoped `compose/up` of that one engine service from the merged tag.
   - Wait for the replica to serve again (`/health` 200, running requests climbing) before touching the next.
   - On each replica, verify the step-2 checks over the in-CVM path.
5. Watch the gateway 429s and backend errors through each step.

## Rollback

- Pin the v12 digest `sha256:da8d4e4b5ec667499b58a21e56050aae9bcfc4df9669b3ac27e2e20af033d681` **and**
  remove the entrypoint wrapper in the same change. v12 with `--admin-api-key` would hand the key out via
  `/server_info`.
- Redeploy one replica at a time.

## SGLang upgrade plan

- **Now: v13.** The smallest change that makes the admin key safe on the fork we already run
  (fc91d24, 0.5.16.dev20260904 + 16 NEAR patches). No serving-path change, so no performance requalification.
- **Next: rebase onto current upstream (≥ e93aacbd0, which has both fixes natively).**
  - Stock SGLang cannot load `graphistry/GLM-5.3-Flash-W4AFP8` without our loader patch (v0.5.20:
    `output_partition_size … not divisible by block_n`), so the upgrade means carrying the NEAR patch
    stack forward: W4AFP8 loader, pool clamp, admission reserve, HiCache/HCC, the DSA indexer split and
    budget, Q8 prefill, FP8 NoPE KV, KDA zero-sync, the event-loop patches and the observability hooks.
  - Many may already be upstream-equivalent. The 09-24 v0.5.20 port found 31 of 46 fork commits moot.
  - Lab on gpu31 with a branch per patch group. Qualify with GSM8K/MMLU/passkey, the perception check and
    the tee-bench prod-mix load.
  - Then an A/B against v13 on the same argv, and a TEE canary on one replica before the fleet.
  - Watch the upstream default changes, such as `SGLANG_EXPERIMENTAL_DSA_KPOOL_METADATA_FUSION`.
- **Ongoing:** track upstream security fixes in `entrypoints/` and `utils/auth.py`, and backport them into
  the recipe as numbered patches until the rebase lands.
