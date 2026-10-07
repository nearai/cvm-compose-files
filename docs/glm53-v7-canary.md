# GLM-5.3 Flash v7 bundle: one base replica and one long-context replica, 3-4 hour canary

Status: **prepared, not deployed.** Merging this changes no running service. The bundle exists only when someone sets the per-replica variables below in ONE host's compose-manager env map. The engine image is published: `docker.io/nearaidev/sglang@sha256:fa730e6e62b2ae8058114ce540487ade33ab93bc42b1179ae78edc92bd563fc5` (tag `glm53-hicache-w4afp8-v7`, publish run 37693106399 from main `29a7db9`, #345).

The **v7 bundle** is one engine image: `glm53-hicache-w4afp8-v6` + the FP8 KV-cache patch + the preprocessing stall fix + the self-profiling hook (#343, enabled on the two canary replicas only) + the tool-schema depth cap. It canaries on exactly:

| File (shared by two hosts) | Canary host | Canary replica | Same-host controls | Other host |
|---|---|---|---|---|
| `prod/GLM-5.3-Flash-SGL-TP2x4-W4AFP8.yaml` | **gpu03** | `model-sg-glm53-w4afp8-tp2-r4` (GPUs 6,7, `instance` 4) | r1, r2, r3 (r3 is the island-mate on GPUs 4,5) | gpu04: never sets the variables; its r4 is the same-position cross-host control |
| `prod/GLM-5.3-Flash-SGL-TP4-W4AFP8-LongContext.yaml` | **gpu02** | `model-sg-glm53-w4afp8-tp2-r2a` (GPUs 4,5, `instance` 2a) | r2b (island-mate on GPUs 6,7), r1a, r1b (same argv apart from the rendezvous port) | gpu23: never sets the variables; its r2a is the same-slot cross-host reference |

This change **supersedes #344** (gpu04 overlap-scheduler-off A/B): it carries #344's mechanism (a per-host env override that is empty by default, a variant-suffix label, validator checks and render tests), generalised from one flag to one replica's whole set, and puts `--disable-overlap-schedule` inside the bundle instead of testing it alone. Overlap-off alone is therefore not measured by this canary; see "What this canary cannot show".

## Why these replicas

- **Base: gpu03 r4** (user decision; gpu04 stays fully on today's config). r4 is the slot compiled into the file (GPUs 6,7); r3 is its island-mate and r1, r2 the other island. Pre-check, and the reason to stop: gpu03 has recently shown extra prefill/decode (pf/dc) replicas in metrics. The canary's CVM RAM budget (a 325 GiB HiCache host tier per base replica, 4 x 325 GiB) and GPUs 6,7 must not be shared with any other service: dump `docker/ps` and the env map of gpu03 first. If any extra service holds GPUs 6,7 or enough host RAM to break the 1,300 GiB plan, do not deploy; the slot cannot move to another replica without a one-line generator change (`V7_REPLICA`), which is a new PR. All four gpu03 replicas must already run the memory-optimized candidate argv (48 running, mem 0.86) so the siblings are a matched control. gpu04's r4 is the same-position cross-host reference.
- **Long: gpu02 r2a.** Per `cvm-ansible-playbooks` `docs/openrouter-capacity-rollout.md` (compose-manager `/docker/ps`, 2026-10-06), gpu02 and gpu23 each run four TP2 replicas at 12 running / 4 queued, so either host gives the canary three same-host siblings on an identical argv (check at pre-check; a host still running a TP4 replica is not a matched control). gpu02 rather than gpu23 because it is the host whose pool was measured KV-bound (token-pool usage 0.96-1.00 in most hours after its first TP2 replicas started): FP8 KV and a higher running cap can only show a throughput change where KV or slots bind, so gpu02 is where a +25% claim is testable at all, and where a regression would show first. The cost is the same reason: the canary host is the one closest to its KV limit, which is what the abort criteria watch. gpu23 stays the unchanged cross-host reference. r2a (user recommendation) because both observed preprocessor wedges hit r2a slots, probably because of routing-pinned traffic, so the canary there exercises the preprocessing fix under the traffic that triggered the stalls. This can confound the throughput read: if the fix turns pathological requests into fast 422s or timeouts, the canary serves fewer and cheaper requests than its siblings. The read therefore reports request outcomes (status mix, preprocess timeouts) and restricts the matched-bin comparison to requests the siblings also accepted (same prompt-length bucket, status 200); if the mix differs materially, say so in the PR and re-run the canary on r2b (the generator slot is one constant, `V7_SLOT`). r2a also had the better pre-change baseline in the 2026-10-06 busy hour (TTFT p95 55 s against 64 s, 26 aborts against 38; `docs/long-context-glm53-2xtp2-rollout.md`), so a measured improvement cannot come from choosing the weaker replica, and a regression cannot be blamed on it. r2a's island-mate r2b (GPUs 6,7, same NVLink island) is the primary control; r1a and r1b (GPUs 0-3, the other island) are the pooled secondary.
- One replica per tier keeps the blast radius at 1 of 8 base replicas (12.5% of base slots) and 1 of gpu02's 4 engines.

## What changes in the files, and what does not

Both files stay byte-identical in effect for every other service and for the other host: **with the variables unset, `docker compose config` renders exactly what the previous tag rendered** (checked on compose v5.1.4, see the PR; CI runs without the compose plugin, so the in-CI render tests use a stand-in interpolator and the dry-run in the deploy steps is the production proof). Only one replica per file reads `GLM53_V7_<slot>_*` variables, each pinned by the validator to today's value, or to empty, as its default:

| Variable (`R4` base, `R2A` long) | Default (today) | Canary value | Used as |
|---|---|---|---|
| `GLM53_V7_<slot>_IMAGE` | the v6 image reference | `docker.io/nearaidev/sglang@sha256:<v7 digest>` | the replica's `image:` |
| `..._IMAGE_LABEL` | `9c6ddd4319c4` | first 12 hex of the digest | `engine_image` in the log tag, OTel label and scrape job |
| `..._PRECISION` | `int4-weights-fp8-activations-bf16-kv` | `int4-weights-fp8-activations-fp8-kv` | `precision` in the log tag and scrape job |
| `..._KV_DTYPE` | `bfloat16` | `fp8_e4m3` | `--kv-cache-dtype` |
| `..._DSA_BACKEND` | `tilelang` | `flashmla_kv` | `--dsa-prefill-backend` and `--dsa-decode-backend` (one variable: they travel together) |
| `..._MAX_RUNNING` | 48 (base) / 12 (long) | 64 (base) / 16 (long) | `--max-running-requests` and `--cuda-graph-max-bs-decode` (one variable: the validator requires them equal) |
| `GLM53_V7_R4_MAMBA_SLOTS` | 330 | 380 | `--max-mamba-cache-size` (5 slots per running request: 64 x 5 = 320 <= 380) |
| `GLM53_V7_R2A_MAX_QUEUED` | 4 | 4 (unchanged) | `--max-queued-requests` |
| `..._VARIANT_SUFFIX` | empty | `-v7` (base), `-v7-mr16q4` (long) | suffix on `config_variant` in all three telemetry places |
| `..._EXTRA_ARGS` | empty | `--disable-overlap-schedule` | last token of the command |
| `..._ENV_PREFIX` | empty | `env SGLANG_PREPROCESS_WORKERS=4 SGLANG_PREPROCESS_TIMEOUT_S=60 SGLANG_TOOL_SCHEMA_MAX_DEPTH=32 SGLANG_TOOL_SCHEMA_MAX_NODES=25000 SGLANG_PREPROCESS_LOG_SLOW_S=5 NEAR_SELF_PROFILE=1 NEAR_SELF_PROFILE_AFTER_S=900 NEAR_SELF_PROFILE_STEPS=50` | first tokens of the command |

Everything else is unchanged: mem 0.86, EAGLE fixed 4/1/5, `--prefill-decode-interval 2`, chunk 8192, HiCache (`write_through_selective` on base, `write_through` on long), the admission-reserve environment on base and none on long, the DSA indexer split, the observability environment.

How the mechanism works, and why it looks like this:

- **Flags are replaced in place, never repeated.** A flag the bundle changes is `--flag ${VAR:-today}`, so it appears once. `--disable-overlap-schedule` is not in the argv today, so it arrives through `EXTRA_ARGS`. `command` is a folded string that compose splits like a shell after interpolation: an empty variable leaves whitespace and **no empty argv element** (a list-form command would pass `""` to SGLang's argparse and fail). The env value must be plain words: no quotes, no spaces inside a value.
- **The preprocessing environment travels in the command, as `env NAME=value ... sglang serve ...`.** An environment entry cannot be made per-replica without leaving a trace in every other replica's render: an empty list entry renders as an empty-named variable, and a mapping entry renders as an empty value. The command can: the prefix variable is the first token, empty by default. `env` execs the engine, so PID 1 and the process tree are as before. Verify with `/proc/<pid>/environ` (see "Verify").
- **The image digest is interpolated** as `image: ${GLM53_V7_R4_IMAGE:-docker.io/nearaidev/sglang@sha256:9c6d...}`: the default is today's pinned digest, so every other replica and host keeps its pin, and the canary host's env map supplies a full digest-pinned reference. The digest never appears in the file, so publishing the image changes no file and no compose hash.
- **No new top-level key.** The slot's `image` and `command` are set on the replica itself, so `docker compose config` has no new `x-` entry on any host.
- **The slot is enforced, not trusted.** `scripts/validate_glm53_prod_config.rb` requires: every `GLM53_V7_*` reference is exactly `${GLM53_V7_<slot>_<NAME>:-<pinned default>}` with a pinned count; the variables appear on the slot service and its scrape job only (no other service, anchor, collector job or proxy); the bundle's flags, environment names, `fp8_e4m3`, `flashmla_kv` and the placeholders are never literals outside comments; the slot's argv is the candidate argv with `ENV_PREFIX` first and `EXTRA_ARGS` last. `scripts/test_glm53_v7_bundle.py` and the two generators' test modules render both files under the unset, canary, other-host and per-piece-rollback env maps.

## The env map to set (exactly these keys, on exactly one host)

Print it, never type it: `python3 scripts/glm53_v7_canary_env.py base` (gpu03) or `... long` (gpu02). The printer refuses (exit 2) while the digest is a placeholder. With today's constants it prints, for the long file:

```
GLM53_V7_R2A_IMAGE=docker.io/nearaidev/sglang@sha256:<v7 digest>
GLM53_V7_R2A_IMAGE_LABEL=<first 12 hex>
GLM53_V7_R2A_PRECISION=int4-weights-fp8-activations-fp8-kv
GLM53_V7_R2A_KV_DTYPE=fp8_e4m3
GLM53_V7_R2A_DSA_BACKEND=flashmla_kv
GLM53_V7_R2A_MAX_RUNNING=16
GLM53_V7_R2A_MAX_QUEUED=4
GLM53_V7_R2A_VARIANT_SUFFIX=-v7-mr16q4
GLM53_V7_R2A_EXTRA_ARGS=--disable-overlap-schedule
GLM53_V7_R2A_ENV_PREFIX=env SGLANG_PREPROCESS_WORKERS=4 SGLANG_PREPROCESS_TIMEOUT_S=60 SGLANG_TOOL_SCHEMA_MAX_DEPTH=32 SGLANG_TOOL_SCHEMA_MAX_NODES=25000 SGLANG_PREPROCESS_LOG_SLOW_S=5 NEAR_SELF_PROFILE=1 NEAR_SELF_PROFILE_AFTER_S=900 NEAR_SELF_PROFILE_STEPS=50
```

and for the base file the same ten keys with prefix `GLM53_V7_R4_`, `MAX_RUNNING=64`, `MAMBA_SLOTS=380` in place of `MAX_QUEUED`, and `VARIANT_SUFFIX=-v7`.

- **The long-context caps (16 running, 4 queued) are final per the coordinator** after the 0.09 lab result (FP8 16/6 against bf16 12/4: tok/s +10%, served +12%, TTFT p95 23.3 against 19.3 s, +21%, after +18% at 0.06); 16 running was lab-measured, 16/4 itself was not. They are the single generator constants `V7_LONG_MAX_RUNNING` and `V7_LONG_MAX_QUEUED` in `scripts/prepare_glm53_w4afp8_long_context.py`. Change them there; the printer, the suffix (`-mr16q4`) and the tests follow, and no compose file changes (the file holds only the 12/4 defaults). The mamba slots (330) must hold 5 per running request, so the printer refuses a cap above 66.
- **The preprocessing values are constants in `scripts/glm53_v7_bundle.py`**: `PREPROCESS_WORKERS` (4), `PREPROCESS_TIMEOUT_S` (60), and `TOOL_SCHEMA_MAX_DEPTH` (32), `TOOL_SCHEMA_MAX_NODES` (25000, the real guard: `check_schema` is linear in node count; both default to off in the image) and `PREPROCESS_LOG_SLOW_S` (5). Change them there.
- **gpu04's and gpu23's env maps must never contain any `GLM53_V7_` key.** Nothing on those hosts reads them, but a leftover key would activate the slot the day that host's replica set changes.
- The env map is replaced whole by compose-manager: dump it first, add the ten keys, preserve every existing key.

## Before anything is deployed

1. **Fill in the placeholders** (done) in `scripts/glm53_v7_bundle.py`: `V7_IMAGE_DIGEST` is the published v7 digest and the tool-schema caps are the image PR's (#345) values. `V7ReleaseGateTest.test_committed_bundle_values_are_filled_in` keeps any later placeholder edit red; do not weaken it. Check the digest by pulling it on the canary host first (requirement 3 of the base canary docs: image already on the host).
2. **KMS compose hash.** Both files change, so there are two new compose hashes. Register them with the KMS contract before any deploy of either file on any host (gpu04 and gpu23 are affected even though their running services do not change), and keep the previous tag's hashes registered so the rollback stays deployable. **Open question for the compose-manager owners (deploy blocker):** are env-map values, including the image reference, part of what the TEE attests or what the hash covers? If they are, the canary's image and flags must be registered with it; if they are not, the canary image is not attested by the compose hash and the statement that the TEE environment is otherwise unchanged is weaker. Record the answer in the PR.
3. **Env-map dump** of the canary host and of the other host of the same file, saved as the rollback reference. Assert with `python3 scripts/glm53_v7_check_env_map.py --host <host> --expect none <env-map.json>` (exit 0): no `GLM53_V7_` key on gpu04, gpu23 or the canary host before step "set the env map". Also assert `GLM53_BACKEND_URLS` is as documented for gpu02's stage (`docs/long-context-glm53-2xtp2-rollout.md`).
4. **The v7 image must be split-capable** (`SGLANG_DSA_INDEXER_QSPLIT=1` stays set on the slot) and contain `/usr/bin/env`; the validator checks the default v6 image only.
4a. **Image facts from the image PR's author, written into the PR thread**: the exact startup log line for the preprocess pool (this runbook expects `preprocess pool started` with the worker count), the `server_args` field names, that the hook is the #343 module (inert unless `NEAR_SELF_PROFILE` is exactly `1`, TP rank 0 only), and that the image contains `/usr/bin/env`.
5. **Quality gate, outside this repo.** FP8 KV needs `--kv-cache-dtype fp8_e4m3` AND both `--dsa-prefill-backend flashmla_kv` and `--dsa-decode-backend flashmla_kv`; the patch does not assert the pairing, so the file does (one variable feeds both flags; test `test_fp8_kv_needs_both_dsa_backends_and_the_dtype_together`). FP8 KV changes model outputs; a throughput bake cannot show quality. The owner records the lab parity result for FP8 KV with `flashmla_kv` (GSM8K, perception, and a tool-call suite on the bundle) in the PR thread before GO. The bake below reads quality proxies only.
6. **Both hosts' replicas on the expected baseline.** Base: all four gpu03 replicas and all four gpu04 replicas run the memory-optimized candidate (startup log `mem_fraction_static=0.86`, `max_running_requests=48`). Long: gpu02 r1a, r1b, r2a and r2b all run `max_running_requests=12` / `max_queued_requests=4` on TP2 (config_variant ends `-mr12q4-strict-budget8192-obs-v1`); if any gpu02 replica is still TP4 or at 24/8 it is not a matched sibling, so hold or exclude it from the controls.
7. **A named person** checks the dashboards every 30 minutes during the bake and owns the rollback decision.

## Rollout constraints (user rules)

- **Long context: one replica at a time, and wait until it is back serving and baked.** Only r2a is touched. r2b, r1a, r1b and gpu23 are not recreated, rolled or re-tagged until r2a is back serving (`/backends/list` healthy, a real long-domain completion plus a cache-hit follow-up) **and** its bake has been read. A rollback recreate of r2a is again "one replica at a time".
- **Base: never more than 2 base replicas down fleet-wide.** The canary takes 1 down for about 22 minutes (cold start under CC). Before each canary or rollback recreate, check `docker/ps` and `/backends/list` on both gpu04 and gpu03: start only when every other base replica is up and ready, so that the canary plus at most one unplanned failure stay within 2. If a second replica goes down while the canary is starting, stop and do not start a rollback recreate until one is back, unless the canary itself is the cause (then roll back the canary: it is already one of the two).
- **`compose/down` takes effect immediately and cannot be recalled; it ignores `dry_run`.** This runbook uses `compose/up` of the one replica (compose recreates a replica whose config changed), so no `compose/down` is needed. If a `compose/down` is ever sent, send it with a one-replica `services` list, never dry-run it, and **verify in the host's compose-manager action log** (the attested log the other runbooks read) that the logged `services` list is exactly that one replica and no other service was stopped. Cross-check with `docker/ps`.
- **Every `compose/up` while the variables are set must carry a `services` list.** An unscoped call would recreate every service whose rendering changed, and a collector recreate before the replica flips mislabels it. Allowed lists: `["<the one replica>"]` and `["otelcol-contrib"]`, nothing else.
- `dry_run: true` first on every `compose/up`; use the host's complete env map on every call.

## Deploy: base (gpu03, replica r4)

1. **Dump gpu03's env map**; assert no `GLM53_V7_` key; assert gpu04's has none. Capture `docker/ps`, `/backends/list` (four gpu03 handles) and the container IDs of r1-r3.
2. **Deploy the merged tag with the variables still unset**, `dry_run: true` then apply, with `services` listing each replica you intend to touch: the plan must recreate **nothing** (the rendering is identical to the previous tag). If the compose-manager flow requires a tag deploy to register the new tag, this is it. Confirm the tag is live.
3. **Set the ten keys** from `glm53_v7_canary_env.py base` in gpu03's env map. Re-read the map: ten new keys, every old key unchanged; gpu04 untouched.
4. `compose/up`, `services: ["model-sg-glm53-w4afp8-tp2-r4"]`, `dry_run: true`. The plan must show exactly one recreate (r4) and nothing else. Check the rendered plan's image is the v7 digest, not the v6 one. Apply when no other base replica is down (rule above). Compose stops r4 (5 minute grace); r1-r3 and gpu04 serve meanwhile; r4's conversation-affinity pins re-prefill elsewhere.
5. **Wait for ready (about 22 minutes)**, then run "Verify". Hold at the first failure.
6. **Collector last:** `compose/up`, `services: ["otelcol-contrib"]`, `force_recreate: true`, `dry_run` first (plan: collector only). It is last on purpose: a recreate earlier would label a still-bf16 engine `-v7`. Verify `config_variant` ends `-obs-v1-v7` for `instance` 4 and is unchanged for 1-3, and that `precision` / `engine_image` follow. Datadog log tags are set when the container is created, so they are right as soon as r4 is up.
7. Confirm r1-r3, proxy, nginx, registrar and the soak relay were never recreated (container IDs and uptime unchanged).

## Deploy: long context (gpu02, replica r2a)

1. Preconditions above, plus: gpu02 r1a, r1b, r2b and the proxy are healthy; gpu23 and gpu13 are healthy (the lane's long tier is not already one host short). Pick a low-traffic window: while r2a restarts the host runs three replicas, 12 running slots fewer.
2. Dump gpu02's env map; assert no `GLM53_V7_` key and that `GLM53_BACKEND_URLS` is gpu02's current stage value (`all-tp2` once all four TP2 replicas run; `docs/long-context-glm53-2xtp2-rollout.md`). Capture `docker/ps`, `/backends/list` for the long domain, and the r1a/r1b/r2b container IDs.
3. Deploy the merged tag with the variables unset, `dry_run` first: the plan recreates nothing. Apply.
4. **Set the ten keys** from `glm53_v7_canary_env.py long` in gpu02's env map only. Re-read the map.
5. `compose/up`, `services: ["model-sg-glm53-w4afp8-tp2-r2a"]`, `dry_run: true`: one recreate, nothing else, image = the v7 digest. Apply. Wait until r2a is ready and back in the proxy pool.
6. Run "Verify". Then the collector: `compose/up`, `services: ["otelcol-contrib"]`, `force_recreate: true`, `dry_run` first. Verify `config_variant` ends `-obs-v1-v7-mr16q4` for `instance` 2a and is unchanged for 1, 2, 2b.
7. **Do not touch r2b, r1a, r1b or gpu23 until r2a is baked** (rule above).

## Verify (startup log and runtime, before the bake clock starts)

On the canary replica's startup log (`docker logs`) and `/get_server_info` or the `server_args` line:

| Check | Base r4 | Long r2a |
|---|---|---|
| `kv_cache_dtype=fp8_e4m3` | required | required |
| `dsa_prefill_backend='flashmla_kv'`, `dsa_decode_backend='flashmla_kv'` | required | required |
| `disable_overlap_schedule=True` | required (and `False` on r1-r3) | required (and `False` on r2b, r1) |
| `max_running_requests` | 64 | 16 |
| `max_queued_requests` | 8 (unchanged) | 6 |
| cuda graph batch sizes reach | 64 | 16 |
| `max_mamba_cache_size` | 380 | 330 (unchanged) |
| preprocess pool | `preprocess pool started` with 4 workers (line text per the image PR) | same |
| `/proc/<engine pid>/environ` | `SGLANG_PREPROCESS_WORKERS=4`, `SGLANG_PREPROCESS_TIMEOUT_S=60`, `SGLANG_TOOL_SCHEMA_MAX_DEPTH=32` | same |
| image and command | `docker inspect` shows the v7 digest and `.Path` = `env`, `.Args` = `SGLANG_PREPROCESS_WORKERS=4 ... sglang serve ...`; the container did not exit 127 | same |
| profiling hook | `NEAR_SELF_PROFILE=1` only on the canary; startup does not profile, the one-shot profile runs about 15 minutes later (below) | same |

Also, as in the other canaries: KV pool size and free GPU memory at ready (hold if free memory is under 10 GiB, or the KV pool is under what the log shows for r1-r3 / r2b by more than the FP8 gain explains), no CUDA 801, NCCL, Xid or OOM, `/backends/list` healthy, one real completion and a cache-hit follow-up, and **a tool-call request with a deep schema**: the depth cap must reject or flatten it as the image PR specifies, not crash the pool. If `disable_overlap_schedule` reads `False` on the canary, the env did not reach the container: stop and re-read the env map.

## Self-profiling hook (canary replicas only)

The hook from #343 (`near-self-profile.diff`) runs once per engine start on TP rank 0: after `NEAR_SELF_PROFILE_AFTER_S=900` seconds (about 15 minutes of warm traffic) it profiles `NEAR_SELF_PROFILE_STEPS=50` scheduler steps. It is **inert unless `NEAR_SELF_PROFILE` is exactly `1`**; `0`, empty or unset do nothing. The canary replica gets `1` through its `ENV_PREFIX`; the name is never a literal in either file, so no other replica or host can have it (the validator rejects the name anywhere outside comments, and the render tests assert it appears only on the slot's argv).

- **Expected one-time stall:** a few seconds, up to about 20 s, once per engine start, about 15 minutes after start, while the trace is written and summarised by a niced subprocess. Do not read it as a stall abort (criterion 4) if it matches this: once, TP rank 0, on the canary, 15 minutes after ready.
- **Read the result in Loki:** `{host="gpu03", container_name="model-sg-glm53-w4afp8-tp2-r4"} |= "NEAR_PROFILE"` (long: `{host="gpu02", container_name="model-sg-glm53-w4afp8-tp2-r2a"} |= "NEAR_PROFILE"`). The summary says `cuda+cpu` or `cpu-only` and lists host syncs, D2H copies, launches and GPU idle gaps.
- **CPU-only fallback:** if CUPTI is unavailable under CC the hook retries once CPU-only and logs `retrying cpu-only`; to force it, add `NEAR_SELF_PROFILE_ACTIVITIES=cpu` to the `ENV_PREFIX`. CPU-only still shows time blocked in `item()`, `synchronize()` and copies, but not GPU kernels.
- **Any error** prints one `NEAR_PROFILE error ...` line and disables the hook; that is not an abort.
- **The bake read must exclude the profiling minute** (from 15 minutes after ready, plus one minute either side), symmetrically for all replicas; it already falls inside the warm-up exclusion. Profiling overhead is a one-shot cost, not part of the steady-state comparison.
- It is the first deployment of the hook in prod: confirm in the PR thread that #343's CPU tests passed on the image digest and that the hook is in the bundle.

## The bake (3-4 hours, and it must include a busy period)

**What this bake is.** A safety and smoke gate, not an efficacy trial. In 2-3 read hours it can falsify (crash, Xid, OOM, stall, a gross latency or error regression, a tool-call collapse) and show headroom at the new caps; it cannot establish the +25% claim, attribute anything to one of the five changes, or say anything about capacity. Numbers below come from one replica on one host in one window. The efficacy read, if wanted, is the **same canary left running** for 12-24 h under the rule below; the 3-4 hour verdict is "no gross failure" or "abort", nothing more.

**Readability gate, computed before the bake.** Compute the whole statistic below as a sibling-vs-sibling placebo (the would-be canary against its siblings, no treatment) on the previous 10 days at matching hours, and record its day-level SD in the PR thread. If that SD exceeds 8%, the bake cannot support a +25% claim and is a safety gate only.

**Clock.** It starts at the later of "canary ready" and "collector recreated", plus **60 minutes of warm-up and until the canary's HiCache and prefix hit rate are within 10 points of the siblings' for 30 consecutive minutes** (cold HiCache, no conversation-affinity pins, cold FP8 kernels); the siblings are excluded for the same period, so the exclusion is symmetric. Data before that is excluded. The remaining 2-3 hours are the read window. Schedule the start so that the read window contains **at least one hour of the three busiest hours** of the same weekday over the previous 7 days (fix the hour slots from Grafana before the start and write them in the PR thread). A bake with no busy hour is inconclusive, not a pass.

**What is compared.** The canary against its same-host siblings (base: r1, r2, r3, with r3 the island-mate; long: r2b the island-mate, then r1a and r1b), and the canary's own pre-change ratio:

- **Primary: tok/s per replica at matched running bins.** Cells are `num_running_reqs` bins (1-8, 9-16, 17-32, 33-48) crossed with total context tokens in the batch (`kv_used / max_total`, tertiles of the siblings' distribution), weighted by the siblings' cell occupancy; a cell counts only when both arms have at least 3 of the 5 one-minute windows in it. Output-token rate over 1-minute windows against the time-weighted mean running requests in the same window, never a ratio of sampled gauges. The target is **+25%** against the siblings' pooled value. At a matched cell this is a per-stream speed ratio (about N x decode speed, including the speculative-decoding gain), not capacity. The bins the new caps open (base 49-64, long 13-16) have no sibling control: they are descriptive only, so the matched comparison excludes the regime where the cap and the FP8 KV pool are expected to matter. A capacity claim needs a controlled replay or concurrency sweep with identical request streams on canary and sibling, which this bake is not.
- **Guards: TTFT p95 and ITL p95 no more than +20% worse** than the siblings at matched cells (a guard is breached when the one-sided 90% upper bound of the log ratio is above +20%; long-tier TTFT p95 cannot be bounded at about 1,500 requests, so on long it is reported, not gated), p95 from summed histogram buckets over the window (never averaged across hours or replicas; if the bucket edges around the observed p95 are wider than 5% relative, use per-request logs). TTFT is read by prompt-length bucket and flagged **cache-confounded** until the canary's prefix/HiCache hit rate is within 10 points of the siblings.
- **Safety:** errors (5xx other than queue-full 503, aborts), stalls, restarts, Xid on the canary's GPUs (base 6,7; long 4,5), OOM or CUDA errors, free GPU memory, the preprocess pool's timeouts or restarts.
- **Context, descriptive:** request share, mean prompt and output tokens, cache hit rate by tier, EAGLE accept length and scheduler steps/s (decompose tok/s into tokens per step x steps/s; an accept-length drop above 5% is reported next to the headline as a possible quality signal), tail TTFT (p99), KV retractions or preemptions, HiCache errors, queue time and depth, 503 rate, `kv_used / max_total`, the profiler switch.
- **Quality proxies, descriptive:** finish-reason mix (`length` rate), tool-call parse failures, empty or repeated outputs, per replica. These can raise an abort for a gross failure; they do not show parity.

**Confounds, handled up front.**

- *Newborn conversations.* Affinity keeps existing conversations on the siblings, so after a restart the canary receives new conversations only: shorter, younger contexts for the whole bake, which speeds decode at the same running count. Matching on context tokens (above) addresses part of it; also report conversation age and the prompt-token distribution per arm, and if the canary's mean prompt differs from the siblings' by more than 10%, tok/s and ITL (not only TTFT) are mix-confounded. The restart also dumps the canary's pinned load onto the siblings, so they are not an untreated control for the first hour.
- *Restart.* A freshly started process (cold caches, graph memory) is compared with siblings that have been up for days, and the pre-change ratio contains no restart effect. After the read, recreate the same slot with an unchanged env map (a restart placebo, if the down-replica rule allows) and run the same statistic.
- *Routing.* The proxy is least-connections with conversation affinity, so a faster replica, or one with a larger cap (64 against 48; 16 against 12), attracts more concurrent requests. Running count is a consequence of the treatment, so conditioning on it can hide a throughput gain: report both the matched-bin comparison and the unconditional per-replica totals with request share, mean prompt and output tokens. Do not state a capacity gain from this bake. If the canary's request share differs from its pre-change share by more than 15% relative, flag the latency comparison as mix-confounded and restrict it to prompt-length-matched buckets.
- *Position.* The canary sits on fixed GPUs. The pre-change period is the baseline: compute the canary-to-sibling ratio over the same hours-of-day on the previous 10 days (frozen queries, same cells) and report the bake ratio **against that ratio** (a difference of ratios), not against 1. On base, gpu04's r4 (same GPUs, same file, other host) gives the same ratio against its own siblings: it is a co-primary difference-in-differences, and a claim needs both to agree in sign (long: gpu23's r2a). Its traffic differs, which is why it is a difference and not a pooled comparison.
- *Warm-up.* See the clock above.
- *Same time, same traffic, same host:* siblings run concurrently; no cross-day comparisons except the pre-change ratio.

**Pre-registered read (fixed before the bake; changes need a recorded reason before the data is read).**

- Statistic: log ratio (canary / pooled siblings) minus the pre-change log ratio, from a stratified regression on cell, with accept length as a covariate. Uncertainty by a block bootstrap over 30-minute blocks of the bake and over pre-change days jointly (5-minute blocks are autocorrelated at the 15-60 minute scale). Report the cell-by-cell table and the pooled value with the 90% interval; cells with fewer than 12 blocks are descriptive. Per-cell effects of mixed sign are plausible (overlap-off costs at small batch, FP8 attention helps at long context, the cap acts near saturation), so no same-sign-in-every-cell rule is used.
- **Efficacy is read once**, at a stop fixed before the bake (the later of 12 h and 100 blocks in each main cell). Extension is allowed only up to that stop and by that rule; otherwise the result is "inconclusive". The 30-minute abort checks are safety looks; their results are never reused to stop or extend the efficacy read.
- **+25% claim.** "Consistent with the target" if the interval contains +25% and its lower bound is above +10%; "meets the target" if the point estimate is at least +20% and the lower bound is above +10%; "positive but unproven" if the lower bound is above 0 only; anything else is "no evidence for the target". A rule that demands a point estimate of +25% passes only half the time at a true +25%, so the point clause is +20%. The lab's long-tier result (+11% tok/s) is below the target: expect the lower verdicts there.
- A single canary replica against 3 siblings is one replica-group of evidence on one host in one window. It can falsify and can motivate a larger test; it cannot establish a fleet-wide effect, a capacity gain, or output quality.
- A judgment abort rolls back the whole bundle (full revert). Piece rollbacks are for safety only and restart the warm-up, so no efficacy read is possible after one.

## Abort criteria (pre-registered; any one means roll back)

These are safety looks every 30 minutes by the named person; they are not efficacy reads. Thresholds are one-sided and use trailing windows so that one noisy hour on one replica does not roll back the experiment: a false abort costs a 22 minute cold start and the experiment.

Safety (immediate):

1. Any exit or restart of the canary container, any Xid on its GPUs, any OOM, CUDA error, NCCL error, or a watchdog / scheduler-stall message in its log.
2. Free GPU memory (DCGM `FB_FREE` joined on the canary's GPU indexes) under 2 GiB for 5 minutes.
3. The preprocess pool logs a timeout storm or a worker death more than 3 times in 10 minutes, or requests carrying images or tool schemas fail where the siblings' do not.
4. A stall: the canary has at least one decode-phase request but generates no tokens for 30 seconds, while the siblings do (a prefill-only interval, e.g. a 130K prompt at chunk 8192, is excluded). The one-time profiling stall above is not this.

Judgment (after the 60 minute warm-up, matched bins, summed histograms):

5. ITL p95 more than **20% worse** than the siblings (difference of ratios) when its one-sided 90% lower bound exceeds +20% over a trailing 90 minutes with at least 12 blocks and 300 requests, or its point estimate exceeds +50% over 30 minutes. Base TTFT p95 likewise (prompt buckets whose cache hit rate is within 10 points of the siblings). Long TTFT p95 is reported against the same lines but is not a judgment abort at this sample size, except a point estimate above +50% over 90 minutes.
      **Long-tier TTFT risk, known in advance.** Lab TTFT p95 was +18% (0.06) and +21% (0.09) at 16/6, already at or over the 20% line; the queue cap was cut to 4 for that reason, and 16/4 is not itself lab-measured. TTFT p95 is reported, not a long-tier judgment abort (see criterion 5); if the tail is still bad at 16/4, roll back the cap pieces.
6. 5xx other than queue-full 503, or aborts, when the one-sided 95% exact lower bound on the rate difference to the siblings exceeds 1 percentage point over 90 minutes (long: 2 points).
7. Queue-full 503 rate above the siblings' by more than 3 percentage points (95% lower bound above zero) over 30 minutes while the canary's running count is below its cap.
8. A quality-proxy gross failure: tool-call parse failures or empty outputs above the siblings' rate by more than 2 percentage points over 60 minutes.
9. Canary-only TTFT p95 above 2x its pre-change 7-day same-hour median for 60 minutes (a lane-level criterion is too insensitive for one replica of four). The long lane's first-token p95 is watched by the gateway runbook's own gates; it is a lane-level figure, not a replica's, so it is not repeated here.

Missing-signal checks, reported with every look: canary queue starvation at the larger caps, KV retraction or preemption rate, HiCache errors, p99 TTFT, and unconditional ITL (matched-cell ITL hides the user-visible effect of running a higher batch).

Not an abort by itself: the canary's request share rising (a larger cap and a faster replica attract load), EAGLE accept length moving, a transient TTFT spike inside the warm-up.

## Rollback

Per piece, by editing the canary host's env map and recreating the **one** replica (each recreate is a 22 minute cold start and counts against the rules above; `python3 scripts/glm53_v7_canary_env.py base|long --rollback fp8|overlap|preprocess` prints the edits):

| Piece | Env-map edit | Notes |
|---|---|---|
| FP8 off | delete `..._KV_DTYPE`, `..._DSA_BACKEND`, `..._PRECISION` and the caps (`..._MAX_RUNNING`, and `..._MAMBA_SLOTS` / `..._MAX_QUEUED`) | bf16 KV and the TileLang DSA backends together (`flashmla_kv` needs the fp8 cache); the caps return with them because they were sized for FP8's larger KV pool. The image and other pieces stay. Delete `..._VARIANT_SUFFIX` too if the suffix should stop claiming the bundle. |
| Overlap back on | delete `..._EXTRA_ARGS` | the flag disappears from the argv |
| Profiling off | set `..._ENV_PREFIX` to the same value with `NEAR_SELF_PROFILE=0` (or drop the three `NEAR_SELF_PROFILE*` words) | the hook is one-shot per start, so this only matters at the next recreate |
| Preprocess workers 0 | set `..._ENV_PREFIX` to the same value with `SGLANG_PREPROCESS_WORKERS=0` | `SGLANG_PREPROCESS_WORKERS` defaults to 0 = pool off in the image (confirmed by the #345 author). The tool-schema caps default to 0 = off too: set both `SGLANG_TOOL_SCHEMA_MAX_DEPTH=0 SGLANG_TOOL_SCHEMA_MAX_NODES=0` in the `ENV_PREFIX` to disable them |
| Caps back | set `..._MAX_RUNNING` (and `..._MAX_QUEUED`, `..._MAMBA_SLOTS`) to today's values, or delete them | |
| **Full revert** | delete all ten `GLM53_V7_*` keys | the replica renders exactly what the previous tag rendered (v6, bf16, overlap on, 48 or 12/4). No tag rollback needed. |

After any env-map edit, check the map with `scripts/glm53_v7_check_env_map.py` (`--expect none` after a full revert, `--expect canary` otherwise, with the edited keys compared against the printer's output by eye), then: `compose/up` `["<the one replica>"]` (`dry_run` first: one recreate), wait for ready and a real completion, then `compose/up` `["otelcol-contrib"]` with `force_recreate: true` so the labels follow, then delete the keys from the env map you no longer want to carry. For a full revert also verify `docker inspect` shows the v6 image and the startup log shows `kv_cache_dtype=bfloat16`, `disable_overlap_schedule=False`. If the file itself is faulty, redeploy the previous tag's file with the same scoped lists. The other host, every other replica, the proxy, nginx and the registrar are never touched.

## What this canary cannot show

- **Overlap-off alone.** #344 would have isolated it; this bundle changes five things on one replica, so a result is attributed to the bundle, not to any piece. Piece rollbacks exist for safety; they are not an attribution experiment.
- **Quality of FP8 KV**, a capacity gain at SLO, or anything about prompts longer than the window sees (the long tier's prompts are about 130K at the median, but the 400K+ tail may not appear in 3 hours).
- **Restart or cold-cache effects separate from the bundle** (no placebo unless the restart placebo is run), **any per-stream speedup that is not a capacity gain**, and effects in the cells siblings cannot reach, which is where the cap and FP8 KV changes act.
- **Whether `--disable-overlap-schedule` does anything** under EAGLE in this build: the image PR author is asked to confirm it is not already a no-op; if it is, the bundle is four changes, not five.
- **Anything beyond one replica on one host:** gateway caps, placement behaviour across hosts (cloud-api scores `fullness = (running + queued + pending) / max_running`, so a 16-slot canary reads emptier than its 12-slot sibling and attracts more of the long tier's traffic), and interactions when several replicas run the bundle.

## Gateway caps (companion PR in `nearai/cvm-ansible-playbooks`)

The gateway's admission bounds are **per tier, not per host or per backend**: one `VLLM_PROXY_ADMISSION_LONG_MAX_INFLIGHT_PER_HOST` for every long host and a base-host bound computed from the budget and reserve, so a bound cannot be scoped to the canary host. The base bound (160 per host) is already below gpu03's 192 running slots (208 with the canary at 64), so the base canary's extra slots need no gateway change. The long ceiling (48) equals gpu02's four replicas at 12 running; the canary at 16 adds 4 running slots, so the companion PR raises the long ceiling from 48 to 52, globally, as its own reviewed stage. It deploys only after the canary is healthy at the final caps, and reverts when the canary ends. See that PR for the arithmetic and gates.
