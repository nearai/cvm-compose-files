# gpu13 Qwen handover without restarting shared ingress

This corrects #234's ingress transition, as part of [gpu02 evacuation](gpu02-upgrade.md).
It changes no model IDs and does not withdraw gpu02 or authorize a CVM shutdown.
Keep all source capacity until the complete destination and evacuation gates pass.

## Files and fixed boundaries

- `prod/gpu13-qwen-handover.yaml`: isolated operational project
  **`gpu13-qwen-handover`**, passed explicitly in every compose-manager API request.
  Never omit `project`: the manager's default project is `work`, and its
  `--remove-orphans` would otherwise endanger the serving stack.
- `prod/small-models.yaml`: final work-project Qwen3.6 GPU1 / Qwen3.8 GPU2 topology.
  Do not use it while the old Qwen3.6 r2 still exists. Keep its actual old file and
  immutable revision available for scoped removal and rollback.
- The shared nginx service/config and multi-model registrar service/script remain
  identical to commit `33b77d54c5bc8ef3ac137134e5ff7cab866e7579`. Do not select them
  in a compose operation. Their runtime container identities must remain unchanged.
- Qwen3.8 owns a **separate** ingress: HTTP probe 8000, TLS routing 8010. Its
  `qwen38-model-proxy-registrar` is profile-gated and only owns probe `${HOST_IP}:8000`.
  Qwen3.6 remains on shared TLS8444, preserving its address-derived backend handle.
- The temporary proxy keeps the canonical proxy's pinned image, dstack/cert mounts,
  privileged/NVIDIA attestation surface and single backend (r1 by default). It has no engine GPU
  reservation. No helper/downloader uses the NVIDIA runtime or reserves a GPU.
- The generated overlay embeds `scripts/qwen_handover.py`; the manager fetches only
  one Compose file. Regenerate with `python3 scripts/render_qwen_handover.py`.

Helpers use only the nginx container's PID namespace and external `dstack_default`
network, never host PID or the Docker socket. They need SYS_PTRACE, SYS_CHROOT,
DAC_OVERRIDE and KILL, plus AppArmor unconfined, to reach `/proc/<master>/root` and
signal that one verified master. Their own root is read-only. Only their small
state volume and nginx's candidate/config files are writable through this surface.
Preflight never signals nginx or replaces its live config. It inspects the target
mount table and filesystem flags, and atomically writes/removes a hidden sibling
outside the nginx include glob to prove directory writability. Missing capabilities,
read-only/bind-mounted configs or ambiguous master identity are stop conditions.
The regression deploys the actual inline `configs.content` mechanism with Compose;
this is not a substitute for checking the already-running CVM's mount layout.

## Preconditions

1. Obtain review of the current head and green CI; use an accepted immutable merged
   reference. Do not alter manager tag-age/signature settings to bypass a gate.
2. Capture fresh inventory, service IDs, GPU UUIDs/claims, cached weights/disk
   headroom, both peer registries and source/destination queues/errors. No concurrent
   compose operation. Preserve credentials in the control plane, not report files.
3. Verify guest ports8000/8010 are free, forwarded and reachable. Confirm GPU1/2
   still belong to the expected Qwen3.6 r1/r2 and the existing source paths work.
4. Establish the effective signature-cache TTL from the exact running proxy
   image/config. Its default is1200s, but a default is not deployment evidence.
   If unknown, retain both proxy caches; do not recreate or remove either proxy.
5. Follow [GPU preflight](migration-gpu-preflight.md) on the actual destination;
   inspect supported GPU fields, kernel coverage/alignment and repeated samples.
   Missing telemetry is not zero. Do not upgrade a production manager/CVM to obtain
   logs: the existing application collector is the supported production transport.

Every following mutation needs a fresh exact dry-run with the same file, project,
revision, complete environment and explicit service list. Reject unrelated creates,
recreates/removals. Reconcile terminal events and runtime before retrying an
interrupted call. The helper project must not affect any `work` container.

## Sequence

1. Start only `handover-canary` in the isolated project. Fetch its exact fresh
   operation ID and terminal `status=ok` from Loki with host/model/deployment labels.
   Helpers repeat the terminal record for95s; this is not delivery proof. Require
   the record and then absence from running containers. `/docker/ps` does not expose
   exited-container exit codes. Missing or ambiguous output blocks all HUP actions.
2. Run `handover-model-downloader-qwen38` against the existing
   `work_hugginface_cache`, then start only `proxy-qwen36-handover`. Qualify the
   temporary proxy before any traffic change: model ID, adequate-budget semantic,
   streaming/tools, both signature algorithms, metrics and nonce-bound attestation.
   Compare signing identities, TLS binding and OHTTP key configurations with the
   canonical proxy. Prove the temporary proxy accepts a request encrypted with the
   canonical proxy's previously fetched key. Never persist keys or raw reports.
   Use `handover-qualify` inside the isolated network: it has no host ports, PID
   sharing, GPU devices, extra capabilities or host mounts. It checks the unchanged
   nginx TLS certificate with the canonical SNI, then compares its SPKI with both
   fresh attestation responses. Dependencies are fully pinned with hashes and installed
   into disposable tmpfs (`exec` permits native wheel loading, while `nosuid`,
   `nodev`, dropped capabilities and the read-only root remain enforced).
   Installation or verification failure is a failed gate.
   This validates structural nonce/key/TLS bindings and live request signatures,
   not an independent Intel/NVIDIA certificate-chain verification. Retrieve the
   exact-operation terminal record from Loki before proceeding.
   The qualifier also tests two synthetic images and a four-frame synthetic video,
   in JSON and complete streaming responses, directly on the selected engine and
   through temporary. Text, readiness and GPU counters do not prove multimodal health.
3. Run `handover-preflight`, requiring unchanged installed config hash, master and
   workers, successful candidate `nginx -t`, and `no_signal=true`. It stages a
   separate main config outside the live `*.conf` include. The existing periodic
   six-hour reload loop must be identified and at least two minutes from firing.
4. Create a synthetic canonical completion, retaining only its test chat ID in
   restricted evidence; start long HTTP/2 streams for Qwen3.6 and an unaffected
   shared model. Apply `handover-to-temp`. Require the exact operation's old/new
   hashes, same master/container and newly started worker identities. Verify all
   pre-HUP streams finish normally. On fresh connections, prove both old-chat
   signature algorithms fall back to canonical, new signatures come from temporary,
   cached-key OHTTP still works, and every unrelated route remains healthy.
5. Poll `handover-drain` until all captured pre-switch workers have exited naturally.
   The helper tracks PID plus process start time, not just PIDs. Never force workers
   out or infer this drain from a response-header-lifetime proxy gauge. Require
   three fresh spaced zero-work samples for r2. Remove **only** old r2 using the
   actual old work-project file, `volumes=false`; omit the reserved `work` project
   name in the API. `/compose/down` has no dry-run: validate its exact body and use
   a same-original-file service-scoped up dry-run to confirm resolution first.
6. Verify GPU2 released, then against final `small-models.yaml` start only Qwen3.8's
   engine, proxy, dedicated nginx and required telemetry changes, one stage at a time.
   Preserve Qwen3.6r1, shared nginx/registrar, GLM and all utility IDs. Qualify the
   real GPU2 path through HTTP8000 and TLS8010: semantics, stream completion, tools,
   both signatures, private inference and attestation, plus GPU/queue/error metrics.
   Explicitly activate only `qwen38-model-proxy-registrar` after passing. Verify
   probe8000/routing8010 and model/domain on **every** peer, retaining source routes.
7. After the pre-to-temp workers drained, retain the old canonical proxy for the
   proven signature TTL plus120s, with no new completion traffic sent to it. Then
   recreate **only** canonical `proxy-qwen36-35b-a3b` from the final file, off-path,
   and qualify it against temporary, including key/signature identity continuity.
8. Keep test streams open and apply `handover-to-canonical`; verify temporary-chat
   signature fallback, new canonical signatures and cached-key OHTTP. Wait for the
   captured workers to drain, then retain temporary for its proven TTL plus120s.
   Apply `handover-to-steady`, prove exact original config restoration and complete
   all old streams. Wait for its captured workers to drain before stopping helpers
   or temporary proxy. Keep the small state volume and all caches/certificates.
9. Soak both single-replica destination paths for at least30minutes of representative
   traffic with no failed probes, sustained queue growth, SLO-breaking latency or
   new GPU/proxy/signature/attestation/OHTTP errors. Complete a final30minute steady
   observation and inventory/routing reconciliation. Both source Qwens remain
   registered until the separate all-model gpu02 evacuation checklist passes.

## Failure and rollback

- Failed pre-HUP validation leaves the live file and workers unchanged. A failure
  after installation restores the previous config and, when needed, HUPs only the
  same verified master. The state is `failed`; do not infer which path owns all
  active streams. Reconcile config, generations and functional paths before any
  subsequent mutation. Both proxies must remain running during recovery.
- Missing helper logs, signature fallback failure, unequal required signing/OHTTP
  identities, truncated streams, concurrent reload/config changes or non-draining
  workers stop the sequence. Do not restart shared nginx/registrar or the CVM.
- After a successful to-temp transition, rollback is to-canonical **with temporary
  signature fallback**, then TTL/drain and steady restoration. Going directly to a
  no-fallback config would strand signatures generated by temporary.
- If Qwen3.8 fails, withdraw only its dedicated registrar, verify peer withdrawal
  and drain that listener/engine while source Qwen3.8 remains serving. Do not restore
  r2 on GPU2 until Qwen3.8 is stopped and GPU release is verified.

### Recover an unhealthy retained r1 before consolidation

`QWEN_HANDOVER_REPLICA` selects only `r1` (default) or `r2` for the temporary
proxy and qualification helpers; other values fail qualification before requests.
It never changes the canonical work-project proxy, engines, GPU allocation,
nginx candidates or registrar. Never change it on an active temporary proxy.

1. If temporary has served traffic, first complete the existing to-canonical
   rollback, natural worker drain, proven temporary signature TTL plus120s,
   to-steady restoration and its natural worker drain. Keep temporary until all
   gates pass. New completion traffic restarts its cache-retention clock.
2. Run only `handover-model-check` with `QWEN_HANDOVER_REPLICA=r2`. It receives
   no credentials, uses the fixed local r2 engine name, and must pass fresh
   image/video JSON and streaming tests. A failed check stops recovery.
3. With temporary off-path and its caches expired, recreate only temporary with
   the same explicit r2 setting. Reconcile the dry-run and actual backend config;
   run `handover-qualify` with r2, preserving all existing crypto and TLS gates.
4. Follow the normal preflight/long-stream/to-temp/natural-drain procedure.
   Prove three spaced zero-work samples on r1 before a service-scoped rolling
   r1 process recovery using its actual old work-project file and immutable
   image/checkpoint. No GPU reset, CVM/manager restart or shared ingress restart.
   Preserve r2, all source replicas and both proxies throughout.
5. Run only `handover-model-check` with r1 after it is ready; inspect fresh GPU,
   image/video error and queue evidence. Do not remove r2 on a text-only pass.
   Roll back to canonical with temporary signature fallback, validate continuity,
   drain and retain caches, restore steady, then return temporary to r1 off-path.
   Repeat full qualification before resuming the original consolidation sequence.

This is a recovery procedure, not a claim to fix the initiating CUDA fault. A
repeated fault blocks consolidation and requires a separate root-cause change.

## Verification

`validate_qwen_handover.py` uses synthetic real-Docker HTTP/2 streams, verifies
forward/reverse signature fallback without POST retries, negative config/SHA guards,
same nginx container and natural worker drain, exact steady restoration, and scoped
Compose down preserving a peer/external network. This does not replace the real-CVM
cryptographic, capacity, telemetry and log-delivery gates above.
`validate_qwen_qualification.py` installs the hash-locked dependencies in the exact
read-only helper sandbox and checks semantic/stream termination, both signature
algorithms and payload binding, binary HTTP field/trailer bounds and zero padding,
encrypted response integrity, mandatory authenticated final markers and truncated
encrypted-response rejection. Entirely empty binary HTTP trailers may be omitted
as required by [RFC9292 section3.8](https://datatracker.ietf.org/doc/html/rfc9292#section-3.8);
partial nonempty sections and invalid padding are rejected.

Related: #234, #235, GPU diagnostic #237 and project-scoped logs
[compose-manager#60](https://github.com/nearai/compose-manager/pull/60). Production
can use Loki without upgrading its manager; staging can also use project-scoped logs.
