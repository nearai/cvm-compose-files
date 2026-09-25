# Rotate the compose-manager `BEARER_TOKEN`

Use `prod/rotate-compose-manager-token.yaml` through compose-manager to rotate
a running compose-manager CVM's `BEARER_TOKEN` **without restarting the CVM
and without touching any model container**. It only recreates the single
`compose-manager` container inside its own app-compose project (`dstack`).

Verified against `nearai/compose-manager` at commit `8e07c35` (prod tag
`prod-20260702-8e07c35`, the pinned digest running on every compose-manager
CVM as of 2026-09-25) and `nearai/cvm-ansible-playbooks`
`cvm_configurations/compose-manager.yaml.j2`. See the compose file's own
header comment for the full mechanism writeup.

## Always scope the request to its own project

Set the API field `"project": "cm-token-rotate"` explicitly on every
`POST /compose/up`, `POST /compose/down` and `POST /compose/logs` call
against this file. The file's own `name: cm-token-rotate` is only a Compose
fallback default — it does not protect a request that explicitly names a
different project. compose-manager itself refuses `"project": "work"` or
`"project": "dstack"` (see `resolve_compose_project` in
`nearai/compose-manager` `src/main.rs`), so those two are safe regardless,
but always pass the project explicitly rather than relying on that refusal.

## Canary, then apply

1. **Dry run with a placeholder token first.** `dry_run: true` is genuinely
   read-only (no action recorded, nothing pulled/built/spawned) — but
   compose-manager still writes the `env` map to a short-lived temp file on
   disk before short-circuiting, so never pass the real new token on a
   dry-run call:

   ```json
   {
     "tag": "<reviewed tag>",
     "file": "prod/rotate-compose-manager-token.yaml",
     "project": "cm-token-rotate",
     "services": ["rotate-token"],
     "env": { "NEW_BEARER_TOKEN": "placeholder-for-dry-run-only" },
     "dry_run": true
   }
   ```

   Expected plan: create only `rotate-token` under project `cm-token-rotate`.
   No existing container (compose-manager, certbot, datadog, any model) is
   listed as recreated or removed.

2. **Apply with the real token.** Repeat the identical payload with the real
   `NEW_BEARER_TOKEN` and `"dry_run": false`.

   `compose-manager`'s streaming response to this call only confirms the
   `rotate-token` container **started** (`docker compose up -d` returns as
   soon as the detached container is created — it does not wait for the
   container's own script to finish). To know whether the rotation itself
   succeeded, check the container's own exit status and logs afterward:

   ```json
   { "project": "cm-token-rotate", "services": ["rotate-token"] }
   ```

   against `POST /docker/ps` and `POST /compose/logs`.

3. **Interpret the exit code:**
   - `0` — rotated; compose-manager came back healthy on the **new** token.
   - `2` — rotation failed; automatically rolled back; compose-manager is
     healthy again on the **old** token. Investigate before retrying.
   - `3` — CRITICAL: the rollback recreate also failed to come up healthy.
     Manual intervention is required on this host — do not assume either
     token works until checked by hand.

4. **Verify directly against the host**, independent of the exit code:

   ```bash
   curl -s -o /dev/null -w '%{http_code}\n' http://<host_ip>:8080/version \
     -H "Authorization: Bearer <OLD_TOKEN>"   # expect 401 on success (0)
   curl -s -o /dev/null -w '%{http_code}\n' http://<host_ip>:8080/version \
     -H "Authorization: Bearer <NEW_TOKEN>"   # expect 200 on success (0)
   ```

5. **Update the gpu-manager dashboard's per-host token** (`dashboard.json` /
   `COMPOSE_MANAGER_INSTANCE_TOKENS`, per the current rotation plan) so future
   dashboard-driven calls to this host use the new token. This file does not
   do that for you.

## Rollback

The `rotate-token` container already auto-rolls-back on a failed health
check (exit `2`/`3` above). If a rotation reports success (`0`) but the new
token turns out to be wrong for some other reason, run this same file again
with `"env": {"NEW_BEARER_TOKEN": "<the OLD token value>"}` — it will back up
the current `.env.launcher`, write the old token back in, and recreate
compose-manager again. A timestamped backup
(`.env.launcher.bak.<UTC timestamp>`) is also left behind inside
`compose-manager-launcher`'s `/app/work` on every run; restoring it by hand
(`docker exec compose-manager-launcher cp <backup> /app/work/.env.launcher`
followed by the same recreate command in the mechanism writeup) is the
fallback if the compose file itself cannot be re-run.

## Give Brave a heads-up before running

Brave's public verifier watches the attestation report's append-only actions
log, which records every `compose_up` (`timestamp`, `action`, `tag`,
`commit`, `file`, `file_sha256`, `services`) — it will see this rotation
happen (file `prod/rotate-compose-manager-token.yaml`) even though the token
**value** is never recorded there or anywhere else (`env` values only ever
reach compose-manager's short-lived per-call temp env file, deleted when the
request's stream ends — never written to git, the action log, or any log
line). Coordinate with Brave before running this in prod so an expected
`compose_up` entry doesn't read as a surprise.

## Inner script

The `rotate-token` service's `command:` carries the bulk of its logic (the
part that runs inside `compose-manager-launcher`) as a base64 blob instead of
writing it inline with `docker compose`'s `$$`-escaping on every line —
`docker compose` interpolates `$VAR` / `${VAR}` / `${VAR:-x}` anywhere in the
compose file, including inside multi-line `command:` strings, and base64's
alphabet contains no `$`, so the blob is immune to that. This is the exact
plaintext it decodes to (sha256 `21b469e48311c159bbdcbf665a091f8146f4cf6339cbf7a72ea6ee7de3fc6e96`,
checked by `scripts/validate_rotate_compose_manager_token.rb` so the compose
file and this doc can never silently drift apart):

```sh
set -eu
ENV_FILE="${LAUNCHER_ENV_FILE:-/app/work/.env.launcher}"
BASE_ENV_FILE="${LAUNCHER_BASE_ENV_FILE:-/dstack-env}"
COMPOSE_FILE="${LAUNCHER_COMPOSE_FILE:-/app/work/compose-manager.yml}"
PROJECT="${LAUNCHER_COMPOSE_PROJECT:-dstack}"
HEALTH_URL="${LAUNCHER_HEALTH_URL:-http://127.0.0.1:8080/version}"
TIMEOUT="${LAUNCHER_HEALTH_TIMEOUT:-180}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
BACKUP=""

if [ ! -f "$COMPOSE_FILE" ]; then
  echo "[rotate-token] $COMPOSE_FILE is missing inside compose-manager-launcher -- aborting, nothing changed" >&2
  exit 1
fi

if [ -f "$ENV_FILE" ]; then
  BACKUP="${ENV_FILE}.bak.${TS}"
  cp -p "$ENV_FILE" "$BACKUP"
else
  : > "$ENV_FILE"
fi

rewrite_token() {
  tmp="${ENV_FILE}.tmp.$$"
  mode="0644"
  [ -f "$ENV_FILE" ] && mode="$(stat -c %a "$ENV_FILE" 2>/dev/null || echo 0644)"
  awk -v k="BEARER_TOKEN" 'BEGIN{v=ENVIRON["NEW_BEARER_TOKEN"]; found=0}
    index($0,k"=")==1{print k"="v; found=1; next}
    {print}
    END{if(!found) print k"="v}' "$ENV_FILE" > "$tmp"
  chmod "$mode" "$tmp"
  mv "$tmp" "$ENV_FILE"
}

recreate() {
  docker compose -p "$PROJECT" -f "$COMPOSE_FILE" \
    --env-file "$BASE_ENV_FILE" --env-file "$ENV_FILE" \
    up -d --no-deps compose-manager
}

wait_healthy() {
  i=0
  steps=$(( TIMEOUT / 3 ))
  while [ "$i" -lt "$steps" ]; do
    if curl -fsSL --max-time 5 "$HEALTH_URL" >/dev/null 2>&1; then
      return 0
    fi
    i=$((i + 1))
    sleep 3
  done
  return 1
}

echo "[rotate-token] rewriting BEARER_TOKEN in $ENV_FILE (backup: ${BACKUP:-none created, file was missing})"
rewrite_token

echo "[rotate-token] recreating compose-manager (project $PROJECT, --no-deps, no other service touched)"
recreate

echo "[rotate-token] waiting up to ${TIMEOUT}s for $HEALTH_URL"
if wait_healthy; then
  echo "[rotate-token] compose-manager healthy on the NEW token"
  exit 0
fi

echo "[rotate-token] compose-manager did NOT become healthy within ${TIMEOUT}s -- rolling back" >&2
if [ -n "$BACKUP" ] && [ -f "$BACKUP" ]; then
  cp -p "$BACKUP" "$ENV_FILE"
else
  echo "[rotate-token] no backup existed (file was newly created this run) -- leaving as-is" >&2
fi
recreate
if wait_healthy; then
  echo "[rotate-token] rollback recreate succeeded -- compose-manager is healthy again on the OLD token" >&2
  exit 2
fi
echo "[rotate-token] CRITICAL: rollback recreate ALSO failed to become healthy -- manual intervention required on this host" >&2
exit 3
```

`LAUNCHER_ENV_FILE`, `LAUNCHER_BASE_ENV_FILE`, `LAUNCHER_COMPOSE_FILE`,
`LAUNCHER_COMPOSE_PROJECT`, `LAUNCHER_HEALTH_URL` and
`LAUNCHER_HEALTH_TIMEOUT` are not passed in explicitly — `docker exec` into
an already-running container automatically inherits that container's own
environment, so this script reads them live from `compose-manager-launcher`
itself (`cvm_configurations/compose-manager.yaml.j2` in
`nearai/cvm-ansible-playbooks` sets all six on that service today). The
defaults shown after `:-` match `launcher.sh`'s own defaults at `8e07c35` and
only apply if a future launcher build drops one of these vars entirely.

This mechanism, including the auto-rollback branch, was validated end-to-end
against a local Docker sandbox standing in for a compose-manager CVM (a fake
`compose-manager-launcher` with the same mounts/env, a stand-in
`compose-manager` service, `docker compose ... up`/`--dry-run` run for real)
before this file was opened for review — not just `docker compose config`.
