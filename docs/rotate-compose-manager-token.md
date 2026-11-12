# Rotate the compose-manager `BEARER_TOKEN`

`prod/rotate-compose-manager-token.yaml` replaces the `BEARER_TOKEN` of a running compose-manager. Its
one service, `rotate-token`, recreates only the `compose-manager` container; model containers are not
touched. The service has no network, only the Docker socket, and never restarts.

## Usage

Send the file through compose-manager with `"project": "cm-token-rotate"` on every request for it
(`/compose/up`, `/compose/down`, `/compose/logs`), and pass the new token as `NEW_BEARER_TOKEN`:

```json
{
  "tag": "<release tag>",
  "file": "prod/rotate-compose-manager-token.yaml",
  "project": "cm-token-rotate",
  "services": ["rotate-token"],
  "env": { "NEW_BEARER_TOKEN": "<new token>" }
}
```

The container exits `0` when the new token is accepted, `2` when the rotation was rolled back to the
old token, and `3` when the rollback also failed.

## Inner script

The `rotate-token` command carries the script that runs inside `compose-manager-launcher` as a base64
blob, because Compose interpolates `$` anywhere in the file and base64 has no `$`. The blob decodes to
the script below (sha256 `f74936ed5b57269ae823d6de2f47501ee1e75fc6483a6e5351541153d3718f42`, checked
by `scripts/validate_rotate_compose_manager_token.rb`). Change the compose file, this block and the
validator's hash together.

```sh
set -eu
ENV_FILE="${LAUNCHER_ENV_FILE:-/app/work/.env.launcher}"
BASE_ENV_FILE="${LAUNCHER_BASE_ENV_FILE:-/dstack-env}"
COMPOSE_FILE="${LAUNCHER_COMPOSE_FILE:-/app/work/compose-manager.yml}"
PROJECT="${LAUNCHER_COMPOSE_PROJECT:-dstack}"
HEALTH_URL="${LAUNCHER_HEALTH_URL:-http://127.0.0.1:8080/version}"
# /version is unauthenticated (verified live: no BEARER_TOKEN required) --
# it only proves the container process is up and answering, NOT that the
# BEARER_TOKEN override actually took effect. Derive the authenticated
# GET /docker/ps endpoint (verify_bearer_token-gated in compose-manager's
# own src/main.rs) from the same host:port instead of a second hardcoded
# assumption.
AUTH_CHECK_URL="${HEALTH_URL%/version}/docker/ps"
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

# Polls the AUTHENTICATED GET /docker/ps with NEW_BEARER_TOKEN, building the
# Authorization header via curl's -K config-from-stdin so the token is never
# a command-line argument (never visible in `ps`/`docker top`) and is never
# echoed/logged. NEW_BEARER_TOKEN was already format-validated (no quotes,
# no whitespace) by the outer wrapper before this script ever runs.
wait_new_token_accepted() {
  i=0
  steps=$(( TIMEOUT / 3 ))
  while [ "$i" -lt "$steps" ]; do
    if printf 'header = "Authorization: Bearer %s"\n' "$NEW_BEARER_TOKEN" \
         | curl -fsSL -K - --max-time 5 -o /dev/null "$AUTH_CHECK_URL" 2>/dev/null; then
      return 0
    fi
    i=$((i + 1))
    sleep 3
  done
  return 1
}

rollback_and_exit() {
  reason="$1"
  echo "[rotate-token] $reason -- rolling back" >&2
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
}

echo "[rotate-token] rewriting BEARER_TOKEN in $ENV_FILE (backup: ${BACKUP:-none created, file was missing})"
rewrite_token

echo "[rotate-token] recreating compose-manager (project $PROJECT, --no-deps, no other service touched)"
recreate

echo "[rotate-token] waiting up to ${TIMEOUT}s for $HEALTH_URL"
if ! wait_healthy; then
  rollback_and_exit "compose-manager did NOT become healthy within ${TIMEOUT}s"
fi

# /version answering 200 only proves the container process is up -- it does
# NOT prove the BEARER_TOKEN override actually took effect (e.g. an
# interpolation/ordering regression could leave the OLD token live while the
# container still answers /version normally). Separately confirm the NEW
# token is actually accepted before declaring success.
echo "[rotate-token] compose-manager healthy; verifying the NEW token is accepted at $AUTH_CHECK_URL"
if ! wait_new_token_accepted; then
  rollback_and_exit "compose-manager is healthy but the NEW token was rejected at $AUTH_CHECK_URL (override did not take effect)"
fi

echo "[rotate-token] compose-manager healthy AND the NEW token is accepted"
exit 0
```

The `LAUNCHER_*` variables are not passed in: `docker exec` inherits the environment of
`compose-manager-launcher`, which sets them. The defaults after `:-` apply only if one is missing.
