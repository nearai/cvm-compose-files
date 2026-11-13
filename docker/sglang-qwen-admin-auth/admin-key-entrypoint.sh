#!/bin/sh
set -eu
if [ "${1:-}" = "sglang" ] && [ "${2:-}" = "serve" ]; then
  for arg in "$@"; do
    if [ "$arg" = "--admin-api-key" ]; then exec "$@"; fi
  done
  key="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
  [ -n "$key" ]
  exec "$@" --admin-api-key "$key"
fi
exec "$@"
