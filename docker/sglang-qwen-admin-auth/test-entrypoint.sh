#!/bin/sh
set -eu
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT HUP INT TERM
mkdir "$tmp/bin"
cat > "$tmp/bin/sglang" <<'STUB'
#!/bin/sh
printf '%s\n' "$@" > "$CAPTURE"
STUB
cat > "$tmp/bin/passthrough" <<'STUB'
#!/bin/sh
printf '%s\n' "$@" > "$CAPTURE"
STUB
chmod +x "$tmp/bin/sglang" "$tmp/bin/passthrough"
export CAPTURE="$tmp/capture"
PATH="$tmp/bin:$PATH"
export PATH
config='{"enable_multithread_load": "true", "num_threads": 64}'
"$here/admin-key-entrypoint.sh" sglang serve --model-path x --model-loader-extra-config "$config"
{ printf '%s\n' serve --model-path x --model-loader-extra-config "$config" --admin-api-key; sed -n '7p' "$CAPTURE"; } > "$tmp/expected"
cmp "$tmp/expected" "$CAPTURE"
key1=$(sed -n '7p' "$CAPTURE")
[ "${#key1}" -eq 43 ]
case "$key1" in *[!A-Za-z0-9_-]*) exit 1;; esac
"$here/admin-key-entrypoint.sh" sglang serve --model-path x
key2=$(sed -n '5p' "$CAPTURE")
[ "${#key2}" -eq 43 ]
[ "$key1" != "$key2" ]
"$here/admin-key-entrypoint.sh" sglang serve --admin-api-key existing --model-path x
printf '%s\n' serve --admin-api-key existing --model-path x > "$tmp/expected"
cmp "$tmp/expected" "$CAPTURE"
"$here/admin-key-entrypoint.sh" passthrough 'argument with spaces' ''
printf '%s\n' 'argument with spaces' '' > "$tmp/expected"
cmp "$tmp/expected" "$CAPTURE"
rm "$CAPTURE"
cat > "$tmp/bin/python3" <<'STUB'
#!/bin/sh
exit 1
STUB
chmod +x "$tmp/bin/python3"
if "$here/admin-key-entrypoint.sh" sglang serve; then
  echo 'Expected key generation failure' >&2
  exit 1
fi
[ ! -e "$CAPTURE" ]
echo 'Entrypoint checks passed'
