#!/usr/bin/env ruby
# Structural + isolation contract for prod/rotate-compose-manager-token.yaml,
# the one-shot compose-manager BEARER_TOKEN rotation tool. Excluded from the
# normal serving-stack contracts (validate_otel_labels.rb) since it is a
# maintenance tool, not a model stack -- this script is its replacement
# safety net, mirroring validate_migration_gpu_preflight.rb for the other
# root-owned one-shot maintenance file in this repo.

require "base64"
require "digest"
require "json"
require "yaml"

ROOT = File.expand_path("..", __dir__)
FILE = File.join(ROOT, "prod", "rotate-compose-manager-token.yaml")
# See docs/rotate-compose-manager-token.md#inner-script for the plaintext
# this decodes to. Keeping the hash here (rather than trusting eyeballing)
# means the compose file, the doc, and this check can never silently drift
# apart -- any edit to the inner script must update all three together.
INNER_SCRIPT_SHA256 = "f74936ed5b57269ae823d6de2f47501ee1e75fc6483a6e5351541153d3718f42".freeze

def yaml_load(content)
  YAML.load(content, aliases: true)
rescue ArgumentError
  YAML.load(content)
end

doc = yaml_load(File.read(FILE))

raise "Unsafe default project" unless doc["name"] == "cm-token-rotate"
raise "Unexpected service set" unless doc.fetch("services").keys == ["rotate-token"]

service = doc["services"]["rotate-token"]

raise "Image not pinned" unless service["image"].match?(/@sha256:[a-f0-9]{64}$/)
raise "Must never auto-restart" unless service["restart"] == "no"
raise "Must be profile-gated (defense in depth against a blanket up)" \
  unless service["profiles"] == ["cm-token-rotate"]
raise "Isolation weakened" unless service["read_only"] == true && service["network_mode"] == "none"
raise "Capability scope changed" unless service["cap_drop"] == ["ALL"]
raise "Privilege escalation allowed" unless service["security_opt"] == ["no-new-privileges:true"]
raise "Must run as root to reach the Docker socket" unless service["user"] == "0:0"
raise "Unexpected tmpfs" unless service["tmpfs"] == ["/tmp"]

# The Docker socket is the ONLY host access this service may have -- no CVM
# decrypted env, no model/HF-cache volume, no dstack.sock, no extra ports.
raise "Unexpected volumes" \
  unless service["volumes"] == ["/var/run/docker.sock:/var/run/docker.sock"]
raise "Unexpected host attachment" \
  if %w[privileged pid ipc devices depends_on extra_hosts ports].any? { |key| service.key?(key) }

raise "Unexpected environment" unless service["environment"] == ['NEW_BEARER_TOKEN=${NEW_BEARER_TOKEN}']

raise "Diagnostic log metadata missing" \
  unless JSON.parse(service.fetch("labels").fetch("com.datadoghq.ad.logs")) ==
         [{ "source" => "cm-token-rotate", "service" => "cm-token-rotate", "tags" => ["deployment:cm-token-rotate"] }]

command = service.fetch("command").fetch(0)

# The command must never contain a literal, un-escaped compose interpolation
# target beyond the two intentional ones (NEW_BEARER_TOKEN in `environment:`,
# checked above, is a separate field) -- every other `$` in this string must
# already be doubled to `$$` so compose passes it through to the shell
# unresolved. A single un-escaped `${...}` here would silently blank out at
# render time (compose only warns) instead of failing loudly, which is
# exactly the class of bug this check exists to catch.
if command.scan(/(?<!\$)\$\{[A-Za-z_][A-Za-z0-9_]*(?::-[^}]*)?\}/).any?
  raise "Un-escaped ${VAR} interpolation target found in command -- must be $${VAR} for the shell, not compose"
end

match = command.match(/INNER_B64='([^']*)'/m)
raise "Could not find INNER_B64 blob in command" unless match

inner_script = Base64.decode64(match[1])
actual_sha256 = Digest::SHA256.hexdigest(inner_script)
if actual_sha256 != INNER_SCRIPT_SHA256
  raise "Inner script sha256 mismatch: compose file's INNER_B64 decodes to " \
        "#{actual_sha256}, expected #{INNER_SCRIPT_SHA256}. Update " \
        "INNER_SCRIPT_SHA256 here AND docs/rotate-compose-manager-token.md#inner-script " \
        "together with the compose file, never one alone."
end

raise "Inner script must only recreate compose-manager, never remove orphans" \
  if inner_script.include?("--remove-orphans")
raise "Inner script must target the launcher container by its known name" \
  unless inner_script.include?("compose-manager-launcher")
raise "Inner script must scope the recreate to --no-deps compose-manager only" \
  unless inner_script.include?("up -d --no-deps compose-manager")
raise "Inner script must never write the sealed base env file" \
  if inner_script.match?(/>\s*"?\$\{?BASE_ENV_FILE\b/)

# compose-manager's own GET /version is unauthenticated (verify_bearer_token
# is never called by its handler), so it can only prove the container process
# answers -- not that the BEARER_TOKEN override actually took effect. The
# inner script must separately confirm the NEW token against an authenticated
# endpoint (GET /docker/ps, which does call verify_bearer_token) before
# declaring success, and must never put the token on a curl command line
# (only ever through -K/config-from-stdin, so it can't show up in `ps`/
# `docker top`).
raise "Inner script must verify the new token against the authenticated /docker/ps endpoint" \
  unless inner_script.include?("/docker/ps") && inner_script.include?("Authorization: Bearer")
raise "Inner script must build the Authorization header via curl -K (never -H) so the token never reaches argv" \
  unless inner_script.match?(/curl\s.*-K\s+-/) && !inner_script.match?(/-H\s+["']?Authorization:/)
raise "Inner script must log a distinct message when the new token is rejected despite a healthy /version" \
  unless inner_script.include?("was rejected")

puts "Compose-manager token rotation contract OK (#{command.bytesize} byte command, " \
     "inner script #{inner_script.bytesize} bytes, sha256 #{actual_sha256[0, 12]}...)"
