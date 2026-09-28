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

# Compose interpolates BOTH the braced `${VAR}` / `${VAR:-x}` form AND the
# un-braced `$VAR` form, anywhere in the file -- including inside this
# multi-line `command:` string (see the compose file's own header comment).
# Every `$` that must reach the shell (not compose) has to be doubled to
# `$$`, compose's own escape for a literal `$`. This walks the string
# instead of using a single regex because escaping is about RUNS of `$`,
# not individual characters: `$$` is one literal `$` (safe), but `$$$VAR`
# is `$$` (one literal `$`) followed by a genuine, un-escaped `$VAR` --
# i.e. a run of an ODD number of `$` always leaves exactly one real
# interpolation attempt behind, however many pairs precede it. A naive
# `(?<!\$)\$...` regex only gets this right for runs of length 1 or 2.
#
# Returns one { line:, token: } per un-escaped interpolation target found,
# `line` counted within `text` (1-indexed) so failures are easy to locate
# in the compose file's `command:` block.
def unescaped_compose_interpolations(text)
  findings = []
  i = 0
  len = text.length
  while i < len
    if text[i] == "$"
      run_start = i
      i += 1 while i < len && text[i] == "$"
      run_len = i - run_start
      next if run_len.even? # fully paired off into literal `$`s -- safe

      rest = text[i..-1]
      token =
        if (m = rest.match(/\A\{[A-Za-z_][A-Za-z0-9_]*(?::-[^}]*)?\}/))
          "$#{m[0]}"
        elsif (m = rest.match(/\A[A-Za-z_][A-Za-z0-9_]*/))
          "$#{m[0]}"
        end
      next unless token # lone trailing `$` not followed by `{`/an identifier
                         # isn't a compose interpolation target at all

      findings << { line: text[0...i].count("\n") + 1, token: token }
    else
      i += 1
    end
  end
  findings
end

# The one interpolation target this whole mechanism deliberately relies on:
# `/compose/up`'s own `env` map is how `NEW_BEARER_TOKEN` reaches the
# container (see the file's header comment) -- checked exactly against
# `service["environment"]` above, so it's matched here by that same exact
# string rather than by field path.
INTENTIONAL_INTERPOLATIONS = ["NEW_BEARER_TOKEN=${NEW_BEARER_TOKEN}"].freeze

# Sweep every string value anywhere in the parsed YAML doc -- not just
# `command` -- so a `$`/`${...}` slipped into some other field (an added
# `environment:` line, a label, a future field) fails loudly too instead of
# being silently blanked out at compose render time (compose only warns).
def each_yaml_string(value, path, &block)
  case value
  when String
    yield(path, value)
  when Array
    value.each_with_index { |v, i| each_yaml_string(v, "#{path}[#{i}]", &block) }
  when Hash
    value.each { |k, v| each_yaml_string(v, "#{path}.#{k}", &block) }
  end
end

# Self-test the guard itself against synthetic fixtures before trusting it
# to judge the real file -- mirrors validate_migration_gpu_preflight.rb
# always running its own embedded tests as part of a normal invocation (no
# separate spec file or flag needed; this always runs when CI runs this
# script).
def assert_interpolations(label, text, expected_tokens)
  found = unescaped_compose_interpolations(text).map { |f| f[:token] }
  return if found == expected_tokens

  raise "Guard self-test failed (#{label}): text=#{text.inspect} " \
        "expected=#{expected_tokens.inspect} got=#{found.inspect}"
end

# (a) un-escaped, un-braced $FOO -- the exact case the review comments on
# this file flagged as missed by the previous ${VAR}-only regex.
assert_interpolations("bare $FOO", "before $FOO after", ["$FOO"])
# (b) un-escaped ${FOO}
assert_interpolations("braced ${FOO}", "before ${FOO} after", ["${FOO}"])
# (c) un-escaped ${FOO:-x} (default-value form)
assert_interpolations("braced with default ${FOO:-x}", "before ${FOO:-x} after", ["${FOO:-x}"])
# $$FOO -- one escaped pair, safe, must NOT be flagged
assert_interpolations("escaped $$FOO", "before $$FOO after", [])
# $${FOO} -- one escaped pair, safe, must NOT be flagged
assert_interpolations("escaped $${FOO}", "before $${FOO} after", [])
# $$$FOO -- an escaped pair (one literal $) PLUS a genuine un-escaped $FOO
# immediately after it; the failure mode a naive `(?<!\$)\$...` regex gets
# wrong (see the comment on unescaped_compose_interpolations above).
assert_interpolations("odd run $$$FOO", "before $$$FOO after", ["$FOO"])
# $$$$FOO -- two escaped pairs, fully safe, must NOT be flagged.
assert_interpolations("even run $$$$FOO", "before $$$$FOO after", [])
# The exact allowed environment line: unescaped_compose_interpolations
# itself must still DETECT it (it genuinely is an interpolation target) --
# it's only exempt because INTENTIONAL_INTERPOLATIONS allow-lists that
# exact string in the full-doc sweep below.
assert_interpolations(
  "allowed env line still detected on its own",
  "NEW_BEARER_TOKEN=${NEW_BEARER_TOKEN}",
  ["${NEW_BEARER_TOKEN}"]
)
raise "Guard self-test failed: allow-list did not exempt the intentional env line" \
  unless INTENTIONAL_INTERPOLATIONS.include?("NEW_BEARER_TOKEN=${NEW_BEARER_TOKEN}")

puts "Interpolation guard self-tests OK (8 cases)"

violations = []
each_yaml_string(doc, "doc") do |path, value|
  next if INTENTIONAL_INTERPOLATIONS.include?(value)

  unescaped_compose_interpolations(value).each do |f|
    violations << "#{path} line #{f[:line]}: #{f[:token]}"
  end
end
if violations.any?
  raise "Un-escaped compose interpolation target(s) found -- must be $$ (for " \
        "the shell, not compose) unless genuinely intentional:\n  " +
        violations.join("\n  ")
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
