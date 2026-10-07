#!/usr/bin/env ruby
# Protect the canonical two-replica GLM-5.3 Flash production contract, the
# separate r2-only HiCache canary, and the long-context r1-control/r2-HiCache
# experiment. Admission reserve remains required in the first two files and is
# forbidden in the long-context experiment pending a pool-clamp image. Every file's
# model-downloader also pre-stages the W4AFP8 snapshot (see REQUIRED_DOWNLOADS).
# The generated W4AFP8 base, 4x TP2 base canary and W4AFP8 long-context files each
# have their own exact contract below.

require "json"
require "shellwords"
require "yaml"

ROOT = File.expand_path("..", __dir__)
COMPOSE_FILE = File.join(ROOT, "prod", "GLM-5.3-Flash-SGL-TP4.yaml")
HICACHE_FILE = File.join(ROOT, "prod", "GLM-5.3-Flash-SGL-TP4-HiCache.yaml")
LONG_CONTEXT_FILE = File.join(ROOT, "prod", "GLM-5.3-Flash-SGL-TP4-LongContext.yaml")
LEGACY_CANARY_FILE = File.join(ROOT, "prod", "GLM-5.3-Flash-SGL-TP4-Canary.yaml")
RELEASED_IMAGE_FILE = File.join(ROOT, "docker", "sglang-glm53-hicache", "RELEASED_IMAGE")
ENGINE_IMAGE = "docker.io/nearaidev/sglang@sha256:e9d29a1cb1cd65284392c4d62d5f2a36669628057e15c60fe93ea40cfe4fc7e7"
PROXY_IMAGE = "nearaidev/vllm-proxy-rs@sha256:d61357da39918a57126864a451eaf054f06a6989c03fe9a1666f7e6374ba6907"
# Engine-side priority scheduling is only safe behind an inference-proxy that
# overwrites the `priority` of every forwarded request (build 2834196 onward,
# nearai/inference-proxy#241). Any other proxy build lets client-chosen or
# missing priorities reach the scheduler, where an untagged request gets the
# lowest possible priority.
PRIORITY_NORMALIZING_PROXY_IMAGES = [
  "nearaidev/vllm-proxy-rs@sha256:b3a8c6260834231271b4356c56a7aa2718608c8a537b35973916e0a56dc88fba",
  # inference-proxy main 0f37728 (includes #241 and #274 replica-state publishing).
  "nearaidev/vllm-proxy-rs@sha256:d61357da39918a57126864a451eaf054f06a6989c03fe9a1666f7e6374ba6907",
].freeze
PRIORITY_SWITCHES = %w[--enable-priority-scheduling --disable-priority-preemption].freeze
# The proxy assigns every request's priority; an engine-side default would
# silently re-admit untagged requests at a chosen level instead of failing loud.
FORBIDDEN_OPTIONS = %w[--default-priority-value].freeze
REPLICAS = {
  "model-sg-glm53-fp8-tp4-r1" => %w[0 1 2 3],
  "model-sg-glm53-fp8-tp4-r2" => %w[4 5 6 7],
}.freeze
EXPECTED_SERVICES = [
  "model-downloader",
  "hf-cleanup",
  "nginx",
  "model-proxy-registrar",
  "proxy-glm53",
  *REPLICAS.keys,
  "glm53-perception-check",
  "glm53-soak-relay",
  "dcgm-glm53",
  "otelcol-contrib",
].freeze
REQUIRED_OPTIONS = {
  "--model-path" => "/root/.cache/huggingface/hub/models--zai-org--GLM-5.3-Flash/snapshots/84c6a6aa9497188e15a635ba793b0f95a79b1033",
  "--revision" => "84c6a6aa9497188e15a635ba793b0f95a79b1033",
  "--served-model-name" => "z-ai/glm-5.3-flash",
  "--tp-size" => "4",
  "--ep-size" => "4",
  "--mem-fraction-static" => "0.80",
  "--max-running-requests" => "32",
  "--max-queued-requests" => "8",
  "--chunked-prefill-size" => "4096",
  "--prefill-decode-interval" => "1",
  "--cuda-graph-max-bs-decode" => "32",
  "--dsa-prefill-backend" => "tilelang",
  "--dsa-decode-backend" => "tilelang",
  "--kv-cache-dtype" => "bfloat16",
  "--moe-runner-backend" => "deep_gemm",
  "--speculative-algorithm" => "EAGLE",
  "--speculative-num-steps" => "5",
  "--speculative-eagle-topk" => "1",
  "--speculative-num-draft-tokens" => "6",
  "--reasoning-parser" => "glm45",
  "--grammar-backend" => "xgrammar",
  "--tool-call-parser" => "glm47",
  "--chat-template" => "/root/.cache/huggingface/hub/models--zai-org--GLM-5.3-Flash/snapshots/3f1971b7b5f7a528c9c4ef6212c8785298a8c24a/chat_template.jinja",
  "--context-length" => "1048576",
  "--limit-mm-data-per-request" => '{"image": 64}',
  "--log-requests-level" => "0",
}.freeze
REQUIRED_SWITCHES = %w[
  --enable-priority-scheduling
  --disable-priority-preemption
  --speculative-adaptive
  --enable-strict-thinking
  --enable-metrics
  --enable-cache-report
  --disable-fast-image-processor
].freeze
REPLICA_IDENTITY_FIELDS = %w[container_name deploy labels].freeze
# Admission-reserve v10 must remain enabled in the canonical and standalone
# HiCache files. The long-context experiment has a separate reserve-free contract.
REQUIRED_ENV = {
  "SGLANG_CHUNKED_PREFILL_ADMISSION_RESERVE" => "4096",
  "SGLANG_ADMISSION_RESERVE_MAX_FRACTION" => "0.75",
}.freeze
ADMISSION_RESERVE_ENV = REQUIRED_ENV.keys.freeze
# SGLANG_ADMISSION_RESERVE_MIN_WAIT_S: wall-clock gate desyncs the TP ranks and
# crashes the engine. SGLANG_ADMISSION_RESERVE_MIN_ITERS/FIT/DEBUG: unmeasured
# or debug-only in production.
FORBIDDEN_ENV = %w[
  SGLANG_ADMISSION_RESERVE_MIN_WAIT_S
  SGLANG_ADMISSION_RESERVE_MIN_ITERS
  SGLANG_ADMISSION_RESERVE_FIT
  SGLANG_ADMISSION_RESERVE_DEBUG
].freeze
HICACHE_OPTIONS = {
  "--hicache-write-policy" => "write_through",
  "--hicache-io-backend" => "direct",
  "--hicache-mem-layout" => "page_first_direct",
}.freeze
HICACHE_ENV = {
  "SGLANG_HICACHE_RAM_BUDGET" => "${GLM53_HICACHE_RAM_BUDGET:-80%}",
  "SGLANG_HICACHE_CUDA_HOST_MEMORY" => "${GLM53_HICACHE_CUDA_HOST_MEMORY:-1}",
  "SGLANG_HICACHE_POOLED_TRANSFERS" => "1",
  "SGLANG_HICACHE_STAGING_PAGES" => "64",
}.freeze
HICACHE_VARIANT = "fc91d24-hicache-cuda-host-pooled-v1-admission-reserve-v10-pdi1-h200-tp4-ep4-eagle-adaptive-5-1-6-strict-budget8192"
OFFICIAL_VARIANT = "fc91d24-admission-reserve-v10-pdi1-h200-tp4-ep4-eagle-adaptive-5-1-6-strict-budget8192"
LONG_CONTEXT_CONTROL_VARIANT = "fc91d24-long-context-admission-reserve-disabled-hicache-disabled-pdi1-h200-tp4-ep4-eagle-adaptive-5-1-6-strict-budget8192"
LONG_CONTEXT_HICACHE_VARIANT = "fc91d24-long-context-admission-reserve-disabled-hicache-cuda-host-pooled-v1-pdi1-h200-tp4-ep4-eagle-adaptive-5-1-6-strict-budget8192"
# model-downloader fetches the FP8 snapshot, the corrected chat template and the W4AFP8
# snapshot in every dedicated GLM-5.3 Flash TP4 file (prod/small-models.yaml is out of
# scope), so a host switches between the FP8 and W4AFP8 files paying only the engine
# cold start: compose-manager's `up --remove-orphans` removes engines a new file does
# not define, so a new file's downloader cannot pre-stage while the old engines serve.
REQUIRED_DOWNLOADS = [
  "hf download zai-org/GLM-5.3-Flash --revision 84c6a6aa9497188e15a635ba793b0f95a79b1033",
  "hf download zai-org/GLM-5.3-Flash chat_template.jinja --revision 3f1971b7b5f7a528c9c4ef6212c8785298a8c24a",
  "hf download graphistry/GLM-5.3-Flash-W4AFP8 --revision 99f1fa70408c52b007d4fd69e02e5a522422e755",
].freeze


def yaml_load(content)
  YAML.load(content, aliases: true)
rescue ArgumentError
  YAML.load(content)
end

# Reads and parses a top-level compose file. A missing file, invalid YAML, or
# a document that isn't a mapping is reported as one error and returns nil
# instead of raising, so the caller can skip the checks that depend on it.
def load_compose_file(errors, label, path)
  content = File.read(path)
  document = yaml_load(content)
  unless document.is_a?(Hash)
    errors << "#{label} must be a YAML mapping"
    return nil
  end
  document
rescue Errno::ENOENT
  errors << "#{label} not found at #{path}"
  nil
rescue Psych::SyntaxError => error
  errors << "#{label} is not valid YAML: #{error.message}"
  nil
end

# Same contract as load_compose_file, for a YAML document embedded as a
# string field inside a compose file (e.g. a `content:` block).
def load_embedded_yaml(errors, label, content)
  if content.nil?
    errors << "#{label} is missing"
    return nil
  end

  document = yaml_load(content)
  unless document.is_a?(Hash)
    errors << "#{label} must be a YAML mapping"
    return nil
  end
  document
rescue Psych::SyntaxError => error
  errors << "#{label} is not valid YAML: #{error.message}"
  nil
end

# Finds one Prometheus scrape job by name inside a parsed otelcol collector
# config. A nil collector, a non-mapping scrape entry, or a missing job is
# reported as an error and returns nil instead of raising.
def scrape_job(errors, label, collector, job_name)
  return nil if collector.nil?

  scrape_configs = collector.dig("receivers", "prometheus/apps", "config", "scrape_configs")
  job = Array(scrape_configs).find { |entry| entry.is_a?(Hash) && entry["job_name"] == job_name }
  errors << "missing #{job_name} scrape job in #{label} collector config" if job.nil?
  job
end

def environment_map(service)
  environment = service["environment"]
  return {} if environment.nil?
  return environment.transform_values(&:to_s) if environment.is_a?(Hash)

  Array(environment).to_h do |entry|
    key, value = entry.to_s.split("=", 2)
    [key, value]
  end
end

def command_text(service)
  command = service["command"]
  command.is_a?(Array) ? command.join("\n") : command.to_s
end

def log_config_variant_tags(metadata)
  Array(JSON.parse(metadata.to_s)).flat_map do |entry|
    next [] unless entry.is_a?(Hash)

    Array(entry["tags"]).select { |tag| tag.to_s.start_with?("config_variant:") }
  end
rescue JSON::ParserError
  []
end

def validate_command(errors, name, command)
  arguments = Shellwords.split(command)
  errors << "#{name} command must start with sglang serve" unless arguments.first(2) == %w[sglang serve]

  REQUIRED_OPTIONS.each do |option, expected_value|
    positions = arguments.each_index.select { |index| arguments[index] == option }
    if positions.length != 1
      errors << "#{name} command must contain #{option} exactly once"
      next
    end

    actual_value = arguments[positions.first + 1]
    errors << "#{name} #{option} must be #{expected_value.inspect}, got #{actual_value.inspect}" unless actual_value == expected_value
  end

  REQUIRED_SWITCHES.each do |option|
    count = arguments.count(option)
    errors << "#{name} command must contain #{option} exactly once" unless count == 1
  end

  FORBIDDEN_OPTIONS.each do |option|
    errors << "#{name} must not set #{option}; the inference-proxy assigns every request's priority" if arguments.include?(option)
  end
rescue ArgumentError => error
  errors << "#{name} command cannot be parsed: #{error.message}"
end

# The shared model cache: every pinned snapshot is downloaded exactly once and nothing
# else is, and the maintenance-only hf-cleanup service stays behind its profile with no
# default MODEL_NAME, so no normal apply can evict a snapshot an engine depends on.
def validate_model_cache(errors, label, services)
  downloader = services["model-downloader"]
  if downloader
    downloads = command_text(downloader).scan(/hf download [^\n]*/).map(&:strip)
    REQUIRED_DOWNLOADS.each do |download|
      errors << "#{label} model-downloader must run `#{download}` exactly once" unless downloads.count(download) == 1
    end
    unexpected = downloads - REQUIRED_DOWNLOADS
    errors << "#{label} model-downloader has unexpected downloads: #{unexpected.join('; ')}" unless unexpected.empty?
  end

  cleanup = services["hf-cleanup"]
  return unless cleanup

  errors << "#{label} hf-cleanup must stay behind the maintenance profile" unless Array(cleanup["profiles"]) == ["maintenance"]
  errors << "#{label} hf-cleanup must not default MODEL_NAME" unless environment_map(cleanup)["MODEL_NAME"] == "${MODEL_NAME:-}"
end

def runtime_contract(service)
  service.reject { |key, _value| REPLICA_IDENTITY_FIELDS.include?(key) }
end

# Checks shared by the canonical, HiCache, and long-context files: expected service
# set, per-replica serving contract (excluding image, which differs by file),
# GPU device assignment, the perception-check image and the proxy contract.
# Returns the replica services found, keyed by name.
def validate_common(errors, label, services, required_env = REQUIRED_ENV)
  missing_services = EXPECTED_SERVICES - services.keys
  extra_services = services.keys - EXPECTED_SERVICES
  errors << "#{label} is missing services: #{missing_services.join(', ')}" unless missing_services.empty?
  errors << "#{label} has unexpected services: #{extra_services.join(', ')}" unless extra_services.empty?

  replica_services = {}
  REPLICAS.each do |name, expected_devices|
    service = services[name]
    if service.nil?
      errors << "#{label} missing services.#{name}"
      next
    end

    replica_services[name] = service
    errors << "#{label} #{name} must use the prebuilt signed image, not a host-local build" if service.key?("build")
    validate_command(errors, "#{label} #{name}", command_text(service))

    replica_env = environment_map(service)
    required_env.each do |key, expected_value|
      errors << "#{label} #{name} must set #{key}=#{expected_value}" unless replica_env[key] == expected_value
    end

    device_ids = service.dig("deploy", "resources", "reservations", "devices", 0, "device_ids")
    normalized_ids = Array(device_ids).map(&:to_s)
    errors << "#{label} #{name} must use GPU device_ids #{expected_devices.join(',')}" unless normalized_ids == expected_devices
  end

  services.each do |name, service|
    service_env = environment_map(service)
    FORBIDDEN_ENV.each do |key|
      next unless service_env.key?(key)

      reason = if key == "SGLANG_ADMISSION_RESERVE_MIN_WAIT_S"
                 "not rank-deterministic, crashes the engine"
               else
                 "unmeasured/debug-only in production"
               end
      errors << "#{label} #{name} must not set #{key} (#{reason})"
    end
  end

  perception_check = services["glm53-perception-check"]
  if perception_check
    errors << "#{label} glm53-perception-check image must be #{ENGINE_IMAGE}" unless perception_check["image"] == ENGINE_IMAGE
    errors << "#{label} glm53-perception-check must use the prebuilt signed image, not a host-local build" if perception_check.key?("build")
  end

  proxy = services["proxy-glm53"]
  if proxy.nil?
    errors << "#{label} missing services.proxy-glm53"
  else
    errors << "#{label} proxy-glm53 image must be #{PROXY_IMAGE}" unless proxy["image"] == PROXY_IMAGE
    proxy_env = environment_map(proxy)
    expected_backends = REPLICAS.keys.map { |name| "http://#{name}:8000" }.join(",")
    errors << "#{label} proxy-glm53 must target both canonical replicas" unless proxy_env["VLLM_BACKEND_URLS"] == expected_backends
    errors << "#{label} proxy-glm53 must enable conversation affinity" unless proxy_env["VLLM_BACKEND_CONVERSATION_AFFINITY"] == "1"

    priority_enabled = replica_services.values.any? do |service|
      arguments = Shellwords.split(command_text(service)) rescue []
      PRIORITY_SWITCHES.any? { |switch| arguments.include?(switch) }
    end
    if priority_enabled && !PRIORITY_NORMALIZING_PROXY_IMAGES.include?(proxy["image"])
      errors << "#{label} enables SGLang priority scheduling but proxy-glm53 image #{proxy['image'].inspect} is not a priority-normalizing inference-proxy build (expected one of: #{PRIORITY_NORMALIZING_PROXY_IMAGES.join(', ')})"
    end
  end

  replica_services
end

# Asserts that `service_name`'s telemetry (the nearai.otel.config_variant
# label, the config_variant: tag in its com.datadoghq.ad.logs metadata, and
# its Prometheus scrape job's config_variant label) all carry
# `expected_variant`. A missing labels hash, a missing scrape job, or a
# missing variant anywhere is an explicit error, never a silent skip.
def check_variant(errors, file_label, service, service_name, collector, expected_variant)
  labels = service["labels"]
  if labels.is_a?(Hash)
    metric_variant = labels["nearai.otel.config_variant"]
    errors << "#{file_label} #{service_name} nearai.otel.config_variant must be #{expected_variant}, got #{metric_variant.inspect}" unless metric_variant == expected_variant
    log_tag = labels["com.datadoghq.ad.logs"]
    expected_log_tag = "config_variant:#{expected_variant}"
    actual_log_tags = log_config_variant_tags(log_tag)
    unless actual_log_tags == [expected_log_tag]
      errors << "#{file_label} #{service_name} log metadata must carry exactly #{expected_log_tag}, got #{actual_log_tags.inspect}"
    end
  else
    errors << "#{file_label} #{service_name} is missing labels"
  end

  job_name = "sglang-#{service_name}"
  scrape = scrape_job(errors, file_label, collector, job_name)
  return unless scrape

  scrape_variant = scrape.dig("static_configs", 0, "labels", "config_variant")
  errors << "#{file_label} #{job_name} scrape label config_variant must be #{expected_variant}, got #{scrape_variant.inspect}" unless scrape_variant == expected_variant
end

# Canonical file: both replicas run the plain engine image with an identical
# runtime contract, no service anywhere carries a HiCache flag or env var,
# and both replicas' telemetry is pinned to OFFICIAL_VARIANT.
def validate_canonical(errors, compose, services, replica_services)
  REPLICAS.each_key do |name|
    service = replica_services[name]
    next unless service

    errors << "canonical #{name} image must be #{ENGINE_IMAGE}" unless service["image"] == ENGINE_IMAGE
  end

  if replica_services.length == REPLICAS.length
    runtime_contracts = replica_services.values.map { |service| runtime_contract(service) }
    errors << "canonical GLM-5.3 replicas must use identical runtime configuration" unless runtime_contracts.uniq.length == 1
  end

  services.each do |name, service|
    text = command_text(service)
    if text.include?("--enable-hierarchical-cache") || text.include?("--hicache")
      errors << "canonical #{name} must not enable HiCache; use prod/GLM-5.3-Flash-SGL-TP4-HiCache.yaml for the canary instead"
    end
    hicache_env = environment_map(service).keys.select { |key| key.start_with?("SGLANG_HICACHE_") }
    unless hicache_env.empty?
      errors << "canonical #{name} must not set #{hicache_env.join(', ')}; use prod/GLM-5.3-Flash-SGL-TP4-HiCache.yaml for the canary instead"
    end
  end

  collector = load_embedded_yaml(errors, "canonical file otelcol_app_config", compose.dig("configs", "otelcol_app_config", "content"))
  REPLICAS.each_key do |name|
    service = replica_services[name]
    next unless service

    check_variant(errors, "canonical", service, name, collector, OFFICIAL_VARIANT)
  end
end

# Opt-in observability (docker/sglang-glm53-hicache-w4afp8 v6: ghost-prefix-cache.diff and
# kv-tier-metrics.diff), required on every replica of the W4AFP8 long-context and 4x TP2 files:
# the ghost prefix cache in shared mode (one key file and one aggregator socket per CVM on the
# in-memory ghost volume, a distinct replica name per engine), the KV tier metrics, and one
# glm53-ghost-aggregator sidecar per CVM on the engines' image, scraped like the engines.
# scripts/glm53_observability.py renders the same values; change them together.
GHOST_SERVICE = "glm53-ghost-aggregator"
GHOST_VOLUME = "ghost"
GHOST_MOUNT = "#{GHOST_VOLUME}:/ghost"
GHOST_KEY_FILE = "/ghost/key"
GHOST_SOCKET = "/ghost/aggregator.sock"
GHOST_SAMPLE = "16"
GHOST_PORT = "9464"
GHOST_JOB = "ghost-aggregator-#{GHOST_SERVICE}"
GHOST_AGGREGATOR_ARGV = [
  "python3", "-m", "sglang.srt.observability.ghost_aggregator",
  "--socket", GHOST_SOCKET, "--port", GHOST_PORT, "--sample", GHOST_SAMPLE, "--model-name", "z-ai/glm-5.3-flash",
].freeze
OBSERVABILITY_VARIANT_SUFFIX = "-obs-v1"

# v8 bundle canary slots (docs/glm53-v8-bundle-canary.md). The base file and the long-context file are each
# deployed by two hosts, so neither may hardcode the bundle: exactly one replica per file reads per-replica
# override variables (GLM53_V8_<slot>_*) that default to today's prod value, or to nothing, and are set only in the
# canary host's compose-manager env map. These helpers pin that contract: the variables exist only on the slot,
# only as ${NAME:-<today's value>} with the exact counts below, and the bundle's flags/environment/image never
# appear as literals anywhere in the file.
V8_PLACEHOLDERS = ["REPLACE_WITH_V8", "<tbd>"].freeze
V8_LITERALS = %w[--disable-overlap-schedule SGLANG_PREPROCESS_ SGLANG_TOOL_SCHEMA NEAR_SELF_PROFILE fp8_e4m3 flashmla_kv].freeze

def v8_expr(prefix, name, default)
  "${#{prefix}#{name}:-#{default}}"
end

# What an unset env map renders: every ${GLM53_V8_*:-default} becomes its default.
def v8_resolve(text)
  text.to_s.gsub(/\$\{GLM53_V8_[A-Z0-9_]+:-([^}]*)\}/) { Regexp.last_match(1) }
end

def v8_body(raw)
  raw.lines.reject { |line| line.lstrip.start_with?("#") }.join
end

# slot = { "service", "prefix", "variables" => { NAME => [expected count in the file, default] } }
def validate_v8_slot(errors, label, compose, raw, slot)
  body = v8_body(raw)
  prefix = slot["prefix"]
  expected = slot["variables"].to_h { |name, (count, default)| [v8_expr(prefix, name, default), count] }
  found = body.scan(/\$\{GLM53_V8_[A-Z0-9_]+[^}]*\}/).group_by(&:itself).transform_values(&:length)
  found.each do |expression, count|
    if !expected.key?(expression)
      errors << "#{label} #{expression} is not a v8 slot variable with its pinned default (each must default to today's value, or to empty for the flag, environment and suffix variables)"
    elsif expected[expression] != count
      errors << "#{label} #{expression} must appear exactly #{expected[expression]} time(s) outside comments, found #{count}"
    end
  end
  expected.each_key { |expression| errors << "#{label} is missing #{expression}" unless found.key?(expression) }
  stray = body.scan("GLM53_V8_").length - found.values.sum
  errors << "#{label} references GLM53_V8_ outside a ${NAME:-default} expression" unless stray.zero?
  V8_PLACEHOLDERS.each { |placeholder| errors << "#{label} contains the v8 placeholder #{placeholder}; the image digest and depth belong only in the canary host's env map" if raw.include?(placeholder) }
  V8_LITERALS.each do |literal|
    errors << "#{label} must not hardcode #{literal} outside comments; the bundle arrives only through the #{prefix}* variables of #{slot['service']}" if body.include?(literal)
  end
  compose.fetch("services", {}).each do |name, service|
    next if name == slot["service"]

    errors << "#{label} #{name} must not reference GLM53_V8_ variables; only #{slot['service']} does" if JSON.generate(service).include?("GLM53_V8_")
  end
  compose.each do |key, value|
    next unless key.to_s.start_with?("x-")

    errors << "#{label} #{key} must not reference GLM53_V8_ variables or the bundle's flags" if JSON.generate(value).match?(/GLM53_V8_|#{Regexp.union(V8_LITERALS)}/)
  end
end

def observability_env(replica_label)
  {
    "SGLANG_GHOST_CACHE" => "1",
    "SGLANG_GHOST_CACHE_SAMPLE" => GHOST_SAMPLE,
    "SGLANG_GHOST_CACHE_KEY_FILE" => GHOST_KEY_FILE,
    "SGLANG_GHOST_CACHE_SOCKET" => GHOST_SOCKET,
    "SGLANG_GHOST_CACHE_REPLICA" => replica_label,
    "SGLANG_KV_TIER_METRICS" => "1",
  }
end

def argv_of(value)
  value.is_a?(Array) ? value.map(&:to_s) : Shellwords.split(value.to_s)
rescue ArgumentError
  []
end

# `replicas` maps each engine service name to its ghost-cache replica name. Every engine must
# carry the full observability environment and mount the ghost volume; the CVM's engines must
# agree on one key file, one socket and one sample rate, with distinct replica names; the sidecar
# must read that socket at that sample rate on the engines' image, and be scraped.
def validate_observability(errors, label, compose, collector, replicas, engine_image, deployment)
  services = compose.fetch("services", {})
  replicas.each do |name, replica_label|
    service = services[name]
    next unless service

    env = environment_map(service)
    observability_env(replica_label).each do |key, value|
      next if env[key] == value

      errors << "#{label} #{name} must set #{key}=#{value} (opt-in observability is required on every replica), got #{env[key].inspect}"
    end
    errors << "#{label} #{name} must mount #{GHOST_MOUNT}" unless Array(service["volumes"]).map(&:to_s).include?(GHOST_MOUNT)
  end
  present = replicas.keys.select { |name| services[name] }
  %w[SGLANG_GHOST_CACHE_KEY_FILE SGLANG_GHOST_CACHE_SOCKET SGLANG_GHOST_CACHE_SAMPLE].each do |key|
    values = present.map { |name| environment_map(services[name])[key] }.uniq
    errors << "#{label} replicas must share one #{key}, got #{values.inspect}" unless values.length <= 1
  end
  names = present.map { |name| environment_map(services[name])["SGLANG_GHOST_CACHE_REPLICA"] }
  errors << "#{label} replicas must use distinct SGLANG_GHOST_CACHE_REPLICA names, got #{names.inspect}" unless names.uniq.length == names.length

  sidecar = services[GHOST_SERVICE]
  if sidecar.nil?
    errors << "#{label} is missing the #{GHOST_SERVICE} sidecar"
  else
    errors << "#{label} #{GHOST_SERVICE} image must be the engines' image #{engine_image}" unless sidecar["image"] == engine_image
    errors << "#{label} #{GHOST_SERVICE} must use the prebuilt signed image, not a host-local build" if sidecar.key?("build")
    argv = argv_of(sidecar["entrypoint"]) + argv_of(sidecar["command"])
    errors << "#{label} #{GHOST_SERVICE} must run #{GHOST_AGGREGATOR_ARGV.join(' ')}, got #{argv.join(' ')}" unless argv == GHOST_AGGREGATOR_ARGV
    errors << "#{label} #{GHOST_SERVICE} must mount only #{GHOST_MOUNT}" unless Array(sidecar["volumes"]).map(&:to_s) == [GHOST_MOUNT]
    errors << "#{label} #{GHOST_SERVICE} must not reserve GPUs" if sidecar.key?("deploy")
    errors << "#{label} #{GHOST_SERVICE} must run under runc" unless sidecar["runtime"] == "runc"
    errors << "#{label} #{GHOST_SERVICE} must not publish ports" if sidecar.key?("ports")
    labels = sidecar["labels"].is_a?(Hash) ? sidecar["labels"] : {}
    { "nearai.otel.scrape" => "true", "nearai.otel.port" => GHOST_PORT, "nearai.otel.path" => "/metrics",
      "nearai.otel.container_name" => GHOST_SERVICE, "nearai.otel.deployment" => deployment }.each do |key, value|
      errors << "#{label} #{GHOST_SERVICE} #{key} must be #{value.inspect}, got #{labels[key].inspect}" unless labels[key] == value
    end
  end

  volume = compose.dig("volumes", GHOST_VOLUME)
  unless volume.is_a?(Hash) && volume.dig("driver_opts", "type") == "tmpfs"
    errors << "#{label} must declare the #{GHOST_VOLUME} volume as tmpfs (the shared key must never touch disk)"
  end

  scrape = scrape_job(errors, label, collector, GHOST_JOB)
  return unless scrape

  errors << "#{label} #{GHOST_JOB} must scrape #{GHOST_SERVICE}:#{GHOST_PORT}" unless scrape.dig("static_configs", 0, "targets") == ["#{GHOST_SERVICE}:#{GHOST_PORT}"]
  scrape_labels = scrape.dig("static_configs", 0, "labels") || {}
  { "container_name" => GHOST_SERVICE, "host" => "${CVM_HOST}", "deployment" => deployment }.each do |key, value|
    errors << "#{label} #{GHOST_JOB} scrape label #{key} must be #{value.inspect}, got #{scrape_labels[key].inspect}" unless scrape_labels[key] == value
  end
end

# Removes the observability additions (sidecar, its volume and scrape job) from a file view
# compared with a source file that predates them; validate_observability asserts them.
def strip_observability(view, collector)
  view.fetch("services", {}).delete(GHOST_SERVICE)
  view["volumes"]&.delete(GHOST_VOLUME)
  return unless collector

  Array(collector.dig("receivers", "prometheus/apps", "config", "scrape_configs")).reject! do |job|
    job.is_a?(Hash) && job["job_name"] == GHOST_JOB
  end
end

# W4AFP8 base file (gpu03, gpu04, gpu23): both replicas run gpu31 campaign-2 arm B5,
# exactly the argv gpu02's W4AFP8 r1 runs, on the w4afp8 image with the canonical engine
# environment (admission reserve on, no HiCache). Outside the two engines and their
# truthful telemetry it must equal the canonical file, and the two replicas must share
# one runtime configuration.
W4AFP8_BASE_FILE = File.join(ROOT, "prod", "GLM-5.3-Flash-SGL-TP4-W4AFP8.yaml")
W4AFP8_BASE_IMAGE = "docker.io/nearaidev/sglang@sha256:8bce6a7cc872a80faded3bd1ef0a64873a1d7abae34c94e5358775ca21f133cc"
W4AFP8_BASE_VARIANT = "fc91d24-w4afp8-c4096-admission-reserve-v10-pool-clamp-pdi1-h200-tp4-ep4-eagle-adaptive-5-1-6-strict-budget8192"
W4AFP8_BASE_CHECKPOINT = "graphistry/GLM-5.3-Flash-W4AFP8"
W4AFP8_BASE_PRECISION = "int4-weights-fp8-activations-bf16-kv"
W4AFP8_BASE_REPLICAS = {
  "model-sg-glm53-w4afp8-tp4-r1" => { "devices" => %w[0 1 2 3], "instance" => "1" },
  "model-sg-glm53-w4afp8-tp4-r2" => { "devices" => %w[4 5 6 7], "instance" => "2" },
}.freeze
W4AFP8_BASE_ARGV = Shellwords.split(<<~'ARGV').freeze
  sglang serve
  --model-path /root/.cache/huggingface/hub/models--graphistry--GLM-5.3-Flash-W4AFP8/snapshots/99f1fa70408c52b007d4fd69e02e5a522422e755
  --served-model-name z-ai/glm-5.3-flash
  --tp-size 4 --ep-size 4
  --mem-fraction-static 0.80
  --max-running-requests 32 --max-queued-requests 8
  --enable-priority-scheduling --disable-priority-preemption
  --chunked-prefill-size 4096 --max-prefill-tokens 32768 --prefill-decode-interval 1
  --cuda-graph-max-bs-decode 32
  --dsa-prefill-backend tilelang --dsa-decode-backend tilelang
  --kv-cache-dtype bfloat16
  --speculative-algorithm EAGLE --speculative-num-steps 5 --speculative-eagle-topk 1
  --speculative-num-draft-tokens 6 --speculative-adaptive
  --reasoning-parser glm45 --enable-strict-thinking --grammar-backend xgrammar --tool-call-parser glm47
  --chat-template /root/.cache/huggingface/hub/models--zai-org--GLM-5.3-Flash/snapshots/3f1971b7b5f7a528c9c4ef6212c8785298a8c24a/chat_template.jinja
  --context-length 1048576
  --dist-init-addr 127.0.0.1:29510
  --watchdog-timeout 1800 --host 0.0.0.0 --port 8000
  --enable-metrics --enable-cache-report --log-requests-level 0
  --disable-fast-image-processor --limit-mm-data-per-request '{"image": 64}'
ARGV

# The canonical file and the W4AFP8 base file, reduced to what must be identical:
# engines, the engine anchor and the replicas' scrape jobs removed, replica names and
# the checkpoint behind the telemetry normalized.
def w4afp8_base_view(errors, file_label, compose, replica_names)
  view = Marshal.load(Marshal.dump(compose))
  view.delete("x-sg-glm53-flash-common")
  replica_names.each { |name| view.fetch("services", {}).delete(name) }
  otel = view.dig("configs", "otelcol_app_config")
  if otel && otel["content"]
    collector = load_embedded_yaml(errors, "#{file_label} otelcol_app_config", otel["content"])
    if collector
      Array(collector.dig("receivers", "prometheus/apps", "config", "scrape_configs")).reject! do |job|
        job.is_a?(Hash) && replica_names.any? { |name| job["job_name"] == "sglang-#{name}" }
      end
      otel["content"] = collector
    end
  end
  normalized = JSON.generate(view)
                   .gsub("model-sg-glm53-w4afp8-tp4-r", "REPLICA-r").gsub("model-sg-glm53-fp8-tp4-r", "REPLICA-r")
                   .gsub(W4AFP8_BASE_CHECKPOINT, "CHECKPOINT").gsub("zai-org/GLM-5.3-Flash", "CHECKPOINT")
  JSON.parse(normalized)
end

def validate_w4afp8_base(errors, compose, canonical)
  label = "W4AFP8 base"
  services = compose.fetch("services", {})
  expected_services = EXPECTED_SERVICES - REPLICAS.keys + W4AFP8_BASE_REPLICAS.keys
  missing = expected_services - services.keys
  extra = services.keys - expected_services
  errors << "#{label} is missing services: #{missing.join(', ')}" unless missing.empty?
  errors << "#{label} has unexpected services: #{extra.join(', ')}" unless extra.empty?

  expected_env = environment_map(canonical.dig("services", "model-sg-glm53-fp8-tp4-r1") || {})
  collector = load_embedded_yaml(errors, "#{label} file otelcol_app_config", compose.dig("configs", "otelcol_app_config", "content"))
  engine_image_label = W4AFP8_BASE_IMAGE.split(":").last[0, 12]
  replicas = {}
  W4AFP8_BASE_REPLICAS.each do |name, spec|
    service = services[name]
    next errors << "#{label} missing services.#{name}" if service.nil?

    replicas[name] = service
    errors << "#{label} #{name} image must be #{W4AFP8_BASE_IMAGE}" unless service["image"] == W4AFP8_BASE_IMAGE
    errors << "#{label} #{name} must use the prebuilt signed image, not a host-local build" if service.key?("build")
    actual_argv = begin
      Shellwords.split(command_text(service))
    rescue ArgumentError => error
      errors << "#{label} #{name} command cannot be parsed: #{error.message}"
      []
    end
    unless actual_argv == W4AFP8_BASE_ARGV
      drift = ((actual_argv - W4AFP8_BASE_ARGV) + (W4AFP8_BASE_ARGV - actual_argv)).uniq
      errors << "#{label} #{name} argv must be campaign-2 arm B5 exactly; differing tokens: #{drift.first(8).join(' ')}"
    end
    env = environment_map(service)
    REQUIRED_ENV.each do |key, value|
      errors << "#{label} #{name} must set #{key}=#{value}" unless env[key] == value
    end
    hicache = env.keys.select { |key| key.start_with?("SGLANG_HICACHE_") }
    errors << "#{label} #{name} must not set #{hicache.join(', ')}" unless hicache.empty?
    (env.keys & FORBIDDEN_ENV).each { |key| errors << "#{label} #{name} must not set #{key}" }
    unless env == expected_env
      diff = (env.to_a - expected_env.to_a) + (expected_env.to_a - env.to_a)
      errors << "#{label} #{name} environment must equal the canonical engine environment; differing: #{diff.map { |key, value| "#{key}=#{value}" }.uniq.join(' ')}"
    end
    device_ids = Array(service.dig("deploy", "resources", "reservations", "devices", 0, "device_ids")).map(&:to_s)
    errors << "#{label} #{name} must use GPU device_ids #{spec['devices'].join(',')}" unless device_ids == spec["devices"]

    labels = service["labels"].is_a?(Hash) ? service["labels"] : {}
    { "nearai.otel.model_path" => W4AFP8_BASE_CHECKPOINT, "nearai.otel.engine_image" => engine_image_label, "nearai.otel.instance" => spec["instance"] }.each do |key, value|
      errors << "#{label} #{name} #{key} must be #{value.inspect}, got #{labels[key].inspect}" unless labels[key] == value
    end
    tags = begin
      Array(JSON.parse(labels["com.datadoghq.ad.logs"].to_s).first&.fetch("tags", []))
    rescue JSON::ParserError
      []
    end
    ["model_path:#{W4AFP8_BASE_CHECKPOINT}", "precision:#{W4AFP8_BASE_PRECISION}", "engine_image:#{engine_image_label}", "instance:#{spec['instance']}"].each do |tag|
      errors << "#{label} #{name} log metadata must carry #{tag}" unless tags.include?(tag)
    end
    check_variant(errors, label, service, name, collector, W4AFP8_BASE_VARIANT)
    scrape = scrape_job(errors, label, collector, "sglang-#{name}")
    scrape_labels = scrape&.dig("static_configs", 0, "labels") || {}
    { "model_path" => W4AFP8_BASE_CHECKPOINT, "precision" => W4AFP8_BASE_PRECISION, "engine_image" => engine_image_label, "instance" => spec["instance"] }.each do |key, value|
      errors << "#{label} sglang-#{name} scrape label #{key} must be #{value.inspect}, got #{scrape_labels[key].inspect}" if scrape && scrape_labels[key] != value
    end
  end

  if replicas.length == 2
    contracts = replicas.values.map { |service| runtime_contract(service) }
    errors << "#{label} replicas must use identical runtime configuration" unless contracts.uniq.length == 1
  end

  dcgm_labels = services.dig("dcgm-glm53", "labels") || {}
  errors << "#{label} dcgm-glm53 nearai.otel.model_path must be #{W4AFP8_BASE_CHECKPOINT}" unless dcgm_labels["nearai.otel.model_path"] == W4AFP8_BASE_CHECKPOINT
  errors << "#{label} dcgm-glm53 log metadata must carry model_path:#{W4AFP8_BASE_CHECKPOINT}" unless dcgm_labels["com.datadoghq.ad.logs"].to_s.include?("model_path:#{W4AFP8_BASE_CHECKPOINT}")

  proxy = services["proxy-glm53"] || {}
  proxy_env = environment_map(proxy)
  expected_backends = W4AFP8_BASE_REPLICAS.keys.map { |name| "http://#{name}:8000" }.join(",")
  errors << "#{label} proxy-glm53 must pool both W4AFP8 replicas" unless proxy_env["VLLM_BACKEND_URLS"] == expected_backends
  errors << "#{label} proxy-glm53 must enable conversation affinity" unless proxy_env["VLLM_BACKEND_CONVERSATION_AFFINITY"] == "1"
  unless PRIORITY_NORMALIZING_PROXY_IMAGES.include?(proxy["image"])
    errors << "#{label} enables SGLang priority scheduling but proxy-glm53 image #{proxy['image'].inspect} is not a priority-normalizing inference-proxy build"
  end

  canonical_view = w4afp8_base_view(errors, "canonical file", canonical, REPLICAS.keys)
  target_view = w4afp8_base_view(errors, "#{label} file", compose, W4AFP8_BASE_REPLICAS.keys)
  return if canonical_view == target_view

  difference = first_difference(canonical_view, target_view)
  errors << "#{label} file must match the canonical file outside the two engines and their telemetry (first difference: #{difference})"
end

# 4x TP2 base-tier canary (generated from the W4AFP8 base file): four TP2/EP2 replicas, one
# per NVLink GPU pair, on the HiCache + W4AFP8 image the long tier runs, with exactly the
# lab-qualified TP2 argv, the base engine environment plus a 325 GiB HiCache host tier and the
# DSA indexer split. Outside the engines, their telemetry, the four-way fan-out (proxy pool,
# perception loop, soak relay) and the deployment label it must equal the W4AFP8 base file.
W4AFP8_TP2X4_FILE = File.join(ROOT, "prod", "GLM-5.3-Flash-SGL-TP2x4-W4AFP8.yaml")
W4AFP8_TP2X4_IMAGE = "docker.io/nearaidev/sglang@sha256:9c6ddd4319c4ab00e351d8650459e68b8830e36ffcc029d67fa5e19d0ac3ed17"
W4AFP8_TP2X4_VARIANT = "hicache-w4afp8-qsplit-selective325-mamba165-bf16state-c8192-admission-reserve-v10-pdi1-h200-tp2-ep2-eagle-adaptive-5-1-6-strict-budget8192#{OBSERVABILITY_VARIANT_SUFFIX}"
W4AFP8_TP2X4_DEPLOYMENT = "glm53-flash-sgl-tp2x4"
W4AFP8_BASE_DEPLOYMENT = "glm53-flash-sgl-tp4"
W4AFP8_TP2X4_PREFIX = "model-sg-glm53-w4afp8-tp2-r"
# All four replicas run the memory-optimized argv (tee-bench exp 25/25b/25c), promoted from the
# r3/r4 canary after the gpu03 same-host bake (2026-10-06). The "control" role (the previous prod
# argv, W4AFP8_TP2X4_ARGV) is kept for reverting a replica and is what the candidate edits derive from.
# r4 is also the v8 bundle canary slot ("v8-slot": the candidate behind per-replica override variables).
W4AFP8_TP2X4_REPLICAS = {
  "#{W4AFP8_TP2X4_PREFIX}1" => { "devices" => %w[0 1], "instance" => "1", "soak_port" => "8008", "role" => "candidate", "ghost_replica" => "r1" },
  "#{W4AFP8_TP2X4_PREFIX}2" => { "devices" => %w[2 3], "instance" => "2", "soak_port" => "8009", "role" => "candidate", "ghost_replica" => "r2" },
  "#{W4AFP8_TP2X4_PREFIX}3" => { "devices" => %w[4 5], "instance" => "3", "soak_port" => "8010", "role" => "candidate", "ghost_replica" => "r3" },
  "#{W4AFP8_TP2X4_PREFIX}4" => { "devices" => %w[6 7], "instance" => "4", "soak_port" => "8011", "role" => "v8-slot", "ghost_replica" => "r4" },
}.freeze
W4AFP8_TP2X4_CANDIDATE_ANCHOR = "x-sg-glm53-flash-candidate"
W4AFP8_TP2X4_CANDIDATE_PDI = "2"
W4AFP8_TP2X4_CANDIDATE_VARIANT = "hicache-w4afp8-qsplit-selective325-mamba330-bf16state-memopt086-mr48-c8192-admission-reserve-v10-pdi#{W4AFP8_TP2X4_CANDIDATE_PDI}-h200-tp2-ep2-eagle-fixed-4-1-5-strict-budget8192#{OBSERVABILITY_VARIANT_SUFFIX}"
W4AFP8_TP2X4_ARGV = Shellwords.split(<<~'ARGV').freeze
  sglang serve
  --model-path /root/.cache/huggingface/hub/models--graphistry--GLM-5.3-Flash-W4AFP8/snapshots/99f1fa70408c52b007d4fd69e02e5a522422e755
  --served-model-name z-ai/glm-5.3-flash
  --tp-size 2 --ep-size 2
  --mem-fraction-static 0.80
  --max-running-requests 32 --max-queued-requests 8
  --enable-priority-scheduling --disable-priority-preemption
  --chunked-prefill-size 8192 --max-prefill-tokens 32768 --prefill-decode-interval 1
  --cuda-graph-max-bs-decode 32
  --dsa-prefill-backend tilelang --dsa-decode-backend tilelang
  --kv-cache-dtype bfloat16
  --speculative-algorithm EAGLE --speculative-num-steps 5 --speculative-eagle-topk 1
  --speculative-num-draft-tokens 6 --speculative-adaptive
  --reasoning-parser glm45 --enable-strict-thinking --grammar-backend xgrammar --tool-call-parser glm47
  --chat-template /root/.cache/huggingface/hub/models--zai-org--GLM-5.3-Flash/snapshots/3f1971b7b5f7a528c9c4ef6212c8785298a8c24a/chat_template.jinja
  --context-length 1048576
  --dist-init-addr 127.0.0.1:29510
  --watchdog-timeout 1800 --host 0.0.0.0 --port 8000
  --enable-metrics --enable-cache-report --log-requests-level 0
  --disable-fast-image-processor --limit-mm-data-per-request '{"image": 64}'
  --enable-hierarchical-cache --hicache-write-policy write_through_selective
  --hicache-io-backend direct --hicache-mem-layout page_first_direct
  --max-mamba-cache-size 165 --mamba-ssm-dtype bfloat16
ARGV
# The candidate argv is the control argv with exactly these token edits, minus --speculative-adaptive.
W4AFP8_TP2X4_CANDIDATE_EDITS = [
  ["--mem-fraction-static", "0.86"], ["--max-running-requests", "48"], ["--prefill-decode-interval", W4AFP8_TP2X4_CANDIDATE_PDI],
  ["--cuda-graph-max-bs-decode", "48"], ["--speculative-num-steps", "4"], ["--speculative-num-draft-tokens", "5"],
  ["--max-mamba-cache-size", "330"],
].freeze
W4AFP8_TP2X4_CANDIDATE_ARGV = begin
  argv = W4AFP8_TP2X4_ARGV.dup
  W4AFP8_TP2X4_CANDIDATE_EDITS.each { |flag, value| argv[argv.index(flag) + 1] = value }
  argv.delete("--speculative-adaptive")
  argv.freeze
end
# The v8 slot (r4): the candidate argv with the values the bundle changes behind variables (default = the candidate's
# value), the environment prefix first and the extra args last. Both are empty tokens by default.
W4AFP8_TP2X4_V8_PREFIX = "GLM53_V8_R4_"
W4AFP8_TP2X4_V8_VALUE_FLAGS = {
  "--kv-cache-dtype" => ["KV_DTYPE", "bfloat16"], "--dsa-prefill-backend" => ["DSA_BACKEND", "tilelang"],
  "--dsa-decode-backend" => ["DSA_BACKEND", "tilelang"], "--max-running-requests" => ["MAX_RUNNING", "48"],
  "--cuda-graph-max-bs-decode" => ["MAX_RUNNING", "48"], "--max-mamba-cache-size" => ["MAMBA_SLOTS", "330"],
}.freeze
W4AFP8_TP2X4_V8_ARGV = begin
  argv = W4AFP8_TP2X4_CANDIDATE_ARGV.dup
  W4AFP8_TP2X4_V8_VALUE_FLAGS.each { |flag, (name, default)| argv[argv.index(flag) + 1] = v8_expr(W4AFP8_TP2X4_V8_PREFIX, name, default) }
  ([v8_expr(W4AFP8_TP2X4_V8_PREFIX, "ENV_PREFIX", "")] + argv + [v8_expr(W4AFP8_TP2X4_V8_PREFIX, "EXTRA_ARGS", "")]).freeze
end
W4AFP8_TP2X4_V8_IMAGE = v8_expr(W4AFP8_TP2X4_V8_PREFIX, "IMAGE", W4AFP8_TP2X4_IMAGE)
W4AFP8_TP2X4_V8_VARIANT = W4AFP8_TP2X4_CANDIDATE_VARIANT + v8_expr(W4AFP8_TP2X4_V8_PREFIX, "VARIANT_SUFFIX", "")
W4AFP8_TP2X4_V8_SLOT = {
  "service" => "#{W4AFP8_TP2X4_PREFIX}4", "prefix" => W4AFP8_TP2X4_V8_PREFIX,
  "variables" => {
    "IMAGE" => [1, W4AFP8_TP2X4_IMAGE], "IMAGE_LABEL" => [3, W4AFP8_TP2X4_IMAGE.split(":").last[0, 12]],
    "PRECISION" => [2, "int4-weights-fp8-activations-bf16-kv"], "KV_DTYPE" => [1, "bfloat16"],
    "DSA_BACKEND" => [2, "tilelang"], "MAX_RUNNING" => [2, "48"], "MAMBA_SLOTS" => [1, "330"],
    "VARIANT_SUFFIX" => [3, ""], "EXTRA_ARGS" => [1, ""], "ENV_PREFIX" => [1, ""],
  },
}.freeze
# Every running request needs 5 mamba state slots (prod: 165 slots for 32 running; candidate: 330 for 48).
W4AFP8_TP2X4_MAMBA_SLOTS_PER_REQUEST = 5
W4AFP8_TP2X4_EXTRA_ENV = HICACHE_ENV.merge(
  "SGLANG_HICACHE_RAM_BUDGET" => "${GLM53_HICACHE_RAM_BUDGET:-325GiB}",
  "SGLANG_DSA_INDEXER_QSPLIT" => "1",
).freeze

# The W4AFP8 base file and the 4x TP2 file, reduced to what must be identical: engines, the
# engine anchor, the replicas' scrape jobs, the proxy pool, the perception command and the
# soak relay's ports/config removed (each asserted separately), names and labels normalized.
def w4afp8_tp2x4_view(errors, file_label, compose, replica_names)
  view = Marshal.load(Marshal.dump(compose))
  view.delete("x-sg-glm53-flash-common")
  view.delete(W4AFP8_TP2X4_CANDIDATE_ANCHOR)
  services = view.fetch("services", {})
  replica_names.each { |name| services.delete(name) }
  services["glm53-perception-check"]&.delete("command")
  services["glm53-soak-relay"]&.delete("ports")
  view.dig("configs", "glm53_soak_nginx_conf")&.delete("content")
  proxy = services["proxy-glm53"]
  proxy["environment"] = Array(proxy["environment"]).reject { |entry| entry.to_s.start_with?("VLLM_BACKEND_URLS=") } if proxy
  otel = view.dig("configs", "otelcol_app_config")
  collector = nil
  if otel && otel["content"]
    collector = load_embedded_yaml(errors, "#{file_label} otelcol_app_config", otel["content"])
    if collector
      Array(collector.dig("receivers", "prometheus/apps", "config", "scrape_configs")).reject! do |job|
        job.is_a?(Hash) && replica_names.any? { |name| job["job_name"] == "sglang-#{name}" }
      end
      otel["content"] = collector
    end
  end
  strip_observability(view, collector)
  JSON.parse(
    JSON.generate(view)
        .gsub(W4AFP8_TP2X4_DEPLOYMENT, W4AFP8_BASE_DEPLOYMENT)
        .gsub("(4x TP2/EP2)", "(2x TP4/EP4)"),
  )
end

def validate_w4afp8_tp2x4(errors, compose, base, raw)
  label = "W4AFP8 4x TP2"
  services = compose.fetch("services", {})
  expected_services = EXPECTED_SERVICES - REPLICAS.keys + W4AFP8_TP2X4_REPLICAS.keys + [GHOST_SERVICE]
  missing = expected_services - services.keys
  extra = services.keys - expected_services
  errors << "#{label} is missing services: #{missing.join(', ')}" unless missing.empty?
  errors << "#{label} has unexpected services: #{extra.join(', ')}" unless extra.empty?
  errors << "#{label} file must not reference any TP4 engine (tp4-r)" if raw.include?("tp4-r")
  errors << "#{label} file must not carry the #{W4AFP8_BASE_DEPLOYMENT} deployment label" if raw.include?("#{W4AFP8_BASE_DEPLOYMENT}\"")

  # No replica merges the common anchor directly any more, but it is the previous prod argv a
  # replica is reverted to and the base the candidate edits derive from, so it stays pinned.
  common_argv = begin
    Shellwords.split(command_text(compose["x-sg-glm53-flash-common"] || {}))
  rescue ArgumentError => error
    errors << "#{label} x-sg-glm53-flash-common command cannot be parsed: #{error.message}"
    []
  end
  unless common_argv == W4AFP8_TP2X4_ARGV
    drift = ((common_argv - W4AFP8_TP2X4_ARGV) + (W4AFP8_TP2X4_ARGV - common_argv)).uniq
    errors << "#{label} x-sg-glm53-flash-common argv must be the lab-qualified TP2 control argv exactly; differing tokens: #{drift.first(8).join(' ')}"
  end

  base_env = environment_map(base.dig("services", W4AFP8_BASE_REPLICAS.keys.first) || {}).merge(W4AFP8_TP2X4_EXTRA_ENV)
  collector = load_embedded_yaml(errors, "#{label} file otelcol_app_config", compose.dig("configs", "otelcol_app_config", "content"))
  engine_image_label = W4AFP8_TP2X4_IMAGE.split(":").last[0, 12]
  replicas = {}
  W4AFP8_TP2X4_REPLICAS.each do |name, spec|
    service = services[name]
    next errors << "#{label} missing services.#{name}" if service.nil?

    replicas[name] = service
    errors << "#{label} #{name} container_name must be #{name}" unless service["container_name"] == name
    v8_slot = spec["role"] == "v8-slot"
    expected_image = v8_slot ? W4AFP8_TP2X4_V8_IMAGE : W4AFP8_TP2X4_IMAGE
    errors << "#{label} #{name} image must be #{expected_image}" unless service["image"] == expected_image
    errors << "#{label} #{name} must use the prebuilt signed image, not a host-local build" if service.key?("build")
    actual_argv = begin
      Shellwords.split(command_text(service))
    rescue ArgumentError => error
      errors << "#{label} #{name} command cannot be parsed: #{error.message}"
      []
    end
    candidate = spec["role"] == "candidate" || v8_slot
    expected_argv = v8_slot ? W4AFP8_TP2X4_V8_ARGV : (candidate ? W4AFP8_TP2X4_CANDIDATE_ARGV : W4AFP8_TP2X4_ARGV)
    expected_variant = v8_slot ? W4AFP8_TP2X4_V8_VARIANT : (candidate ? W4AFP8_TP2X4_CANDIDATE_VARIANT : W4AFP8_TP2X4_VARIANT)
    unless actual_argv == expected_argv
      drift = ((actual_argv - expected_argv) + (expected_argv - actual_argv)).uniq
      errors << "#{label} #{name} argv must be the #{v8_slot ? 'v8 slot (candidate behind GLM53_V8_R4_ variables, ENV_PREFIX first, EXTRA_ARGS last)' : (candidate ? 'memory-optimized candidate' : 'lab-qualified TP2 control')} argv exactly; differing tokens: #{drift.first(8).join(' ')}"
    end
    # Capacity invariants are asserted on what an unset env map renders: every variable resolved to its default.
    resolved_argv = actual_argv.map { |token| v8_resolve(token) }.reject(&:empty?)
    # Capacity invariants, asserted on what the file says rather than on the expected argv.
    flag_value = lambda do |flag|
      found = resolved_argv.each_cons(2).find { |token, _| token == flag }
      errors << "#{label} #{name} must set #{flag}" if found.nil?
      found ? found.last.to_i : 0
    end
    slots = flag_value.call("--max-mamba-cache-size")
    running = flag_value.call("--max-running-requests")
    if slots < W4AFP8_TP2X4_MAMBA_SLOTS_PER_REQUEST * running
      errors << "#{label} #{name} --max-mamba-cache-size #{slots} cannot hold #{running} running requests (needs >= #{W4AFP8_TP2X4_MAMBA_SLOTS_PER_REQUEST} slots each)"
    end
    errors << "#{label} #{name} --cuda-graph-max-bs-decode must equal --max-running-requests (#{running})" unless flag_value.call("--cuda-graph-max-bs-decode") == running
    env = environment_map(service)
    expected_env = base_env.merge(observability_env(spec["ghost_replica"]))
    REQUIRED_ENV.each do |key, value|
      errors << "#{label} #{name} must set #{key}=#{value}" unless env[key] == value
    end
    (env.keys & FORBIDDEN_ENV).each { |key| errors << "#{label} #{name} must not set #{key}" }
    unless env == expected_env
      diff = (env.to_a - expected_env.to_a) + (expected_env.to_a - env.to_a)
      errors << "#{label} #{name} environment must be the W4AFP8 base engine environment plus #{W4AFP8_TP2X4_EXTRA_ENV.map { |key, value| "#{key}=#{value}" }.join(' ')} and the observability environment; differing: #{diff.map { |key, value| "#{key}=#{value}" }.uniq.join(' ')}"
    end
    if env[W4AFP8_QSPLIT_ENV] == "1" && !W4AFP8_QSPLIT_CAPABLE_IMAGES.include?(v8_resolve(service["image"]))
      errors << "#{label} #{name} sets #{W4AFP8_QSPLIT_ENV}=1 but does not run an approved split-capable image"
    end
    device_ids = Array(service.dig("deploy", "resources", "reservations", "devices", 0, "device_ids")).map(&:to_s)
    errors << "#{label} #{name} must use GPU device_ids #{spec['devices'].join(',')}" unless device_ids == spec["devices"]

    slot_engine_label = v8_slot ? v8_expr(W4AFP8_TP2X4_V8_PREFIX, "IMAGE_LABEL", engine_image_label) : engine_image_label
    slot_precision = v8_slot ? v8_expr(W4AFP8_TP2X4_V8_PREFIX, "PRECISION", W4AFP8_PRECISION) : W4AFP8_PRECISION
    labels = service["labels"].is_a?(Hash) ? service["labels"] : {}
    {
      "nearai.otel.container_name" => name, "nearai.otel.model_path" => W4AFP8_CHECKPOINT, "nearai.otel.engine_image" => slot_engine_label,
      "nearai.otel.instance" => spec["instance"], "nearai.otel.deployment" => W4AFP8_TP2X4_DEPLOYMENT,
    }.each do |key, value|
      errors << "#{label} #{name} #{key} must be #{value.inspect}, got #{labels[key].inspect}" unless labels[key] == value
    end
    tags = begin
      Array(JSON.parse(labels["com.datadoghq.ad.logs"].to_s).first&.fetch("tags", []))
    rescue JSON::ParserError
      []
    end
    ["model_path:#{W4AFP8_CHECKPOINT}", "precision:#{slot_precision}", "engine_image:#{slot_engine_label}", "instance:#{spec['instance']}", "deployment:#{W4AFP8_TP2X4_DEPLOYMENT}"].each do |tag|
      errors << "#{label} #{name} log metadata must carry #{tag}" unless tags.include?(tag)
    end
    check_variant(errors, label, service, name, collector, expected_variant)
    scrape = scrape_job(errors, label, collector, "sglang-#{name}")
    scrape_labels = scrape&.dig("static_configs", 0, "labels") || {}
    errors << "#{label} sglang-#{name} must scrape #{name}:8000" if scrape && scrape.dig("static_configs", 0, "targets") != ["#{name}:8000"]
    {
      "container_name" => name, "model_path" => W4AFP8_CHECKPOINT, "precision" => slot_precision, "engine_image" => slot_engine_label,
      "instance" => spec["instance"], "deployment" => W4AFP8_TP2X4_DEPLOYMENT,
    }.each do |key, value|
      errors << "#{label} sglang-#{name} scrape label #{key} must be #{value.inspect}, got #{scrape_labels[key].inspect}" if scrape && scrape_labels[key] != value
    end
  end

  if replicas.length == W4AFP8_TP2X4_REPLICAS.length
    # The argv is checked exactly per role and the environment in full per replica above (it
    # differs only in the ghost-cache replica name); everything else is identical.
    # (the image is asserted per replica above: the v8 slot's is an interpolated default)
    contracts = replicas.values.map { |service| runtime_contract(service).reject { |key, _value| %w[command environment image].include?(key) } }
    errors << "#{label} replicas must use identical runtime configuration" unless contracts.uniq.length == 1
  end
  validate_v8_slot(errors, label, compose, raw, W4AFP8_TP2X4_V8_SLOT)
  validate_observability(errors, label, compose, collector, W4AFP8_TP2X4_REPLICAS.transform_values { |spec| spec["ghost_replica"] },
                         W4AFP8_TP2X4_IMAGE, W4AFP8_TP2X4_DEPLOYMENT)

  dcgm_labels = services.dig("dcgm-glm53", "labels") || {}
  errors << "#{label} dcgm-glm53 nearai.otel.model_path must be #{W4AFP8_CHECKPOINT}" unless dcgm_labels["nearai.otel.model_path"] == W4AFP8_CHECKPOINT

  proxy = services["proxy-glm53"] || {}
  proxy_env = environment_map(proxy)
  expected_backends = W4AFP8_TP2X4_REPLICAS.keys.map { |name| "http://#{name}:8000" }.join(",")
  errors << "#{label} proxy-glm53 must pool all four TP2 replicas" unless proxy_env["VLLM_BACKEND_URLS"] == expected_backends
  errors << "#{label} proxy-glm53 must enable conversation affinity" unless proxy_env["VLLM_BACKEND_CONVERSATION_AFFINITY"] == "1"
  unless PRIORITY_NORMALIZING_PROXY_IMAGES.include?(proxy["image"])
    errors << "#{label} enables SGLang priority scheduling but proxy-glm53 image #{proxy['image'].inspect} is not a priority-normalizing inference-proxy build"
  end

  errors << "#{label} glm53-perception-check image must be #{ENGINE_IMAGE}" unless services.dig("glm53-perception-check", "image") == ENGINE_IMAGE
  perception = command_text(services["glm53-perception-check"] || {})
  unless perception.include?("for replica in (1, 2, 3, 4):") && perception.include?("base = f\"http://#{W4AFP8_TP2X4_PREFIX}{replica}:8000\"")
    errors << "#{label} glm53-perception-check must check replicas 1-4 at http://#{W4AFP8_TP2X4_PREFIX}{replica}:8000"
  end

  relay_ports = Array(services.dig("glm53-soak-relay", "ports")).map(&:to_s)
  expected_ports = W4AFP8_TP2X4_REPLICAS.values.map { |spec| "#{spec['soak_port']}:#{spec['soak_port']}" }
  errors << "#{label} glm53-soak-relay must publish #{expected_ports.join(', ')}" unless relay_ports == expected_ports
  relay_conf = compose.dig("configs", "glm53_soak_nginx_conf", "content").to_s
  relay_servers = relay_conf.scan(/listen (\d+) ssl;.*?set \$\$backend http:\/\/([^:;]+):8000;/m)
  expected_servers = W4AFP8_TP2X4_REPLICAS.map { |name, spec| [spec["soak_port"], name] }
  errors << "#{label} glm53-soak-relay must map #{expected_servers.map { |port, name| "#{port}->#{name}" }.join(', ')}" unless relay_servers == expected_servers

  base_view = w4afp8_tp2x4_view(errors, "W4AFP8 base file", base, W4AFP8_BASE_REPLICAS.keys)
  base_relay = base.dig("configs", "glm53_soak_nginx_conf", "content").to_s
  target_view = w4afp8_tp2x4_view(errors, "#{label} file", compose, W4AFP8_TP2X4_REPLICAS.keys)
  # The relay's shared http block (auth, TLS, timeouts) must be unchanged: compare it with the
  # per-replica server blocks removed.
  strip_servers = ->(conf) { conf.gsub(%r{^ *server \{\n.*?^ *location / \{ return 404; \}\n *\}\n}m, "") }
  errors << "#{label} glm53-soak-relay config must match the W4AFP8 base file outside its per-replica servers" unless strip_servers.call(relay_conf) == strip_servers.call(base_relay)
  return if base_view == target_view

  errors << "#{label} file must match the W4AFP8 base file outside the engines, their fan-out and telemetry (first difference: #{first_difference(base_view, target_view)})"
end

# HiCache file: r1 stays the disabled control on the plain engine image and
# is telemetry-pinned to OFFICIAL_VARIANT; r2 pins RELEASED_IMAGE, enables
# HiCache with exactly the pinned options/env, is otherwise identical to r1,
# and is telemetry-pinned to HICACHE_VARIANT.
def validate_hicache(errors, compose, replica_services, label = "HiCache", control_variant = OFFICIAL_VARIANT, hicache_variant = HICACHE_VARIANT, hicache_env = HICACHE_ENV)
  released_image = File.exist?(RELEASED_IMAGE_FILE) ? File.read(RELEASED_IMAGE_FILE).strip : nil
  unless released_image && released_image.match?(%r{\Adocker\.io/nearaidev/sglang@sha256:[a-f0-9]{64}\z}) && released_image != ENGINE_IMAGE
    errors << "RELEASED_IMAGE must be an immutable docker.io/nearaidev/sglang digest distinct from ENGINE_IMAGE"
    return
  end

  r1 = replica_services["model-sg-glm53-fp8-tp4-r1"]
  r2 = replica_services["model-sg-glm53-fp8-tp4-r2"]
  return unless r1 && r2

  errors << "#{label} r1 image must be #{ENGINE_IMAGE}" unless r1["image"] == ENGINE_IMAGE
  r1_text = command_text(r1)
  if r1_text.include?("--enable-hierarchical-cache") || r1_text.include?("--hicache") ||
     environment_map(r1).keys.any? { |key| key.start_with?("SGLANG_HICACHE_") }
    errors << "#{label} r1 must remain the HiCache-disabled control"
  end

  errors << "#{label} r2 image must be RELEASED_IMAGE (#{released_image})" unless r2["image"] == released_image

  r1_arguments = Shellwords.split(command_text(r1))
  normalized = Marshal.load(Marshal.dump(r2))
  arguments = Shellwords.split(command_text(normalized))
  errors << "#{label} r2 must enable HiCache exactly once" unless arguments.count("--enable-hierarchical-cache") == 1
  arguments.delete("--enable-hierarchical-cache")
  HICACHE_OPTIONS.each do |key, expected|
    positions = arguments.each_index.select { |index| arguments[index] == key }
    if positions.length != 1 || arguments[positions.first.to_i + 1] != expected
      errors << "#{label} r2 must set #{key} #{expected} exactly once"
    else
      arguments.slice!(positions.first, 2)
    end
  end
  errors << "#{label} r2 must preserve all control serving arguments outside HiCache" unless arguments == r1_arguments

  env = environment_map(normalized)
  hicache_env.each do |key, expected|
    errors << "#{label} r2 must set #{key}=#{expected}" unless env.delete(key) == expected
  end
  errors << "#{label} r2 must preserve all control environment outside HiCache" unless env == environment_map(r1)

  normalized["image"] = r1["image"]
  normalized["command"] = r1["command"]
  normalized["environment"] = r1["environment"]
  errors << "#{label} r2 must preserve the control runtime outside HiCache" unless runtime_contract(normalized) == runtime_contract(r1)

  collector = load_embedded_yaml(errors, "#{label} file otelcol_app_config", compose.dig("configs", "otelcol_app_config", "content"))
  check_variant(errors, label, r1, "model-sg-glm53-fp8-tp4-r1", collector, control_variant)
  check_variant(errors, label, r2, "model-sg-glm53-fp8-tp4-r2", collector, hicache_variant)
end

def validate_long_context(errors, compose, replica_services)
  reserve_env = replica_services.values.flat_map do |service|
    environment_map(service).keys & ADMISSION_RESERVE_ENV
  end.uniq
  unless reserve_env.empty?
    errors << "long-context replicas must not set admission-reserve environment: #{reserve_env.join(', ')}"
  end

  validate_hicache(
    errors,
    compose,
    replica_services,
    "long-context",
    LONG_CONTEXT_CONTROL_VARIANT,
    LONG_CONTEXT_HICACHE_VARIANT,
    HICACHE_ENV,
  )
end

# W4AFP8 + HiCache long-context file (gpu02): both replicas run gpu31 campaign-2 arm L2
# with exactly the argv below (only --dist-init-addr and --prefill-decode-interval differ:
# both pin the original #308 offloop-v3 image, with pdi1 on r1 and pdi2 on r2),
# the long-context control environment plus the per-replica 406 GiB HiCache environment, and
# no admission reserve. Outside the two engines and their truthful telemetry it must
# equal the long-context file, so the long-domain routing contract (nginx and the :8001
# discovery stub, registrar, proxy pooling) cannot drift.
W4AFP8_LONG_CONTEXT_FILE = File.join(ROOT, "prod", "GLM-5.3-Flash-SGL-TP4-W4AFP8-LongContext.yaml")
# The v1 digest is retained only so the historical rollback target stays greppable.
W4AFP8_LONG_CONTEXT_V1_IMAGE = "docker.io/nearaidev/sglang@sha256:fde25985aea3ebabf1eb581ae21d53be8540e32933eef942ee8b962a1bfbea20"
W4AFP8_LONG_CONTEXT_V2_IMAGE = "docker.io/nearaidev/sglang@sha256:8ff1a487b98a52fe08b781715bebd7c8c445d4fe068f312f03f527d5a3c77e84"
W4AFP8_LONG_CONTEXT_V3_IMAGE = "docker.io/nearaidev/sglang@sha256:47aff791090003a37f893e998c44794c410d3f7bdfc7fdd2dfab5eb5592b30bb"
# glm53-hicache-w4afp8-v6 (#340): the v3 recipe plus the opt-in ghost prefix cache and KV tier
# metrics (#336) and the PyJWT CVE fix; published, signed and attested by workflow run 37505271073.
W4AFP8_LONG_CONTEXT_V6_IMAGE = "docker.io/nearaidev/sglang@sha256:9c6ddd4319c4ab00e351d8650459e68b8830e36ffcc029d67fa5e19d0ac3ed17"
W4AFP8_QSPLIT_CAPABLE_IMAGES = [W4AFP8_LONG_CONTEXT_V2_IMAGE, W4AFP8_LONG_CONTEXT_V3_IMAGE, W4AFP8_LONG_CONTEXT_V6_IMAGE].freeze
W4AFP8_LONG_CONTEXT_VARIANT = "fc91d24-long-context-w4afp8-cCHUNK-QSPLITOFFLOOPhicache-cuda-host-pooled-v1-HOSTadmission-reserve-disabled-pool-clamp-pdiPDI-h200-tp4-ep4-eagle-adaptive-5-1-6-strict-budget8192#{OBSERVABILITY_VARIANT_SUFFIX}"
W4AFP8_QSPLIT_ENV = "SGLANG_DSA_INDEXER_QSPLIT"
# c16384 is only memory-safe WITH the split: without it a concurrent long burst left 0.04-0.65 GB
# free, the condition that preceded the gpu02 crash. Enforced below for every replica.
W4AFP8_SPLIT_REQUIRED_CHUNK = "16384"
W4AFP8_CHECKPOINT = "graphistry/GLM-5.3-Flash-W4AFP8"
W4AFP8_PRECISION = "int4-weights-fp8-activations-bf16-kv"
W4AFP8_LONG_CONTEXT_REPLICAS = {
  "model-sg-glm53-w4afp8-tp4-r1" => { "devices" => %w[0 1 2 3], "dist_init" => "127.0.0.1:29510", "instance" => "1", "pdi" => "1",
                                     "image" => W4AFP8_LONG_CONTEXT_V6_IMAGE, "chunk" => "8192", "qsplit" => "1", "offloop" => "offloop-v3",
                                     "budget" => "${GLM53_HICACHE_RAM_BUDGET:-406GiB}", "host_variant" => "", "ghost_replica" => "r1" },
  "model-sg-glm53-w4afp8-tp4-r2" => { "devices" => %w[4 5 6 7], "dist_init" => "127.0.0.1:29511", "instance" => "2", "pdi" => "2",
                                     "image" => W4AFP8_LONG_CONTEXT_V6_IMAGE, "chunk" => "8192", "qsplit" => "1", "offloop" => "offloop-v3",
                                     # HiCache host-tier canary: write_through keeps the host tier an inclusive
                                     # copy of the ~3.52M-token device pool, so 406 GiB (~4.99M tokens) adds only
                                     # ~1.5M; 650 GiB (~8M) adds ~4.5M. r1 stays at 406 GiB as the control.
                                     "budget" => "${GLM53_R2_HICACHE_RAM_BUDGET:-650GiB}", "host_variant" => "host650g-", "ghost_replica" => "r2" },
}.freeze
W4AFP8_LONG_CONTEXT_ARGV = Shellwords.split(<<~'ARGV').freeze
  sglang serve
  --model-path /root/.cache/huggingface/hub/models--graphistry--GLM-5.3-Flash-W4AFP8/snapshots/99f1fa70408c52b007d4fd69e02e5a522422e755
  --served-model-name z-ai/glm-5.3-flash
  --tp-size 4 --ep-size 4
  --mem-fraction-static 0.80
  --max-running-requests 32 --max-queued-requests 8
  --enable-priority-scheduling --disable-priority-preemption
  --chunked-prefill-size CHUNK --max-prefill-tokens 32768 --prefill-decode-interval PDI
  --cuda-graph-max-bs-decode 32
  --dsa-prefill-backend tilelang --dsa-decode-backend tilelang
  --kv-cache-dtype bfloat16
  --speculative-algorithm EAGLE --speculative-num-steps 5 --speculative-eagle-topk 1
  --speculative-num-draft-tokens 6 --speculative-adaptive
  --reasoning-parser glm45 --enable-strict-thinking --grammar-backend xgrammar --tool-call-parser glm47
  --chat-template /root/.cache/huggingface/hub/models--zai-org--GLM-5.3-Flash/snapshots/3f1971b7b5f7a528c9c4ef6212c8785298a8c24a/chat_template.jinja
  --context-length 1048576
  --dist-init-addr DIST_INIT
  --watchdog-timeout 1800 --host 0.0.0.0 --port 8000
  --enable-metrics --enable-cache-report --log-requests-level 0
  --disable-fast-image-processor --limit-mm-data-per-request '{"image": 64}'
  --enable-hierarchical-cache --hicache-write-policy write_through
  --hicache-io-backend direct --hicache-mem-layout page_first_direct
ARGV
W4AFP8_LONG_CONTEXT_HICACHE_ENV = HICACHE_ENV.merge("SGLANG_HICACHE_RAM_BUDGET" => "${GLM53_HICACHE_RAM_BUDGET:-406GiB}").freeze

# 2xTP2 memory-optimized replicas. The file is shared by gpu02 and gpu23, so the TP4 r1/r2
# above stay defined (a host not yet converted keeps deploying them) and these TP2 services are ADDED
# to start in their place (same GPUs, so a TP4 replica and its pair are never up together). Lab-validated
# (tee-bench exp 19): mem 0.86, 330 mamba slots, fixed EAGLE 4/1/5. Caps are 12 running / 4 queued with decode
# graphs capped at 12 (user decision, prod KV-bound evidence in docs/long-context-glm53-2xtp2-rollout.md; the lab ran 24/8).
W4AFP8_TP2_CANARY_REPLICAS = {
  # -r2a is also the v8 bundle canary slot (gpu02): the same argv behind per-replica GLM53_V8_R2A_ variables.
  "model-sg-glm53-w4afp8-tp2-r2a" => { "devices" => %w[4 5], "dist_init" => "127.0.0.1:29512", "instance" => "2a", "gpu_pair" => "4-5",
                                       "budget" => "${GLM53_R2A_HICACHE_RAM_BUDGET:-325GiB}", "ghost_replica" => "r2a", "v8" => true },
  "model-sg-glm53-w4afp8-tp2-r2b" => { "devices" => %w[6 7], "dist_init" => "127.0.0.1:29513", "instance" => "2b", "gpu_pair" => "6-7",
                                       "budget" => "${GLM53_R2B_HICACHE_RAM_BUDGET:-325GiB}", "ghost_replica" => "r2b" },
  # Long-context 2xTP2 rollout: the same pair in r1's place (GPUs 0-3).
  "model-sg-glm53-w4afp8-tp2-r1a" => { "devices" => %w[0 1], "dist_init" => "127.0.0.1:29514", "instance" => "1a", "gpu_pair" => "0-1",
                                       "budget" => "${GLM53_R1A_HICACHE_RAM_BUDGET:-325GiB}", "ghost_replica" => "r1a" },
  "model-sg-glm53-w4afp8-tp2-r1b" => { "devices" => %w[2 3], "dist_init" => "127.0.0.1:29515", "instance" => "1b", "gpu_pair" => "2-3",
                                       "budget" => "${GLM53_R1B_HICACHE_RAM_BUDGET:-325GiB}", "ghost_replica" => "r1b" },
}.freeze
# Each TP2 replica replaces one TP4 replica and must stay inside that replica's GPUs.
W4AFP8_TP2_PARENT = {
  "model-sg-glm53-w4afp8-tp2-r1a" => "model-sg-glm53-w4afp8-tp4-r1", "model-sg-glm53-w4afp8-tp2-r1b" => "model-sg-glm53-w4afp8-tp4-r1",
  "model-sg-glm53-w4afp8-tp2-r2a" => "model-sg-glm53-w4afp8-tp4-r2", "model-sg-glm53-w4afp8-tp2-r2b" => "model-sg-glm53-w4afp8-tp4-r2",
}.freeze
W4AFP8_TP2_CANARY_VARIANT = "fc91d24-long-context-w4afp8-c8192-qsplit-offloop-v3-hicache-cuda-host-pooled-v1-host325g-memopt-mamba330-bf16state-admission-reserve-disabled-pool-clamp-pdi2-h200-tp2-ep2-eagle-fixed-4-1-5-mr12q4-strict-budget8192#{OBSERVABILITY_VARIANT_SUFFIX}"
W4AFP8_TP2_CANARY_ARGV = Shellwords.split(<<~'ARGV').freeze
  sglang serve
  --model-path /root/.cache/huggingface/hub/models--graphistry--GLM-5.3-Flash-W4AFP8/snapshots/99f1fa70408c52b007d4fd69e02e5a522422e755
  --served-model-name z-ai/glm-5.3-flash
  --tp-size 2 --ep-size 2
  --mem-fraction-static 0.86
  --max-running-requests 12 --max-queued-requests 4
  --enable-priority-scheduling --disable-priority-preemption
  --chunked-prefill-size 8192 --max-prefill-tokens 32768 --prefill-decode-interval 2
  --cuda-graph-max-bs-decode 12
  --dsa-prefill-backend tilelang --dsa-decode-backend tilelang
  --kv-cache-dtype bfloat16
  --speculative-algorithm EAGLE --speculative-num-steps 4 --speculative-eagle-topk 1
  --speculative-num-draft-tokens 5
  --reasoning-parser glm45 --enable-strict-thinking --grammar-backend xgrammar --tool-call-parser glm47
  --chat-template /root/.cache/huggingface/hub/models--zai-org--GLM-5.3-Flash/snapshots/3f1971b7b5f7a528c9c4ef6212c8785298a8c24a/chat_template.jinja
  --context-length 1048576
  --dist-init-addr DIST_INIT
  --watchdog-timeout 1800 --host 0.0.0.0 --port 8000
  --enable-metrics --enable-cache-report --log-requests-level 0
  --disable-fast-image-processor --limit-mm-data-per-request '{"image": 64}'
  --enable-hierarchical-cache --hicache-write-policy write_through
  --hicache-io-backend direct --hicache-mem-layout page_first_direct
  --max-mamba-cache-size 330 --mamba-ssm-dtype bfloat16
ARGV
# The v8 slot of the long-context file (r2a): the TP2 argv with the values the bundle changes behind variables
# (default = today's 12/4/bf16/tilelang), the environment prefix first and the extra args last.
W4AFP8_TP2_V8_PREFIX = "GLM53_V8_R2A_"
W4AFP8_TP2_V8_VALUE_FLAGS = {
  "--kv-cache-dtype" => ["KV_DTYPE", "bfloat16"], "--dsa-prefill-backend" => ["DSA_BACKEND", "tilelang"],
  "--dsa-decode-backend" => ["DSA_BACKEND", "tilelang"], "--max-running-requests" => ["MAX_RUNNING", "12"],
  "--cuda-graph-max-bs-decode" => ["MAX_RUNNING", "12"], "--max-queued-requests" => ["MAX_QUEUED", "4"],
}.freeze
W4AFP8_TP2_V8_ARGV = begin
  argv = W4AFP8_TP2_CANARY_ARGV.dup
  W4AFP8_TP2_V8_VALUE_FLAGS.each { |flag, (name, default)| argv[argv.index(flag) + 1] = v8_expr(W4AFP8_TP2_V8_PREFIX, name, default) }
  ([v8_expr(W4AFP8_TP2_V8_PREFIX, "ENV_PREFIX", "")] + argv + [v8_expr(W4AFP8_TP2_V8_PREFIX, "EXTRA_ARGS", "")]).freeze
end
W4AFP8_TP2_V8_IMAGE = v8_expr(W4AFP8_TP2_V8_PREFIX, "IMAGE", W4AFP8_LONG_CONTEXT_V6_IMAGE)
W4AFP8_TP2_V8_VARIANT = W4AFP8_TP2_CANARY_VARIANT + v8_expr(W4AFP8_TP2_V8_PREFIX, "VARIANT_SUFFIX", "")
W4AFP8_TP2_V8_SLOT = {
  "service" => "model-sg-glm53-w4afp8-tp2-r2a", "prefix" => W4AFP8_TP2_V8_PREFIX,
  "variables" => {
    "IMAGE" => [1, W4AFP8_LONG_CONTEXT_V6_IMAGE], "IMAGE_LABEL" => [3, W4AFP8_LONG_CONTEXT_V6_IMAGE.split(":").last[0, 12]],
    "PRECISION" => [2, "int4-weights-fp8-activations-bf16-kv"], "KV_DTYPE" => [1, "bfloat16"], "DSA_BACKEND" => [2, "tilelang"],
    "MAX_RUNNING" => [2, "12"], "MAX_QUEUED" => [1, "4"], "VARIANT_SUFFIX" => [3, ""], "EXTRA_ARGS" => [1, ""], "ENV_PREFIX" => [1, ""],
  },
}.freeze
# Flags that must never appear on a TP2 canary replica: 32K chunks conflict with 0.86 at TP2,
# adaptive EAGLE is replaced by the fixed 4/1/5 arm, and gpu13 is the separate overlap-off canary.
W4AFP8_TP2_CANARY_FORBIDDEN_FLAGS = %w[--disable-overlap-schedule --speculative-adaptive].freeze
# The proxy pool is host-overridable; its default MUST stay r1 + r2 so a host that has not
# converted (and never sets GLM53_BACKEND_URLS) keeps its exact current backend list.
W4AFP8_PROXY_BACKENDS_DEFAULT = "http://model-sg-glm53-w4afp8-tp4-r1:8000,http://model-sg-glm53-w4afp8-tp4-r2:8000"
W4AFP8_PROXY_BACKENDS_VALUE = "${GLM53_BACKEND_URLS:-#{W4AFP8_PROXY_BACKENDS_DEFAULT}}"

# The long-context file and the W4AFP8 long-context file, reduced to what must be
# identical: engines, the engine anchor and the replicas' scrape jobs removed, replica
# names and the checkpoint behind the telemetry normalized.
def w4afp8_long_context_view(errors, file_label, compose, replica_names)
  view = Marshal.load(Marshal.dump(compose))
  view.delete("x-sg-glm53-flash-common")
  replica_names.each { |name| view.fetch("services", {}).delete(name) }
  # The gpu02 2xTP2 canary services and the host-overridable proxy pool are validated separately.
  W4AFP8_TP2_CANARY_REPLICAS.each_key { |name| view.fetch("services", {}).delete(name) }
  proxy_environment = view.dig("services", "proxy-glm53", "environment")
  if proxy_environment.is_a?(Array)
    proxy_environment.map! { |item| item == "VLLM_BACKEND_URLS=#{W4AFP8_PROXY_BACKENDS_VALUE}" ? "VLLM_BACKEND_URLS=#{W4AFP8_PROXY_BACKENDS_DEFAULT}" : item }
  end
  otel = view.dig("configs", "otelcol_app_config")
  collector = nil
  if otel && otel["content"]
    collector = load_embedded_yaml(errors, "#{file_label} otelcol_app_config", otel["content"])
    if collector
      Array(collector.dig("receivers", "prometheus/apps", "config", "scrape_configs")).reject! do |job|
        job.is_a?(Hash) && (replica_names + W4AFP8_TP2_CANARY_REPLICAS.keys).any? { |name| job["job_name"] == "sglang-#{name}" }
      end
      otel["content"] = collector
    end
  end
  strip_observability(view, collector)
  normalized = JSON.generate(view)
                   .gsub("model-sg-glm53-w4afp8-tp4-r", "REPLICA-r").gsub("model-sg-glm53-fp8-tp4-r", "REPLICA-r")
                   .gsub(W4AFP8_CHECKPOINT, "CHECKPOINT").gsub("zai-org/GLM-5.3-Flash", "CHECKPOINT")
  JSON.parse(normalized)
end

def validate_w4afp8_long_context(errors, compose, reference, raw)
  label = "W4AFP8 long-context"
  services = compose.fetch("services", {})
  expected_services = EXPECTED_SERVICES - REPLICAS.keys + W4AFP8_LONG_CONTEXT_REPLICAS.keys + W4AFP8_TP2_CANARY_REPLICAS.keys + [GHOST_SERVICE]
  missing = expected_services - services.keys
  extra = services.keys - expected_services
  errors << "#{label} is missing services: #{missing.join(', ')}" unless missing.empty?
  errors << "#{label} has unexpected services: #{extra.join(', ')}" unless extra.empty?

  reference_env = environment_map(reference.dig("services", "model-sg-glm53-fp8-tp4-r1") || {})
  expected_env = reference_env.merge(W4AFP8_LONG_CONTEXT_HICACHE_ENV)
  collector = load_embedded_yaml(errors, "#{label} file otelcol_app_config", compose.dig("configs", "otelcol_app_config", "content"))
  replicas = {}
  W4AFP8_LONG_CONTEXT_REPLICAS.each do |name, spec|
    service = services[name]
    next errors << "#{label} missing services.#{name}" if service.nil?

    replicas[name] = service
    engine_image_label = spec["image"].split(":").last[0, 12]
    errors << "#{label} #{name} image must be #{spec['image']}" unless service["image"] == spec["image"]
    errors << "#{label} #{name} must use the prebuilt signed image, not a host-local build" if service.key?("build")
    expected_argv = W4AFP8_LONG_CONTEXT_ARGV.map { |token| { "DIST_INIT" => spec["dist_init"], "PDI" => spec["pdi"], "CHUNK" => spec["chunk"] }.fetch(token, token) }
    actual_argv = begin
      Shellwords.split(command_text(service))
    rescue ArgumentError => error
      errors << "#{label} #{name} command cannot be parsed: #{error.message}"
      []
    end
    unless actual_argv == expected_argv
      drift = ((actual_argv - expected_argv) + (expected_argv - actual_argv)).uniq
      errors << "#{label} #{name} argv must be campaign-2 arm L2 exactly (with --dist-init-addr #{spec['dist_init']} --prefill-decode-interval #{spec['pdi']}); differing tokens: #{drift.first(8).join(' ')}"
    end
    env = environment_map(service)
    replica_expected_env = expected_env.merge("SGLANG_HICACHE_RAM_BUDGET" => spec["budget"])
    replica_expected_env = replica_expected_env.merge(W4AFP8_QSPLIT_ENV => spec["qsplit"]) if spec["qsplit"]
    replica_expected_env = replica_expected_env.merge(observability_env(spec["ghost_replica"]))
    reserve = env.keys & ADMISSION_RESERVE_ENV
    errors << "#{label} #{name} must not set admission-reserve environment: #{reserve.join(', ')}" unless reserve.empty?
    (env.keys & FORBIDDEN_ENV).each { |key| errors << "#{label} #{name} must not set #{key}" }
    unless env == replica_expected_env
      diff = (env.to_a - replica_expected_env.to_a) + (replica_expected_env.to_a - env.to_a)
      errors << "#{label} #{name} environment must be the long-context control environment plus #{W4AFP8_LONG_CONTEXT_HICACHE_ENV.merge("SGLANG_HICACHE_RAM_BUDGET" => spec["budget"]).map { |key, value| "#{key}=#{value}" }.join(' ')} and the observability environment; differing: #{diff.map { |key, value| "#{key}=#{value}" }.uniq.join(' ')}"
    end
    # Hard pairing: a 16384 chunk without the indexer split is the pre-crash memory profile.
    # Assert it against what the file actually says, not against the expected spec, so the gate
    # still fires if someone edits the compose by hand or changes the spec above.
    actual_chunk = actual_argv.each_cons(2).find { |flag, _| flag == "--chunked-prefill-size" }&.last
    if actual_chunk == W4AFP8_SPLIT_REQUIRED_CHUNK && env[W4AFP8_QSPLIT_ENV] != "1"
      errors << "#{label} #{name} sets --chunked-prefill-size #{W4AFP8_SPLIT_REQUIRED_CHUNK} without #{W4AFP8_QSPLIT_ENV}=1; that pairing is required (without the split a concurrent long burst left 0.04-0.65 GB free, the condition that preceded the gpu02 crash)"
    end
    if env[W4AFP8_QSPLIT_ENV] == "1" && !W4AFP8_QSPLIT_CAPABLE_IMAGES.include?(service["image"])
      errors << "#{label} #{name} sets #{W4AFP8_QSPLIT_ENV}=1 but does not run an approved split-capable image (expected one of #{W4AFP8_QSPLIT_CAPABLE_IMAGES.join(', ')})"
    end
    device_ids = Array(service.dig("deploy", "resources", "reservations", "devices", 0, "device_ids")).map(&:to_s)
    errors << "#{label} #{name} must use GPU device_ids #{spec['devices'].join(',')}" unless device_ids == spec["devices"]

    labels = service["labels"].is_a?(Hash) ? service["labels"] : {}
    { "nearai.otel.model_path" => W4AFP8_CHECKPOINT, "nearai.otel.engine_image" => engine_image_label, "nearai.otel.instance" => spec["instance"] }.each do |key, value|
      errors << "#{label} #{name} #{key} must be #{value.inspect}, got #{labels[key].inspect}" unless labels[key] == value
    end
    tags = begin
      Array(JSON.parse(labels["com.datadoghq.ad.logs"].to_s).first&.fetch("tags", []))
    rescue JSON::ParserError
      []
    end
    ["model_path:#{W4AFP8_CHECKPOINT}", "precision:#{W4AFP8_PRECISION}", "engine_image:#{engine_image_label}", "instance:#{spec['instance']}"].each do |tag|
      errors << "#{label} #{name} log metadata must carry #{tag}" unless tags.include?(tag)
    end
    expected_variant = W4AFP8_LONG_CONTEXT_VARIANT
                       .sub("cCHUNK", "c#{spec['chunk']}")
                       .sub("QSPLIT", spec["qsplit"] ? "qsplit-" : "")
                       .sub("OFFLOOP", spec["offloop"] ? "#{spec['offloop']}-" : "")
                       .sub("pdiPDI", "pdi#{spec['pdi']}")
                       .sub("HOST", spec["host_variant"])
    check_variant(errors, label, service, name, collector, expected_variant)
    scrape = scrape_job(errors, label, collector, "sglang-#{name}")
    scrape_labels = scrape&.dig("static_configs", 0, "labels") || {}
    { "model_path" => W4AFP8_CHECKPOINT, "precision" => W4AFP8_PRECISION, "engine_image" => engine_image_label, "instance" => spec["instance"] }.each do |key, value|
      errors << "#{label} sglang-#{name} scrape label #{key} must be #{value.inspect}, got #{scrape_labels[key].inspect}" if scrape && scrape_labels[key] != value
    end
  end

  if replicas.length == 2
    # Image, command and environment are asserted in full per replica above. Runtime equality
    # below covers every remaining service property.
    canary_divergent = %w[command image environment]
    contracts = replicas.values.map { |service| runtime_contract(service).reject { |key, _value| canary_divergent.include?(key) } }
    errors << "#{label} replicas must share one runtime configuration outside image, environment and command (each asserted per replica)" unless contracts.uniq.length == 1
  end
  images = W4AFP8_LONG_CONTEXT_REPLICAS.values.map { |spec| spec["image"] }.uniq
  errors << "#{label} replicas must share one image for the #{GHOST_SERVICE} sidecar to follow" unless images.length == 1
  # Every GLM engine in the file: TP4 r1/r2 (gpu23, gpu02 r1) and the gpu02 TP2 pair r2a/r2b.
  ghost_replicas = W4AFP8_LONG_CONTEXT_REPLICAS.merge(W4AFP8_TP2_CANARY_REPLICAS).transform_values { |spec| spec["ghost_replica"] }
  validate_observability(errors, label, compose, collector, ghost_replicas,
                         images.first, W4AFP8_BASE_DEPLOYMENT)

  dcgm_labels = services.dig("dcgm-glm53", "labels") || {}
  errors << "#{label} dcgm-glm53 nearai.otel.model_path must be #{W4AFP8_CHECKPOINT}" unless dcgm_labels["nearai.otel.model_path"] == W4AFP8_CHECKPOINT
  errors << "#{label} dcgm-glm53 log metadata must carry model_path:#{W4AFP8_CHECKPOINT}" unless dcgm_labels["com.datadoghq.ad.logs"].to_s.include?("model_path:#{W4AFP8_CHECKPOINT}")

  proxy = services["proxy-glm53"] || {}
  proxy_env = environment_map(proxy)
  expected_backends = W4AFP8_LONG_CONTEXT_REPLICAS.keys.map { |name| "http://#{name}:8000" }.join(",")
  errors << "#{label} proxy-glm53 default pool must stay exactly #{expected_backends} (gpu23 never sets GLM53_BACKEND_URLS and must keep it)" unless W4AFP8_PROXY_BACKENDS_DEFAULT == expected_backends
  errors << "#{label} proxy-glm53 must pool both W4AFP8 replicas by default and stay host-overridable: VLLM_BACKEND_URLS=#{W4AFP8_PROXY_BACKENDS_VALUE}" unless proxy_env["VLLM_BACKEND_URLS"] == W4AFP8_PROXY_BACKENDS_VALUE
  errors << "#{label} proxy-glm53 must enable conversation affinity" unless proxy_env["VLLM_BACKEND_CONVERSATION_AFFINITY"] == "1"
  unless PRIORITY_NORMALIZING_PROXY_IMAGES.include?(proxy["image"])
    errors << "#{label} enables SGLang priority scheduling but proxy-glm53 image #{proxy['image'].inspect} is not a priority-normalizing inference-proxy build"
  end

  validate_w4afp8_tp2_canary(errors, label, services, collector, replicas)
  validate_v8_slot(errors, label, compose, raw, W4AFP8_TP2_V8_SLOT)

  reference_view = w4afp8_long_context_view(errors, "long-context file", reference, REPLICAS.keys)
  target_view = w4afp8_long_context_view(errors, "#{label} file", compose, W4AFP8_LONG_CONTEXT_REPLICAS.keys)
  return if reference_view == target_view

  difference = first_difference(reference_view, target_view)
  errors << "#{label} file must match the long-context file outside the two engines and their telemetry (first difference: #{difference})"
end

# The gpu02 2xTP2 memory-optimized pair, checked as rigorously as r1/r2: exact argv and
# environment, GPU pinning, no admission reserve, telemetry, and scrape jobs. `replicas` holds the
# TP4 r1/r2 services so port and GPU collisions against them can be rejected.
def validate_w4afp8_tp2_canary(errors, label, services, collector, replicas)
  reference_env = environment_map(services.fetch("model-sg-glm53-w4afp8-tp4-r1", {}))
  tp2_services = {}
  W4AFP8_TP2_CANARY_REPLICAS.each do |name, spec|
    service = services[name]
    next errors << "#{label} missing services.#{name}" if service.nil?

    tp2_services[name] = service
    v8_slot = spec["v8"] == true
    engine_image_label = W4AFP8_LONG_CONTEXT_V6_IMAGE.split(":").last[0, 12]
    slot_engine_label = v8_slot ? v8_expr(W4AFP8_TP2_V8_PREFIX, "IMAGE_LABEL", engine_image_label) : engine_image_label
    slot_precision = v8_slot ? v8_expr(W4AFP8_TP2_V8_PREFIX, "PRECISION", W4AFP8_PRECISION) : W4AFP8_PRECISION
    expected_image = v8_slot ? W4AFP8_TP2_V8_IMAGE : W4AFP8_LONG_CONTEXT_V6_IMAGE
    errors << "#{label} #{name} image must be #{expected_image}" unless service["image"] == expected_image
    errors << "#{label} #{name} must use the prebuilt signed image, not a host-local build" if service.key?("build")
    expected_argv = (v8_slot ? W4AFP8_TP2_V8_ARGV : W4AFP8_TP2_CANARY_ARGV).map { |token| token == "DIST_INIT" ? spec["dist_init"] : token }
    actual_argv = begin
      Shellwords.split(command_text(service))
    rescue ArgumentError => error
      errors << "#{label} #{name} command cannot be parsed: #{error.message}"
      []
    end
    unless actual_argv == expected_argv
      drift = ((actual_argv - expected_argv) + (expected_argv - actual_argv)).uniq
      errors << "#{label} #{name} argv must be the memory-optimized TP2 argv exactly (tp2/ep2, 0.86, 330 mamba slots, bf16 state, fixed EAGLE 4/1/5, 12 running/4 queued, graphs 12, chunk 8192, write_through, --dist-init-addr #{spec['dist_init']}#{v8_slot ? ', behind the GLM53_V8_R2A_ variables with ENV_PREFIX first and EXTRA_ARGS last' : ''}); differing tokens: #{drift.first(8).join(' ')}"
    end
    (actual_argv & W4AFP8_TP2_CANARY_FORBIDDEN_FLAGS).each { |flag| errors << "#{label} #{name} must not set #{flag}" }
    actual_argv.each_cons(2) do |flag, value|
      errors << "#{label} #{name} must not enable 32K prefill chunks (conflicts with 0.86 at TP2)" if flag == "--chunked-prefill-size" && value.to_i > 8192
    end

    env = environment_map(service)
    expected_env = reference_env.merge(W4AFP8_LONG_CONTEXT_HICACHE_ENV)
                                .merge("SGLANG_HICACHE_RAM_BUDGET" => spec["budget"], W4AFP8_QSPLIT_ENV => "1")
                                .merge(observability_env(spec["ghost_replica"]))
    reserve = env.keys & ADMISSION_RESERVE_ENV
    errors << "#{label} #{name} must not set admission-reserve environment: #{reserve.join(', ')}" unless reserve.empty?
    (env.keys & FORBIDDEN_ENV).each { |key| errors << "#{label} #{name} must not set #{key}" }
    unless env == expected_env
      diff = (env.to_a - expected_env.to_a) + (expected_env.to_a - env.to_a)
      errors << "#{label} #{name} environment must be the long-context environment plus per-replica SGLANG_HICACHE_RAM_BUDGET=#{spec['budget']}, #{W4AFP8_QSPLIT_ENV}=1 and the observability environment; differing: #{diff.map { |key, value| "#{key}=#{value}" }.uniq.join(' ')}"
    end

    device_ids = Array(service.dig("deploy", "resources", "reservations", "devices", 0, "device_ids")).map(&:to_s)
    errors << "#{label} #{name} must use GPU device_ids #{spec['devices'].join(',')}" unless device_ids == spec["devices"]

    labels = service["labels"].is_a?(Hash) ? service["labels"] : {}
    { "nearai.otel.model_path" => W4AFP8_CHECKPOINT, "nearai.otel.engine_image" => slot_engine_label, "nearai.otel.instance" => spec["instance"],
      "nearai.otel.gpu_pair" => spec["gpu_pair"], "nearai.otel.deployment" => labels["nearai.otel.deployment"] }.each do |key, value|
      errors << "#{label} #{name} #{key} must be #{value.inspect}, got #{labels[key].inspect}" unless labels[key] == value
    end
    errors << "#{label} #{name} nearai.otel.deployment must match proxy-glm53 so dashboards line up" unless labels["nearai.otel.deployment"] == services.dig("proxy-glm53", "labels", "nearai.otel.deployment")
    tags = begin
      Array(JSON.parse(labels["com.datadoghq.ad.logs"].to_s).first&.fetch("tags", []))
    rescue JSON::ParserError
      []
    end
    ["model_path:#{W4AFP8_CHECKPOINT}", "precision:#{slot_precision}", "engine_image:#{slot_engine_label}", "instance:#{spec['instance']}", "gpu_pair:#{spec['gpu_pair']}"].each do |tag|
      errors << "#{label} #{name} log metadata must carry #{tag}" unless tags.include?(tag)
    end
    check_variant(errors, label, service, name, collector, v8_slot ? W4AFP8_TP2_V8_VARIANT : W4AFP8_TP2_CANARY_VARIANT)
    scrape = scrape_job(errors, label, collector, "sglang-#{name}")
    if scrape
      targets = scrape.dig("static_configs", 0, "targets")
      errors << "#{label} sglang-#{name} must scrape #{name}:8000, got #{targets.inspect}" unless targets == ["#{name}:8000"]
      scrape_labels = scrape.dig("static_configs", 0, "labels") || {}
      { "container_name" => name, "model_path" => W4AFP8_CHECKPOINT, "precision" => slot_precision, "engine_image" => slot_engine_label,
        "instance" => spec["instance"], "gpu_pair" => spec["gpu_pair"], "deployment" => labels["nearai.otel.deployment"] }.each do |key, value|
        errors << "#{label} sglang-#{name} scrape label #{key} must be #{value.inspect}, got #{scrape_labels[key].inspect}" unless scrape_labels[key] == value
      end
    end
  end

  # Pair parity: everything outside identity, command, environment and devices is shared.
  if tp2_services.length == W4AFP8_TP2_CANARY_REPLICAS.length
    divergent = %w[command environment image]
    contracts = tp2_services.values.map { |service| runtime_contract(service).reject { |key, _value| divergent.include?(key) } }
    errors << "#{label} the TP2 canary replicas must share one runtime configuration outside command, environment and image (each asserted per replica)" unless contracts.uniq.length == 1
  end

  # Unique rendezvous ports across every engine in the file, and no GPU overlap with r1. The pair
  # deliberately reuses r2's GPUs 4-7 (r2 is stopped before the pair starts), so each TP2 pair's
  # devices must be a subset of r2's and the pair must not overlap each other.
  all_engines = replicas.merge(tp2_services)
  ports = all_engines.map do |name, service|
    argv = begin Shellwords.split(command_text(service)) rescue [] end
    [name, argv.each_cons(2).find { |flag, _| flag == "--dist-init-addr" }&.last]
  end
  duplicated = ports.group_by(&:last).select { |port, group| port && group.length > 1 }
  duplicated.each { |port, group| errors << "#{label} --dist-init-addr #{port} is used by more than one engine: #{group.map(&:first).join(', ')}" }
  device_sets = all_engines.transform_values { |service| Array(service.dig("deploy", "resources", "reservations", "devices", 0, "device_ids")).map(&:to_s) }
  # Each TP2 replica stays inside the GPUs of the TP4 replica it replaces (never the other one's),
  # and no two TP2 replicas share a GPU.
  tp2_services.each_key do |name|
    parent = W4AFP8_TP2_PARENT.fetch(name)
    parent_devices = device_sets[parent] || []
    other_devices = (device_sets.select { |other, _| W4AFP8_TP2_PARENT.value?(other) && other != parent }).values.flatten
    errors << "#{label} #{name} must stay within #{parent}'s GPUs #{parent_devices.join(',')} (it replaces that replica)" unless (device_sets[name] - parent_devices).empty?
    errors << "#{label} #{name} must not share GPUs with the other TP4 replica" unless (device_sets[name] & other_devices).empty?
  end
  tp2_services.keys.combination(2).each do |left, right|
    errors << "#{label} #{left} and #{right} must not share GPUs" unless (device_sets[left] & device_sets[right]).empty?
  end
end

# Walks two equal-shaped (or not) structures and returns a dotted path to the
# first point where they differ, for a more actionable failure message.
def first_difference(a, b, path = [])
  return path.join(".") if a == b

  if a.is_a?(Hash) && b.is_a?(Hash)
    keys = (a.keys | b.keys).sort_by(&:to_s)
    keys.each do |key|
      next if a[key] == b[key]

      return first_difference(a[key], b[key], path + [key])
    end
  elsif a.is_a?(Array) && b.is_a?(Array)
    length = [a.length, b.length].max
    (0...length).each do |index|
      next if a[index] == b[index]

      return first_difference(a[index], b[index], path + [index])
    end
  end

  path.join(".")
end

# Cross-file: outside r2 (and its telemetry variant), the HiCache file must be
# a byte-for-byte equal contract to the canonical file — every other service,
# top-level extension block, volume/network declaration and config content.
def cross_file_view(errors, file_label, compose)
  view = Marshal.load(Marshal.dump(compose))
  view["services"]&.delete("model-sg-glm53-fp8-tp4-r2")
  otel = view.dig("configs", "otelcol_app_config")
  if otel && otel["content"]
    collector = load_embedded_yaml(errors, "#{file_label} otelcol_app_config", otel["content"])
    if collector
      scrape = scrape_job(errors, file_label, collector, "sglang-model-sg-glm53-fp8-tp4-r2")
      if scrape
        labels = scrape.dig("static_configs", 0, "labels")
        if labels.is_a?(Hash)
          labels["config_variant"] = OFFICIAL_VARIANT
        else
          errors << "missing labels on sglang-model-sg-glm53-fp8-tp4-r2 scrape job in #{file_label} collector config"
        end
      end
      otel["content"] = collector
    end
  end
  view
end

errors = []

hicache_present = File.exist?(HICACHE_FILE)
released_present = File.exist?(RELEASED_IMAGE_FILE)
if hicache_present != released_present
  errors << "docker/sglang-glm53-hicache/RELEASED_IMAGE and prod/GLM-5.3-Flash-SGL-TP4-HiCache.yaml must both be present or both be absent"
end
errors << "legacy canary compose file must be removed" if File.exist?(LEGACY_CANARY_FILE)

canonical_compose = load_compose_file(errors, "canonical file", COMPOSE_FILE)
if canonical_compose
  canonical_services = canonical_compose.fetch("services", {})
  canonical_replicas = validate_common(errors, "canonical", canonical_services)
  validate_canonical(errors, canonical_compose, canonical_services, canonical_replicas)
  validate_model_cache(errors, "canonical", canonical_services)
end

w4afp8_base_present = File.exist?(W4AFP8_BASE_FILE)
if w4afp8_base_present
  w4afp8_base_compose = load_compose_file(errors, "W4AFP8 base file", W4AFP8_BASE_FILE)
  if w4afp8_base_compose && canonical_compose
    validate_w4afp8_base(errors, w4afp8_base_compose, canonical_compose)
    validate_model_cache(errors, "W4AFP8 base", w4afp8_base_compose.fetch("services", {}))
  end
end

w4afp8_tp2x4_present = File.exist?(W4AFP8_TP2X4_FILE)
if w4afp8_tp2x4_present
  w4afp8_tp2x4_compose = load_compose_file(errors, "W4AFP8 4x TP2 file", W4AFP8_TP2X4_FILE)
  if w4afp8_tp2x4_compose && w4afp8_base_compose
    validate_w4afp8_tp2x4(errors, w4afp8_tp2x4_compose, w4afp8_base_compose, File.read(W4AFP8_TP2X4_FILE))
    validate_model_cache(errors, "W4AFP8 4x TP2", w4afp8_tp2x4_compose.fetch("services", {}))
  elsif w4afp8_tp2x4_compose
    errors << "W4AFP8 4x TP2 file requires the W4AFP8 base file it is generated from"
  end
end

hicache_compose = nil
if hicache_present
  hicache_compose = load_compose_file(errors, "HiCache file", HICACHE_FILE)
  if hicache_compose
    hicache_services = hicache_compose.fetch("services", {})
    hicache_replicas = validate_common(errors, "HiCache", hicache_services)
    validate_hicache(errors, hicache_compose, hicache_replicas)
    validate_model_cache(errors, "HiCache", hicache_services)
  end

  if canonical_compose && hicache_compose
    canonical_view = cross_file_view(errors, "canonical file", canonical_compose)
    hicache_view = cross_file_view(errors, "prod/GLM-5.3-Flash-SGL-TP4-HiCache.yaml", hicache_compose)
    unless canonical_view == hicache_view
      diff_path = first_difference(canonical_view, hicache_view)
      suffix = diff_path.to_s.empty? ? "" : " (first difference: #{diff_path})"
      errors << "prod/GLM-5.3-Flash-SGL-TP4-HiCache.yaml must match the canonical file outside r2 and its telemetry variant#{suffix}"
    end
  end
end

long_context_present = File.exist?(LONG_CONTEXT_FILE)
if hicache_present && released_present && !long_context_present
  errors << "long-context file not found at #{LONG_CONTEXT_FILE}"
end
if long_context_present
  long_context_compose = load_compose_file(errors, "long-context file", LONG_CONTEXT_FILE)
  if long_context_compose
    long_context_services = long_context_compose.fetch("services", {})
    long_context_replicas = validate_common(errors, "long-context", long_context_services, {})
    validate_long_context(errors, long_context_compose, long_context_replicas)
    validate_model_cache(errors, "long-context", long_context_services)
  end
end

w4afp8_long_context_present = File.exist?(W4AFP8_LONG_CONTEXT_FILE)
if w4afp8_long_context_present
  w4afp8_long_context_compose = load_compose_file(errors, "W4AFP8 long-context file", W4AFP8_LONG_CONTEXT_FILE)
  if w4afp8_long_context_compose && long_context_compose
    validate_w4afp8_long_context(errors, w4afp8_long_context_compose, long_context_compose, File.read(W4AFP8_LONG_CONTEXT_FILE))
    validate_model_cache(errors, "W4AFP8 long-context", w4afp8_long_context_compose.fetch("services", {}))
  elsif w4afp8_long_context_compose
    errors << "W4AFP8 long-context file requires the long-context file it is generated from"
  end
end

if errors.any?
  warn "GLM-5.3 production contract failed:"
  errors.each { |error| warn "  - #{error}" }
  exit 1
end

puts "GLM-5.3 production contract OK (prod/GLM-5.3-Flash-SGL-TP4.yaml)"
puts "GLM-5.3 production contract OK (prod/GLM-5.3-Flash-SGL-TP4-W4AFP8.yaml)" if w4afp8_base_present
puts "GLM-5.3 production contract OK (prod/GLM-5.3-Flash-SGL-TP2x4-W4AFP8.yaml)" if w4afp8_tp2x4_present
puts "GLM-5.3 production contract OK (prod/GLM-5.3-Flash-SGL-TP4-HiCache.yaml)" if hicache_present
puts "GLM-5.3 production contract OK (prod/GLM-5.3-Flash-SGL-TP4-LongContext.yaml)" if long_context_present
puts "GLM-5.3 production contract OK (prod/GLM-5.3-Flash-SGL-TP4-W4AFP8-LongContext.yaml)" if w4afp8_long_context_present
