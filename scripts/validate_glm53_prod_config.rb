#!/usr/bin/env ruby
# Protect the canonical two-replica GLM-5.3 Flash production contract, the
# separate r2-only HiCache canary, and the long-context r1-control/r2-HiCache
# experiment. Admission reserve remains required in the first two files and is
# forbidden in the long-context experiment pending a pool-clamp image. Every file's
# model-downloader also pre-stages the W4AFP8 snapshot (see REQUIRED_DOWNLOADS).

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
PROXY_IMAGE = "nearaidev/vllm-proxy-rs@sha256:b3a8c6260834231271b4356c56a7aa2718608c8a537b35973916e0a56dc88fba"
# Engine-side priority scheduling is only safe behind an inference-proxy that
# overwrites the `priority` of every forwarded request (build 2834196 onward,
# nearai/inference-proxy#241). Any other proxy build lets client-chosen or
# missing priorities reach the scheduler, where an untagged request gets the
# lowest possible priority.
PRIORITY_NORMALIZING_PROXY_IMAGES = [
  "nearaidev/vllm-proxy-rs@sha256:b3a8c6260834231271b4356c56a7aa2718608c8a537b35973916e0a56dc88fba",
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
# with exactly the argv below (only --dist-init-addr differs), the hicache-w4afp8 image,
# the long-context control environment plus the per-replica 406 GiB HiCache environment, and
# no admission reserve. Outside the two engines and their truthful telemetry it must
# equal the long-context file, so the long-domain routing contract (nginx and the :8001
# discovery stub, registrar, proxy pooling) cannot drift.
W4AFP8_LONG_CONTEXT_FILE = File.join(ROOT, "prod", "GLM-5.3-Flash-SGL-TP4-W4AFP8-LongContext.yaml")
W4AFP8_LONG_CONTEXT_IMAGE = "docker.io/nearaidev/sglang@sha256:fde25985aea3ebabf1eb581ae21d53be8540e32933eef942ee8b962a1bfbea20"
W4AFP8_LONG_CONTEXT_VARIANT = "fc91d24-long-context-w4afp8-c8192-hicache-cuda-host-pooled-v1-admission-reserve-disabled-pool-clamp-pdi1-h200-tp4-ep4-eagle-adaptive-5-1-6-strict-budget8192"
W4AFP8_CHECKPOINT = "graphistry/GLM-5.3-Flash-W4AFP8"
W4AFP8_PRECISION = "int4-weights-fp8-activations-bf16-kv"
W4AFP8_LONG_CONTEXT_REPLICAS = {
  "model-sg-glm53-w4afp8-tp4-r1" => { "devices" => %w[0 1 2 3], "dist_init" => "127.0.0.1:29510", "instance" => "1" },
  "model-sg-glm53-w4afp8-tp4-r2" => { "devices" => %w[4 5 6 7], "dist_init" => "127.0.0.1:29511", "instance" => "2" },
}.freeze
W4AFP8_LONG_CONTEXT_ARGV = Shellwords.split(<<~'ARGV').freeze
  sglang serve
  --model-path /root/.cache/huggingface/hub/models--graphistry--GLM-5.3-Flash-W4AFP8/snapshots/99f1fa70408c52b007d4fd69e02e5a522422e755
  --served-model-name z-ai/glm-5.3-flash
  --tp-size 4 --ep-size 4
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
  --dist-init-addr DIST_INIT
  --watchdog-timeout 1800 --host 0.0.0.0 --port 8000
  --enable-metrics --enable-cache-report --log-requests-level 0
  --disable-fast-image-processor --limit-mm-data-per-request '{"image": 64}'
  --enable-hierarchical-cache --hicache-write-policy write_through
  --hicache-io-backend direct --hicache-mem-layout page_first_direct
ARGV
W4AFP8_LONG_CONTEXT_HICACHE_ENV = HICACHE_ENV.merge("SGLANG_HICACHE_RAM_BUDGET" => "${GLM53_HICACHE_RAM_BUDGET:-406GiB}").freeze

# The long-context file and the W4AFP8 long-context file, reduced to what must be
# identical: engines, the engine anchor and the replicas' scrape jobs removed, replica
# names and the checkpoint behind the telemetry normalized.
def w4afp8_long_context_view(errors, file_label, compose, replica_names)
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
                   .gsub(W4AFP8_CHECKPOINT, "CHECKPOINT").gsub("zai-org/GLM-5.3-Flash", "CHECKPOINT")
  JSON.parse(normalized)
end

def validate_w4afp8_long_context(errors, compose, reference)
  label = "W4AFP8 long-context"
  services = compose.fetch("services", {})
  expected_services = EXPECTED_SERVICES - REPLICAS.keys + W4AFP8_LONG_CONTEXT_REPLICAS.keys
  missing = expected_services - services.keys
  extra = services.keys - expected_services
  errors << "#{label} is missing services: #{missing.join(', ')}" unless missing.empty?
  errors << "#{label} has unexpected services: #{extra.join(', ')}" unless extra.empty?

  reference_env = environment_map(reference.dig("services", "model-sg-glm53-fp8-tp4-r1") || {})
  expected_env = reference_env.merge(W4AFP8_LONG_CONTEXT_HICACHE_ENV)
  collector = load_embedded_yaml(errors, "#{label} file otelcol_app_config", compose.dig("configs", "otelcol_app_config", "content"))
  engine_image_label = W4AFP8_LONG_CONTEXT_IMAGE.split(":").last[0, 12]
  replicas = {}
  W4AFP8_LONG_CONTEXT_REPLICAS.each do |name, spec|
    service = services[name]
    next errors << "#{label} missing services.#{name}" if service.nil?

    replicas[name] = service
    errors << "#{label} #{name} image must be #{W4AFP8_LONG_CONTEXT_IMAGE}" unless service["image"] == W4AFP8_LONG_CONTEXT_IMAGE
    errors << "#{label} #{name} must use the prebuilt signed image, not a host-local build" if service.key?("build")
    expected_argv = W4AFP8_LONG_CONTEXT_ARGV.map { |token| token == "DIST_INIT" ? spec["dist_init"] : token }
    actual_argv = begin
      Shellwords.split(command_text(service))
    rescue ArgumentError => error
      errors << "#{label} #{name} command cannot be parsed: #{error.message}"
      []
    end
    unless actual_argv == expected_argv
      drift = ((actual_argv - expected_argv) + (expected_argv - actual_argv)).uniq
      errors << "#{label} #{name} argv must be campaign-2 arm L2 exactly (with --dist-init-addr #{spec['dist_init']}); differing tokens: #{drift.first(8).join(' ')}"
    end
    env = environment_map(service)
    reserve = env.keys & ADMISSION_RESERVE_ENV
    errors << "#{label} #{name} must not set admission-reserve environment: #{reserve.join(', ')}" unless reserve.empty?
    (env.keys & FORBIDDEN_ENV).each { |key| errors << "#{label} #{name} must not set #{key}" }
    unless env == expected_env
      diff = (env.to_a - expected_env.to_a) + (expected_env.to_a - env.to_a)
      errors << "#{label} #{name} environment must be the long-context control environment plus #{W4AFP8_LONG_CONTEXT_HICACHE_ENV.map { |key, value| "#{key}=#{value}" }.join(' ')}; differing: #{diff.map { |key, value| "#{key}=#{value}" }.uniq.join(' ')}"
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
    check_variant(errors, label, service, name, collector, W4AFP8_LONG_CONTEXT_VARIANT)
    scrape = scrape_job(errors, label, collector, "sglang-#{name}")
    scrape_labels = scrape&.dig("static_configs", 0, "labels") || {}
    { "model_path" => W4AFP8_CHECKPOINT, "precision" => W4AFP8_PRECISION, "engine_image" => engine_image_label, "instance" => spec["instance"] }.each do |key, value|
      errors << "#{label} sglang-#{name} scrape label #{key} must be #{value.inspect}, got #{scrape_labels[key].inspect}" if scrape && scrape_labels[key] != value
    end
  end

  if replicas.length == 2
    contracts = replicas.values.map { |service| runtime_contract(service).reject { |key, _value| key == "command" } }
    errors << "#{label} replicas must share one runtime configuration outside --dist-init-addr" unless contracts.uniq.length == 1
  end

  dcgm_labels = services.dig("dcgm-glm53", "labels") || {}
  errors << "#{label} dcgm-glm53 nearai.otel.model_path must be #{W4AFP8_CHECKPOINT}" unless dcgm_labels["nearai.otel.model_path"] == W4AFP8_CHECKPOINT
  errors << "#{label} dcgm-glm53 log metadata must carry model_path:#{W4AFP8_CHECKPOINT}" unless dcgm_labels["com.datadoghq.ad.logs"].to_s.include?("model_path:#{W4AFP8_CHECKPOINT}")

  proxy = services["proxy-glm53"] || {}
  proxy_env = environment_map(proxy)
  expected_backends = W4AFP8_LONG_CONTEXT_REPLICAS.keys.map { |name| "http://#{name}:8000" }.join(",")
  errors << "#{label} proxy-glm53 must pool both W4AFP8 replicas" unless proxy_env["VLLM_BACKEND_URLS"] == expected_backends
  errors << "#{label} proxy-glm53 must enable conversation affinity" unless proxy_env["VLLM_BACKEND_CONVERSATION_AFFINITY"] == "1"
  unless PRIORITY_NORMALIZING_PROXY_IMAGES.include?(proxy["image"])
    errors << "#{label} enables SGLang priority scheduling but proxy-glm53 image #{proxy['image'].inspect} is not a priority-normalizing inference-proxy build"
  end

  reference_view = w4afp8_long_context_view(errors, "long-context file", reference, REPLICAS.keys)
  target_view = w4afp8_long_context_view(errors, "#{label} file", compose, W4AFP8_LONG_CONTEXT_REPLICAS.keys)
  return if reference_view == target_view

  difference = first_difference(reference_view, target_view)
  errors << "#{label} file must match the long-context file outside the two engines and their telemetry (first difference: #{difference})"
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
    validate_w4afp8_long_context(errors, w4afp8_long_context_compose, long_context_compose)
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
puts "GLM-5.3 production contract OK (prod/GLM-5.3-Flash-SGL-TP4-HiCache.yaml)" if hicache_present
puts "GLM-5.3 production contract OK (prod/GLM-5.3-Flash-SGL-TP4-LongContext.yaml)" if long_context_present
puts "GLM-5.3 production contract OK (prod/GLM-5.3-Flash-SGL-TP4-W4AFP8-LongContext.yaml)" if w4afp8_long_context_present
