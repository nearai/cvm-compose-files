#!/usr/bin/env ruby
# Protect the canonical two-replica GLM-5.3 Flash production contract.

require "shellwords"
require "yaml"

ROOT = File.expand_path("..", __dir__)
COMPOSE_FILE = File.join(ROOT, "prod", "GLM-5.3-Flash-SGL-TP4.yaml")
LEGACY_CANARY_FILE = File.join(ROOT, "prod", "GLM-5.3-Flash-SGL-TP4-Canary.yaml")
ENGINE_IMAGE = "docker.io/nearaidev/sglang@sha256:a7b7136abcf5e07522289d96e96fec9b42a1a30f9dfda57e957f142680d2d67b"
PROXY_IMAGE = "nearaidev/vllm-proxy-rs@sha256:98b57ad7aa4f9afd8ffff3ac9f4773303997f667e1d221be076c2f064e4a1284"
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
  "--max-queued-requests" => "32",
  "--chunked-prefill-size" => "4096",
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
  --speculative-adaptive
  --enable-strict-thinking
  --enable-metrics
  --enable-cache-report
  --disable-fast-image-processor
].freeze
REPLICA_IDENTITY_FIELDS = %w[container_name deploy labels].freeze
CANARY_IMAGE_FILE = File.join(ROOT, "docker", "sglang-glm53-hicache", "RELEASED_IMAGE")
CANARY_IMAGE = File.exist?(CANARY_IMAGE_FILE) ? File.read(CANARY_IMAGE_FILE).strip : nil
HICACHE_OPTIONS = {
  "--hicache-write-policy" => "write_through",
  "--hicache-io-backend" => "direct",
  "--hicache-mem-layout" => "page_first_direct",
}.freeze
HICACHE_ENV = {
  "SGLANG_HICACHE_RAM_BUDGET" => "${GLM53_HICACHE_RAM_BUDGET:-80%}",
  "SGLANG_HICACHE_POOLED_TRANSFERS" => "1",
  "SGLANG_HICACHE_STAGING_PAGES" => "64",
}.freeze
CANARY_VARIANT = "fc91d24-hicache-pooled-v1-h200-tp4-ep4-eagle-adaptive-5-1-6-strict-budget8192"


def yaml_load(content)
  YAML.load(content, aliases: true)
rescue ArgumentError
  YAML.load(content)
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
rescue ArgumentError => error
  errors << "#{name} command cannot be parsed: #{error.message}"
end

def runtime_contract(service)
  service.reject { |key, _value| REPLICA_IDENTITY_FIELDS.include?(key) }
end

errors = []
if CANARY_IMAGE && (!CANARY_IMAGE.match?(%r{\Adocker\.io/nearaidev/sglang@sha256:[a-f0-9]{64}\z}) || CANARY_IMAGE == ENGINE_IMAGE)
  errors << "HiCache release must pin a distinct immutable image from the signed publishing workflow"
end
errors << "legacy canary compose file must be removed" if File.exist?(LEGACY_CANARY_FILE)

compose = yaml_load(File.read(COMPOSE_FILE))
services = compose.fetch("services", {})

missing_services = EXPECTED_SERVICES - services.keys
extra_services = services.keys - EXPECTED_SERVICES
errors << "canonical config is missing services: #{missing_services.join(', ')}" unless missing_services.empty?
errors << "canonical config has unexpected services: #{extra_services.join(', ')}" unless extra_services.empty?

replica_services = []
REPLICAS.each do |name, expected_devices|
  service = services[name]
  if service.nil?
    errors << "missing services.#{name}"
    next
  end

  replica_services << service
  expected_image = name.end_with?("-r2") && CANARY_IMAGE ? CANARY_IMAGE : ENGINE_IMAGE
  errors << "#{name} image must be #{expected_image}" unless service["image"] == expected_image
  errors << "#{name} must use the prebuilt signed image, not a host-local build" if service.key?("build")

  validate_command(errors, name, service["command"].to_s)

  device_ids = service.dig("deploy", "resources", "reservations", "devices", 0, "device_ids")
  normalized_ids = Array(device_ids).map(&:to_s)
  errors << "#{name} must use GPU device_ids #{expected_devices.join(',')}" unless normalized_ids == expected_devices
end

if replica_services.length == REPLICAS.length
  control, canary = replica_services
  control_arguments = Shellwords.split(control["command"].to_s)
  if control_arguments.any? { |arg| arg.include?("hicache") || arg == "--enable-hierarchical-cache" } ||
     environment_map(control).keys.any? { |key| key.start_with?("SGLANG_HICACHE_") }
    errors << "r1 must remain the HiCache-disabled control"
  end
  if CANARY_IMAGE
    normalized = Marshal.load(Marshal.dump(canary))
    arguments = Shellwords.split(normalized["command"].to_s)
    errors << "r2 must enable HiCache exactly once" unless arguments.count("--enable-hierarchical-cache") == 1
    arguments.delete("--enable-hierarchical-cache")
    HICACHE_OPTIONS.each do |key, expected|
      positions = arguments.each_index.select { |index| arguments[index] == key }
      if positions.length != 1 || arguments[positions.first.to_i + 1] != expected
        errors << "r2 must set #{key} #{expected} exactly once"
      else
        arguments.slice!(positions.first, 2)
      end
    end
    errors << "r2 must preserve all control serving arguments outside HiCache" unless arguments == control_arguments
    env = environment_map(normalized)
    HICACHE_ENV.each do |key, expected|
      errors << "r2 must set #{key}=#{expected}" unless env.delete(key) == expected
    end
    errors << "r2 must preserve all control environment outside pooled transfers" unless env == environment_map(control)
    normalized["image"] = control["image"]
    normalized["command"] = control["command"]
    normalized["environment"] = control["environment"]
    errors << "r2 must preserve the control runtime outside HiCache" unless runtime_contract(normalized) == runtime_contract(control)
    labels = canary.fetch("labels", {})
    errors << "r2 must identify the pooled canary in metric labels" unless labels["nearai.otel.config_variant"] == CANARY_VARIANT
    errors << "r2 must identify the pooled canary in log metadata" unless labels["com.datadoghq.ad.logs"].to_s.include?("config_variant:#{CANARY_VARIANT}")
    collector = yaml_load(compose.dig("configs", "otelcol_app_config", "content"))
    scrape = collector.dig("receivers", "prometheus/apps", "config", "scrape_configs").find { |job| job["job_name"] == "sglang-model-sg-glm53-fp8-tp4-r2" }
    errors << "r2 collector must carry the pooled canary variant" unless scrape.dig("static_configs", 0, "labels", "config_variant") == CANARY_VARIANT
  else
    runtime_contracts = replica_services.map { |service| runtime_contract(service) }
    errors << "GLM-5.3 replicas must use identical runtime configuration until a signed HiCache image is pinned" unless runtime_contracts.uniq.length == 1
  end
end

perception_check = services["glm53-perception-check"]
if perception_check
  errors << "glm53-perception-check image must be #{ENGINE_IMAGE}" unless perception_check["image"] == ENGINE_IMAGE
  errors << "glm53-perception-check must use the prebuilt signed image, not a host-local build" if perception_check.key?("build")
end

proxy = services["proxy-glm53"]
if proxy.nil?
  errors << "missing services.proxy-glm53"
else
  errors << "proxy-glm53 image must be #{PROXY_IMAGE}" unless proxy["image"] == PROXY_IMAGE
  proxy_env = environment_map(proxy)
  expected_backends = REPLICAS.keys.map { |name| "http://#{name}:8000" }.join(",")
  errors << "proxy-glm53 must target both canonical replicas" unless proxy_env["VLLM_BACKEND_URLS"] == expected_backends
  errors << "proxy-glm53 must enable conversation affinity" unless proxy_env["VLLM_BACKEND_CONVERSATION_AFFINITY"] == "1"
end

if errors.any?
  warn "GLM-5.3 production contract failed:"
  errors.each { |error| warn "  - #{error}" }
  exit 1
end

puts "GLM-5.3 production contract OK"
