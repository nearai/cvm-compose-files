#!/usr/bin/env ruby
# Protect the canonical two-replica GLM-5.3 Flash production contract.

require "digest"
require "shellwords"
require "yaml"

ROOT = File.expand_path("..", __dir__)
COMPOSE_FILE = File.join(ROOT, "prod", "GLM-5.3-Flash-SGL-TP4.yaml")
LEGACY_CANARY_FILE = File.join(ROOT, "prod", "GLM-5.3-Flash-SGL-TP4-Canary.yaml")
ENGINE_IMAGE = "nearai/glm53-sglang-upstream-fc91d24:local"
ENGINE_BUILD_SHA256 = "9d559a2f7c5e7ce9c4a0c2377c0552aab74d9f5e26416bd17e19d9b8643e03be"
PROXY_IMAGE = "nearaidev/vllm-proxy-rs@sha256:27265f5a10abcacc3fe80604a00ed2ba32c6e082895115e4b652b40525cb554d"
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
  errors << "#{name} image must be #{ENGINE_IMAGE}" unless service["image"] == ENGINE_IMAGE

  build = service["build"] || {}
  errors << "#{name} build context must be ." unless build["context"] == "."
  build_sha256 = Digest::SHA256.hexdigest(build["dockerfile_inline"].to_s)
  errors << "#{name} engine build recipe changed (expected SHA-256 #{ENGINE_BUILD_SHA256}, got #{build_sha256})" unless build_sha256 == ENGINE_BUILD_SHA256

  validate_command(errors, name, service["command"].to_s)

  device_ids = service.dig("deploy", "resources", "reservations", "devices", 0, "device_ids")
  normalized_ids = Array(device_ids).map(&:to_s)
  errors << "#{name} must use GPU device_ids #{expected_devices.join(',')}" unless normalized_ids == expected_devices
end

if replica_services.length == REPLICAS.length
  runtime_contracts = replica_services.map { |service| runtime_contract(service) }
  errors << "GLM-5.3 replicas must use identical runtime configuration outside identity, labels, and GPU allocation" unless runtime_contracts.uniq.length == 1
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
