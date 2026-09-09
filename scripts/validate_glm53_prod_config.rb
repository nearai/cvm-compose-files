#!/usr/bin/env ruby
# Protect the canonical two-replica GLM-5.3 Flash production contract.

require "yaml"

ROOT = File.expand_path("..", __dir__)
COMPOSE_FILE = File.join(ROOT, "prod", "GLM-5.3-Flash-SGL-TP4.yaml")
LEGACY_CANARY_FILE = File.join(ROOT, "prod", "GLM-5.3-Flash-SGL-TP4-Canary.yaml")
ENGINE_IMAGE = "nearai/glm53-sglang-upstream-fc91d24:local"
REPLICAS = {
  "model-sg-glm53-fp8-tp4-r1" => %w[0 1 2 3],
  "model-sg-glm53-fp8-tp4-r2" => %w[4 5 6 7],
}.freeze
REQUIRED_COMMAND_FRAGMENTS = [
  "--revision 84c6a6aa9497188e15a635ba793b0f95a79b1033",
  "--served-model-name z-ai/glm-5.3-flash",
  "--tp-size 4",
  "--ep-size 4",
  "--mem-fraction-static 0.80",
  "--max-running-requests 32",
  "--max-queued-requests 32",
  "--chunked-prefill-size 4096",
  "--cuda-graph-max-bs-decode 32",
  "--kv-cache-dtype bfloat16",
  "--moe-runner-backend deep_gemm",
  "--speculative-num-steps 5",
  "--speculative-eagle-topk 1",
  "--speculative-num-draft-tokens 6",
  "--speculative-adaptive",
  "--reasoning-parser glm45",
  "--enable-strict-thinking",
  "--tool-call-parser glm47",
  "--context-length 1048576",
  "--disable-fast-image-processor",
  "--limit-mm-data-per-request '{\"image\": 64}'",
  "--enable-metrics",
  "--log-requests-level 0",
].freeze

def yaml_load(content)
  YAML.load(content, aliases: true)
rescue ArgumentError
  YAML.load(content)
end

def environment_map(service)
  Array(service["environment"]).to_h do |entry|
    key, value = entry.split("=", 2)
    [key, value]
  end
end

errors = []
errors << "legacy canary compose file must be removed" if File.exist?(LEGACY_CANARY_FILE)

compose = yaml_load(File.read(COMPOSE_FILE))
services = compose.fetch("services", {})

forbidden_services = services.keys.grep(/dsv4|deepseek/i)
unless forbidden_services.empty?
  errors << "canonical GLM-5.3 config must not include other model services: #{forbidden_services.join(', ')}"
end

commands = []
REPLICAS.each do |name, expected_devices|
  service = services[name]
  if service.nil?
    errors << "missing services.#{name}"
    next
  end

  errors << "#{name} image must be #{ENGINE_IMAGE}" unless service["image"] == ENGINE_IMAGE

  command = service["command"].to_s
  commands << command
  REQUIRED_COMMAND_FRAGMENTS.each do |fragment|
    errors << "#{name} command missing #{fragment}" unless command.include?(fragment)
  end

  device_ids = service.dig("deploy", "resources", "reservations", "devices", 0, "device_ids")
  errors << "#{name} must use GPU device_ids #{expected_devices.join(',')}" unless device_ids == expected_devices
end

if commands.length == REPLICAS.length && commands.uniq.length != 1
  errors << "GLM-5.3 replicas must use identical engine commands"
end

proxy = services["proxy-glm53"]
if proxy.nil?
  errors << "missing services.proxy-glm53"
else
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
