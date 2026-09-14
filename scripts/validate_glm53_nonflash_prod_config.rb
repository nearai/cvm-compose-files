#!/usr/bin/env ruby
# Protect the GLM-5.3 non-Flash TP8/full-context production contract.

require "shellwords"
require "yaml"

ROOT = File.expand_path("..", __dir__)
COMPOSE_FILE = File.join(ROOT, "prod", "GLM-5.3-W4AFP8-SGL-TP8.yaml")
RELEASE_FILE = File.join(ROOT, "docker", "sglang-glm53-v0519", "RELEASED_IMAGE")
ENGINE_SERVICE = "model-sg-glm53-w4afp8-tp8"
MODEL_REVISION = "03179e95916f9f79a92536e76da63f9251d93d4f"
EXPECTED_SERVICES = %w[
  model-downloader
  hf-cleanup
  nginx
  model-proxy-registrar
  proxy-glm53
  model-sg-glm53-w4afp8-tp8
  dcgm-glm53
  otelcol-contrib
].freeze
REQUIRED_OPTIONS = {
  "--model-path" => "/root/.cache/huggingface/hub/models--PhalaCloud--GLM-5.3-W4AFP8/snapshots/#{MODEL_REVISION}",
  "--revision" => MODEL_REVISION,
  "--served-model-name" => "z-ai/glm-5.3",
  "--quantization" => "w4afp8",
  "--tp" => "8",
  "--kv-cache-dtype" => "fp8_e4m3",
  "--dsa-prefill-backend" => "flashmla_sparse_q8",
  "--dsa-decode-backend" => "flashmla_kv",
  "--speculative-algorithm" => "EAGLE",
  "--speculative-num-steps" => "3",
  "--speculative-eagle-topk" => "1",
  "--speculative-num-draft-tokens" => "4",
  "--reasoning-parser" => "glm45",
  "--tool-call-parser" => "glm47",
  "--chunked-prefill-size" => "32768",
  "--mem-fraction-static" => "0.76",
  "--max-running-requests" => "48",
  "--max-queued-requests" => "64",
  "--cuda-graph-max-bs" => "64",
  "--context-length" => "1048576",
  "--dist-init-addr" => "127.0.0.1:29500",
  "--watchdog-timeout" => "1800",
  "--log-requests-level" => "0",
  "--api-key" => "$$ENGINE_API_TOKEN",
}.freeze
REQUIRED_SWITCHES = %w[
  --disable-shared-experts-fusion
  --disable-custom-all-reduce
  --enable-cache-report
  --enable-metrics
  --enable-trace
  --enable-request-time-stats-logging
  --trust-remote-code
].freeze

def yaml_load(path)
  YAML.load_file(path, aliases: true)
rescue ArgumentError
  YAML.load_file(path)
end

def environment_map(service)
  environment = service["environment"]
  return environment.transform_values(&:to_s) if environment.is_a?(Hash)

  Array(environment).to_h do |entry|
    key, value = entry.to_s.split("=", 2)
    [key, value]
  end
end

errors = []
unless File.exist?(RELEASE_FILE)
  errors << "missing #{RELEASE_FILE}; publish and verify the protected-main image before production"
  engine_image = nil
else
  engine_image = File.read(RELEASE_FILE).strip
  unless engine_image.match?(%r{\Adocker\.io/nearaidev/sglang@sha256:[a-f0-9]{64}\z})
    errors << "RELEASED_IMAGE must contain one immutable nearaidev/sglang digest"
  end
end

compose = yaml_load(COMPOSE_FILE)
services = compose.fetch("services", {})
missing = EXPECTED_SERVICES - services.keys
extra = services.keys - EXPECTED_SERVICES
errors << "missing services: #{missing.join(', ')}" unless missing.empty?
errors << "unexpected services: #{extra.join(', ')}" unless extra.empty?

engine = services[ENGINE_SERVICE]
if engine
  errors << "#{ENGINE_SERVICE} image must match RELEASED_IMAGE" unless engine_image && engine["image"] == engine_image
  errors << "#{ENGINE_SERVICE} must not use a host-local build" if engine.key?("build")
  errors << "#{ENGINE_SERVICE} must not publish a host port" if engine.key?("ports")
  expected_entrypoint = ["/opt/nvidia/nvidia_entrypoint.sh", "/bin/bash", "-lc"]
  errors << "#{ENGINE_SERVICE} must use the token-expanding entrypoint" unless engine["entrypoint"] == expected_entrypoint

  command = Array(engine["command"]).join("\n")
  arguments = Shellwords.split(command)
  errors << "#{ENGINE_SERVICE} command must start with exec sglang serve" unless arguments.first(3) == %w[exec sglang serve]
  REQUIRED_OPTIONS.each do |option, expected|
    positions = arguments.each_index.select { |index| arguments[index] == option }
    if positions.length != 1
      errors << "#{ENGINE_SERVICE} must contain #{option} exactly once"
      next
    end
    actual = arguments[positions.first + 1]
    errors << "#{ENGINE_SERVICE} #{option} must be #{expected.inspect}, got #{actual.inspect}" unless actual == expected
  end
  REQUIRED_SWITCHES.each do |option|
    errors << "#{ENGINE_SERVICE} must contain #{option} exactly once" unless arguments.count(option) == 1
  end
  if arguments.any? { |argument| argument.include?("hicache") || argument == "--enable-hierarchical-cache" }
    errors << "#{ENGINE_SERVICE} must remain HiCache-disabled"
  end

  environment = environment_map(engine)
  errors << "SGLANG_ENABLE_JIT_DEEPGEMM must remain 0" unless environment["SGLANG_ENABLE_JIT_DEEPGEMM"] == "0"
  errors << "ENGINE_API_TOKEN must be required for the engine" unless environment["ENGINE_API_TOKEN"] == "${ENGINE_API_TOKEN:?ENGINE_API_TOKEN is required}"
  if environment.keys.any? { |key| key.start_with?("SGLANG_HICACHE_") }
    errors << "#{ENGINE_SERVICE} must not set experimental HiCache environment variables"
  end

  device_ids = engine.dig("deploy", "resources", "reservations", "devices", 0, "device_ids")
  errors << "#{ENGINE_SERVICE} must own GPUs 0-7" unless Array(device_ids).map(&:to_s) == (0..7).map(&:to_s)
else
  errors << "missing services.#{ENGINE_SERVICE}"
end

proxy = services["proxy-glm53"]
if proxy
  proxy_env = environment_map(proxy)
  errors << "proxy must target only the TP8 engine" unless proxy_env["VLLM_BACKEND_URLS"] == "http://#{ENGINE_SERVICE}:8000"
  errors << "proxy must authenticate with PROXY_TOKEN" unless proxy_env["TOKEN"] == "${PROXY_TOKEN}"
  errors << "proxy must require the dedicated engine credential" unless proxy_env["VLLM_BACKEND_API_KEY"] == "${ENGINE_API_TOKEN:?ENGINE_API_TOKEN is required}"
  errors << "proxy model identity must be z-ai/glm-5.3" unless proxy_env["MODEL_NAME"] == "z-ai/glm-5.3"
end

registrar = compose.dig("configs", "registrar_script", "content").to_s
errors << "registrar must use the canonical GLM-5.3 SNI" unless registrar.include?('register_model "z-ai/glm-5.3" "glm-5-3.completions.near.ai"')
errors << "registrar readiness must authenticate" unless registrar.include?('-H "Authorization: Bearer $$INFERENCE_TOKEN"')

nginx = compose.dig("configs", "nginx_conf", "content").to_s
%w[glm-5-3.completions.near.ai glm-5-3.completions-stg.near.ai].each do |domain|
  errors << "nginx must accept #{domain}" unless nginx.include?(domain)
end

downloader = services["model-downloader"]
if downloader
  command = Array(downloader["command"]).join("\n")
  errors << "downloader must pin PhalaCloud/GLM-5.3-W4AFP8" unless command.include?("PhalaCloud/GLM-5.3-W4AFP8")
  errors << "downloader must pin #{MODEL_REVISION}" unless command.include?(MODEL_REVISION)
end

if errors.any?
  warn "GLM-5.3 non-Flash production contract failed:"
  errors.each { |error| warn "  - #{error}" }
  exit 1
end

puts "GLM-5.3 non-Flash production contract OK"
