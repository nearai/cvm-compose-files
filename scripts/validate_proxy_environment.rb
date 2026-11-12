#!/usr/bin/env ruby
# Ensure inference-proxy services pass through required deploy-time env vars.

require "yaml"

ROOT = File.expand_path("..", __dir__)
# Bare-metal OpenRouter stacks: no cloud-api traffic, so no web search or media policy env.
EXCLUDED_FILES = [
  "gpt-oss-120b-openrouter-baremetal.yaml",
  "qwen38-27b-openrouter-baremetal.yaml",
].freeze
REQUIRED_PROXY_ENV = %w[
  WEB_CONTEXT_SEARCH_URL
  WEB_CONTEXT_SEARCH_API_KEY
].freeze
PROXY_IMAGE_PREFIX = "nearaidev/vllm-proxy-rs"
# Under prod/, every proxy inlines remote media and refuses video parts by default;
# a deploy can still override the value through the environment.
REQUIRED_PROD_PROXY_ENV = {
  "VLLM_PROXY_INLINE_MEDIA" => "${VLLM_PROXY_INLINE_MEDIA:-1}",
  "VLLM_PROXY_REJECTED_CONTENT_PART_TYPES" => "${VLLM_PROXY_REJECTED_CONTENT_PART_TYPES:-video_url}",
}.freeze

def yaml_load(content)
  YAML.load(content, aliases: true)
rescue ArgumentError
  YAML.load(content)
end

def service_environment(service)
  env = service["environment"] || []
  case env
  when Array
    env.each_with_object({}) do |entry, memo|
      key, value = entry.to_s.split("=", 2)
      memo[key] = value
    end
  when Hash
    env
  else
    {}
  end
end

def inference_proxy_service?(file, service, env)
  return true if env.key?("MODEL_NAME") && env.key?("CLOUD_API_URL")

  # Under prod/ the image alone identifies a proxy, so no proxy escapes the checks below.
  file.start_with?("prod/") && service["image"].to_s.start_with?(PROXY_IMAGE_PREFIX)
end

errors = []

compose_files = Dir.glob(File.join(ROOT, "prod", "*.yaml"))

compose_files.sort.each do |path|
  file = path.sub("#{ROOT}/", "")
  next if EXCLUDED_FILES.include?(File.basename(path))

  compose = yaml_load(File.read(path))
  services = compose.fetch("services", {})

  services.each do |service_name, service|
    env = service_environment(service)
    next unless inference_proxy_service?(file, service, env)

    if file.start_with?("prod/")
      REQUIRED_PROD_PROXY_ENV.each do |name, expected|
        actual = env[name]
        next if actual == expected

        problem = actual.nil? ? "is missing" : "is #{actual.inspect}"
        errors << "#{file}: services.#{service_name}.environment #{name} #{problem}, expected #{expected.inspect}"
      end
    end

    missing = REQUIRED_PROXY_ENV.reject do |name|
      env[name] == "${#{name}}"
    end
    next if missing.empty?

    errors << "#{file}: services.#{service_name}.environment missing #{missing.join(", ")}"
  end
rescue StandardError => e
  errors << "#{file}: invalid YAML: #{e.message}"
end

if errors.any?
  warn "Proxy environment contract failed:"
  errors.each { |error| warn "  - #{error}" }
  exit 1
end

puts "Proxy environment contract OK"
