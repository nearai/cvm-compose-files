#!/usr/bin/env ruby
# Ensure inference-proxy services pass through required deploy-time env vars.

require "yaml"

ROOT = File.expand_path("..", __dir__)
EXCLUDED_FILES = ["cleanup-hf-model.yaml"].freeze
REQUIRED_PROXY_ENV = %w[
  WEB_CONTEXT_SEARCH_URL
  WEB_CONTEXT_SEARCH_API_KEY
].freeze
# Advisory only: prod proxies should publish replica state (see AGENT.md). Warn, never fail.
REPLICA_STATE_ENV = "REPLICA_STATE_REDIS_URL".freeze

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

def inference_proxy_service?(env)
  env.key?("MODEL_NAME") && env.key?("CLOUD_API_URL")
end

errors = []
warnings = []

compose_files = Dir.glob(File.join(ROOT, "prod", "*.yaml")) +
                Dir.glob(File.join(ROOT, "experiments", "*.yaml")) +
                Dir.glob(File.join(ROOT, "*.yaml"))

compose_files.sort.each do |path|
  file = path.sub("#{ROOT}/", "")
  next if EXCLUDED_FILES.include?(File.basename(path))

  compose = yaml_load(File.read(path))
  services = compose.fetch("services", {})

  services.each do |service_name, service|
    env = service_environment(service)
    next unless inference_proxy_service?(env)

    if file.start_with?("prod/") && !env.key?(REPLICA_STATE_ENV)
      warnings << "#{file}: services.#{service_name}.environment lacks #{REPLICA_STATE_ENV} " \
                  "(replica-state publishing; required for new prod files, see AGENT.md)"
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

warnings.each { |w| warn "::warning::#{w}" }

if errors.any?
  warn "Proxy environment contract failed:"
  errors.each { |error| warn "  - #{error}" }
  exit 1
end

puts "Proxy environment contract OK"
