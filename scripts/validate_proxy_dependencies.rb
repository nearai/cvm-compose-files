#!/usr/bin/env ruby
# Prevent proxy-only rollouts from pulling model containers into compose operations.

require "yaml"

ROOT = File.expand_path("..", __dir__)
EXCLUDED_FILES = ["cleanup-hf-model.yaml"].freeze

def yaml_load(content)
  YAML.load(content, aliases: true)
rescue ArgumentError
  YAML.load(content)
end

def dependency_names(raw)
  case raw
  when Hash
    raw.keys
  when Array
    raw
  else
    []
  end
end

def proxy_service?(name)
  name == "proxy-nginx" || name.start_with?("proxy-", "vllm-proxy-")
end

def model_service?(name)
  name == "model-downloader" || name.start_with?("model-")
end

errors = []

Dir.glob(File.join(ROOT, "*.yaml")).sort.each do |path|
  file = File.basename(path)
  next if EXCLUDED_FILES.include?(file)

  compose = yaml_load(File.read(path))
  services = compose.fetch("services", {})

  services.each do |service_name, service|
    next unless proxy_service?(service_name)

    model_deps = dependency_names(service["depends_on"]).select { |dep| model_service?(dep.to_s) }
    next if model_deps.empty?

    errors << "#{file}: services.#{service_name}.depends_on must not include model services: #{model_deps.join(", ")}"
  end
rescue StandardError => e
  errors << "#{file}: invalid YAML: #{e.message}"
end

if errors.any?
  warn "Proxy dependency contract failed:"
  errors.each { |error| warn "  - #{error}" }
  exit 1
end

puts "Proxy dependency contract OK"
