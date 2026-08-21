#!/usr/bin/env ruby
# Keep proxy startup dependencies explicit without pulling model containers into
# proxy-only compose operations.

require "yaml"

ROOT = File.expand_path("..", __dir__)
EXCLUDED_FILES = ["cleanup-hf-model.yaml"].freeze
QWEN36_DOMAIN = "qwen3-6-35b.completions.near.ai".freeze
QWEN30_DOMAIN = "qwen3-30b.completions.near.ai".freeze
QWEN30_STAGING_DOMAIN = "qwen3-30b.completions-stg.near.ai".freeze

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

def nginx_service?(name)
  name == "nginx" || name == "proxy-nginx"
end

def model_service?(name)
  name == "model-downloader" || name.start_with?("model-")
end

def config_sources(service)
  Array(service["configs"]).map do |config|
    case config
    when Hash
      config["source"]
    else
      config.to_s
    end
  end.compact
end

def proxy_pass_upstreams(compose, service)
  configs = compose.fetch("configs", {})
  config_sources(service).flat_map do |source|
    configs.fetch(source, {}).fetch("content", "").scan(%r{proxy_pass\s+http://([A-Za-z0-9_.-]+)(?::\d+)?}).flatten
  end.uniq
end

def validate_qwen30_compat(compose, file, errors)
  return unless file.start_with?("prod/")

  services = compose.fetch("services", {})
  configs = compose.fetch("configs", {})
  nginx = services.find { |name, _service| nginx_service?(name) }&.last
  return if nginx.nil?

  nginx_content = config_sources(nginx).map do |source|
    configs.fetch(source, {}).fetch("content", "")
  end.join("\n")
  return unless nginx_content.include?(QWEN36_DOMAIN)

  qwen_vhost = nginx_content.match(
    %r{server_name\s+#{Regexp.escape(QWEN36_DOMAIN)}(?<names>.*?);\s+.*?proxy_pass\s+http://proxy-qwen36-35b-a3b:8000;}m,
  )
  if qwen_vhost.nil?
    errors << "#{file}: Qwen3.6 nginx vhost is missing or does not target proxy-qwen36-35b-a3b:8000"
  else
    names = qwen_vhost[:names]
    [QWEN30_DOMAIN, QWEN30_STAGING_DOMAIN, "qwen3-30b-i"].each do |required_name|
      next if names.include?(required_name)

      errors << "#{file}: Qwen3.6 nginx vhost missing compatibility SNI #{required_name}"
    end
  end

  registrar = configs.dig("registrar_script", "content").to_s
  unless registrar.include?('"https://completions-stg.near.ai"')
    errors << "#{file}: registrar alias refresh must cover the staging model-proxy"
  end
  expected_alias = %(register_alias "#{QWEN30_DOMAIN}" "#{QWEN36_DOMAIN}")
  unless registrar.include?(expected_alias)
    errors << "#{file}: registrar missing #{expected_alias}"
  end
end

errors = []

# Find compose files in prod/, experiments/, and root (utilities). Relative
# paths from the repo root are used in error messages.
compose_files = Dir.glob(File.join(ROOT, "prod", "*.yaml")) +
                Dir.glob(File.join(ROOT, "experiments", "*.yaml")) +
                Dir.glob(File.join(ROOT, "*.yaml"))
compose_files.sort.each do |path|
  file = path.sub("#{ROOT}/", "")
  next if EXCLUDED_FILES.include?(File.basename(path))

  compose = yaml_load(File.read(path))
  services = compose.fetch("services", {})
  validate_qwen30_compat(compose, file, errors)

  services.each do |service_name, service|
    deps = dependency_names(service["depends_on"]).map(&:to_s)

    if proxy_service?(service_name)
      model_deps = deps.select { |dep| model_service?(dep) }
      unless model_deps.empty?
        errors << "#{file}: services.#{service_name}.depends_on must not include model services: #{model_deps.join(", ")}"
      end
    end

    next unless nginx_service?(service_name)

    missing_proxy_deps = proxy_pass_upstreams(compose, service).select do |upstream|
      services.key?(upstream) && proxy_service?(upstream) && !deps.include?(upstream)
    end
    next if missing_proxy_deps.empty?

    errors << "#{file}: services.#{service_name}.depends_on missing proxy_pass upstreams: #{missing_proxy_deps.join(", ")}"
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
