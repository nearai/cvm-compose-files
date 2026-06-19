#!/usr/bin/env ruby
# Validates the label contract shared by Datadog logs and the app OTel collector.

require "json"
require "yaml"

ROOT = File.expand_path("..", __dir__)
EXCLUDED_FILES = ["cleanup-hf-model.yaml"].freeze
REQUIRED_COLLECTOR_ENV = %w[
  MONITORING_INGEST_TOKEN
  CVM_NAME
  CVM_HOST
  DD_HOSTNAME
  ENV
  HOST_IP
].freeze
REQUIRED_LOG_TAGS = %w[deployment env host ip].freeze

def add_error(errors, file, path, message)
  errors << [file, path, message]
end

def yaml_load(content)
  YAML.load(content, aliases: true)
rescue ArgumentError
  YAML.load(content)
end

def load_compose(path, errors)
  yaml_load(File.read(path))
rescue StandardError => e
  add_error(errors, path.sub("#{ROOT}/", ""), "$", "invalid YAML: #{e.message}")
  nil
end

def docker_log_config(raw)
  parsed = JSON.parse(raw)
  raise "expected non-empty JSON array" unless parsed.is_a?(Array) && parsed.first.is_a?(Hash)

  parsed.first
end

def tag_map(tags)
  Array(tags).each_with_object({}) do |tag, memo|
    key, value = tag.to_s.split(":", 2)
    memo[key] = value if key && value
  end
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

def scrape_targets(collector_config)
  parsed = yaml_load(collector_config)
  scrape_configs = parsed.dig("receivers", "prometheus/apps", "config", "scrape_configs") || []
  scrape_configs.each_with_object({}) do |scrape, memo|
    Array(scrape["static_configs"]).each do |static_config|
      Array(static_config["targets"]).each do |target|
        service_name, scrape_port = target.to_s.split(":", 2)
        memo[service_name] = {
          "target" => target,
          "scrape_port" => scrape_port,
          "metrics_path" => scrape["metrics_path"],
          "labels" => static_config["labels"] || {},
        }
      end
    end
  end
end

def public_ports_by_proxy(compose)
  content = compose.dig("configs", "nginx_conf", "content").to_s
  content.scan(/listen\s+(\d+);(?:(?!listen\s+\d+;).)*?proxy_pass\s+http:\/\/([^:;]+):8000;/m)
         .each_with_object({}) do |(port, proxy_service), memo|
    memo[proxy_service] ||= port
  end
end

def expected_public_ports(compose)
  proxy_ports = public_ports_by_proxy(compose)
  expected = proxy_ports.dup

  (compose["services"] || {}).each do |service_name, service|
    next unless proxy_ports.key?(service_name)

    env = service_environment(service)
    backend_urls = [env["VLLM_BASE_URL"], env["VLLM_BACKEND_URLS"]].compact.join(",")
    backend_urls.scan(/http:\/\/([^:\/,]+):\d+/).flatten.each do |backend_service|
      expected[backend_service] ||= proxy_ports[service_name]
    end
  end

  expected
end

def validate_log_label(file, service_name, service, errors)
  raw = service.dig("labels", "com.datadoghq.ad.logs")
  return nil unless raw

  config = docker_log_config(raw)
  source = config["source"]
  service_label = config["service"]
  tags = tag_map(config["tags"])

  add_error(errors, file, "services.#{service_name}.labels.com.datadoghq.ad.logs", "missing source") if source.to_s.empty?
  add_error(errors, file, "services.#{service_name}.labels.com.datadoghq.ad.logs", "missing service") if service_label.to_s.empty?

  REQUIRED_LOG_TAGS.each do |key|
    add_error(errors, file, "services.#{service_name}.labels.com.datadoghq.ad.logs", "missing #{key}: tag") unless tags.key?(key)
  end

  unless service_name == "otelcol-contrib"
    add_error(errors, file, "services.#{service_name}.labels.com.datadoghq.ad.logs", "missing model: tag") unless tags.key?("model")
  end

  otel_service = service.dig("labels", "nearai.otel.service")
  if otel_service && source != otel_service
    add_error(errors, file, "services.#{service_name}.labels.com.datadoghq.ad.logs", "source #{source.inspect} does not match nearai.otel.service #{otel_service.inspect}")
  end
  if otel_service && service_label != otel_service
    add_error(errors, file, "services.#{service_name}.labels.com.datadoghq.ad.logs", "service #{service_label.inspect} does not match nearai.otel.service #{otel_service.inspect}")
  end

  tags
rescue StandardError => e
  add_error(errors, file, "services.#{service_name}.labels.com.datadoghq.ad.logs", "invalid JSON: #{e.message}")
  nil
end

def validate_datadog_instances(file, service_name, service, log_tags, errors)
  raw = service.dig("labels", "com.datadoghq.ad.instances")
  return unless raw && log_tags

  instances = JSON.parse(raw)
  first = instances.first || {}
  instance_tags = tag_map(first["tags"])
  %w[model port].each do |key|
    next unless log_tags[key] && instance_tags[key]
    next if log_tags[key] == instance_tags[key]

    add_error(errors, file, "services.#{service_name}.labels.com.datadoghq.ad.instances", "#{key} tag #{instance_tags[key].inspect} does not match log tag #{log_tags[key].inspect}")
  end
rescue StandardError => e
  add_error(errors, file, "services.#{service_name}.labels.com.datadoghq.ad.instances", "invalid JSON: #{e.message}")
end

def validate_collector_service(file, compose, errors)
  services = compose["services"] || {}
  configs = compose["configs"] || {}
  volumes = compose["volumes"] || {}
  collector = services["otelcol-contrib"]

  add_error(errors, file, "services.otelcol-contrib", "missing app collector service") unless collector
  add_error(errors, file, "configs.otelcol_app_config", "missing app collector config") unless configs.dig("otelcol_app_config", "content")
  add_error(errors, file, "volumes.otelcol_app_storage", "missing app collector storage volume") unless volumes.key?("otelcol_app_storage")
  return unless collector

  env_entries = Array(collector["environment"]).map(&:to_s)
  REQUIRED_COLLECTOR_ENV.each do |var|
    marker = "${#{var}:-"
    unless env_entries.any? { |entry| entry.start_with?("#{var}=") && entry.include?(marker) }
      add_error(errors, file, "services.otelcol-contrib.environment", "missing soft-default marker for #{var}")
    end
  end
end

def validate_scrape_contract(file, compose, log_tags_by_service, errors)
  config = compose.dig("configs", "otelcol_app_config", "content")
  return unless config

  if config.include?("datadog-agent:4317") || config.include?("otlp/datadog_agent")
    add_error(errors, file, "configs.otelcol_app_config", "standalone inference compose files must not export OTLP to datadog-agent:4317")
  end

  targets = scrape_targets(config)
  expected_ports = expected_public_ports(compose)
  scrape_services = {}

  (compose["services"] || {}).each do |service_name, service|
    labels = service["labels"] || {}
    next unless labels["nearai.otel.scrape"].to_s == "true"

    scrape_services[service_name] = true
    target = targets[service_name]
    unless target
      add_error(errors, file, "configs.otelcol_app_config", "missing scrape target for #{service_name}")
      next
    end

    expected_target = "#{service_name}:#{labels["nearai.otel.port"]}"
    if target["target"] != expected_target
      add_error(errors, file, "configs.otelcol_app_config", "#{service_name} target #{target["target"].inspect} should be #{expected_target.inspect}")
    end
    if target["metrics_path"] != labels["nearai.otel.path"]
      add_error(errors, file, "configs.otelcol_app_config", "#{service_name} metrics_path #{target["metrics_path"].inspect} does not match nearai.otel.path")
    end

    target_labels = target["labels"]
    {
      "service" => labels["nearai.otel.service"],
      "container_name" => service_name,
      "model" => labels["nearai.otel.model"],
      "deployment" => labels["nearai.otel.deployment"],
      "env" => labels["nearai.otel.env"],
      "host" => labels["nearai.otel.host"],
      "ip" => labels["nearai.otel.ip"],
    }.each do |key, expected|
      next if expected.to_s.empty?
      next if target_labels[key] == expected

      add_error(errors, file, "configs.otelcol_app_config", "#{service_name} #{key} label #{target_labels[key].inspect} should be #{expected.inspect}")
    end

    expected_public_port = expected_ports[service_name] || log_tags_by_service.dig(service_name, "port") || labels["nearai.otel.port"]
    if expected_public_port && target_labels["port"] != expected_public_port
      add_error(errors, file, "configs.otelcol_app_config", "#{service_name} public port label #{target_labels["port"].inspect} should be #{expected_public_port.inspect}")
    end
  end

  extra_targets = targets.keys - scrape_services.keys
  extra_targets.each do |service_name|
    add_error(errors, file, "configs.otelcol_app_config", "extra scrape target #{service_name} has no nearai.otel.scrape=true service")
  end
rescue StandardError => e
  add_error(errors, file, "configs.otelcol_app_config", "invalid collector config: #{e.message}")
end

errors = []

# Recursively find all compose files under prod/ and experiments/, plus any
# root-level utilities (cleanup-hf-model.yaml). Sort for deterministic error
# output. Relative paths from the repo root are used in error messages so
# same-named files in different dirs are distinguishable.
compose_files = Dir.glob(File.join(ROOT, "prod", "*.yaml")) +
                Dir.glob(File.join(ROOT, "experiments", "*.yaml")) +
                Dir.glob(File.join(ROOT, "*.yaml"))
compose_files.sort.each do |path|
  file = path.sub("#{ROOT}/", "")
  compose = load_compose(path, errors)
  next unless compose
  next if EXCLUDED_FILES.include?(File.basename(path))

  validate_collector_service(file, compose, errors)

  log_tags_by_service = {}
  (compose["services"] || {}).each do |service_name, service|
    log_tags = validate_log_label(file, service_name, service, errors)
    log_tags_by_service[service_name] = log_tags if log_tags
    validate_datadog_instances(file, service_name, service, log_tags, errors)
  end

  expected_public_ports(compose).each do |service_name, port|
    log_port = log_tags_by_service.dig(service_name, "port")
    next unless log_port && log_port != port

    add_error(errors, file, "services.#{service_name}.labels.com.datadoghq.ad.logs", "port tag #{log_port.inspect} should match nginx public port #{port.inspect}")
  end

  validate_scrape_contract(file, compose, log_tags_by_service, errors)
end

if errors.any?
  errors.each do |file, path, message|
    warn "::error file=#{file}::#{path}: #{message}"
  end
  exit 1
end

puts "OTel label contract OK"
