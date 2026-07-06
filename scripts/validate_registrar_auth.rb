#!/usr/bin/env ruby
require "yaml"

ROOT = File.expand_path("..", __dir__)
EXCLUDED_FILES = ["cleanup-hf-model.yaml"].freeze
AUTH_HEADER = 'Authorization: Bearer $$INFERENCE_TOKEN'.freeze
INFERENCE_TOKEN = 'INFERENCE_TOKEN="$${PROXY_TOKEN}"'.freeze
PROXY_TOKEN_ENV = "${PROXY_TOKEN}".freeze

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

def curl_blocks(script)
  lines = script.lines
  blocks = []
  index = 0

  while index < lines.length
    line = lines[index]
    unless line.include?("curl ")
      index += 1
      next
    end

    block = line.dup
    while block.lines.last&.rstrip&.end_with?("\\") && index + 1 < lines.length
      index += 1
      block << lines[index]
    end
    blocks << block
    index += 1
  end

  blocks
end

def registrar_probe_blocks(script)
  ready_url_requires_auth = script.match?(/READY_URL="http:\/\/\$\$\{HOST_IP\}:[^"]+\/v1\/models"/)

  curl_blocks(script).select do |block|
    block.include?("/v1/chat/completions") ||
      block.include?("/v1/models") ||
      (ready_url_requires_auth && block.include?("$$READY_URL"))
  end
end

errors = []
compose_files = Dir.glob(File.join(ROOT, "prod", "*.yaml"))

compose_files.sort.each do |path|
  file = path.sub("#{ROOT}/", "")
  next if EXCLUDED_FILES.include?(File.basename(path))

  compose = yaml_load(File.read(path))
  services = compose.fetch("services", {})
  registrar = services["model-proxy-registrar"]
  script = compose.dig("configs", "registrar_script", "content").to_s
  next if registrar.nil? || script.empty?

  probe_blocks = registrar_probe_blocks(script)
  next if probe_blocks.empty?

  env = service_environment(registrar)
  unless env["PROXY_TOKEN"] == PROXY_TOKEN_ENV
    errors << "#{file}: services.model-proxy-registrar.environment missing PROXY_TOKEN=#{PROXY_TOKEN_ENV}"
  end

  unless script.include?(INFERENCE_TOKEN)
    errors << "#{file}: configs.registrar_script.content missing #{INFERENCE_TOKEN}"
  end

  probe_blocks.each_with_index do |block, index|
    next if block.include?(AUTH_HEADER)

    first_line = block.lines.first.to_s.strip
    errors << "#{file}: registrar inference probe #{index + 1} missing #{AUTH_HEADER}: #{first_line}"
  end
rescue StandardError => e
  errors << "#{file}: invalid YAML: #{e.message}"
end

if errors.any?
  warn "Registrar auth contract failed:"
  errors.each { |error| warn "  - #{error}" }
  exit 1
end

puts "Registrar auth contract OK"
