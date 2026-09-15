#!/usr/bin/env ruby
# Ensure every production inference proxy has its own directory on the
# Compose-managed durable usage-outbox volume.

require "yaml"

ROOT = File.expand_path("..", __dir__)
VOLUME = "inference-proxy-usage-outbox"
MOUNT = "/var/lib/inference-proxy/usage-outbox"

def yaml_load(content)
  YAML.load(content, aliases: true)
rescue ArgumentError
  YAML.load(content)
end

def service_environment(service)
  environment = service["environment"] || []
  case environment
  when Array
    environment.each_with_object({}) do |entry, memo|
      key, value = entry.to_s.split("=", 2)
      memo[key] = value
    end
  when Hash
    environment
  else
    {}
  end
end

def service_volumes(service)
  Array(service["volumes"]).map do |volume|
    case volume
    when Hash
      [volume["source"].to_s, volume["target"].to_s]
    else
      parts = volume.to_s.split(":", 3)
      [parts[0], parts[1]]
    end
  end
end

errors = []
proxy_count = 0

Dir.glob(File.join(ROOT, "prod", "*.yaml")).sort.each do |path|
  file = path.sub("#{ROOT}/", "")
  compose = yaml_load(File.read(path))
  services = compose.fetch("services", {})
  proxy_directories = {}

  services.each do |service_name, service|
    env = service_environment(service)
    next unless env.key?("MODEL_NAME") && env.key?("CLOUD_API_URL")

    proxy_count += 1
    expected_directory = "#{MOUNT}/#{service_name}"
    actual_directory = env["CLOUD_API_USAGE_OUTBOX_DIR"]
    if actual_directory != expected_directory
      errors << "#{file}: services.#{service_name}.environment CLOUD_API_USAGE_OUTBOX_DIR must be #{expected_directory.inspect}, got #{actual_directory.inspect}"
    end

    if env["CLOUD_API_USAGE_TOKEN"].to_s.empty?
      errors << "#{file}: services.#{service_name}.environment missing CLOUD_API_USAGE_TOKEN"
    end

    unless service_volumes(service).include?([VOLUME, MOUNT])
      errors << "#{file}: services.#{service_name}.volumes must mount #{VOLUME.inspect} at #{MOUNT.inspect}"
    end

    if proxy_directories.key?(actual_directory)
      errors << "#{file}: services.#{service_name} and #{proxy_directories[actual_directory]} share usage outbox directory #{actual_directory.inspect}"
    else
      proxy_directories[actual_directory] = service_name
    end
  end

  next if proxy_directories.empty?

  unless compose.fetch("volumes", {}).key?(VOLUME)
    errors << "#{file}: top-level volumes missing #{VOLUME.inspect}"
  end
rescue StandardError => e
  errors << "#{file}: invalid YAML: #{e.message}"
end

errors << "no production inference proxies found" if proxy_count.zero?

if errors.any?
  warn "Durable usage outbox contract failed:"
  errors.each { |error| warn "  - #{error}" }
  exit 1
end

puts "Durable usage outbox contract OK (#{proxy_count} proxy services)"
