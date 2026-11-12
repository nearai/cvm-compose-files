#!/usr/bin/env ruby
# Engine isolation contract: an inference engine joins internal networks only,
# publishes no ports, carries no Hugging Face token and runs with the hub
# offline. Files opt in through ISOLATED_FILES; every engine service in a
# listed file must satisfy the contract.

require "yaml"

ROOT = File.expand_path("..", __dir__)
ISOLATED_FILES = %w[
  prod/qwen38-27b-abliterated-cc-1gpu.yaml
].freeze
ENGINE_IMAGE = %r{\A(docker\.io/)?(lmsysorg/sglang|vllm/vllm-openai|nearaidev/sglang|nearaidev/vllm)[@:]}
FORBIDDEN_ENV = %w[HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN].freeze

def yaml_load(content)
  YAML.load(content, aliases: true)
rescue ArgumentError
  YAML.load(content)
end

def environment(service)
  env = service["environment"] || []
  return env.transform_values(&:to_s) if env.is_a?(Hash)

  env.each_with_object({}) do |entry, memo|
    key, value = entry.to_s.split("=", 2)
    memo[key] = value.to_s
  end
end

def network_names(service)
  nets = service["networks"]
  case nets
  when Array then nets.map(&:to_s)
  when Hash then nets.keys.map(&:to_s)
  else []
  end
end

def check(file, compose)
  errors = []
  networks = compose["networks"] || {}
  engines = (compose["services"] || {}).select { |_, s| s["image"].to_s.match?(ENGINE_IMAGE) }
  errors << "#{file}: no engine service found" if engines.empty?

  engines.each do |name, service|
    where = "#{file}: services.#{name}"
    nets = network_names(service)
    errors << "#{where}: must list its networks explicitly" if nets.empty?
    errors << "#{where}: network_mode is not allowed" if service.key?("network_mode")
    errors << "#{where}: must not publish ports" if service.key?("ports")
    nets.each do |net|
      errors << "#{where}: network #{net} is not internal" unless networks.dig(net, "internal") == true
    end
    env = environment(service)
    FORBIDDEN_ENV.each do |key|
      errors << "#{where}: #{key} must not be set" if env.key?(key)
    end
    errors << "#{where}: HF_HUB_OFFLINE must be the literal 1" unless env["HF_HUB_OFFLINE"] == "1"
  end
  errors
end

if ARGV.first == "--self-test"
  good = {
    "networks" => { "eng" => { "internal" => true } },
    "services" => { "e" => { "image" => "lmsysorg/sglang@sha256:0", "networks" => ["eng"],
                             "environment" => ["HF_HUB_OFFLINE=1"] } }
  }
  bad = Marshal.load(Marshal.dump(good))
  bad["services"]["e"]["networks"] = ["default"]
  bad["services"]["e"]["environment"] = ["HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}", "HF_TOKEN=x"]
  bad["services"]["e"]["ports"] = ["8000:8000"]
  raise "self-test: clean file rejected" unless check("good", good).empty?
  raise "self-test: bad file accepted" unless check("bad", bad).length == 4

  puts "Engine isolation self-tests OK"
  exit 0
end

errors = ISOLATED_FILES.flat_map do |file|
  path = File.join(ROOT, file)
  next ["#{file}: listed in ISOLATED_FILES but missing"] unless File.exist?(path)

  check(file, yaml_load(File.read(path)))
end

if errors.any?
  warn "Engine isolation contract failed:"
  errors.each { |error| warn "  - #{error}" }
  exit 1
end

puts "Engine isolation contract OK (#{ISOLATED_FILES.length} file(s))"
