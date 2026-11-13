#!/usr/bin/env ruby
# Engine isolation contract: an inference engine joins internal networks only,
# publishes no ports, carries no Hugging Face token and runs with the hub
# offline. Only its proxies, the metrics collector and the egress probe may
# share a network with it, and the collector must not listen for OTLP there.
# An engine is any service that runs an engine image or that a proxy uses as
# its backend. Files opt in through ISOLATED_FILES; every engine service in a
# listed file must satisfy the contract, except the services listed as pending
# for that file.

require "yaml"

ROOT = File.expand_path("..", __dir__)
# File => engine services that do not follow the contract yet.
ISOLATED_FILES = {
  "prod/pack-small-models.yaml" => %w[
    model-sg-qwen36-35b-a3b-fp8-tp1-r1
    model-sg-glm53-w4afp8-tp2-r1
    model-sg-glm53-w4afp8-tp2-r2
    glm53-ghost-aggregator
    model-sg-flux2-klein-4b-tp1
    model-vllm-qwen3vl-30b-a3b-fp8-tp1
    model-vllm-qwen3-embedding-0.6b-tp1
    model-vllm-qwen3-reranker-0.6b-tp1
    model-vllm-whisper-large-v3-tp1
    model-privacy-filter
  ],
  "prod/qwen38-27b-abliterated-cc-1gpu.yaml" => [],
  "prod/small-models.yaml" => %w[
    model-sg-qwen36-35b-a3b-fp8-tp1-r1
    model-sg-glm53-w4afp8-tp2-r1
    model-sg-glm53-w4afp8-tp2-r2
    glm53-ghost-aggregator
    model-sg-flux2-klein-4b-tp1
    model-vllm-qwen3vl-30b-a3b-fp8-tp1
    model-vllm-qwen3-embedding-0.6b-tp1
    model-vllm-qwen3-reranker-0.6b-tp1
    model-vllm-whisper-large-v3-tp1
    model-privacy-filter
  ]
}.freeze
BACKEND_ENV = %w[VLLM_BACKEND_URLS VLLM_BASE_URL].freeze
ENGINE_IMAGE = %r{\A(docker\.io/)?(lmsysorg/sglang|vllm/vllm-openai|nearaidev/sglang|nearaidev/vllm)[@:]}
FORBIDDEN_ENV = %w[HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN].freeze
PROXY_IMAGE = %r{\A(docker\.io/)?nearaidev/vllm-proxy-rs[@:]}
COLLECTOR_IMAGE = %r{\A(docker\.io/)?otel/opentelemetry-collector-contrib[@:]}
PROBE_IMAGE = %r{\A(docker\.io/)?curlimages/curl[@:]}
LOOPBACK = /\A(127\.0\.0\.1|localhost):\d+\z/

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

# Collector configs mounted by the service, parsed from the compose `configs`.
def collector_configs(compose, service)
  configs = (service["configs"] || []).map do |entry|
    source = entry.is_a?(Hash) ? entry["source"] : entry
    content = compose.dig("configs", source.to_s, "content")
    content && yaml_load(content.gsub("$$", "$"))
  end
  configs.compact
end

def otlp_endpoints(config)
  receivers = config.is_a?(Hash) ? config["receivers"] || {} : {}
  receivers.select { |name, _| name.to_s.split("/").first == "otlp" }.flat_map do |_, receiver|
    protocols = (receiver || {})["protocols"] || {}
    # A protocol without an endpoint listens on the collector default.
    protocols.map { |protocol, settings| [protocol, (settings || {})["endpoint"].to_s] }
  end
end

# Everything else that shares a network with an engine.
def check_members(file, compose, engine_networks, engines)
  errors = []
  (compose["services"] || {}).each do |name, service|
    next if engines.key?(name)

    nets = network_names(service)
    shared = nets & engine_networks
    next if shared.empty?

    where = "#{file}: services.#{name}"
    image = service["image"].to_s
    if image.match?(PROXY_IMAGE)
      next
    elsif image.match?(COLLECTOR_IMAGE)
      collector_configs(compose, service).each do |config|
        otlp_endpoints(config).each do |protocol, endpoint|
          next if endpoint.match?(LOOPBACK)

          errors << "#{where}: OTLP #{protocol} receiver must listen on loopback"
        end
      end
    elsif image.match?(PROBE_IMAGE)
      errors << "#{where}: the probe must join engine networks only" unless (nets - engine_networks).empty?
    else
      errors << "#{where}: not allowed on engine network #{shared.join(', ')}"
    end
  end
  errors
end

# Service names the proxies in this file send requests to.
def proxy_backends(file, services, errors)
  by_container = services.to_h { |name, service| [service["container_name"].to_s, name] }
  found = services.select { |_, s| s["image"].to_s.match?(PROXY_IMAGE) }.flat_map do |name, service|
    env = environment(service)
    BACKEND_ENV.flat_map { |key| env[key].to_s.split(",", -1) if env.key?(key) }.compact.map do |raw|
      url = raw.strip
      # Scheme optional; the host is a name, an IPv4 address or a bracketed IPv6 address.
      host = url[%r{\A(?:\w+://)?(\[[^\]]+\]|[^:/\[\]$]+)(?::\d+)?(?:/.*)?\z}, 1]
      backend = services.key?(host) ? host : by_container[host]
      errors << "#{file}: services.#{name}: backend URL #{url.inspect} does not name a service in this file" unless backend
      backend
    end
  end
  found.compact
end

def engine_errors(where, service, networks)
  errors = []
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
  errors
end

def check(file, compose, pending = [])
  errors = []
  networks = compose["networks"] || {}
  services = compose["services"] || {}
  backends = proxy_backends(file, services, errors)
  detected = services.select { |name, s| s["image"].to_s.match?(ENGINE_IMAGE) || backends.include?(name) }
  errors << "#{file}: no engine service found" if detected.empty?
  pending.each do |name|
    if !detected.key?(name)
      errors << "#{file}: pending service #{name} is not an engine in this file"
    elsif engine_errors(name, detected[name], networks).empty?
      errors << "#{file}: pending service #{name} follows the contract now; remove it from the pending list"
    end
  end
  engines = detected.reject { |name, _| pending.include?(name) }
  engine_networks = engines.values.flat_map { |service| network_names(service) }.uniq
  errors.concat(check_members(file, compose, engine_networks, engines))

  engines.each do |name, service|
    errors.concat(engine_errors("#{file}: services.#{name}", service, networks))
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

  members = Marshal.load(Marshal.dump(good))
  members["configs"] = { "col" => { "content" => "receivers:\n  otlp:\n    protocols:\n      grpc:\n        endpoint: 127.0.0.1:4317\n" } }
  members["services"].merge!(
    "proxy" => { "image" => "nearaidev/vllm-proxy-rs@sha256:0", "networks" => %w[default eng] },
    "collector" => { "image" => "otel/opentelemetry-collector-contrib@sha256:0", "networks" => %w[default eng],
                     "configs" => [{ "source" => "col" }] },
    "probe" => { "image" => "curlimages/curl@sha256:0", "networks" => ["eng"] },
    "other" => { "image" => "nginx@sha256:0" }
  )
  raise "self-test: allowed members rejected" unless check("members", members).empty?

  members["configs"]["col"]["content"] = "receivers:\n  otlp:\n    protocols:\n      grpc:\n        endpoint: 0.0.0.0:4317\n      http:\n"
  members["services"]["probe"]["networks"] = %w[default eng]
  members["services"]["other"]["networks"] = %w[default eng]
  raise "self-test: bad members accepted" unless check("members", members).length == 4

  custom = Marshal.load(Marshal.dump(good))
  custom["services"].merge!(
    "proxy" => { "image" => "nearaidev/vllm-proxy-rs@sha256:0", "networks" => %w[default eng],
                 "environment" => ["VLLM_BASE_URL=http://custom-engine:8000"] },
    "custom" => { "image" => "local-build", "container_name" => "custom-engine" }
  )
  raise "self-test: custom-image backend not treated as an engine" unless check("custom", custom).length == 2
  raise "self-test: pending engine rejected" unless check("custom", custom, ["custom"]).empty?
  raise "self-test: stale pending entry accepted" unless check("custom", custom, %w[custom gone]).length == 1

  compliant = Marshal.load(Marshal.dump(custom))
  compliant["services"]["custom"].merge!("networks" => ["eng"], "environment" => ["HF_HUB_OFFLINE=1"])
  stale = check("custom", compliant, ["custom"]).select { |error| error.include?("follows the contract now") }
  raise "self-test: compliant pending engine accepted" unless stale.length == 1

  custom["services"]["proxy"]["environment"] = ["VLLM_BACKEND_URLS=http://e:8000,http://elsewhere:8000"]
  raise "self-test: unknown backend accepted" unless check("custom", custom, ["custom"]).length == 2

  # Backends the parser must handle: no scheme, container name, and the forms it must refuse by URL.
  custom["services"]["proxy"]["environment"] = ["VLLM_BACKEND_URLS=custom-engine:8000,e:8000"]
  raise "self-test: scheme-less backend not resolved" unless check("custom", custom, ["custom"]).empty?
  { "http://${ENGINE_HOST}:8000" => "http://${ENGINE_HOST}:8000", "http://e:8000," => "",
    "http://[fd00::2]:8000" => "http://[fd00::2]:8000" }.each do |value, shown|
    custom["services"]["proxy"]["environment"] = ["VLLM_BACKEND_URLS=#{value}"]
    found = check("custom", custom).select { |error| error.include?("backend URL") }
    raise "self-test: backend #{value} accepted" unless found.length == 1
    raise "self-test: backend error does not show the URL" unless found.first.include?(shown.inspect)
  end

  puts "Engine isolation self-tests OK"
  exit 0
end

errors = ISOLATED_FILES.flat_map do |file, pending|
  path = File.join(ROOT, file)
  next ["#{file}: listed in ISOLATED_FILES but missing"] unless File.exist?(path)

  check(file, yaml_load(File.read(path)), pending)
end

if errors.any?
  warn "Engine isolation contract failed:"
  errors.each { |error| warn "  - #{error}" }
  exit 1
end

pending = ISOLATED_FILES.values.sum(&:length)
puts "Engine isolation contract OK (#{ISOLATED_FILES.length} file(s), #{pending} engine(s) pending)"
