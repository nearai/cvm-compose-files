#!/usr/bin/env ruby
# Guard the two-host capacity move, including physical GPU ownership and telemetry.
require "yaml"
require "set"
require "digest"

root = File.expand_path("..", __dir__)
load_yaml = ->(file) { YAML.load_file(File.join(root, file), aliases: true) }
packs = {
  "gpu02" => load_yaml.call("prod/DSV4-GLM53-After-Upgrade.yaml"),
  "gpu13" => load_yaml.call("prod/small-models.yaml")
}
legacy = load_yaml.call("prod/dsv4-qwen38-glm51.yaml")
expected = {
  "gpu02" => {
    "model-sg-dsv4-flash-fp4-tp2-r1" => %w[0 1],
    "model-sg-dsv4-flash-fp4-tp2-r2" => %w[2 3],
    "model-sg-glm53-fp8-tp4" => %w[4 5 6 7]
  },
  "gpu13" => {
    "model-privacy-filter" => %w[0],
    "model-sg-qwen36-35b-a3b-fp8-tp1-r1" => %w[1],
    "model-sg-qwen38-27b-fp8-tp1" => %w[2],
    "model-sg-glm51-awq-tp4" => %w[3 4 5 6],
    "model-sg-flux2-klein-4b-tp1" => %w[7],
    "model-vllm-qwen3vl-30b-a3b-fp8-tp1" => %w[7],
    "model-vllm-qwen3-embedding-0.6b-tp1" => %w[7],
    "model-vllm-qwen3-reranker-0.6b-tp1" => %w[7],
    "model-vllm-whisper-large-v3-tp1" => %w[7]
  }
}
exporters = {
  "gpu02" => {
    "dcgm-dsv4-flash" => [%w[0 1], "deepseek-ai/DeepSeek-V4-Flash"],
    "dcgm-dsv4-flash-r2" => [%w[2 3], "deepseek-ai/DeepSeek-V4-Flash"],
    "dcgm-glm53" => [%w[4 5 6 7], "z-ai/glm-5.3-flash"]
  },
  "gpu13" => {
    "dcgm-privacy-filter" => [%w[0], "openai/privacy-filter"],
    "dcgm-qwen36-35b-a3b" => [%w[1], "Qwen/Qwen3.6-35B-A3B-FP8"],
    "dcgm-qwen38-27b" => [%w[2], "Qwen/Qwen3.8-27B-FP8"],
    "dcgm-glm51" => [%w[3 4 5 6], "QuantTrio/GLM-5.1-AWQ"],
    "dcgm-shared-gpu7" => [%w[7], "shared"]
  }
}
errors = []
check = ->(condition, message) { errors << message unless condition }
%w[model-sg-dsv4-flash-fp4-tp2-r1 model-sg-dsv4-flash-fp4-tp2-r2
   model-sg-qwen38-27b-fp8-tp1-r1 model-sg-qwen38-27b-fp8-tp1-r2
   model-sg-qwen36-35b-a3b-fp8-tp1 model-sg-qwen36-35b-a3b-fp8-tp1-r2].each do |name|
  check.call(legacy.fetch("services").key?(name), "Legacy serving/rollback file lost #{name}")
end
check.call(!legacy.fetch("services").key?("model-sg-glm53-fp8-tp4"), "Final GLM stack must not replace the legacy serving file")
check.call(packs["gpu02"].dig("services", "model-proxy-registrar", "profiles") == ["register-upgraded-models"], "Replacement-CVM registrar must require explicit post-qualification activation")
devices = ->(service) {
  Array(service&.dig("deploy", "resources", "reservations", "devices")).flat_map { |d| Array(d["device_ids"]) }
}
env = ->(service) {
  value = service.fetch("environment", {})
  value.is_a?(Hash) ? value : value.to_h { |item| item.split("=", 2) }
}

packs.each do |host, pack|
  services = pack.fetch("services")
  actual = services.select { |name, service| name.start_with?("model-") && !devices.call(service).empty? }
  check.call(actual.keys.to_set == expected.fetch(host).keys.to_set, "#{host}: unexpected/missing GPU engine")
  expected.fetch(host).each do |name, slots|
    check.call(devices.call(services[name]) == slots, "#{host}: #{name} must reserve #{slots.join(',')}")
  end
  check.call(actual.values.flat_map { |s| devices.call(s) }.to_set == (0..7).map(&:to_s).to_set, "#{host}: allocated physical slots must total eight")
  dcgm = services.select { |name, _| name.start_with?("dcgm-") }
  check.call(dcgm.keys.to_set == exporters.fetch(host).keys.to_set, "#{host}: unexpected/missing GPU exporter")
  exporters.fetch(host).each do |name, (slots, model)|
    service = services[name]
    check.call(devices.call(service) == slots, "#{host}: #{name} must expose only #{slots.join(',')}")
    label = service&.dig("labels", "nearai.otel.model")
    # The shared exporter's established label is a non-additive pool, not a model.
    check.call(label == model, "#{host}: #{name} model label mismatch (#{label.inspect})") unless model == "shared"
  end
  counts = dcgm.values.flat_map { |s| devices.call(s) }.tally
  check.call(counts == (0..7).to_h { |n| [n.to_s, 1] }, "#{host}: GPU telemetry must cover each slot exactly once")
  services.each do |name, service|
    check.call(service["container_name"] == name, "#{host}: #{name} container_name mismatch")
    deps = service.fetch("depends_on", {})
    deps = deps.keys if deps.is_a?(Hash)
    check.call((deps - services.keys).empty?, "#{host}: #{name} has missing dependencies")
  end
end

a, b = packs.values
glm_name = "model-sg-glm53-fp8-tp4"
glm_service = a.fetch("services").fetch(glm_name)
islands = [%w[0 1 2 3], %w[4 5 6 7]]
check.call(islands.include?(devices.call(glm_service)), "GLM TP4 must stay within one complete GPU island")
canary = load_yaml.call("prod/GLM-5.3-Flash-SGL-TP4-Canary.yaml")
qualified = canary.fetch("services").fetch("model-sg-glm53-fp8-tp4-r1")
%w[image build command environment runtime ipc ulimits stop_grace_period].each do |field|
  check.call(glm_service[field] == qualified[field], "GLM #{field} differs from qualified canary recipe")
end
check.call(glm_service.fetch("volumes").include?("glm53_kernel_cache:/root/.cache"), "GLM must use its own kernel cache")
dcgm_reference = load_yaml.call("prod/GLM-5.3-Flash-SGL-TP4.yaml")
%w[image command configs].each do |field|
  check.call(a.dig("services", "dcgm-glm53", field) == dcgm_reference.dig("services", "dcgm-glm53", field), "GLM DCGM #{field} differs from qualified telemetry")
end
check.call(a.dig("configs", "dcgm_h200_metrics") == dcgm_reference.dig("configs", "dcgm_h200_metrics"), "GLM DCGM metric definitions differ")
check.call(env.call(a.dig("services", "proxy-glm53"))["VLLM_BACKEND_URLS"] == "http://#{glm_name}:8000", "GLM proxy must have exactly one local backend")
check.call(a.fetch("services").keys.none? { |name| name.include?("qwen") }, "gpu02 retains a Qwen service")
check.call(env.call(a.dig("services", "model-proxy-registrar"))["REGISTER_GLM53"] == '${REGISTER_GLM53:-false}', "GLM registration must default off until qualification")
check.call(a.dig("configs", "registrar_script", "content").include?('if [ "$$port" = 8000 ] && [ "$${REGISTER_GLM53}" != true ]; then'), "GLM endpoint registration must be gated independently from DS4F")
%w[nginx_conf registrar_script otelcol_app_config].each do |name|
  check.call(!a.dig("configs", name, "content").match?(/(?:proxy|model-sg|dcgm)-qwen|Qwen\/|qwen3-[68]/), "gpu02 #{name} retains a Qwen reference")
end
%w[36-35b-a3b 38-27b].each do |short|
  proxy = b.dig("services", "proxy-qwen#{short}")
  urls = env.call(proxy).fetch("VLLM_BACKEND_URLS").split(",")
  check.call(urls.length == 1, "Qwen #{short} must have one backend")
  urls.each do |url|
    backend = url.delete_prefix("http://").split(":").first
    check.call(b.fetch("services").key?(backend), "Qwen #{short} points at removed backend")
  end
end
qwen = b.dig("services", "model-sg-qwen38-27b-fp8-tp1")
check.call(qwen["image"] == "lmsysorg/sglang@sha256:febfb971c7352570fc445c466ebd6ffc9d896024958e544a60f2137fd85856b1", "Qwen3.8 image changed during relocation")
check.call(qwen["command"].include?("--revision 017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"), "Qwen3.8 checkpoint changed during relocation")
check.call(qwen.fetch("volumes").include?("hugginface_cache:/root/.cache/huggingface"), "Qwen3.8 must use gpu13's existing HF cache name")
check.call(Digest::SHA256.hexdigest(b.dig("configs", "nginx_conf", "content")) == "844aa14ab489a4843e5038d5369b9490b286d6d24b18dc0283b37b9c5974d76b", "Shared nginx must remain byte-identical to the legacy config")
check.call(Digest::SHA256.hexdigest(b.dig("configs", "registrar_script", "content")) == "5b6989a02f997f4bdb25ec4eaacf0351b9b6969f5e47d9608b8b08b32004d70f", "Shared registrar must remain byte-identical to the legacy script")
check.call(!b.dig("services", "nginx", "depends_on").include?("proxy-qwen38-27b"), "Shared nginx must not depend on Qwen3.8")
check.call(b.dig("services", "nginx", "ports") == ["80:80", *(8002..8009).map { |p| "#{p}:#{p}" }, "8444:443"], "Legacy ingress publications changed")
check.call(b.dig("services", "nginx-qwen38-27b", "ports") == ["8000:8000", "8010:443"], "Qwen3.8 must own dedicated probe/TLS ports")
check.call(b.dig("services", "qwen38-model-proxy-registrar", "profiles") == ["register-qwen38"], "Qwen3.8 registration must be gated")
check.call(b.dig("configs", "nginx_qwen38_conf", "content").include?("qwen3-8-27b.completions.near.ai"), "missing dedicated Qwen3.8 TLS vhost")
published = b["services"].values.flat_map { |s| Array(s["ports"]).map { |p| p.split(":").first } }
check.call(published.uniq == published, "Duplicate published host port")

overlay = load_yaml.call("prod/gpu13-qwen-handover.yaml")
check.call(overlay["name"] == "gpu13-qwen-handover", "Unsafe handover project")
check.call(overlay.dig("networks", "default") == {"external" => true, "name" => "dstack_default"}, "Wrong handover network")
check.call(overlay.dig("volumes", "hugginface_cache") == {"external" => true, "name" => "work_hugginface_cache"}, "Wrong existing HF cache")
check.call(overlay.dig("volumes", "certs") == {"external" => true, "name" => "certs"}, "Wrong certificate volume")
temp = overlay.dig("services", "proxy-qwen36-handover")
canonical = b.dig("services", "proxy-qwen36-35b-a3b")
%w[image privileged runtime ipc ulimits user volumes extra_hosts environment restart].each do |key|
  check.call(temp[key] == canonical[key], "Temporary proxy #{key} must match qualified canonical surface")
end
overlay["services"].each do |name, service|
  check.call(service["container_name"] == name, "Handover name collision: #{name}")
  check.call(service["profiles"] == ["qwen-handover"], "Handover must be explicitly profile gated")
  check.call(!service.key?("ports") && !service.key?("devices") && devices.call(service).empty?, "Handover must not publish ports or reserve GPUs")
  check.call(service["image"].match?(/@sha256:[0-9a-f]{64}$/), "Unpinned handover image")
  next if name == "proxy-qwen36-handover"
  check.call(!service.key?("runtime") && !service["privileged"] && service["restart"] == "no", "Helper NVIDIA/privilege/restart surface")
  check.call(Array(service["volumes"]).none? { |v| v.include?("docker.sock") || v.start_with?("/") }, "Helper host mount forbidden")
  check.call(service["pid"].nil? || service["pid"] == "container:nginx", "Helper host PID forbidden")
end
qualifier = overlay.dig("services", "handover-qualify")
check.call(qualifier["environment"] == ['PROXY_TOKEN=${PROXY_TOKEN}'], "Qualifier must receive only the proxy credential")
check.call(qualifier["read_only"] && qualifier["cap_drop"] == ["ALL"] && !qualifier.key?("cap_add"), "Qualifier sandbox weakened")
check.call(qualifier["security_opt"] == ["no-new-privileges:true"] && !qualifier.key?("pid") && !qualifier.key?("network_mode"), "Qualifier namespace/security changed")
check.call(!qualifier.key?("volumes") && qualifier["tmpfs"] == ["/tmp:rw,exec,nosuid,nodev,size=256m"], "Qualifier dependencies must stay in bounded disposable tmpfs")

if errors.any?
  abort "Qwen/GLM capacity contract failed:\n  - #{errors.join("\n  - ")}"
end
puts "Qwen/GLM capacity contract OK: GLM island 4-7, DS4F 0-3, one Qwen replica each, 16 physical slots across two hosts"
