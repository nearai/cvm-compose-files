#!/usr/bin/env ruby
require 'yaml'
require 'json'
require 'open3'

root = File.expand_path('..', __dir__)
load_compose = ->(file) { YAML.load_file(File.join(root, file), aliases: true) }
qwen = load_compose.call('prod/qwen36-qwen38.yaml')
small = load_compose.call('prod/small-models.yaml')
assert = ->(value, message) { raise message unless value }

# Execute the Qwen-only registrar with shell-only HTTP stubs. No network,
# credentials, model requests, or writes outside the disposable container.
assert.call(!File.read(File.join(root, 'prod/qwen36-qwen38.yaml')).match?(/dsv4|deepseek|ds4f/i), 'Qwen recipe must not contain removed model configuration')
{
  'model-sg-qwen38-27b-fp8-tp1-r1' => ['2'],
  'model-sg-qwen38-27b-fp8-tp1-r2' => ['3'],
  'model-sg-qwen36-35b-a3b-fp8-tp1' => ['5'],
  'model-sg-qwen36-35b-a3b-fp8-tp1-r2' => ['4']
}.each do |name, expected|
  devices = qwen.fetch('services').fetch(name).dig('deploy', 'resources', 'reservations', 'devices')
  assert.call(devices.flat_map { |d| d.fetch('device_ids') } == expected, "Qwen placement changed: #{name}")
end
script = qwen['configs']['registrar_script']['content'].gsub('$$', '$')
script = script.sub('sleep 60', 'exit 0')
stub = <<~SH
  curl() {
    printf 'CALL %s\\n' "$*" >&2
    case " $* " in *' -w '*) printf '200';; esac
    return 0
  }
  sleep() { :; }
SH
env = {'MODEL_PROXY_TOKEN'=>'synthetic', 'PROXY_TOKEN'=>'synthetic', 'HOST_IP'=>'127.0.0.1',
       'TLS_PORT_2'=>'8003', 'TLS_PORT_QWEN36_35B'=>'8007'}
# Replace the heartbeat write so the test leaves no local files behind.
test_script = (stub + script).gsub('date +%s > /tmp/registrar_alive', ':')
_, calls, status = Open3.capture3(env, 'sh', stdin_data: test_script)
assert.call(status.success?, 'Qwen registrar test failed')
%w[8002 8006].each { |port| assert.call(calls.include?("127.0.0.1:#{port}"), "Qwen #{port} absent") }
%w[Qwen/Qwen3.8-27B Qwen/Qwen3.6-35B-A3B-FP8].each do |model|
  assert.call(calls.include?(model), "Qwen model registration missing: #{model}")
end
assert.call(!calls.include?('127.0.0.1:8001'), 'Qwen registrar must not probe removed endpoint')

small_services = small.fetch('services')
small_ids = ->(name) do
  small_services.fetch(name).dig('deploy', 'resources', 'reservations', 'devices').flat_map { |d| d.fetch('device_ids') }
end
scalar_strings = lambda do |value|
  case value
  when Hash then value.flat_map { |key, child| [key.to_s] + scalar_strings.call(child) }
  when Array then value.flat_map { |child| scalar_strings.call(child) }
  else [value.to_s]
  end
end
assert.call(scalar_strings.call(small).none? { |value| value.match?(/ds4f|dsv4(?!1)|deepseek[-_.]?v4[-_.]flash/i) }, 'gpu13 rendered configuration must not contain retired DS4F identities')
assert.call(small_services.keys.none? { |name| name.include?('dsv41') }, 'gpu13 must have no DeepSeek-V4.1 service')
glm_engines = { 'model-sg-glm53-w4afp8-tp2-r1' => %w[4 5], 'model-sg-glm53-w4afp8-tp2-r2' => %w[6 7] }
assert.call(small_services.keys.grep(/\Amodel-sg-glm53-/).sort == glm_engines.keys.sort, 'gpu13 GLM engines must be exactly r1 and r2')
assert.call(!small.dig('volumes', 'ghost').nil?, 'gpu13 GLM engines need the ghost volume')
glm_engines.each do |name, gpus|
  assert.call(small_ids.call(name) == gpus, "gpu13 #{name} must use GPUs #{gpus.join(',')}")
  env = small_services.fetch(name).fetch('environment')
  assert.call(env.include?('CUDA_DEVICE_ORDER=PCI_BUS_ID'), "gpu13 #{name} must set CUDA_DEVICE_ORDER=PCI_BUS_ID")
end
assert.call(small_ids.call('dcgm-glm53') == %w[4 5 6 7], 'gpu13 GLM exporter must use GPUs 4-7')
shared_gpu3 = %w[
  model-sg-flux2-klein-4b-tp1
  model-vllm-qwen3vl-30b-a3b-fp8-tp1
  model-vllm-qwen3-embedding-0.6b-tp1
  model-vllm-qwen3-reranker-0.6b-tp1
  model-vllm-whisper-large-v3-tp1
  dcgm-shared-gpu3
]
shared_gpu3.each do |name|
  assert.call(small_ids.call(name) == ['3'], "gpu13 shared service must use GPU 3: #{name}")
end
assert.call(!small_services.key?('dcgm-shared-gpu7'), 'Retired gpu13 shared GPU 7 exporter must be absent')
small_services.each do |name, service|
  device_ids = service.dig('deploy', 'resources', 'reservations', 'devices')&.flat_map { |device| device.fetch('device_ids', []) } || []
  next unless device_ids.any? { |id| %w[4 5 6 7].include?(id) }

  assert.call((glm_engines.keys + ['dcgm-glm53']).include?(name), "Unexpected gpu13 GPU 4-7 claim: #{name}")
end
# GPUs 4-7 belong to the GLM engines' device_ids; a service selecting one of them
# through CUDA_VISIBLE_DEVICES would collide with them.
small_services.each do |name, service|
  env = service['environment']
  env = env.map { |key, value| "#{key}=#{value}" } if env.is_a?(Hash)
  visible = Array(env).map(&:to_s).grep(/\ACUDA_VISIBLE_DEVICES=/).flat_map { |entry| entry.split('=', 2).last.split(',') }
  next unless visible.any? { |id| %w[4 5 6 7].include?(id.strip) }

  assert.call(false, "Unexpected gpu13 GPU 4-7 claim via CUDA_VISIBLE_DEVICES: #{name}")
end
puts 'Qwen recipe and gpu13 allocation contracts OK'
