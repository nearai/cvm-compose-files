#!/usr/bin/env ruby
require 'yaml'
require 'open3'

root = File.expand_path('..', __dir__)
load_compose = ->(file) { YAML.load_file(File.join(root, file), aliases: true) }
bridge = load_compose.call('prod/GLM-5.1-DSV4-Migration.yaml')
qwen = load_compose.call('prod/qwen36-qwen38.yaml')
glm = load_compose.call('prod/GLM-5.1-SGL-AWQ-TP4.yaml')
services = bridge.fetch('services')
glm_name = 'model-sg-glm51-awq-tp4-r1'
ds_names = %w[model-sg-dsv4-flash-fp4-tp2-r1 model-sg-dsv4-flash-fp4-tp2-r2]
assert = ->(value, message) { raise message unless value }
ids = ->(name) { services.fetch(name).dig('deploy', 'resources', 'reservations', 'devices').flat_map { |d| d.fetch('device_ids') } }
assert.call(ids.call(glm_name) == %w[0 1 2 3], 'GLM must retain GPUs 0-3')
assert.call(ids.call(ds_names[0]) == %w[4 5], 'DS4F r1 must use GPUs 4-5')
assert.call(ids.call(ds_names[1]) == %w[6 7], 'DS4F r2 must use GPUs 6-7')
engine_ids = ([glm_name] + ds_names).flat_map { |name| ids.call(name) }
assert.call(engine_ids.sort == %w[0 1 2 3 4 5 6 7], 'All eight GPUs must be assigned exactly once to engines')
assert.call(ids.call('dcgm-glm51') == %w[0 1 2 3], 'GLM exporter GPU scope')
assert.call(ids.call('dcgm-dsv4-flash') == %w[4 5 6 7], 'DS4F exporter GPU scope')
assert.call(!services.key?('model-sg-glm51-awq-tp4-r2'), 'Drained GLM replica must not restart')
assert.call(services[glm_name] == glm['services'][glm_name], 'Retained GLM engine differs from existing config')
assert.call(services[ds_names[0]]['image'] == 'docker.io/nearaidev/sglang@sha256:ec518148762ea02c23aa8615f69ca79b0c18bcd59b3c21c10229db3df323c615', 'Qualified DS4F image changed')
ds_names.each do |name|
  %w[image command environment ulimits volumes runtime ipc stop_grace_period].each do |key|
    assert.call(services[name][key] == services[ds_names[0]][key], "DS4F replicas must use the same runtime: #{name}.#{key}")
  end
end
assert.call(services['ds4f-migration-registrar']['profiles'] == ['register-ds4f'], 'Registration must be explicitly gated')
disk_check = services.fetch('migration-disk-preflight')
assert.call(disk_check['profiles'] == ['migration-preflight'], 'Disk check must be explicitly gated')
assert.call(disk_check['image'] == services['model-downloader']['image'], 'Disk check must reuse the pinned downloader image')
assert.call(disk_check['volumes'] == ['huggingface_cache:/cache:ro'], 'Disk check must mount only the read-only cache')
assert.call(disk_check['network_mode'] == 'none' && disk_check['read_only'] == true, 'Disk check must be isolated and read-only')
assert.call(disk_check['user'] == '65534:65534' && disk_check['cap_drop'] == ['ALL'], 'Disk check must be unprivileged')
assert.call(disk_check['security_opt'] == ['no-new-privileges:true'], 'Disk check must forbid privilege escalation')
assert.call(%w[environment ports deploy runtime privileged].none? { |key| disk_check.key?(key) }, 'Disk check must not receive credentials, ports or GPUs')
pool = ds_names.map { |name| "http://#{name}:8000" }.join(',')
assert.call(services['proxy-dsv4-flash']['environment'].include?("VLLM_BACKEND_URLS=#{pool}"), 'DS4F pool must contain both destination replicas')
collector = YAML.safe_load(bridge['configs']['otelcol_app_config']['content'])
jobs = collector['receivers']['prometheus/apps']['config']['scrape_configs']
assert.call(jobs.length == 7, 'Expected three engine scrapes, two proxies and two exporters')
assert.call(jobs.none? { |job| job['job_name'] == 'sglang-model-sg-glm51-awq-tp4-r2' }, 'Removed GLM replica must not be scraped')
ds_names.each_with_index do |name, index|
  scrape = jobs.find { |job| job['job_name'] == "sglang-#{name}" }
  assert.call(scrape&.dig('static_configs', 0, 'targets') == ["#{name}:8000"], "Missing DS4F scrape: #{name}")
  pair = index.zero? ? '4-5' : '6-7'
  assert.call(scrape.dig('static_configs', 0, 'labels', 'gpu_pair') == pair, "Wrong DS4F scrape pair: #{name}")
  assert.call(scrape.dig('static_configs', 0, 'labels', 'instance') == (index + 1).to_s, "Wrong DS4F scrape instance: #{name}")
  assert.call(services[name].dig('labels', 'nearai.otel.gpu_pair') == pair, "Wrong DS4F service pair: #{name}")
end
dcgm = jobs.find { |job| job['job_name'] == 'dcgm-dcgm-dsv4-flash' }
assert.call(dcgm.dig('static_configs', 0, 'labels', 'gpu_pair') == '4-7', 'DS4F DCGM scrape must cover both replicas')
assert.call(services['dcgm-dsv4-flash'].dig('labels', 'nearai.otel.gpu_pair') == '4-7', 'DS4F DCGM service must cover both replicas')

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
puts 'DS4F destination and Qwen-only allocation, telemetry and registrar contracts OK'
