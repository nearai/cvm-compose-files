#!/usr/bin/env ruby
require 'yaml'
require 'open3'

root = File.expand_path('..', __dir__)
load_compose = ->(file) { YAML.load_file(File.join(root, file), aliases: true) }
bridge = load_compose.call('prod/GLM-5.1-DSV4-Migration.yaml')
source = load_compose.call('prod/dsv4-qwen38-glm51.yaml')
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
ds_names.each do |name|
  %w[image command environment ulimits volumes runtime ipc stop_grace_period].each do |key|
    assert.call(services[name][key] == source['services'][name][key], "DS4F runtime changed: #{name}.#{key}")
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

# Execute the real source registrar with shell-only HTTP stubs. No network,
# credentials, model requests, or writes outside the disposable container.
script = source['configs']['registrar_script']['content'].gsub('$$', '$')
script = script.sub('sleep 60', 'exit 0')
stub = <<~SH
  curl() {
    printf 'CALL %s\\n' "$*" >&2
    case " $* " in *' -w '*) printf '200';; esac
    return 0
  }
  sleep() { :; }
SH
%w[true false invalid].each do |enabled|
  env = {'MODEL_PROXY_TOKEN'=>'synthetic', 'PROXY_TOKEN'=>'synthetic', 'HOST_IP'=>'127.0.0.1',
         'TLS_PORT'=>'8444', 'TLS_PORT_2'=>'8003', 'TLS_PORT_QWEN36_35B'=>'8007', 'REGISTER_DSV4'=>enabled}
  # Replace the heartbeat write so the test leaves no local files behind.
  test_script = (stub + script).gsub('date +%s > /tmp/registrar_alive', ':')
  _, calls, status = Open3.capture3(env, 'sh', stdin_data: test_script)
  if enabled == 'invalid'
    assert.call(!status.success? && calls.empty?, 'Invalid gate must fail before HTTP')
    next
  end
  assert.call(status.success?, "Registrar test failed: #{enabled}")
  %w[8002 8006].each { |port| assert.call(calls.include?("127.0.0.1:#{port}"), "Qwen #{port} absent") }
  assert.call(calls.include?('127.0.0.1:8001') == (enabled == 'true'), 'DS4F registration/readiness gate ignored')
  assert.call(calls.include?('deepseek-ai/DeepSeek-V4-Flash') == (enabled == 'true'), 'DS4F model registration gate ignored')
end
puts 'DS4F migration allocation, runtime preservation, telemetry and registrar contracts OK'
