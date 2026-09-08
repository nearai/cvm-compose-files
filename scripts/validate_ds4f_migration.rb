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
ds_name = 'model-sg-dsv4-flash-fp4-tp2-r1'
assert = ->(value, message) { raise message unless value }
ids = ->(name) { services.fetch(name).dig('deploy', 'resources', 'reservations', 'devices').flat_map { |d| d.fetch('device_ids') } }
assert.call(ids.call(glm_name) == %w[0 1 2 3], 'GLM must retain GPUs 0-3')
assert.call(ids.call(ds_name) == %w[4 5], 'DS4F must use GPUs 4-5')
assert.call(ids.call('dcgm-glm51') == %w[0 1 2 3], 'GLM exporter GPU scope')
assert.call(ids.call('dcgm-dsv4-flash') == %w[4 5], 'DS4F exporter GPU scope')
assert.call(!services.key?('model-sg-glm51-awq-tp4-r2'), 'Drained GLM replica must not restart')
assert.call(!services.key?('model-sg-dsv4-flash-fp4-tp2-r2'), 'Only one destination DS4F replica')
assert.call(services[glm_name] == glm['services'][glm_name], 'Retained GLM engine differs from existing config')
%w[image command environment ulimits volumes runtime ipc stop_grace_period].each do |key|
  assert.call(services[ds_name][key] == source['services'][ds_name][key], "DS4F runtime changed: #{key}")
end
assert.call(services['ds4f-migration-registrar']['profiles'] == ['register-ds4f'], 'Registration must be explicitly gated')
assert.call(services['proxy-dsv4-flash']['environment'].include?("VLLM_BACKEND_URLS=http://#{ds_name}:8000"), 'DS4F pool must contain only destination r1')
collector = YAML.safe_load(bridge['configs']['otelcol_app_config']['content'])
jobs = collector['receivers']['prometheus/apps']['config']['scrape_configs']
assert.call(jobs.length == 6, 'Expected one engine, proxy and DCGM scrape per model')
assert.call(jobs.none? { |job| job['job_name'].end_with?('-r2') }, 'Removed replicas must not be scraped')

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
