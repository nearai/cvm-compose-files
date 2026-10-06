#!/usr/bin/env ruby
require 'yaml'
require 'json'
require 'open3'

root = File.expand_path('..', __dir__)
load_compose = ->(file) { YAML.load_file(File.join(root, file), aliases: true) }
bridge = load_compose.call('prod/GLM-5.1-DSV4-Migration.yaml')
qwen = load_compose.call('prod/qwen36-qwen38.yaml')
glm = load_compose.call('prod/GLM-5.1-SGL-AWQ-TP4.yaml')
small = load_compose.call('prod/small-models.yaml')
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
assert.call(scalar_strings.call(small).none? { |value| value.match?(/dsv4|deepseek|ds4f/i) }, 'gpu13 rendered configuration must not contain retired DS4F identities')
gpu13_glm = { 'model-sg-glm53-w4afp8-tp2-r1a' => %w[4 5], 'model-sg-glm53-w4afp8-tp2-r1b' => %w[6 7] }
gpu13_glm.each { |name, devices| assert.call(small_ids.call(name) == devices, "gpu13 GLM replica #{name} must use GPUs #{devices.join(',')}") }
assert.call(!small_services.key?('model-sg-glm53-fp8-tp4'), 'gpu13 must no longer run the TP4 GLM replica (replaced by two TP2 replicas)')
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
  device_ids = service.dig('deploy', 'resources', 'reservations', 'devices')&.flat_map { |device| device.fetch('device_ids') } || []
  next unless device_ids.any? { |id| %w[4 5 6 7].include?(id) }

  assert.call((gpu13_glm.keys + %w[dcgm-glm53]).include?(name), "Unexpected gpu13 GPU 4-7 claim: #{name}")
end
# gpu13's GLM is two memory-optimized TP2/EP2 replicas (GPUs 4,5 and 6,7) with the same
# per-replica argv as the long-context file's tp2-r2a/r2b (W4AFP8 + HiCache, campaign-2 L2 base):
# prod/GLM-5.3-Flash-SGL-TP4-W4AFP8-LongContext.yaml, docs/long-context-glm53-2xtp2-rollout.md.
# The #330 overlap-off canary ended when the TP4 replica was replaced; overlap stays ON.
gpu13_variant = 'fc91d24-long-context-w4afp8-c8192-qsplit-offloop-v3-hicache-cuda-host-pooled-v1-host325g-memopt-mamba330-bf16state-admission-reserve-disabled-pool-clamp-pdi2-gpu13-h200-tp2-ep2-eagle-fixed-4-1-5-mr12q4-strict-budget8192'
gpu13_ports = {}
gpu13_glm.each_key do |name|
  engine = small_services.fetch(name)
  assert.call(engine['image'] == 'docker.io/nearaidev/sglang@sha256:47aff791090003a37f893e998c44794c410d3f7bdfc7fdd2dfab5eb5592b30bb', "Qualified gpu13 GLM image changed: #{name}")
  command = engine.fetch('command').to_s.split.each_slice(1).to_a.flatten.join(' ')
  assert.call(command.include?('--model-path /root/.cache/huggingface/hub/models--graphistry--GLM-5.3-Flash-W4AFP8/snapshots/99f1fa70408c52b007d4fd69e02e5a522422e755'), "gpu13 GLM must serve the qualified W4AFP8 snapshot: #{name}")
  {
    '--tp-size' => '2', '--ep-size' => '2', '--mem-fraction-static' => '0.86',
    '--max-running-requests' => '12', '--max-queued-requests' => '4',
    '--chunked-prefill-size' => '8192', '--max-prefill-tokens' => '32768', '--prefill-decode-interval' => '2',
    '--cuda-graph-max-bs-decode' => '12', '--speculative-num-steps' => '4', '--speculative-eagle-topk' => '1',
    '--speculative-num-draft-tokens' => '5', '--max-mamba-cache-size' => '330', '--mamba-ssm-dtype' => 'bfloat16'
  }.each do |flag, value|
    assert.call(command.match?(/(^| )#{Regexp.escape(flag)} #{Regexp.escape(value)}( |$)/), "gpu13 GLM runtime flag changed on #{name}: #{flag} #{value}")
  end
  # Overlap scheduling is ON and EAGLE is the fixed 4/1/5 arm: neither flag may return.
  assert.call(!command.include?('--disable-overlap-schedule'), "gpu13 GLM #{name} must not disable the overlap scheduler (the #330 canary ended)")
  assert.call(!command.include?('--speculative-adaptive'), "gpu13 GLM #{name} must use fixed EAGLE 4/1/5, not adaptive")
  assert.call(command.scan('--chunked-prefill-size').length == 1, "gpu13 GLM #{name} must set the chunk exactly once")
  [
    '--enable-hierarchical-cache', '--hicache-write-policy write_through',
    '--hicache-io-backend direct', '--hicache-mem-layout page_first_direct'
  ].each { |flag| assert.call(command.include?(flag), "gpu13 GLM HiCache contract changed on #{name}: #{flag}") }
  port = command[/--dist-init-addr (\S+)/, 1]
  assert.call(port && !gpu13_ports.key?(port), "gpu13 GLM #{name} needs a unique --dist-init-addr, got #{port.inspect}")
  gpu13_ports[port] = name
  env = engine.fetch('environment')
  assert.call(env.count('SGLANG_DSA_INDEXER_QSPLIT=1') == 1, "gpu13 GLM #{name} must enable DSA indexer query split exactly once")
  [
    "SGLANG_HICACHE_RAM_BUDGET=${GLM53_#{name.end_with?('a') ? 'R1A' : 'R1B'}_HICACHE_RAM_BUDGET:-325GiB}",
    'SGLANG_HICACHE_CUDA_HOST_MEMORY=${GLM53_HICACHE_CUDA_HOST_MEMORY:-1}'
  ].each { |entry| assert.call(env.include?(entry), "gpu13 GLM HiCache host-memory contract changed on #{name}: #{entry}") }
  # The admission reserve crashed gpu02's long r2 with a Prefill OOM on 2026-09-18 and is unsafe
  # on the long tier; every long arm runs without it.
  %w[SGLANG_CHUNKED_PREFILL_ADMISSION_RESERVE SGLANG_ADMISSION_RESERVE_MAX_FRACTION].each do |var|
    assert.call(env.none? { |entry| entry.to_s.start_with?("#{var}=") }, "gpu13 GLM #{name} must not set #{var}: the admission reserve is unsafe on the long tier")
  end
end
# Both replicas must run the identical argv apart from the rendezvous port (r1b carries a full copy of the command), and the identical environment apart from the budget variable.
argv_without_port = ->(name) { small_services.fetch(name).fetch('command').to_s.gsub(/--dist-init-addr \S+/, '').split }
assert.call(argv_without_port.call('model-sg-glm53-w4afp8-tp2-r1a') == argv_without_port.call('model-sg-glm53-w4afp8-tp2-r1b'), 'gpu13 GLM replicas must have identical argv apart from --dist-init-addr')
env_without_budget = ->(name) { small_services.fetch(name).fetch('environment').reject { |entry| entry.start_with?('SGLANG_HICACHE_RAM_BUDGET=') } }
assert.call(env_without_budget.call('model-sg-glm53-w4afp8-tp2-r1a') == env_without_budget.call('model-sg-glm53-w4afp8-tp2-r1b'), 'gpu13 GLM replicas must have identical environment apart from the HiCache budget variable')
# Nothing else in the file may reuse a GLM rendezvous port.
small_services.each do |name, service|
  next if gpu13_glm.key?(name)

  assert.call(!service.fetch('command', '').to_s.match?(/--dist-init-addr 127\.0\.0\.1:2951\d/), "gpu13 #{name} must not reuse a GLM dist-init port")
end
small_proxy = small_services.fetch('proxy-glm53')
assert.call(small_proxy['image'] == 'nearaidev/vllm-proxy-rs@sha256:d61357da39918a57126864a451eaf054f06a6989c03fe9a1666f7e6374ba6907', 'Qualified gpu13 GLM proxy image changed')
assert.call(small_proxy.fetch('environment').include?('VLLM_BACKEND_URLS=http://model-sg-glm53-w4afp8-tp2-r1a:8000,http://model-sg-glm53-w4afp8-tp2-r1b:8000'), 'gpu13 GLM proxy must pool both TP2 replicas')
assert.call(small_proxy.fetch('environment').include?('VLLM_BACKEND_CONVERSATION_AFFINITY=1'), 'gpu13 GLM affinity contract changed')
dcgm_image = 'nvcr.io/nvidia/k8s/dcgm-exporter@sha256:ed594cf53fe6942e84b07b0740cdcbb249fa4b39cb21feeebf93881ae51f0b5e'
assert.call(small_services.fetch('dcgm-glm53')['image'] == dcgm_image, 'gpu13 GLM exporter image must be pinned')
assert.call(small_services.fetch('dcgm-shared-gpu3')['image'] == dcgm_image, 'gpu13 shared DCGM image must be pinned')
registrar = small.fetch('configs').fetch('registrar_script').fetch('content')
nginx = small.fetch('configs').fetch('nginx_conf').fetch('content')
assert.call(small_services.fetch('nginx').fetch('ports').map(&:to_s).include?('8009:8009'), 'gpu13 nginx must publish host port 8009')
# GLM (:8009) is the OpenRouter-only lane: the gateway reaches it directly on
# the host port, so the registrar must never probe or register it. Comments are
# stripped first so the explanatory ":8009" comment block is not a false match.
registrar_code = registrar.lines.reject { |line| line.strip.start_with?('#') }.join
assert.call(!registrar_code.include?('8009'), 'gpu13 registrar must not reference port 8009 (GLM is the OpenRouter-only lane)')
assert.call(!registrar.match?(/register_model "z-ai\/glm-5\.3-flash/), 'gpu13 registrar must not register GLM with model-proxy')
assert.call(nginx.match?(/listen 8009;\s+location \/ \{ proxy_pass http:\/\/proxy-glm53:8000; \}/), 'gpu13 nginx port 8009 must route to the GLM proxy')
assert.call(nginx.match?(/server_name glm-5-3-flash\.completions\.near\.ai.*?location \/ \{ proxy_pass http:\/\/proxy-glm53:8000; \}/m), 'gpu13 GLM SNI must route to the GLM proxy')
glm_sni = nginx[/server_name glm-5-3-flash\.completions\.near\.ai.*?location \/ \{ proxy_pass http:\/\/proxy-glm53:8000; \}/m]
assert.call(glm_sni.include?('"~^glm-5-3-flash-b[0-9a-f]{12}\.completions(-stg)?\.near\.ai$$"'), 'gpu13 GLM SNI must accept model-proxy backend handles')
assert.call(glm_sni.include?('gpu13.hosts.near.ai;'), 'gpu13 GLM SNI must accept the direct OpenRouter TLS hostname')
# The host-level name must resolve to GLM alone: in any other TLS vhost it would
# hand the OpenRouter lane a different model.
tls_server_blocks = nginx.scan(/^server \{\n(?:.*\n)*?^\}$/).select { |block| block.include?('listen 443 ssl') }
host_name_blocks = tls_server_blocks.select { |block| block.include?('gpu13.hosts.near.ai') }
assert.call(host_name_blocks.length == 1 && host_name_blocks.first.include?('proxy_pass http://proxy-glm53:8000;'), 'gpu13.hosts.near.ai must be bound to the GLM vhost only')
small_jobs = YAML.safe_load(small.fetch('configs').fetch('otelcol_app_config').fetch('content')).dig('receivers', 'prometheus/apps', 'config', 'scrape_configs')
%w[sglang-model-sg-glm53-w4afp8-tp2-r1a sglang-model-sg-glm53-w4afp8-tp2-r1b dcgm-dcgm-glm53 dcgm-dcgm-shared-gpu3 inference-proxy-proxy-glm53].each do |job|
  assert.call(small_jobs.any? { |entry| entry['job_name'] == job }, "gpu13 OTel scrape missing: #{job}")
end
# Each replica's OTel label, scrape job and log tag must carry the same truthful config_variant,
# its own instance and gpu_pair, and the proxy and exporter must advertise the same variant.
{ 'model-sg-glm53-w4afp8-tp2-r1a' => ['1a', '4-5'], 'model-sg-glm53-w4afp8-tp2-r1b' => ['1b', '6-7'] }.each do |name, (instance, pair)|
  labels = small_services.fetch(name).fetch('labels')
  log_tags = JSON.parse(labels.fetch('com.datadoghq.ad.logs')).flat_map { |entry| Array(entry['tags']) }
  scrape = small_jobs.find { |entry| entry['job_name'] == "sglang-#{name}" }
  scrape_labels = scrape.dig('static_configs', 0, 'labels')
  assert.call(scrape.dig('static_configs', 0, 'targets') == ["#{name}:8000"], "gpu13 #{name} scrape target changed")
  variants = [labels['nearai.otel.config_variant'], scrape_labels['config_variant'], *log_tags.select { |tag| tag.start_with?('config_variant:') }.map { |tag| tag.sub('config_variant:', '') }]
  assert.call(variants.length == 3 && variants.all? { |variant| variant == gpu13_variant }, "gpu13 #{name} config_variant must be the TP2 variant on the label, scrape job and log tag, got #{variants.inspect}")
  assert.call(labels['nearai.otel.instance'] == instance && scrape_labels['instance'] == instance && log_tags.include?("instance:#{instance}"), "gpu13 #{name} instance must be #{instance}")
  assert.call(labels['nearai.otel.gpu_pair'] == pair && scrape_labels['gpu_pair'] == pair && log_tags.include?("gpu_pair:#{pair}"), "gpu13 #{name} gpu_pair must be #{pair}")
  assert.call(labels['nearai.otel.max_running_requests'] == '12' && scrape_labels['max_running_requests'] == '12' && labels['nearai.otel.max_queued_requests'] == '4' && scrape_labels['max_queued_requests'] == '4', "gpu13 #{name} max_running/max_queued labels must be 12/4")
end
%w[proxy-glm53 dcgm-glm53].each do |name|
  assert.call(small_services.fetch(name).fetch('labels')['nearai.otel.config_variant'] == gpu13_variant, "gpu13 #{name} config_variant must match the TP2 engines")
end
%w[inference-proxy-proxy-glm53 dcgm-dcgm-glm53].each do |job|
  entry = small_jobs.find { |candidate| candidate['job_name'] == job }
  assert.call(entry.dig('static_configs', 0, 'labels', 'config_variant') == gpu13_variant, "gpu13 scrape job #{job} config_variant must match the TP2 engines")
end

puts 'DS4F migration and gpu13 GLM replacement allocation, telemetry and registrar contracts OK'
