#!/usr/bin/env ruby
require 'yaml'
require 'open3'
require 'json'
root = File.expand_path('..', __dir__)
doc = YAML.load_file(File.join(root, 'prod/migration-gpu-preflight.yaml'), aliases: true)
raise 'Unexpected service' unless doc.fetch('services').keys == ['migration-gpu-preflight']
service = doc['services']['migration-gpu-preflight']
raise 'Isolation missing' unless service['network_mode'] == 'none' && service['read_only'] == true
raise 'Capability scope changed' unless service['cap_drop'] == ['ALL'] && service['cap_add'] == ['SYSLOG']
raise 'Privilege escalation allowed' unless service['security_opt'] == ['no-new-privileges:true']
raise 'Unexpected host attachment' unless %w[privileged volumes ports pid ipc devices depends_on].none? { |key| service.key?(key) }
raise 'Unexpected environment' unless service['environment'] == {'NVIDIA_VISIBLE_DEVICES'=>'all', 'NVIDIA_DRIVER_CAPABILITIES'=>'utility'}
raise 'Implicit activation' unless service['profiles'] == ['migration-preflight'] && service['restart'] == 'no'
raise 'Image not pinned' unless service['image'].match?(/@sha256:[a-f0-9]{64}$/)
code = doc.fetch('configs').fetch('gpu_preflight_script').fetch('content')
tests = <<~'PY'
  import sys
  ns = {'__name__': 'test'}
  exec(sys.stdin.read(), ns)
  xml = '<nvidia_smi_log><driver_version>test</driver_version><gpu id="0000:01:00.0"><uuid>GPU-synthetic</uuid><ecc_errors><volatile><dram_uncorrected>2</dram_uncorrected></volatile></ecc_errors><remapped_rows><failure>Yes</failure></remapped_rows></gpu></nvidia_smi_log>'
  row = ns['parse_gpu_xml'](xml)['gpus'][0]
  assert row['health']['ecc_errors']['volatile']['dram_uncorrected'] == '2'
  assert row['health']['remapped_rows']['failure'] == 'Yes'
  assert row['health']['gpu_recovery_action'] is None
  for bad in ('<nvidia_smi_log/>', '<nvidia_smi_log><gpu/></nvidia_smi_log>'):
      try: ns['parse_gpu_xml'](bad)
      except ValueError: pass
      else: raise AssertionError('Missing GPU identity accepted')
  text = '<6>[ 100.0] boot\n<3>[ 7400.0] NVRM: Xid (PCI:synthetic): 31\n<3>[ 7500.0] AER: Uncorrected fatal\n<3>[ 7800.0] Out of memory: Killed process synthetic\n<3>[ 7900.0] uncorrectable ECC\n'
  result = ns['kernel_summary'](text, 8000, 7200)
  assert result['full_window_available'] and result['unparsed_lines'] == 0
  assert result['counts'] == {'xid':1, 'uncorrectable_ecc':1, 'pcie_fatal':1, 'oom_kill':1}
  recent = ns['kernel_summary'](text, 8000, 300)
  assert recent['counts']['xid'] == 0 and recent['counts']['oom_kill'] == 1
  partial = ns['kernel_summary']('[ 7900.0] clean', 8000, 7200)
  assert not partial['full_window_available']
  empty = ns['kernel_summary']('unparseable', 8000)
  assert not empty['ok'] and empty['unparsed_lines'] == 1
  print('GPU preflight isolation and sanitized evidence parser tests passed')
PY
if ARGV == ['--emit-python-tests']
  puts "source = #{JSON.generate(code)}"
  puts tests.sub('sys.stdin.read()', 'source')
  exit
end
stdout, stderr, status = Open3.capture3('python3', '-c', tests, stdin_data: code)
puts stdout
warn stderr unless status.success?
raise 'GPU preflight tests failed' unless status.success?
