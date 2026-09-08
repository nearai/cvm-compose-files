#!/usr/bin/env python3
"""Render the self-contained Compose file fetched by compose-manager.

Only the generated file is deployed. --check verifies it is synchronized.
"""
import copy
import hashlib
import json
from pathlib import Path
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'prod/gpu13-qwen-handover.yaml'
UTILITY = 'ghcr.io/astral-sh/uv:python3.11-bookworm-slim@sha256:4f5d923c9dcea037f57bda425dd209f3ec643da2f0b74227f68d09dab0b3bb36'
STEADY_SHA = '844aa14ab489a4843e5038d5369b9490b286d6d24b18dc0283b37b9c5974d76b'


def candidates(steady):
    assert hashlib.sha256(steady.encode()).hexdigest() == STEADY_SHA
    needle = 'location / { proxy_pass http://proxy-qwen36-35b-a3b:8000; }'
    assert steady.count(needle) == 2

    def routes(primary, fallback):
        # Only signature GET may fall back. Completion/OHTTP POSTs never retry.
        return f'''location / {{ proxy_pass http://{primary}:8000; }}
          location ^~ /v1/signature/ {{
              limit_except GET {{ deny all; }}
              proxy_intercept_errors on;
              error_page 404 502 503 504 = @qwen36_signature_fallback;
              proxy_pass http://{primary}:8000;
          }}
          location @qwen36_signature_fallback {{
              limit_except GET {{ deny all; }}
              proxy_intercept_errors off;
              proxy_pass http://{fallback}:8000;
          }}'''

    return {'steady': steady,
            'to-temp': steady.replace(needle, routes('proxy-qwen36-handover', 'proxy-qwen36-35b-a3b')),
            'to-canonical': steady.replace(needle, routes('proxy-qwen36-35b-a3b', 'proxy-qwen36-handover')),
            'to-steady': steady}


def label(source):
    return {'com.datadoghq.ad.logs': json.dumps([{
        'source': source, 'service': source,
        'tags': ['model:Qwen/Qwen3.6-35B-A3B-FP8', 'deployment:gpu13-qwen-handover',
                 'env:${ENV}', 'host:${CVM_HOST}', 'ip:${HOST_IP}']}], separators=(',', ':'))}


class LiteralDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        if isinstance(data, str) and len(data) > 1000:
            return False
        return super().ignore_aliases(data)


def represent_str(dumper, value):
    return dumper.represent_scalar('tag:yaml.org,2002:str', value, style='|' if '\n' in value else None)


LiteralDumper.add_representer(str, represent_str)


def render():
    pack = yaml.safe_load((ROOT / 'prod/small-models.yaml').read_text())
    services = pack['services']
    logging = copy.deepcopy(pack['x-logging-conf'])
    script = (ROOT / 'scripts/qwen_handover.py').read_text()
    # Encode candidate mapping as an argument, retaining Compose's $$ escaping.
    configs = json.dumps(candidates(pack['configs']['nginx_conf']['content']))
    proxy = copy.deepcopy(services['proxy-qwen36-35b-a3b'])
    proxy['container_name'] = 'proxy-qwen36-handover'
    proxy['labels'] = label('vllm-proxy')
    proxy['profiles'] = ['qwen-handover']
    proxy['environment'] = [
        'VLLM_BACKEND_URLS=http://model-sg-qwen36-35b-a3b-fp8-tp1-${QWEN_HANDOVER_REPLICA:-r1}:8000'
        if entry.startswith('VLLM_BACKEND_URLS=') else entry for entry in proxy['environment']]
    warmer = copy.deepcopy(services['model-downloader-qwen38'])
    warmer['container_name'] = 'handover-model-downloader-qwen38'
    warmer['restart'] = 'no'
    warmer['profiles'] = ['qwen-handover']
    warmer['labels'] = label('model-downloader')
    warmer['volumes'] = ['hugginface_cache:/root/.cache/huggingface']
    result = {'name': 'gpu13-qwen-handover',
              'services': {'proxy-qwen36-handover': proxy,
                           'handover-model-downloader-qwen38': warmer},
              'networks': {'default': {'external': True, 'name': 'dstack_default'}},
              'volumes': {'certs': {'external': True, 'name': 'certs'},
                          'hugginface_cache': {'external': True, 'name': 'work_hugginface_cache'},
                          'handover_state': {}}}
    for mode in ['canary', 'preflight', 'to-temp', 'to-canonical', 'to-steady', 'drain']:
        name = 'handover-' + mode
        helper = {'image': UTILITY, 'container_name': name, 'profiles': ['qwen-handover'],
                  'entrypoint': ['python3', '-c'], 'command': [script, mode, configs],
                  'user': '0', 'read_only': True, 'cap_drop': ['ALL'],
                  'security_opt': ['no-new-privileges:true'], 'restart': 'no',
                  'logging': logging, 'labels': label('qwen-handover')}
        if mode == 'canary':
            helper['network_mode'] = 'none'
            helper['command'] = [script, mode]
        else:
            helper['pid'] = 'container:nginx'
            helper['volumes'] = ['handover_state:/handover-state']
            helper['cap_add'] = ['SYS_PTRACE', 'SYS_CHROOT', 'DAC_OVERRIDE', 'KILL']
            # Default AppArmor disallows /proc/<master>/root writes. The helper
            # retains seccomp, limited capabilities, no host PID or Docker socket.
            helper['security_opt'].append('apparmor:unconfined')
        result['services'][name] = helper
    result['services']['handover-qualify'] = {
        'image': UTILITY, 'container_name': 'handover-qualify', 'profiles': ['qwen-handover'],
        'entrypoint': ['python3', '-c'],
        'command': [(ROOT / 'scripts/qwen_qualify.py').read_text(),
                    (ROOT / 'scripts/qwen_qualification.lock').read_text(),
                    (ROOT / 'scripts/handover_ohttp.py').read_text()],
        'environment': ['PROXY_TOKEN=${PROXY_TOKEN}', 'QWEN_HANDOVER_REPLICA=${QWEN_HANDOVER_REPLICA:-r1}'],
        'user': '0', 'read_only': True, 'cap_drop': ['ALL'],
        'security_opt': ['no-new-privileges:true'], 'restart': 'no',
        # Native hash-locked wheels need executable mmap; root stays read-only.
        'tmpfs': ['/tmp:rw,exec,nosuid,nodev,size=256m'],
        'logging': logging, 'labels': label('qwen-handover')}
    model_check = copy.deepcopy(result['services']['handover-qualify'])
    model_check['container_name'] = 'handover-model-check'
    model_check['command'].append('model-check')
    model_check['environment'] = ['QWEN_HANDOVER_REPLICA=${QWEN_HANDOVER_REPLICA:-r1}']
    result['services']['handover-model-check'] = model_check
    return '# Generated by scripts/render_qwen_handover.py; do not edit by hand.\n' + yaml.dump(
        result, Dumper=LiteralDumper, sort_keys=False, width=100000)


if __name__ == '__main__':
    text = render()
    if '--check' in sys.argv:
        assert OUTPUT.read_text() == text, 'run python3 scripts/render_qwen_handover.py'
    else:
        OUTPUT.write_text(text)
