#!/usr/bin/env python3
"""Real-Docker synthetic tests: scoped namespace HUP, HTTP/2 and signature continuity."""
import io
import json
from pathlib import Path
import subprocess
import tarfile
import time
import uuid
import yaml

from render_qwen_handover import ROOT, UTILITY, candidates, render
from validate_streaming_keepalive import TEST_CERTIFICATE, TEST_PRIVATE_KEY

BACKEND = r'''
import os, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
identity = os.environ['IDENTITY']
released = threading.Event()
class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    def log_message(self, *args): pass
    def do_GET(self):
        if self.path == '/release': released.set()
        if self.path == '/stream':
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Connection', 'close')
            self.end_headers()
            self.wfile.write((identity + '\n').encode()); self.wfile.flush()
            if not released.wait(90): return
            self.wfile.write(b'DONE\n'); self.wfile.flush()
            self.close_connection = True
            return
        status = 200
        body = identity
        if self.path.startswith('/v1/signature/'):
            requested = self.path.rsplit('/', 1)[1]
            if requested != identity: status, body = 404, 'missing'
        self.send_response(status)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers(); self.wfile.write(body.encode())
    def do_POST(self):
        self.send_response(503)
        self.send_header('Content-Length', '0')
        self.end_headers()
ThreadingHTTPServer(('0.0.0.0', 8000), Handler).serve_forever()
'''


def run(*args, check=True, timeout=60, input=None):
    result = subprocess.run(args, input=input, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            timeout=timeout)
    if check and result.returncode:
        raise AssertionError(f'{args[:3]} exit={result.returncode}: {result.stderr.decode()[-1200:]}')
    return result


def copy_files(container, files):
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode='w') as archive:
        for path, content in files.items():
            raw = content.encode()
            entry = tarfile.TarInfo(path.lstrip('/'))
            entry.size, entry.mode = len(raw), 0o644
            archive.addfile(entry, io.BytesIO(raw))
    run('docker', 'cp', '-', container + ':/', input=data.getvalue())


def port(container, internal):
    return run('docker', 'port', container, str(internal)).stdout.decode().strip().rsplit(':', 1)[1]


def main():
    assert (ROOT / 'prod/gpu13-qwen-handover.yaml').read_text() == render()
    pack = yaml.safe_load((ROOT / 'prod/small-models.yaml').read_text())
    cfg = {key: value.replace('$$', '$') for key, value in candidates(pack['configs']['nginx_conf']['content']).items()}
    code = (ROOT / 'scripts/qwen_handover.py').read_text().rsplit("if __name__ == '__main__':", 1)[0]
    code += "\nSTATE.mkdir(exist_ok=True)\nprint(json.dumps(operate(sys.argv[1], json.loads(sys.argv[2]), uuid.uuid4().hex)))\n"
    suffix = uuid.uuid4().hex[:12]
    network = 'qwen-hup-test-' + suffix
    volume = network + '-state'
    old, new, target = [network + '-' + n for n in ['old', 'new', 'nginx']]
    created = []
    streams = []
    try:
        run('docker', 'network', 'create', network)
        run('docker', 'volume', 'create', volume)
        old_aliases = ['proxy-qwen36-35b-a3b', 'proxy-qwen38-27b', 'proxy-glm51', 'proxy-flux2-klein-4b',
                       'proxy-qwen3vl-30b-a3b', 'proxy-qwen3-embedding-0.6b',
                       'proxy-qwen3-reranker-0.6b', 'proxy-whisper-large-v3', 'proxy-privacy-filter']
        for name, identity, aliases in [(old, 'old', old_aliases), (new, 'new', ['proxy-qwen36-handover'])]:
            args = ['docker', 'run', '-d', '--name', name, '--network', network,
                    '-p', '127.0.0.1::8000', '-e', 'IDENTITY=' + identity]
            for alias in aliases: args += ['--network-alias', alias]
            run(*args, UTILITY, 'python3', '-c', BACKEND)
            created.append(name)
        image = pack['services']['nginx']['image']
        run('docker', 'create', '--name', target, '--network', network, '-p', '127.0.0.1::443',
            '--entrypoint', '/bin/sh', image, '-c', pack['services']['nginx']['command'].split("'", 1)[1][:-1])
        created.append(target)
        copy_files(target, {'/etc/nginx/conf.d/default.conf': cfg['steady'],
                            '/etc/letsencrypt/live/completions.near.ai/fullchain.pem': TEST_CERTIFICATE,
                            '/etc/letsencrypt/live/completions.near.ai/privkey.pem': TEST_PRIVATE_KEY})
        # Lower only fixture worker count, preserving the production include.
        run('docker', 'start', target)
        original_id = run('docker', 'inspect', '-f', '{{.Id}}', target).stdout
        tls_port = port(target, 443)

        def curl(path='/', domain='qwen3-6-35b.completions.near.ai', extra=(), stream=False):
            args = ['curl', '--http2', '--insecure', '--noproxy', '*', '--silent', '--show-error',
                    '--max-time', '100' if stream else '10', '--resolve', f'{domain}:{tls_port}:127.0.0.1',
                    *extra, f'https://{domain}:{tls_port}{path}']
            if stream:
                return subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return run(*args).stdout.decode()

        deadline = time.monotonic() + 30
        while True:
            try:
                assert curl() == 'old'
                break
            except AssertionError:
                if time.monotonic() > deadline: raise
                time.sleep(0.25)

        def helper(mode, mapping=cfg, check=True):
            result = run('docker', 'run', '--rm', '--network', network,
                         '--pid', 'container:' + target, '--read-only', '--cap-drop', 'ALL',
                         '--cap-add', 'SYS_PTRACE', '--cap-add', 'SYS_CHROOT',
                         '--cap-add', 'DAC_OVERRIDE', '--cap-add', 'KILL',
                         '--security-opt', 'no-new-privileges:true', '--security-opt', 'apparmor:unconfined',
                         '-v', volume + ':/handover-state', UTILITY,
                         'python3', '-c', code, mode, json.dumps(mapping), check=check)
            return json.loads(result.stdout) if result.returncode == 0 else result

        def route_is(expected):
            # HUP is asynchronous: observing a forked worker is not yet proof
            # that the master finished transferring new-connection admission.
            deadline = time.monotonic() + 15
            while True:
                observed = curl()
                if observed == expected: return
                assert time.monotonic() < deadline, f'new connections stayed on {observed!r}, expected {expected!r}'
                time.sleep(0.1)

        assert helper('preflight')['no_signal']
        wrong = dict(cfg, steady='wrong pre-state')
        assert helper('to-temp', wrong, check=False).returncode != 0
        bad = dict(cfg, **{'to-temp': 'invalid nginx syntax;\n'})
        assert helper('to-temp', bad, check=False).returncode != 0
        assert curl() == 'old'
        for domain in ['qwen3-6-35b.completions.near.ai', 'glm-5-1.completions.near.ai']:
            process = curl('/stream', domain, extra=('--no-buffer', '--write-out', 'HTTP%{http_version}\n'), stream=True)
            streams.append(process)
            assert process.stdout.readline() == b'old\n'
        result = helper('to-temp')
        assert result['status'] == 'ok' and result['old_workers'] and result['new_workers']
        assert not helper('drain')['drained']
        route_is('new')
        assert curl('/v1/signature/old') == 'old'
        assert curl('/v1/signature/new') == 'new'
        assert curl('/v1/chat/completions', extra=('-X', 'POST', '--write-out', '%{http_code}')) == '503'
        assert curl('/v1/signature/old', extra=('-X', 'POST', '--write-out', '%{http_code}')).endswith('403')
        assert curl('/', 'glm-5-1.completions.near.ai') == 'old'
        run('curl', '--silent', '--fail', f'http://127.0.0.1:{port(old, 8000)}/release')
        for process in streams:
            output, stderr = process.communicate(timeout=15)
            assert process.returncode == 0 and b'DONE\nHTTP2\n' in output, (output, stderr)
        streams.clear()

        def drained():
            deadline = time.monotonic() + 30
            while not helper('drain')['drained']:
                assert time.monotonic() < deadline, 'workers did not drain naturally'
                time.sleep(0.25)

        drained()
        assert helper('to-canonical')['status'] == 'ok'
        route_is('old')
        assert curl('/v1/signature/new') == 'new'
        assert curl('/v1/signature/old') == 'old'
        drained()
        assert helper('to-steady')['status'] == 'ok'
        drained()
        assert curl() == 'old'
        assert run('docker', 'inspect', '-f', '{{.Id}}', target).stdout == original_id
        dedicated = network + '-qwen38'
        run('docker', 'create', '--name', dedicated, '--network', network,
            '-p', '127.0.0.1::8000', '-p', '127.0.0.1::443',
            '--entrypoint', '/bin/sh', pack['services']['nginx-qwen38-27b']['image'],
            '-c', pack['services']['nginx-qwen38-27b']['command'].split("'", 1)[1][:-1])
        created.append(dedicated)
        copy_files(dedicated, {
            '/etc/nginx/conf.d/default.conf': pack['configs']['nginx_qwen38_conf']['content'].replace('$$', '$'),
            '/etc/letsencrypt/live/completions.near.ai/fullchain.pem': TEST_CERTIFICATE,
            '/etc/letsencrypt/live/completions.near.ai/privkey.pem': TEST_PRIVATE_KEY})
        run('docker', 'start', dedicated)
        dedicated_port = port(dedicated, 443)
        probe_port = port(dedicated, 8000)
        deadline = time.monotonic() + 30
        while True:
            ready = run('curl', '--silent', '--fail', '--max-time', '2',
                        f'http://127.0.0.1:{probe_port}/', check=False)
            if ready.returncode == 0 and ready.stdout == b'old': break
            assert time.monotonic() < deadline, 'dedicated ingress did not become ready'
            time.sleep(0.25)
        for host in ['qwen3-8-27b.completions.near.ai', 'qwen3-8-27b-i1.completions.near.ai']:
            result = run('curl', '--silent', '--show-error', '--fail', '--insecure', '--noproxy', '*',
                         '--retry', '10', '--retry-connrefused', '--retry-delay', '1', '--max-time', '15',
                         '--http2', '--write-out', 'HTTP%{http_version}',
                         '--resolve', f'{host}:{dedicated_port}:127.0.0.1', f'https://{host}:{dedicated_port}/')
            assert result.stdout == b'oldHTTP2'
        assert run('curl', '--silent', '--fail', f'http://127.0.0.1:{probe_port}/').stdout == b'old'
        assert run('docker', 'inspect', '-f', '{{.Id}}', target).stdout == original_id
        scoped = network + '-scope'
        scoped_file = yaml.safe_dump({'services': {
            name: {'image': UTILITY, 'command': ['sleep', '600']}
            for name in ['one', 'two']},
            'networks': {'default': {'external': True, 'name': network}}}).encode()
        def compose(*args):
            return run('docker', 'compose', '-f', '-', '-p', scoped, *args, input=scoped_file)
        try:
            compose('up', '-d')
            peer = compose('ps', '-q', 'two').stdout.strip()
            network_id = run('docker', 'network', 'inspect', '-f', '{{.Id}}', network).stdout
            compose('down', 'one')
            assert compose('ps', '-q', 'one').stdout.strip() == b''
            assert compose('ps', '-q', 'two').stdout.strip() == peer and peer
            assert run('docker', 'network', 'inspect', '-f', '{{.Id}}', network).stdout == network_id
        finally:
            compose('down')
        print('PASS: same nginx; two HTTP/2 streams complete; forward/reverse signature fallback; no POST retry; exact steady restore; negative guards; natural worker drain')
        print('PASS: scoped Compose down preserves peer identity and external network')
        print('PASS: separate Qwen3.8 HTTP probe and HTTP/2 TLS listener with canonical/indexed SNI')
    finally:
        for process in streams:
            process.terminate()
            process.communicate(timeout=10)
        for name in reversed(created): run('docker', 'rm', '-f', name, check=False)
        run('docker', 'volume', 'rm', volume, check=False)
        run('docker', 'network', 'rm', network, check=False)


if __name__ == '__main__':
    main()
