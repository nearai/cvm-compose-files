#!/usr/bin/env python3
"""Bounded nginx HUP helper. Embedded in the isolated Compose file by the renderer.

No Docker socket, host PID namespace, raw config/log output, or worker signals.
Results describe a transition; they do not authorize stopping either proxy.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time
import uuid

STATE = Path('/handover-state')
CONFIG = '/etc/nginx/conf.d/default.conf'
MAIN_CONFIG = '/etc/nginx/nginx.conf'


def require(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def proc(pid):
    try:
        directory = Path('/proc') / str(pid)
        raw = (directory / 'stat').read_text()
        tail = raw[raw.rindex(')') + 2:].split()
        return {'pid': int(pid), 'start': int(tail[19]), 'ppid': int(tail[1]),
                'cmd': (directory / 'cmdline').read_bytes().replace(b'\0', b' ').strip()}
    except (OSError, ValueError):
        return None


def identity(row):
    return {'pid': row['pid'], 'start': row['start']}


def alive(row):
    current = proc(row['pid'])
    return current is not None and current['start'] == row['start']


def processes():
    return [row for path in Path('/proc').iterdir()
            if path.name.isdigit() and (row := proc(path.name)) is not None]


def topology():
    rows = processes()
    masters = [p for p in rows if p['cmd'].startswith(b'nginx: master process ')]
    require(len(masters) == 1, 'expected_one_nginx_master')
    master = masters[0]
    workers = [identity(p) for p in rows if p['ppid'] == master['pid']
               and p['cmd'].startswith(b'nginx: worker process')]
    require(workers, 'missing_nginx_workers')
    sleepers = [p for p in rows if p['cmd'] == b'sleep 6h']
    require(len(sleepers) == 1, 'expected_one_periodic_reload_sleeper')
    # Include parent identity: an unrelated sleep is not the reload loop.
    parent = proc(sleepers[0]['ppid'])
    require(parent is not None and b'nginx -s reload' in parent['cmd'], 'reload_loop_identity')
    elapsed = float(Path('/proc/uptime').read_text().split()[0])
    elapsed -= sleepers[0]['start'] / os.sysconf('SC_CLK_TCK')
    require(0 <= elapsed < 21600 - 120, 'periodic_reload_too_close')
    return {'master': identity(master), 'workers': sorted(workers, key=lambda p: p['pid']),
            'sleeper': identity(sleepers[0]), 'loop': identity(parent)}


def target_root(topo):
    root = Path('/proc') / str(topo['master']['pid']) / 'root'
    require(alive(topo['master']), 'master_changed')
    exe = os.readlink(root.parent / 'exe')
    require(exe == '/usr/sbin/nginx', 'unexpected_nginx_executable')
    return root


def read_config(root):
    path = root / CONFIG.lstrip('/')
    require(stat.S_ISREG(path.lstat().st_mode), 'config_not_regular')
    mounts = [line.split()[4] for line in (root.parent / 'mountinfo').read_text().splitlines()]
    require(CONFIG not in mounts and not os.path.ismount(path), 'config_is_mountpoint')
    require(not os.statvfs(path).f_flag & os.ST_RDONLY, 'config_filesystem_readonly')
    return path, path.read_bytes()


def atomic_write(path, data):
    tmp = path.with_name('.handover-' + uuid.uuid4().hex)
    try:
        with tmp.open('xb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def validate_candidate(root, candidate, operation):
    """Test a separate main config: live *.conf remains untouched, even on HUP."""
    original_main = (root / MAIN_CONFIG.lstrip('/')).read_bytes()
    include = b'include /etc/nginx/conf.d/*.conf;'
    require(original_main.count(include) == 1, 'unexpected_main_include')
    directory = root / 'etc/nginx'
    fragment = directory / ('.handover-' + operation + '.fragment')
    main = directory / ('.handover-' + operation + '.main')
    try:
        fragment.write_bytes(candidate)
        # Include other conf.d files too, but never include the live default.conf.
        others = sorted((root / 'etc/nginx/conf.d').glob('*.conf'))
        require(all(p.is_file() and not p.is_symlink() for p in others), 'unexpected_config_entry')
        includes = [f'include /etc/nginx/{fragment.name};']
        includes += [f'include /etc/nginx/conf.d/{p.name};' for p in others if p.name != 'default.conf']
        require(all(re.fullmatch(r'[A-Za-z0-9_.-]+', p.name) for p in others), 'unsafe_config_name')
        main.write_bytes(original_main.replace(include, '\n'.join(includes).encode()))
        # Capture and discard stderr: nginx can include paths/config content.
        tested = subprocess.run(['/usr/sbin/chroot', str(root), '/usr/sbin/nginx', '-t',
                                 '-c', '/etc/nginx/' + main.name],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
        require(tested.returncode == 0, 'nginx_candidate_test_failed')
    finally:
        fragment.unlink(missing_ok=True)
        main.unlink(missing_ok=True)


def read_state():
    path = STATE / 'transition.json'
    return json.loads(path.read_text()) if path.exists() else None


def save_state(data):
    atomic_write(STATE / 'transition.json', json.dumps(data, sort_keys=True).encode())


def wait_generation(master, old_workers, timeout=15):
    deadline = time.monotonic() + timeout
    old = {(p['pid'], p['start']) for p in old_workers}
    while time.monotonic() < deadline:
        require(alive(master), 'master_changed_after_hup')
        rows = [identity(p) for p in processes() if p['ppid'] == master['pid']
                and p['cmd'].startswith(b'nginx: worker process')]
        new = [p for p in rows if (p['pid'], p['start']) not in old]
        if new:
            return new
        time.sleep(0.1)
    raise RuntimeError('new_workers_missing')


def operate(mode, candidates, operation):
    require(mode in ('preflight', 'to-temp', 'to-canonical', 'to-steady', 'drain'), 'unknown_mode')
    topo = topology()
    root = target_root(topo)
    path, original = read_config(root)
    current_sha = sha(original)
    prior = read_state()
    if mode == 'drain':
        require(prior is not None and prior['status'] == 'ok', 'missing_successful_transition')
        require(prior['master'] == topo['master'], 'master_changed_since_transition')
        require(prior['new_sha'] == current_sha, 'config_changed_since_transition')
        pending = [p for p in prior['old_workers'] if alive(p)]
        return {'status': 'ok', 'phase': prior['phase'], 'master': topo['master'],
                'config_sha': current_sha, 'pending_workers': pending, 'drained': not pending}

    phase = 'to-temp' if mode == 'preflight' else mode
    previous = {'to-temp': 'steady', 'to-canonical': 'to-temp', 'to-steady': 'to-canonical'}[phase]
    require(current_sha == sha(candidates[previous].encode()), 'unexpected_config_sha')
    if mode != 'preflight' and phase != 'to-temp':
        require(prior is not None and prior['status'] == 'ok', 'missing_prior_transition')
        require(prior['phase'] == previous and prior['master'] == topo['master'], 'unexpected_prior_transition')
        require(not any(alive(p) for p in prior['old_workers']), 'prior_workers_not_drained')
    candidate = candidates[phase].encode()
    # Prove the actual config directory supports atomic writes before signaling.
    # The hidden sibling is outside nginx's *.conf glob; the live file is intact.
    probe = path.with_name('.handover-write-probe-' + operation)
    try:
        atomic_write(probe, b'probe')
    finally:
        probe.unlink(missing_ok=True)
    validate_candidate(root, candidate, operation)
    require(topology() == topo and read_config(root)[1] == original, 'concurrent_reload_or_config_change')
    if mode == 'preflight':
        return {'status': 'ok', 'config_sha': current_sha, 'candidate_sha': sha(candidate),
                'master': topo['master'], 'workers': topo['workers'], 'no_signal': True,
                'config_regular_unmounted': True, 'config_directory_writable': True}

    record = {'phase': phase, 'status': 'pending', 'operation_id': operation,
              'master': topo['master'], 'old_workers': topo['workers'],
              'old_sha': current_sha, 'new_sha': sha(candidate), 'utc': time.time()}
    save_state(record)
    # Only an already-tested candidate can ever become visible to the periodic HUP.
    installed = False
    signalled = False
    try:
        require(topology() == topo and read_config(root)[1] == original, 'concurrent_change_before_install')
        atomic_write(path, candidate)
        installed = True
        require(topology() == topo and path.read_bytes() == candidate, 'concurrent_change_before_hup')
        require(alive(topo['master']), 'master_changed_before_hup')
        os.kill(topo['master']['pid'], signal.SIGHUP)
        signalled = True
        record['new_workers'] = wait_generation(topo['master'], topo['workers'])
        require(path.read_bytes() == candidate, 'config_changed_after_hup')
        record['status'] = 'ok'
        save_state(record)
        return record
    except Exception:
        record['status'] = 'failed'
        # Reconcile before retrying; both proxies must remain available regardless.
        if installed and alive(topo['master']):
            current = path.read_bytes()
            if current in (candidate, original):
                atomic_write(path, original)
                if signalled:
                    os.kill(topo['master']['pid'], signal.SIGHUP)
                record['restored_sha'] = current_sha
            else:
                # Do not overwrite another operator's concurrent modification.
                record['rollback_skipped_current_sha'] = sha(current)
        save_state(record)
        raise


def main():
    mode = sys.argv[1]
    operation = uuid.uuid4().hex
    print(json.dumps({'operation_id': operation, 'stage': 'started', 'mode': mode}), flush=True)
    try:
        if mode == 'canary':
            result = {'status': 'ok', 'no_signal': True}
        else:
            candidates = json.loads(sys.argv[2])
            require(set(candidates) == {'steady', 'to-temp', 'to-canonical', 'to-steady'}, 'invalid_candidates')
            require(candidates['steady'] == candidates['to-steady'], 'steady_restore_mismatch')
            STATE.mkdir(exist_ok=True)
            with (STATE / 'lock').open('w') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                result = operate(mode, candidates, operation)
    except Exception as error:
        # Only fixed error codes, never exceptions containing paths/secrets/config.
        reason = str(error) if type(error) is RuntimeError else type(error).__name__
        result = {'status': 'failed', 'reason': reason}
    result.update(operation_id=operation, mode=mode, terminal=True, emitted_utc=time.time())
    # The external log collector starts new files at end. Repeat, then require
    # actual exact-operation retrieval rather than equating exit with delivery.
    for _ in range(19):
        print(json.dumps(result, sort_keys=True), flush=True)
        time.sleep(5)
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__':
    sys.exit(main())
