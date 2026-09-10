#!/usr/bin/env python3
"""Prepare an r2-only promotion after the signed image has been published.

Defaults to a reviewable diff. --write changes local files; never deploys.
"""
import argparse
import difflib
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = Path('prod/GLM-5.3-Flash-SGL-TP4.yaml')
RELEASE = Path('docker/sglang-glm53-hicache/RELEASED_IMAGE')
VARIANT = 'fc91d24-hicache-pooled-v1-h200-tp4-ep4-eagle-adaptive-5-1-6-strict-budget8192'
OPTIONS = {
    '--hicache-size': '64',
    '--hicache-write-policy': 'write_through',
    '--hicache-io-backend': 'direct',
    '--hicache-mem-layout': 'page_first_direct',
}
ENVIRONMENT = {
    'SGLANG_HICACHE_POOLED_TRANSFERS': '1',
    'SGLANG_HICACHE_STAGING_PAGES': '64',
}


def candidate(text, image):
    if not re.fullmatch(r'docker\.io/nearaidev/sglang@sha256:[a-f0-9]{64}', image):
        raise ValueError('Require the immutable docker.io/nearaidev/sglang digest from the signed publishing workflow')
    if '--enable-hierarchical-cache' in text or 'SGLANG_HICACHE_POOLED_TRANSFERS' in text:
        raise ValueError('Compose already has a HiCache configuration; review it explicitly')
    start = text.index('x-sg-glm53-flash-common:')
    end = text.index('\nx-dcgm-common:', start)
    common = text[start:end]
    command = common[common.index('  command: >'):common.index('  volumes:')]
    environment = common[common.index('  environment:'):common.index('  restart:')]
    command = ''.join('  ' + line if line.strip() else line for line in command.splitlines(True))
    command += '        --enable-hierarchical-cache\n'
    command += ''.join(f'        {key} {value}\n' for key, value in OPTIONS.items())
    environment = ''.join('  ' + line if line.strip() else line for line in environment.splitlines(True))
    environment += ''.join(f'      - {key}={value}\n' for key, value in ENVIRONMENT.items())
    start = text.index('  model-sg-glm53-fp8-tp4-r2:\n')
    end = text.index('\n  # Explicit operator-only semantic check;', start)
    service = text[start:end]
    insertion = '    container_name: model-sg-glm53-fp8-tp4-r2\n'
    if service.count(insertion) != 1 or '    image:' in service or '    command:' in service:
        raise ValueError('Unrecognized r2 service shape; refusing to overwrite overrides')
    service = service.replace(insertion, insertion + f'    image: {image}\n' + command + environment)
    original_variant = 'official-upstream-fc91d24-h200-tp4-ep4-eagle-adaptive-5-1-6-strict-budget8192'
    if service.count(original_variant) != 2:
        raise ValueError('Unrecognized r2 telemetry variant')
    service = service.replace(original_variant, VARIANT)
    text = text[:start] + service + text[end:]
    start = text.index('              - job_name: sglang-model-sg-glm53-fp8-tp4-r2\n')
    end = text.index('              - job_name:', start + 1)
    scrape = text[start:end]
    if scrape.count(original_variant) != 1:
        raise ValueError('Unrecognized r2 scrape labels')
    text = text[:start] + scrape.replace(original_variant, VARIANT) + text[end:]
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    if (ROOT / RELEASE).exists():
        parser.error('A release is already recorded; review or revert it before preparing another')
    original = (ROOT / COMPOSE).read_text()
    updated = candidate(original, args.image)
    for name, before, after in ((COMPOSE, original, updated), (RELEASE, '', args.image + '\n')):
        print(''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                          fromfile='a/' + str(name), tofile='b/' + str(name))), end='')
        if args.write:
            (ROOT / name).write_text(after)


if __name__ == '__main__':
    main()
