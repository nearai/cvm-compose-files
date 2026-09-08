#!/usr/bin/env python3
"""Synthetic, in-CVM compatibility gate. Emits no tokens, keys, chats or quotes.

Checks report bindings and live signatures, not an independent Intel/NVIDIA
certificate-chain verification. Only the two fixed local Qwen proxy names are used.
"""
import hashlib
import json
import os
from pathlib import Path
import socket
import ssl
import subprocess
import sys
import time
import uuid

MODEL = 'Qwen/Qwen3.6-35B-A3B-FP8'
DOMAIN = 'qwen3-6-35b.completions.near.ai'
BASES = ('http://proxy-qwen36-35b-a3b:8000', 'http://proxy-qwen36-handover:8000')


def check(condition, name):
    if not condition:
        raise RuntimeError(name)


def sha(value):
    return hashlib.sha256(value).hexdigest()


def deps():
    requirements = Path('/tmp/qualification.lock')
    requirements.write_text(sys.argv[1])
    result = subprocess.run(['uv', 'pip', 'install', '--require-hashes', '--no-deps',
                             '--only-binary', ':all:', '--target', '/tmp/qualification-deps',
                             '-r', str(requirements)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180,
                            env={**os.environ, 'UV_CACHE_DIR': '/tmp/uv-cache', 'UV_NO_PROGRESS': '1'})
    check(result.returncode == 0, 'locked_dependencies_failed')
    sys.path.insert(0, '/tmp/qualification-deps')
    # Protocol code is a reviewed, fixed argument embedded by the renderer.
    exec(compile(sys.argv[2], 'handover_ohttp.py', 'exec'), globals())


def verify_signature(report, signed, expected_text):
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec, ed25519, utils
    from Crypto.Hash import keccak
    check(signed['text'] == expected_text, 'signature_payload_binding')
    check(signed['signing_address'].lower() == report['signing_address'].lower(), 'signature_identity')
    message = signed['text'].encode()
    signature = bytes.fromhex(signed['signature'].removeprefix('0x'))
    public = bytes.fromhex(report['signing_public_key'].removeprefix('0x'))
    if report['signing_algo'] == 'ed25519':
        ed25519.Ed25519PublicKey.from_public_bytes(public).verify(signature, message)
    else:
        check(report['signing_algo'] == 'ecdsa' and len(signature) == 65 and len(public) == 64, 'ecdsa_shape')
        address = keccak.new(digest_bits=256, data=public).digest()[-20:].hex()
        check(report['signing_address'].lower().removeprefix('0x') == address, 'ecdsa_address_binding')
        prefix = b'\x19Ethereum Signed Message:\n' + str(len(message)).encode()
        digest = keccak.new(digest_bits=256, data=prefix + message).digest()
        key = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256K1(), b'\x04' + public)
        der = utils.encode_dss_signature(int.from_bytes(signature[:32], 'big'), int.from_bytes(signature[32:64], 'big'))
        # Prehashed uses SHA256 only for the 32-byte size; digest is EIP191 Keccak.
        key.verify(der, digest, ec.ECDSA(utils.Prehashed(hashes.SHA256())))


def answer(raw, stream=False):
    if not stream:
        data = json.loads(raw)
        choice = data['choices'][0]
        check(choice['message']['content'].strip() == '42' and choice['finish_reason'] == 'stop', 'semantic_answer')
        return data
    text, finish, terminal = '', None, False
    for line in raw.decode().splitlines():
        if not line.startswith('data: '): continue
        check(not terminal, 'event_after_terminal')
        if line == 'data: [DONE]':
            terminal = True
            continue
        event = json.loads(line[6:])
        check('error' not in event, 'stream_error_event')
        choices = event.get('choices', [])
        if not choices:
            check('usage' in event, 'unexpected_empty_stream_event')
            continue
        choice = choices[0]
        text += choice.get('delta', {}).get('content') or ''
        finish = choice.get('finish_reason') or finish
    check(terminal and text.strip() == '42' and finish == 'stop', 'stream_answer')


def tls_fingerprint():
    from cryptography import x509
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
    context = ssl.create_default_context()
    with socket.create_connection(('nginx', 443), timeout=15) as sock:
        with context.wrap_socket(sock, server_hostname=DOMAIN) as connection:
            cert = x509.load_der_x509_certificate(connection.getpeercert(binary_form=True))
            return sha(cert.public_key().public_bytes(Encoding.DER, PublicFormat.SubjectPublicKeyInfo))


def qualify():
    import requests
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    token = os.environ['PROXY_TOKEN']
    check(bool(token), 'missing_proxy_token')
    session = requests.Session()
    session.trust_env = False
    headers = {'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'}
    checks, reports, configs = [], {}, {}
    fingerprint = tls_fingerprint()

    def get(base, path, **kwargs):
        response = session.get(base + path, headers=headers, timeout=90, **kwargs)
        check(response.status_code == 200, 'get_' + str(response.status_code))
        return response

    def post(base, body):
        encoded = json.dumps(body, separators=(',', ':')).encode()
        response = session.post(base + '/v1/chat/completions', headers=headers, data=encoded, timeout=180)
        check(response.status_code == 200, 'completion_' + str(response.status_code))
        return encoded, response.content

    payload = {'model': MODEL, 'messages': [{'role': 'user', 'content': 'What is 19 + 23? Reply with only the number.'}],
               'max_tokens': 1024, 'temperature': 0, 'chat_template_kwargs': {'enable_thinking': False}}
    try:
        for base in BASES:
            check(get(base, '/healthz').json()['status'] == 'ok', 'proxy_health')
            check([row['id'] for row in get(base, '/v1/models').json()['data']] == [MODEL], 'model_listing')
            check(len(get(base, '/metrics').content) > 0, 'proxy_metrics')
            config = get(base, '/.well-known/ohttp-gateway').content
            check(config == get(base, '/v1/ohttp/config').content, 'ohttp_config_alias')
            configs[base] = config
            reports[base] = {}
            for algo in ('ecdsa', 'ed25519'):
                nonce = os.urandom(32).hex()
                report = get(base, '/v1/attestation/report', params={
                    'signing_algo': algo, 'nonce': nonce, 'include_tls_fingerprint': 'true'}).json()
                check(report['model_name'] == MODEL and report['signing_algo'] == algo, 'attestation_model_algorithm')
                check(report['request_nonce'] == nonce and bool(report['intel_quote']) and bool(report['nvidia_payload']), 'fresh_attestation_shape')
                check(report['tls_cert_fingerprint'] == fingerprint, 'live_tls_spki_binding')
                binding = report['ohttp_attestation']
                check(binding['key_config'] == config.hex() and binding['signing_algo'] == 'ed25519', 'ohttp_attestation_binding')
                Ed25519PublicKey.from_public_bytes(bytes.fromhex(binding['signing_key'])).verify(bytes.fromhex(binding['signature']), config)
                if algo == 'ed25519':
                    check(binding['signing_key'] == report['signing_public_key'], 'ohttp_signer_binding')
                reports[base][algo] = report
            request_raw, response_raw = post(base, payload)
            data = answer(response_raw)
            expected = MODEL + ':' + sha(request_raw) + ':' + sha(response_raw)
            for algo in ('ecdsa', 'ed25519'):
                for attempt in range(6):
                    response = session.get(base + '/v1/signature/' + data['id'], headers=headers,
                                           params={'signing_algo': algo}, timeout=20)
                    if response.status_code != 404: break
                    time.sleep(1)
                check(response.status_code == 200, 'signature_unavailable')
                verify_signature(reports[base][algo], response.json(), expected)
            _, streaming = post(base, {**payload, 'stream': True})
            answer(streaming, stream=True)
            tool = {'type': 'function', 'function': {'name': 'add', 'description': 'Add two integers',
                    'parameters': {'type': 'object', 'properties': {'a': {'type': 'integer'}, 'b': {'type': 'integer'}},
                                   'required': ['a', 'b'], 'additionalProperties': False}}}
            _, raw = post(base, {**payload, 'messages': [{'role': 'user', 'content': 'Call add with a=19 and b=23.'}],
                                'tools': [tool], 'tool_choice': {'type': 'function', 'function': {'name': 'add'}}})
            call = json.loads(raw)['choices'][0]['message']['tool_calls'][0]['function']
            check(call['name'] == 'add' and json.loads(call['arguments']) == {'a': 19, 'b': 23}, 'tool_call')
            checks.append({'proxy': 'canonical' if base == BASES[0] else 'temporary',
                           'health_models_metrics': True, 'semantic_stream_tool': True,
                           'both_signatures_verified': True, 'fresh_nonce_and_tls_bindings': True,
                           'ohttp_key_signature_verified': True})
        for algo in ('ecdsa', 'ed25519'):
            for key in ('signing_public_key', 'signing_address'):
                check(reports[BASES[0]][algo][key] == reports[BASES[1]][algo][key], 'proxy_signing_identity_mismatch')
        check(configs[BASES[0]] == configs[BASES[1]], 'proxy_ohttp_key_mismatch')
        for chunked, streaming in [(False, False), (True, False), (True, True)]:
            body = json.dumps({**payload, **({'stream': True} if streaming else {})}, separators=(',', ':')).encode()
            # Key came from canonical before the request; request goes to temporary.
            exchange = Exchange(configs[BASES[0]], chunked=chunked)
            wire = exchange.request(binary_request(DOMAIN, token, body))
            media = 'message/ohttp' + ('-chunked' if chunked else '')
            response = session.post(BASES[1] + '/ohttp', data=wire, headers={'Content-Type': media + '-req'}, timeout=180)
            check(response.status_code == 200 and response.headers.get('Content-Type', '').split(';')[0] == media + '-res', 'ohttp_outer_response')
            answer(binary_response(exchange.response(response.content)), stream=streaming)
        return {'status': 'ok', 'checks': checks, 'signing_identities_equal': True,
                'canonical_cached_key_on_temporary': ['standard', 'chunked', 'chunked_stream'],
                'tls_spki_sha256': fingerprint, 'independent_quote_chain_verification': False}
    finally:
        session.close()


def main():
    operation = uuid.uuid4().hex
    print(json.dumps({'operation_id': operation, 'stage': 'started', 'mode': 'qualify'}), flush=True)
    try:
        deps()
        result = qualify()
    except Exception as error:
        result = {'status': 'failed', 'reason': str(error) if type(error) is RuntimeError else type(error).__name__}
    result.update(operation_id=operation, mode='qualify', terminal=True, emitted_utc=time.time())
    for _ in range(19):
        print(json.dumps(result, sort_keys=True), flush=True)
        time.sleep(5)
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__':
    sys.exit(main())
