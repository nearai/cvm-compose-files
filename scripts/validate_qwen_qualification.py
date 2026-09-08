#!/usr/bin/env python3
"""Synthetic regression tests in the exact hash-locked qualification sandbox."""
import json
from pathlib import Path
import subprocess
import sys
import unittest


def inside():
    import qwen_qualify as gate
    source = Path(__file__).resolve().parent
    sys.argv = ['qualification-test', (source / 'qwen_qualification.lock').read_text(),
                (source / 'handover_ohttp.py').read_text()]
    gate.deps()
    import handover_ohttp as wire
    from cryptography.exceptions import InvalidSignature, InvalidTag
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec, ed25519, utils, x25519
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from Crypto.Hash import keccak

    class Tests(unittest.TestCase):
        def test_plain_semantics_and_finish(self):
            gate.answer(json.dumps({'choices': [{'message': {'content': '42'}, 'finish_reason': 'stop'}]}))
            for content, finish in [('41', 'stop'), ('42', 'length')]:
                with self.assertRaises(RuntimeError):
                    gate.answer(json.dumps({'choices': [{'message': {'content': content}, 'finish_reason': finish}]}))

        def test_stream_usage_terminal_and_error(self):
            complete = b'data: {"choices":[{"delta":{"content":"42"},"finish_reason":"stop"}]}\n\n'
            usage = b'data: {"choices":[],"usage":{"completion_tokens":1}}\n\n'
            done = b'data: [DONE]\n\n'
            gate.answer(complete + usage + done, True)
            for bad in [complete, complete + done + complete,
                        complete + b'data: {"error":{"message":"synthetic"}}\n' + done,
                        complete + b'data: {"choices":[]}\n' + done]:
                with self.assertRaises(RuntimeError): gate.answer(bad, True)

        def test_ed25519_signature_and_payload_binding(self):
            key = ed25519.Ed25519PrivateKey.generate()
            public = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
            report = {'signing_algo': 'ed25519', 'signing_public_key': public.hex(), 'signing_address': public.hex()}
            message = 'synthetic:model:request-hash:response-hash'
            signed = {'text': message, 'signing_address': public.hex(), 'signature': key.sign(message.encode()).hex()}
            gate.verify_signature(report, signed, message)
            with self.assertRaises(RuntimeError): gate.verify_signature(report, signed, message + '-different')
            with self.assertRaises(InvalidSignature):
                gate.verify_signature(report, {**signed, 'signature': bytes(64).hex()}, message)

        def test_ecdsa_signature_and_address_binding(self):
            key = ec.generate_private_key(ec.SECP256K1())
            public = key.public_key().public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)[1:]
            address = keccak.new(digest_bits=256, data=public).digest()[-20:].hex()
            report = {'signing_algo': 'ecdsa', 'signing_public_key': public.hex(), 'signing_address': '0x' + address}
            message = 'synthetic:model:request-hash:response-hash'
            raw = message.encode()
            digest = keccak.new(digest_bits=256, data=b'\x19Ethereum Signed Message:\n' + str(len(raw)).encode() + raw).digest()
            r, s = utils.decode_dss_signature(key.sign(digest, ec.ECDSA(utils.Prehashed(hashes.SHA256()))))
            signed = {'text': message, 'signing_address': '0x' + address,
                      'signature': (r.to_bytes(32, 'big') + s.to_bytes(32, 'big') + b'\x1b').hex()}
            gate.verify_signature(report, signed, message)
            wrong = {**report, 'signing_address': '0x' + bytes(20).hex()}
            with self.assertRaises(RuntimeError):
                gate.verify_signature(wrong, {**signed, 'signing_address': wrong['signing_address']}, message)

        def test_binary_http_boundaries(self):
            body = b'synthetic response'
            self.assertEqual(wire.binary_response(b'\x01' + wire.vint(200) + b'\0' + wire.field(body) + b'\0'), body)
            self.assertEqual(wire.binary_response(b'\x03' + wire.vint(200) + b'\0' + wire.field(body) + b'\0\0'), body)
            for value in [0, 63, 64, 16383, 16384, 2**30, 2**62 - 1]:
                self.assertEqual(wire.Reader(wire.vint(value)).integer(), value)
            for bad in [b'', b'\x01', b'\x01' + wire.vint(503) + b'\0\0',
                        b'\x01' + wire.vint(200) + b'\0\x08abc']:
                with self.assertRaises(AssertionError): wire.binary_response(bad)

        def test_authenticated_response_and_final_marker(self):
            key = x25519.X25519PrivateKey.generate().public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
            config = b'\x01\x00\x20' + key + b'\x00\x04\x00\x01\x00\x01'
            for chunked in [False, True]:
                exchange = wire.Exchange(config, chunked)
                nonce = bytes(range(16))
                exported = wire.lexpand(exchange.suite, exchange.exporter, b'sec', exchange.label + b' response', 16)
                secret = wire.extract(exchange.enc + nonce, exported)
                cipher = AESGCM(wire.expand(secret, b'key', 16))
                base = wire.expand(secret, b'nonce', 12)
                body = b'synthetic authenticated response'
                sealed = cipher.encrypt(base, body, b'')
                if chunked:
                    final = cipher.encrypt(exchange.counter_nonce(base, 1), b'', b'final')
                    response = nonce + wire.field(sealed) + b'\0' + final
                else:
                    response = nonce + sealed
                self.assertEqual(exchange.response(response), body)
                with self.assertRaises((AssertionError, InvalidTag)):
                    exchange.response(response[:-1])
                with self.assertRaises(InvalidTag):
                    exchange.response(response[:-1] + bytes([response[-1] ^ 1]))
                if chunked:
                    with self.assertRaises(AssertionError): exchange.response(response + b'\0')
            for bad in [b'', config[:-1], config[:37] + b'\0\2\0\2']:
                with self.assertRaises(AssertionError): wire.Exchange(bad)

        def test_binary_http_trailers_padding_and_permitted_omission(self):
            body = b'synthetic response'
            trailer = wire.field('x-test') + wire.field('ok')
            known = b'\x01' + wire.vint(200) + b'\0' + wire.field(body)
            unknown = b'\x03' + wire.vint(200) + b'\0' + wire.field(body) + b'\0'
            for base, supplied in [(known, wire.field(trailer)), (unknown, trailer + b'\0')]:
                # Omission of an entirely empty trailer is explicitly legal in
                # RFC9292 3.8. Partial lengths/fields or nonzero padding are not.
                for suffix in [b'', b'\0', b'\0\0\0', supplied, supplied + b'\0\0']:
                    self.assertEqual(wire.binary_response(base + suffix), body)
                for suffix in [b'\x40', supplied[:-1], b'\0\x01', b'\x04ab']:
                    with self.assertRaises(AssertionError): wire.binary_response(base + suffix)
            with self.assertRaises(AssertionError): wire.binary_response(known + wire.field(b'\0\0'))

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    return 0 if result.wasSuccessful() else 1


def main():
    if '--inside' in sys.argv:
        return inside()
    from render_qwen_handover import ROOT, UTILITY
    # No host secrets, Docker socket, GPU access or published ports in this fixture.
    command = ['docker', 'run', '--rm', '--read-only', '--cap-drop', 'ALL',
               '--security-opt', 'no-new-privileges:true',
               '--tmpfs', '/tmp:rw,exec,nosuid,nodev,size=256m',
               '-v', str(ROOT / 'scripts') + ':/qualification:ro',
               UTILITY, 'python3', '/qualification/validate_qwen_qualification.py', '--inside']
    return subprocess.run(command, timeout=300).returncode


if __name__ == '__main__':
    sys.exit(main())
