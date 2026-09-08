"""Small synthetic qualification client; not a general-purpose protocol library.

Implements RFC9180 base mode, RFC9458 encapsulation and RFC9292 binary HTTP.
Only X25519/HKDF-SHA256/AES128GCM is accepted. Chunked mode matches ohttp0.7.2.
No networking, logging or persistent keys. Invalid/truncated input fails closed.

Public interop references (no private test source is vendored):
https://github.com/nearai/nearai-cloud-verifier/blob/94554726fd548676842b7ea603a8173a60c31341/py/ohttp_client.py
https://github.com/martinthomson/ohttp/blob/af304a6d34fd93facbf7094bcd5c132c94b27e03/ohttp/src/stream.rs
"""
import hashlib
import hmac
import struct
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDFExpand


def expand(secret, label, length):
    return HKDFExpand(hashes.SHA256(), length, label).derive(secret)


def extract(salt, material):
    return hmac.new(salt or bytes(32), material, hashlib.sha256).digest()


def lextract(suite, salt, label, material):
    return extract(salt, b'HPKE-v1' + suite + label + material)


def lexpand(suite, secret, label, info, length):
    return expand(secret, struct.pack('!H', length) + b'HPKE-v1' + suite + label + info, length)


def vint(number):
    assert 0 <= number < 2**62
    for size, limit, marker in [(1, 64, 0), (2, 16384, 1), (4, 2**30, 2), (8, 2**62, 3)]:
        if number < limit:
            return (number | marker << (8 * size - 2)).to_bytes(size, 'big')


class Reader:
    def __init__(self, data):
        self.data, self.offset = data, 0

    def take(self, length):
        assert length >= 0 and self.offset + length <= len(self.data)
        part = self.data[self.offset:self.offset + length]
        self.offset += length
        return part

    def integer(self):
        first = self.take(1)[0]
        size = 1 << (first >> 6)
        return int.from_bytes(bytes([first & 63]) + self.take(size - 1), 'big')

    def field(self):
        return self.take(self.integer())


def field(raw):
    if isinstance(raw, str): raw = raw.encode()
    return vint(len(raw)) + raw


def binary_request(authority, token, body):
    headers = field('authorization') + field('Bearer ' + token)
    headers += field('content-type') + field('application/json')
    return b'\x00' + field('POST') + field('https') + field(authority) + field('/v1/chat/completions') + field(headers) + field(body) + b'\x00'


def binary_response(raw):
    reader = Reader(raw)
    framing = reader.integer()
    assert framing in (1, 3)
    status = reader.integer()
    assert status == 200
    if framing == 1:
        reader.field()  # bounded, known-length header section
        return reader.field()
    while reader.field():
        reader.field()
    chunks = []
    while part := reader.field():
        chunks.append(part)
    return b''.join(chunks)


class Exchange:
    def __init__(self, config, chunked=False):
        assert len(config) >= 41 and config[1:3] == b'\x00\x20'
        suites_len = int.from_bytes(config[35:37], 'big')
        assert suites_len % 4 == 0 and len(config) == 37 + suites_len
        assert b'\x00\x01\x00\x01' in [config[i:i+4] for i in range(37, len(config), 4)]
        self.chunked = chunked
        self.header = config[:3] + b'\x00\x01\x00\x01'
        self.suite = b'HPKE' + self.header[1:]
        kem = b'KEM\x00\x20'
        ephemeral = X25519PrivateKey.generate()
        self.enc = ephemeral.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        shared = ephemeral.exchange(X25519PublicKey.from_public_bytes(config[3:35]))
        eae = lextract(kem, b'', b'eae_prk', shared)
        secret = lexpand(kem, eae, b'shared_secret', self.enc + config[3:35], 32)
        self.label = b'message/bhttp' + (b' chunked' if chunked else b'')
        info = self.label + b' request\0' + self.header
        context = b'\0' + lextract(self.suite, b'', b'psk_id_hash', b'')
        context += lextract(self.suite, b'', b'info_hash', info)
        schedule = lextract(self.suite, secret, b'secret', b'')
        self.key = lexpand(self.suite, schedule, b'key', context, 16)
        self.nonce = lexpand(self.suite, schedule, b'base_nonce', context, 12)
        self.exporter = lexpand(self.suite, schedule, b'exp', context, 32)

    @staticmethod
    def counter_nonce(base, sequence):
        return (int.from_bytes(base, 'big') ^ sequence).to_bytes(len(base), 'big')

    def request(self, body):
        cipher = AESGCM(self.key)
        wire = self.header + self.enc
        if not self.chunked:
            return wire + cipher.encrypt(self.nonce, body, b'')
        sequence = 0
        for offset in range(0, len(body), 16384):
            sealed = cipher.encrypt(self.counter_nonce(self.nonce, sequence), body[offset:offset+16384], b'')
            wire += field(sealed)
            sequence += 1
        return wire + b'\0' + cipher.encrypt(self.counter_nonce(self.nonce, sequence), b'', b'final')

    def response(self, wire):
        reader = Reader(wire)
        response_nonce = reader.take(16)
        exported = lexpand(self.suite, self.exporter, b'sec', self.label + b' response', 16)
        secret = extract(self.enc + response_nonce, exported)
        cipher = AESGCM(expand(secret, b'key', 16))
        base = expand(secret, b'nonce', 12)
        if not self.chunked:
            return cipher.decrypt(base, reader.take(len(wire) - reader.offset), b'')
        result, sequence = [], 0
        while True:
            length = reader.integer()
            nonce = self.counter_nonce(base, sequence)
            if length == 0:
                # An authenticated final marker is mandatory; EOF is not success.
                assert cipher.decrypt(nonce, reader.take(16), b'final') == b''
                assert reader.offset == len(wire)
                return b''.join(result)
            result.append(cipher.decrypt(nonce, reader.take(length), b''))
            sequence += 1
