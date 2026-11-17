# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for the vLLM v0.33.0 request bounds in patches/

Each test exercises one patch through the installed `vllm` package, needs no GPU, model or
network, and fails on the stock v0.33.0 image.
"""

import asyncio
import io
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.exceptions import VLLMValidationError


# 01 error-sanitizer-linear-time (#60233, tests from #52120)
def test_01_long_model_name_sanitizes_in_linear_time():
    from vllm.entrypoints.serve.exception_handling.utils import sanitize_message

    msg = "The model `" + "/a" * 10_000 + "` does not exist. Valid aliases: x.y"
    start = time.perf_counter()
    assert sanitize_message(msg) == msg
    assert time.perf_counter() - start < 1.0
    assert "<path>" in sanitize_message("Error in /app/server.py")
    assert "<path>" in sanitize_message("at /workspace/vllm/engine.py")


# 02 remote-prefill-cache-accounting-trust (vllm#64918)
def test_02_client_cannot_set_cached_tokens(monkeypatch):
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

    monkeypatch.delenv("VLLM_TRUST_REMOTE_PREFILL_CACHED_TOKENS", raising=False)
    req = ChatCompletionRequest(
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        kv_transfer_params={"do_remote_prefill": True, "remote_prefill_cached_tokens": 999999},
    )
    with pytest.raises(VLLMValidationError, match="trusted P/D backend"):
        req.to_sampling_params(max_tokens=1, default_sampling_params={})


def test_02_output_processor_ignores_untrusted_cached_tokens(monkeypatch):
    from vllm import SamplingParams
    from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest
    from vllm.v1.engine.output_processor import OutputProcessor
    from vllm.v1.metrics.stats import PrefillStats

    prompt = [1, 2, 3, 4, 5, 6, 7, 8]
    monkeypatch.setenv("VLLM_TRUST_REMOTE_PREFILL_CACHED_TOKENS", "1")
    params = SamplingParams(
        detokenize=False,
        extra_args={
            "kv_transfer_params": {"do_remote_prefill": True, "remote_prefill_cached_tokens": 999999}
        },
    )
    # The request was built while trusted; the output side must still ignore it.
    monkeypatch.setenv("VLLM_TRUST_REMOTE_PREFILL_CACHED_TOKENS", "0")
    op = OutputProcessor(tokenizer=None, log_stats=False)
    op.add_request(
        EngineCoreRequest(
            request_id="r-int",
            external_req_id="r",
            prompt_token_ids=prompt,
            mm_features=None,
            arrival_time=0,
            lora_request=None,
            cache_salt=None,
            data_parallel_rank=None,
            sampling_params=params,
            pooling_params=None,
        ),
        prompt=None,
    )
    stats = PrefillStats()
    stats.set(
        num_prompt_tokens=len(prompt), num_local_cached_tokens=0, num_external_cached_tokens=3
    )
    out = op.process_outputs(
        [EngineCoreOutput(request_id="r-int", new_token_ids=[42], prefill_stats=stats)]
    )
    assert out.request_outputs[0].num_cached_tokens == 3


# 03 stop-and-bad-words-bounds (vllm#54721)
def test_03_stop_token_ids_and_stop_strings_are_capped():
    from vllm import SamplingParams
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

    with pytest.raises(Exception, match="stop_token_ids"):
        SamplingParams(stop_token_ids=list(range(10_000)))
    with pytest.raises(Exception):
        ChatCompletionRequest(
            model="m", messages=[{"role": "user", "content": "x"}], stop_token_ids=list(range(10_000))
        )
    assert SamplingParams(stop=["a", "b", "a"]).stop == ["a", "b"]


def test_03_generation_config_eos_ids_reach_the_stop_set():
    from vllm import SamplingParams

    sp = SamplingParams(stop_token_ids=[5])
    sp.update_from_generation_config({"eos_token_id": [7, 9]}, eos_token_id=7)
    assert sp.stop_token_ids_set == frozenset(sp.stop_token_ids) == {5, 9}


# 04 detok-bounded-holdback (vllm#63668)
def _byte_level_tokenizer():
    """A byte-level BPE tokenizer built in-process, with one token per byte."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import TokenizersBackend

    vocab = {ch: i for i, ch in enumerate(_bytes_to_unicode())}
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    backend = TokenizersBackend(tokenizer_object=tok)
    byte_id = {b: vocab[ch] for b, ch in enumerate(_bytes_to_unicode())}
    return backend, byte_id


def _bytes_to_unicode() -> list[str]:
    bs = [*range(ord("!"), ord("~") + 1), *range(0xA1, 0xAD), *range(0xAE, 0x100)]
    cs, extra = {b: b for b in bs}, 0
    for b in range(256):
        if b not in cs:
            cs[b], extra = 256 + extra, extra + 1
    return [chr(cs[b]) for b in range(256)]


def _stream(tokenizer, fast: bool, prompt: list[int], ids: list[int]) -> tuple[str, int]:
    from vllm import SamplingParams
    from vllm.v1.engine import EngineCoreRequest
    from vllm.v1.engine.detokenizer import (
        FastIncrementalDetokenizer,
        SlowIncrementalDetokenizer,
    )

    request = EngineCoreRequest(
        request_id="r",
        external_req_id="r",
        prompt_token_ids=prompt,
        mm_features=None,
        sampling_params=SamplingParams(skip_special_tokens=True),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )
    cls = FastIncrementalDetokenizer if fast else SlowIncrementalDetokenizer
    detok = cls(tokenizer, request)
    text, most_held = "", 0
    for i, token_id in enumerate(ids):
        detok.update([token_id], False)
        text += detok.get_next_output_text(False, delta=True)
        most_held = max(most_held, i + 1 - len(text))
    return text + detok.get_next_output_text(True, delta=True), most_held


@pytest.mark.parametrize("fast", [True, False])
def test_04_lead_byte_run_streams_with_bounded_holdback(fast):
    from vllm.tokenizers.detokenizer_utils import MAX_PENDING_TOKENS

    tokenizer, byte_id = _byte_level_tokenizer()
    ids = [byte_id[0xE4]] * 1000 + tokenizer.encode(" world", add_special_tokens=False)
    text, most_held = _stream(tokenizer, fast, [byte_id[0x48]], ids)
    assert most_held <= MAX_PENDING_TOKENS
    assert text == "\ufffd" * 1000 + " world"


@pytest.mark.parametrize("fast", [True, False])
@pytest.mark.parametrize("char", ["\u00e9", "\u4e2d", "\U0001f600"])
def test_04_character_across_flush_boundary_matches_one_shot_decode(fast, char):
    from vllm.tokenizers.detokenizer_utils import MAX_PENDING_TOKENS

    tokenizer, byte_id = _byte_level_tokenizer()
    tail = tokenizer.encode(" world", add_special_tokens=False)
    for run in range(MAX_PENDING_TOKENS - 5, MAX_PENDING_TOKENS + 2):
        ids = [byte_id[0xE4]] * run + [byte_id[b] for b in char.encode()] + tail
        text, _ = _stream(tokenizer, fast, [byte_id[0x48]], ids)
        assert text == tokenizer.decode(ids), f"{run=}"
        assert text.endswith(char + " world")


# 05 pooling-prompt-count-bound (vllm#53473)
def test_05_embedding_and_rerank_list_length_capped():
    from vllm.entrypoints.pooling.embed.protocol import EmbeddingCompletionRequest
    from vllm.entrypoints.pooling.scoring.protocol import RerankRequest

    with pytest.raises(Exception, match="exceeds the maximum"):
        EmbeddingCompletionRequest(model="m", input=["x"] * 5000)
    with pytest.raises(Exception, match="exceeds the maximum"):
        RerankRequest(model="m", query="q", documents=["d"] * 5000)
    EmbeddingCompletionRequest(model="m", input=["x"] * 2048)


# 06 vision-rope-cache-byte-bound (vllm#60607)
def test_06_qwen3_vl_rope_cache_is_byte_bounded():
    import inspect

    import vllm.model_executor.models.qwen3_vl as q

    assert "LRUCache(capacity=1024)" not in inspect.getsource(q).replace(" ", "")


# 07 media-reference-script-reject (vllm#63701)
@pytest.mark.parametrize(
    "payload",
    [
        b"ffconcat version 1.0\nfile '/etc/hostname'\n",
        b"#EXTM3U\n#EXTINF:1,\nhttp://169.254.169.254/x.ts\n",
        b"v=0\no=- 0 0 IN IP4 127.0.0.1\nc=IN IP4 127.0.0.1\n",
    ],
)
def test_07_audio_reference_scripts_rejected(payload):
    from vllm.multimodal.media.audio import load_audio

    with pytest.raises(ValueError, match="playlists and stream descriptions"):
        load_audio(io.BytesIO(payload))


# 08 engine-add-rollback (vllm#56212)
def test_08_sampling_ints_beyond_msgpack_rejected():
    from vllm import SamplingParams

    with pytest.raises(VLLMValidationError, match="must be at most"):
        SamplingParams(top_k=2**64)


# 09 audio-channel-cap (vllm#51894)
def _wav(channels: int) -> bytes:
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * channels * 1600)
    return buf.getvalue()


def _audio_backends() -> list[str]:
    import importlib.util

    # "auto" tries torchcodec, then soundfile, then PyAV; the last two run when installed.
    names = ["auto", "torchcodec"]
    names += [m for m, mod in (("soundfile", "soundfile"), ("pyav", "av")) if importlib.util.find_spec(mod)]
    return names


@pytest.mark.parametrize("backend", _audio_backends())
def test_09_audio_channel_count_capped(backend):
    from vllm.multimodal.media.audio import load_audio

    try:
        wav, _ = load_audio(io.BytesIO(_wav(2)), backend=backend)
    except ImportError as e:
        pytest.skip(f"{backend} unavailable: {e}")
    assert wav.ndim == 1
    with pytest.raises(ValueError, match="channel count 64"):
        load_audio(io.BytesIO(_wav(64)), backend=backend)


# 10 mm-uuid-length-cap (vllm#57221)
def test_10_multimodal_uuid_length_capped():
    from vllm.renderers.chat_utils.chat_message import parse_chat_message_content_part

    part = {"type": "image_url", "image_url": {"url": "data:,"}, "uuid": "u" * 4096}
    with pytest.raises(VLLMValidationError, match="uuid"):
        parse_chat_message_content_part(
            part, Mock(), mm_tracker=Mock(), wrap_dicts=False, interleave_strings=False
        )


# 11 media-fetch-public-ips-only (vllm#57825)
def _v4(*octets: int) -> str:
    return ".".join(map(str, octets))


_FAKE_DNS = {
    "private.example.com": [_v4(10, 1, 2, 3)],
    "mixed.example.com": ["1.1.1.1", _v4(10, 0, 0, 7)],
    "mapped.example.com": ["::ffff:169.254.169.254"],
}


@pytest.fixture
def media_net(monkeypatch):
    """Fake DNS for _FAKE_DNS names, and a record of every socket connect.

    A connect to anything but 127.0.0.1 fails the test before a packet is sent.
    """
    import socket

    real_getaddrinfo, real_connect = socket.getaddrinfo, socket.socket.connect
    dialed: list[str] = []

    def getaddrinfo(host, port, *args, **kwargs):
        if host in _FAKE_DNS:
            return [
                (socket.AF_INET6 if ":" in ip else socket.AF_INET, socket.SOCK_STREAM, 6, "",
                 (ip, port or 0))
                for ip in _FAKE_DNS[host]
            ]
        return real_getaddrinfo(host, port, *args, **kwargs)

    def connect(sock, address):
        dialed.append(address[0])
        assert address[0] == "127.0.0.1", "a socket was opened to a refused address"
        return real_connect(sock, address)

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.delenv("VLLM_MEDIA_URL_ALLOW_PRIVATE_IPS", raising=False)
    monkeypatch.setenv("VLLM_MEDIA_FETCH_MAX_RETRIES", "1")
    return dialed


def _fetch_both(url: str, timeout: float = 2) -> list[Exception | bytes]:
    from vllm.connections import HTTPConnection

    results: list[Exception | bytes] = []
    try:
        results.append(HTTPConnection().get_bytes(url, timeout=timeout))
    except Exception as e:
        results.append(e)

    async def fetch():
        conn = HTTPConnection()
        try:
            return await conn.async_get_bytes(url, timeout=timeout)
        except Exception as e:
            return e
        finally:
            if conn._async_client is not None:
                await conn._async_client.close()

    results.append(asyncio.run(fetch()))
    return results


@pytest.mark.parametrize(
    "host",
    [
        "127.0.0.1", _v4(10, 0, 0, 1), _v4(172, 16, 0, 1), _v4(192, 168, 0, 1),
        "169.254.169.254", "0.0.0.0", _v4(224, 0, 0, 1), "[::1]", "[::ffff:127.0.0.1]",
        "[::ffff:169.254.169.254]", "[fe80::1]", "[64:ff9b::a9fe:a9fe]", "2130706433",
        "0x7f000001", "0177.0.0.1", "127.1",
        "private.example.com", "mixed.example.com", "mapped.example.com",
    ],
)
def test_11_media_fetch_refuses_non_public_destinations(media_net, host):
    for result in _fetch_both(f"http://{host}:9/x.png"):
        assert isinstance(result, ValueError), repr(result)
        assert "non-public IP" in str(result)
    assert media_net == []


class _Recorder:
    """Loopback HTTP server that counts requests and answers 200 or a redirect."""

    def __init__(self, location: str | None = None) -> None:
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        recorder = self
        self.hits = 0

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                recorder.hits += 1
                self.send_response(302 if location else 200)
                if location:
                    self.send_header("Location", location)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"ok")

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def _accept_loopback(monkeypatch):
    import vllm.connections as connections

    real = connections._is_public_ip
    monkeypatch.setattr(connections, "_is_public_ip", lambda ip: ip == "127.0.0.1" or real(ip))


@pytest.mark.parametrize(
    "target", ["http://169.254.169.254/x.png", "http://private.example.com/x.png"]
)
def test_11_redirect_to_refused_address_is_not_followed(media_net, monkeypatch, target):
    _accept_loopback(monkeypatch)
    server = _Recorder(location=target)
    try:
        for result in _fetch_both(server.url + "/start"):
            assert isinstance(result, ValueError), repr(result)
            assert "non-public IP" in str(result)
    finally:
        server.close()
    assert server.hits == 2
    assert set(media_net) == {"127.0.0.1"}


def test_11_proxy_environment_is_ignored(media_net, monkeypatch):
    _accept_loopback(monkeypatch)
    proxy, target = _Recorder(), _Recorder()
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy"):
        monkeypatch.setenv(name, proxy.url)
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    try:
        assert _fetch_both(target.url + "/x.png") == [b"ok", b"ok"]
    finally:
        proxy.close()
        target.close()
    assert (proxy.hits, target.hits) == (0, 2)


def test_11_async_fetch_resolves_off_the_event_loop(monkeypatch):
    import socket

    from vllm.connections import HTTPConnection

    def slow_getaddrinfo(*args, **kwargs):
        time.sleep(1.0)
        raise socket.gaierror("no such host")

    monkeypatch.setattr(socket, "getaddrinfo", slow_getaddrinfo)

    async def main() -> float:
        lag = 0.0

        async def ticker():
            nonlocal lag
            for _ in range(20):
                start = time.perf_counter()
                await asyncio.sleep(0.05)
                lag = max(lag, time.perf_counter() - start - 0.05)

        tick = asyncio.create_task(ticker())
        await asyncio.sleep(0.01)  # the ticker is now mid-sleep
        conn = HTTPConnection()
        try:
            with pytest.raises(ValueError, match="Cannot resolve"):
                await conn.async_get_bytes("http://slow.example.com/x.png", timeout=5)
        finally:
            if conn._async_client is not None:
                await conn._async_client.close()
        await tick
        return lag

    assert asyncio.run(main()) < 0.5


# 12 media-client-no-cookies (vllm#56588)
def test_12_media_clients_keep_no_cookies():
    import aiohttp

    from vllm.connections import HTTPConnection

    conn = HTTPConnection()
    jar = conn.get_sync_client().cookies
    assert jar._policy.allowed_domains() == ()

    async def get_jar():
        client = await conn.get_async_client()
        try:
            return type(client.cookie_jar)
        finally:
            await client.close()

    assert asyncio.run(get_jar()) is aiohttp.DummyCookieJar


# 13 whisper-langdetect-fallback
@pytest.mark.parametrize("token,expected", [("<|yue|>", "en"), ("<|de|>", "de"), ("junk", "en")])
def test_13_whisper_unlisted_language_falls_back(token, expected):
    from vllm.model_executor.models.whisper import WhisperForConditionalGeneration

    tok = SimpleNamespace(decode=lambda ids, skip_special_tokens=False: token)
    assert WhisperForConditionalGeneration.parse_language_detection_output([1], tok) == expected


# 14 media-retry-log-fields
def test_14_retry_warning_carries_attempt_count_and_error_type(monkeypatch):
    import vllm.connections as connections

    lines = []
    monkeypatch.setattr(connections.logger, "warning", lambda msg, *args: lines.append(msg % args))
    connections._log_retry(
        (None, "http://example.com/x"), {}, 1, 3, 2.0, ConnectionError("refused"), 2.0, 1.0
    )
    assert lines == ["HTTP fetch failed (attempt 2/3): ConnectionError; retrying"]
