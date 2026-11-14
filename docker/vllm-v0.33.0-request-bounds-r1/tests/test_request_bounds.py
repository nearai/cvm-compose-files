# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for the vLLM v0.33.0 request bounds in patches/

Each test exercises one patch through the installed `vllm` package, needs no GPU, model or
network, and fails on the stock v0.33.0 image. Run:
    python3 -m pytest -q -p no:cacheprovider tests/test_request_bounds.py
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
    # Admission was bypassed (built while trusted); the output side must still ignore it.
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
def test_04_detokenizer_bounds_pending_tokens():
    import inspect

    import vllm.tokenizers.detokenizer_utils as du

    assert "MAX_PENDING" in inspect.getsource(du).upper()


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

    # "auto" tries torchcodec, then soundfile, then PyAV. The base image ships only torchcodec;
    # the Whisper build adds soundfile and PyAV.
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
@pytest.mark.parametrize("url", ["http://127.0.0.1:9/x.png", "http://169.254.169.254/x.png"])
def test_11_media_fetch_refuses_private_destinations(monkeypatch, url):
    from vllm.connections import HTTPConnection

    monkeypatch.delenv("VLLM_MEDIA_URL_ALLOW_PRIVATE_IPS", raising=False)
    with pytest.raises(ValueError, match="non-public IP"):
        HTTPConnection().get_bytes(url, timeout=2)

    async def fetch():
        conn = HTTPConnection()
        try:
            await conn.async_get_bytes(url, timeout=2)
        finally:
            if conn._async_client is not None:
                await conn._async_client.close()

    with pytest.raises(ValueError, match="non-public IP"):
        asyncio.run(fetch())


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


# 13 whisper-langdetect-fallback (same fix as the Whisper inline build step)
@pytest.mark.parametrize("token,expected", [("<|yue|>", "en"), ("<|de|>", "de"), ("junk", "en")])
def test_13_whisper_unlisted_language_falls_back(token, expected):
    from vllm.model_executor.models.whisper import WhisperForConditionalGeneration

    tok = SimpleNamespace(decode=lambda ids, skip_special_tokens=False: token)
    assert WhisperForConditionalGeneration.parse_language_detection_output([1], tok) == expected
