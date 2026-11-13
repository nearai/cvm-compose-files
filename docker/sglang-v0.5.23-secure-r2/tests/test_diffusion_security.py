"""CPU checks for the sglang-diffusion (multimodal_gen) patches 50-53.

Run inside the built image with pytest. No GPU and no network are needed: DNS
answers and HTTP servers are stubbed. Every test fails on stock v0.5.23.
"""

import asyncio
import ipaddress
import socket
from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from sglang.multimodal_gen.runtime.entrypoints import http_server
from sglang.multimodal_gen.runtime.entrypoints.openai import utils
from sglang.multimodal_gen.runtime.managers.memory_managers import layerwise_offload
from sglang.multimodal_gen.runtime.utils import url_guard, vision
from sglang.srt.utils.auth import AuthLevel

PUBLIC_IP = "93.184.215.14"
FAKE_DNS = {
    "public.example.com": PUBLIC_IP,
    "rebind.example.com": "10.1.2.3",
    "metadata.example.com": "169.254.169.254",
    "v6loop.example.com": "::1",
}


@pytest.fixture(autouse=True)
def fake_dns(monkeypatch):
    """Resolve the example.com names above; other names under it do not resolve."""
    real = socket.getaddrinfo
    monkeypatch.delenv(url_guard.ALLOW_PRIVATE_ENV, raising=False)

    def getaddrinfo(host, port, *args, **kwargs):
        if host in FAKE_DNS:
            ip = FAKE_DNS[host]
            family = socket.AF_INET6 if ":" in ip else socket.AF_INET
            return [(family, socket.SOCK_STREAM, 6, "", (ip, port))]
        if str(host).endswith(".example.com"):
            raise socket.gaierror("no such host")
        return real(host, port, *args, **kwargs)

    monkeypatch.setattr(url_guard.socket, "getaddrinfo", getaddrinfo)


# --- 50: media URL SSRF guard -------------------------------------------------


@pytest.mark.parametrize(
    "ip",
    [
        "127.0.0.1",
        "10.0.0.1",
        "172.17.0.2",
        "192.168.1.1",
        "169.254.169.254",
        "100.64.0.1",
        "0.0.0.0",
        "224.0.0.1",
        "255.255.255.255",
        "::1",
        "::",
        "fe80::1",
        "fc00::1",
        "::ffff:127.0.0.1",
        "::127.0.0.1",
        "64:ff9b::a9fe:a9fe",
        "2002:7f00:1::",
        "ff02::1",
    ],
)
def test_non_public_addresses_are_rejected(ip):
    assert not url_guard.is_public_address(ipaddress.ip_address(ip))


@pytest.mark.parametrize("ip", ["8.8.8.8", PUBLIC_IP, "2606:4700:4700::1111"])
def test_public_addresses_are_accepted(ip):
    assert url_guard.is_public_address(ipaddress.ip_address(ip))


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://public.example.com/a.png",
        "gopher://public.example.com/",
        "http:///nohost",
        "http://127.0.0.1/a.png",
        "http://127.1/a.png",
        "http://2130706433/a.png",
        "http://0x7f000001/a.png",
        "http://localhost:30000/get_server_info",
        "http://[::1]/a.png",
        "http://[::ffff:7f00:1]/a.png",
        "http://169.254.169.254/latest/meta-data/",
        "http://metadata.example.com/x",
        "http://rebind.example.com/x",
        "https://v6loop.example.com/x",
        "http://user:pw@127.0.0.1/x",
        "http://nxdomain.example.com/x",
    ],
)
def test_pinned_request_rejects(url):
    with pytest.raises(url_guard.BlockedURLError):
        url_guard.pinned_request(url)


def test_pinned_request_pins_ip_and_keeps_host_and_sni():
    url, headers, ext = url_guard.pinned_request("https://user:pw@public.example.com:8443/a.png?x=1#f")
    assert url == f"https://{PUBLIC_IP}:8443/a.png?x=1"
    assert headers == {"Host": "public.example.com:8443"}
    assert ext == {"sni_hostname": b"public.example.com"}
    url, headers, ext = url_guard.pinned_request(f"http://{PUBLIC_IP}/a.png")
    assert url == f"http://{PUBLIC_IP}/a.png" and ext == {}


def test_allow_private_env_skips_address_check(monkeypatch):
    monkeypatch.setenv(url_guard.ALLOW_PRIVATE_ENV, "1")
    assert url_guard.pinned_request("http://127.0.0.1:8000/a.png")[0] == "http://127.0.0.1:8000/a.png"
    with pytest.raises(url_guard.BlockedURLError):
        url_guard.pinned_request("file:///etc/passwd")


def _mock_async_client(monkeypatch, handler):
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        utils.httpx,
        "AsyncClient",
        lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw),
    )


def test_image_download_of_private_url_never_connects(monkeypatch, tmp_path):
    seen = []
    _mock_async_client(monkeypatch, lambda r: seen.append(r) or httpx.Response(200, content=b"x"))
    for url in ("http://127.0.0.1:30000/flush_cache", "http://rebind.example.com/a.png"):
        with pytest.raises(Exception, match="non-public"):
            asyncio.run(utils._save_url_image_to_path(url, str(tmp_path / "in" / "a.png")))
    assert seen == []


def test_image_download_redirect_to_private_is_blocked(monkeypatch, tmp_path):
    seen = []

    def handler(request):
        seen.append((str(request.url), request.headers["host"]))
        return httpx.Response(302, headers={"location": "http://169.254.169.254/latest/"})

    _mock_async_client(monkeypatch, handler)
    with pytest.raises(Exception, match="non-public"):
        asyncio.run(
            utils._save_url_image_to_path("http://public.example.com/a.png", str(tmp_path / "in" / "a.png"))
        )
    assert seen == [(f"http://{PUBLIC_IP}/a.png", "public.example.com")]


def test_image_download_follows_public_redirect(monkeypatch, tmp_path):
    def handler(request):
        if request.url.path == "/a.png":
            return httpx.Response(301, headers={"location": "/b.png"})
        assert request.headers["host"] == "public.example.com"
        return httpx.Response(200, content=b"img", headers={"content-type": "image/png"})

    _mock_async_client(monkeypatch, handler)
    saved = asyncio.run(
        utils._save_url_image_to_path("http://public.example.com/a.png", str(tmp_path / "in" / "a"))
    )
    assert open(saved, "rb").read() == b"img"


def test_redirect_loop_is_bounded(monkeypatch, tmp_path):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(302, headers={"location": "/again"})

    _mock_async_client(monkeypatch, handler)
    with pytest.raises(Exception, match="redirects"):
        asyncio.run(
            utils._save_url_image_to_path("http://public.example.com/a.png", str(tmp_path / "in" / "a"))
        )
    assert len(calls) == url_guard.MAX_REDIRECTS + 1


def test_prefer_remote_source_no_longer_passes_urls_through(tmp_path):
    with pytest.raises(Exception, match="non-public"):
        asyncio.run(
            utils.save_image_to_path(
                "http://127.0.0.1/a.png",
                str(tmp_path / "up" / "in"),
                uploads_root=str(tmp_path / "up"),
                prefer_remote_source=True,
            )
        )


def test_pipeline_image_and_video_loaders_use_the_guard():
    with pytest.raises(url_guard.BlockedURLError):
        vision.load_image("http://169.254.169.254/latest/meta-data/x.png")
    with pytest.raises(url_guard.BlockedURLError):
        vision.load_video("http://127.0.0.1:30000/v.mp4")


# --- 51: operator-only request fields ------------------------------------------


@pytest.mark.parametrize(
    "field,value",
    [
        ("perf_dump_path", "/root/.bashrc"),
        ("profile", True),
        ("profile_all_stages", True),
        ("upscaling_model_path", "someone/evil-repo:model.pth"),
        ("frame_interpolation_model_path", "someone/evil-repo"),
    ],
)
def test_operator_only_fields_are_rejected(monkeypatch, field, value):
    monkeypatch.delenv("SGLANG_DIFFUSION_ALLOW_REQUEST_DEBUG_FIELDS", raising=False)
    monkeypatch.setattr(utils, "get_global_server_args", lambda: SimpleNamespace())
    with pytest.raises(HTTPException) as exc:
        utils.build_sampling_params("rid", prompt="p", **{field: value})
    assert exc.value.status_code == 400 and field in exc.value.detail


def test_default_values_pass_and_profile_only_fields_are_dropped(monkeypatch):
    monkeypatch.delenv("SGLANG_DIFFUSION_ALLOW_REQUEST_DEBUG_FIELDS", raising=False)
    kwargs = dict(perf_dump_path=None, profile=False, profile_all_stages=False,
                  upscaling_model_path=None, num_profiled_timesteps=3)
    utils._reject_operator_only_fields(kwargs)
    assert "num_profiled_timesteps" not in kwargs


def test_debug_fields_env_opt_in(monkeypatch):
    monkeypatch.setenv("SGLANG_DIFFUSION_ALLOW_REQUEST_DEBUG_FIELDS", "1")
    kwargs = dict(perf_dump_path="/tmp/p.json", profile=True, num_profiled_timesteps=3)
    utils._reject_operator_only_fields(kwargs)
    assert kwargs["num_profiled_timesteps"] == 3


# --- 52: API key / admin key on the diffusion HTTP server ----------------------

ADMIN_PATHS = {
    "/v1/set_lora",
    "/v1/merge_lora_weights",
    "/v1/unmerge_lora_weights",
    "/update_weights_from_disk",
    "/update_weights_from_tensor",
    "/update_weights_from_tensor_checker",
    "/get_weights_checksum",
    "/release_memory_occupation",
    "/resume_memory_occupation",
    "/rollout/generate",
}


def _app(monkeypatch, api_key=None, admin_key=None):
    for env, value in ((http_server.API_KEY_ENV, api_key), (http_server.ADMIN_API_KEY_ENV, admin_key)):
        if value is None:
            monkeypatch.delenv(env, raising=False)
        else:
            monkeypatch.setenv(env, value)
    pipeline_config = SimpleNamespace(
        supports_action_endpoint=lambda: False, supports_openpi_endpoint=lambda: False
    )
    server_args = SimpleNamespace(enable_metrics=False, pipeline_config=pipeline_config)
    return http_server.create_app(server_args)


def test_admin_routes_are_marked_admin_optional():
    from sglang.multimodal_gen.runtime.entrypoints.openai import common_api
    from sglang.multimodal_gen.runtime.entrypoints.post_training import rollout_api, weights_api

    marked = http_server._admin_paths(common_api.router, weights_api.router, rollout_api.router)
    assert marked == ADMIN_PATHS
    for router in (common_api.router, weights_api.router, rollout_api.router):
        for route in router.routes:
            if route.path in ADMIN_PATHS:
                assert route.endpoint._auth_level == AuthLevel.ADMIN_OPTIONAL


def _status(client, path, key=None):
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    return client.post(path, json={}, headers=headers).status_code


def test_admin_key_gates_admin_routes_only(monkeypatch):
    client = TestClient(_app(monkeypatch, admin_key="adm"), raise_server_exceptions=False)
    for path in sorted(ADMIN_PATHS):
        assert _status(client, path) == 401, path
        assert _status(client, path, "wrong") == 401, path
        assert _status(client, path, "adm") != 401, path
    # spelling variants reach the same handler only through the gate
    for variant in ("/v1/set_lora/", "/v1/set%5Flora", "/update_weights_from_disk/"):
        assert _status(client, variant) in (401, 404, 405), variant
    assert client.get("/v1/models").status_code != 401


def test_api_key_gates_everything_but_health(monkeypatch):
    client = TestClient(_app(monkeypatch, api_key="k"), raise_server_exceptions=False)
    assert _status(client, "/v1/set_lora") == 401
    assert _status(client, "/v1/set_lora", "k") != 401
    assert _status(client, "/v1/images/generations") == 401
    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/models", headers={"Authorization": "Bearer k"}).status_code != 401


def test_both_keys_admin_routes_need_admin_key(monkeypatch):
    client = TestClient(_app(monkeypatch, api_key="k", admin_key="adm"), raise_server_exceptions=False)
    assert _status(client, "/update_weights_from_disk", "k") == 401
    assert _status(client, "/update_weights_from_disk", "adm") != 401


def test_no_keys_keeps_stock_behaviour(monkeypatch):
    client = TestClient(_app(monkeypatch), raise_server_exceptions=False)
    assert _status(client, "/v1/set_lora") != 401


# --- 53: cudaHostRegister switch (TDX: cudaErrorNotSupported) -------------------


class _CudartStub:
    def __init__(self, rc):
        self.rc, self.calls = rc, 0

    def cudaHostRegister(self, ptr, nbytes, flags):
        self.calls += 1
        return self.rc

    def cudaHostUnregister(self, ptr):
        return 0


def _cuda_stub(monkeypatch, rc):
    stub = _CudartStub(rc)
    monkeypatch.setattr(layerwise_offload.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(layerwise_offload.torch.cuda, "cudart", lambda: stub)
    return stub


def test_disable_env_skips_cuda_host_register(monkeypatch):
    stub = _cuda_stub(monkeypatch, 0)
    monkeypatch.setenv(layerwise_offload.DISABLE_HOST_REGISTER_ENV, "1")
    assert layerwise_offload._pin_in_place(layerwise_offload.torch.empty(16)) is False
    assert stub.calls == 0


def test_failed_register_clears_the_cuda_runtime_error(monkeypatch):
    stub = _cuda_stub(monkeypatch, 801)  # cudaErrorNotSupported
    monkeypatch.delenv(layerwise_offload.DISABLE_HOST_REGISTER_ENV, raising=False)
    cleared = []
    monkeypatch.setattr(layerwise_offload, "_clear_cuda_runtime_error", lambda: cleared.append(1))
    assert layerwise_offload._pin_in_place(layerwise_offload.torch.empty(16)) is False
    assert stub.calls == 1 and cleared == [1]


def test_clear_cuda_runtime_error_is_safe_without_a_gpu():
    layerwise_offload._clear_cuda_runtime_error()
