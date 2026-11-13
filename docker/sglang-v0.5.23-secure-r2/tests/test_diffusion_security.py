"""CPU checks for the sglang-diffusion (multimodal_gen) patches diffusion-media-url-ssrf-guard and
diffusion-operator-only-request-fields.

Run inside the built image with pytest. No GPU and no network are needed: DNS
answers and HTTP servers are stubbed. Every test fails on stock v0.5.23.
"""

import asyncio
import ipaddress
import os
import socket
from io import BytesIO
from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException

from sglang.multimodal_gen.runtime.entrypoints.openai import utils
from sglang.multimodal_gen.runtime.utils import url_guard, vision

PUBLIC_IP = "93.184.215.14"
FAKE_DNS = {
    "public.example.com": PUBLIC_IP,
    "rebind.example.com": "10.1.2.3",
    "metadata.example.com": "169.254.169.254",
    "v6loop.example.com": "::1",
    "mixed.example.com": [PUBLIC_IP, "10.0.0.5"],
}


@pytest.fixture(autouse=True)
def fake_dns(monkeypatch):
    """Resolve the example.com names above; other names under it do not resolve."""
    real = socket.getaddrinfo
    monkeypatch.delenv(url_guard.ALLOW_PRIVATE_ENV, raising=False)

    def getaddrinfo(host, port, *args, **kwargs):
        if host in FAKE_DNS:
            ips = FAKE_DNS[host]
            return [
                (socket.AF_INET6 if ":" in ip else socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port))
                for ip in ([ips] if isinstance(ips, str) else ips)
            ]
        if str(host).endswith(".example.com"):
            raise socket.gaierror("no such host")
        return real(host, port, *args, **kwargs)

    monkeypatch.setattr(url_guard.socket, "getaddrinfo", getaddrinfo)


# --- diffusion-media-url-ssrf-guard ---------------------------------------------


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
        "http://mixed.example.com/x",
    ],
)
def test_pinned_request_rejects(url):
    with pytest.raises(url_guard.BlockedURLError):
        url_guard.pinned_request(url)


def test_pinned_request_pins_ip_and_keeps_host_and_sni():
    url, headers, ext = url_guard.pinned_request("https://user:pw@public.example.com:8443/a.png?x=1#f")
    assert url == f"https://{PUBLIC_IP}:8443/a.png?x=1"
    assert headers == {"Host": "public.example.com:8443"}
    # httpcore passes this to ssl as server_hostname, which must be a str.
    assert ext == {"sni_hostname": "public.example.com"}
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



def test_every_resolved_address_is_checked():
    # The public answer comes first; the private one behind it still blocks the URL.
    with pytest.raises(url_guard.BlockedURLError, match="non-public"):
        url_guard.resolve_public_addresses("mixed.example.com", 80)


def test_async_client_that_reads_proxy_env_is_refused(monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.example.com:3128")

    async def run():
        async with httpx.AsyncClient(follow_redirects=False) as client:
            async with url_guard.open_public_url_async(client, "http://public.example.com/a.png", timeout=1):
                pass

    with pytest.raises(url_guard.BlockedURLError, match="proxy"):
        asyncio.run(run())


def test_download_clients_ignore_proxy_env(monkeypatch, tmp_path):
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.example.com:3128")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example.com:3128")
    seen = []

    def handler(request):
        seen.append(request.url.host)
        return httpx.Response(200, content=b"img", headers={"content-type": "image/png"})

    real_async, real_sync = httpx.AsyncClient, httpx.Client
    kwargs = []

    def make(real):
        def build(**kw):
            kwargs.append(kw)
            return real(transport=httpx.MockTransport(handler), **kw)
        return build

    monkeypatch.setattr(utils.httpx, "AsyncClient", make(real_async))
    monkeypatch.setattr(url_guard.httpx, "Client", make(real_sync))
    asyncio.run(utils._save_url_image_to_path("http://public.example.com/a.png", str(tmp_path / "in" / "a")))
    url_guard.fetch_public_url_bytes("http://public.example.com/b.png", timeout=1, max_bytes=10)
    assert [kw.get("trust_env") for kw in kwargs] == [False, False]
    assert seen == [PUBLIC_IP, PUBLIC_IP]


def test_pipeline_loaders_refuse_local_paths_without_probing(monkeypatch, tmp_path):
    from sglang.srt.utils import common

    png = tmp_path / "secret.png"
    png.write_bytes(b"\x89PNG")
    monkeypatch.delenv(common.LOCAL_MEDIA_ENV, raising=False)

    def probe(*args, **kwargs):
        raise AssertionError("local path probed")

    monkeypatch.setattr(vision.os.path, "isfile", probe)
    for ref in (str(png), "file://" + str(png), "secret.png", str(png) + ".missing"):
        with pytest.raises(ValueError):
            vision.load_image(ref)
    for ref in (str(tmp_path / "v.mp4"), "v.mp4"):
        with pytest.raises(ValueError, match="local paths need"):
            vision.load_video(ref)


def test_pipeline_image_loader_reads_local_path_with_opt_in(monkeypatch, tmp_path):
    from PIL import Image

    from sglang.srt.utils import common

    png = tmp_path / "ok.png"
    Image.new("RGB", (2, 2)).save(png)
    monkeypatch.setenv(common.LOCAL_MEDIA_ENV, "1")
    assert vision.load_image(str(png)).size == (2, 2)


def _tls_server(tmp_path, cert_host):
    """HTTPS server on 127.0.0.1 with a self-signed cert for `cert_host`; returns (port, ssl context to trust it)."""
    import datetime
    import http.server
    import ssl
    import threading

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cert_host)])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1)).not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName(cert_host)]), critical=False)
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    cert_pem = tmp_path / f"{cert_host}.pem"
    key_pem = tmp_path / f"{cert_host}.key"
    cert_pem.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_pem.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                          serialization.NoEncryption()))

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = f"host={self.headers['Host']}".encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(cert_pem, key_pem)
    server.socket = ctx.wrap_socket(server.socket, server_side=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, ssl.create_default_context(cafile=str(cert_pem))


@pytest.mark.parametrize("cert_host,ok", [("public.example.com", True), ("other.example.com", False)])
def test_https_fetch_through_httpcore_verifies_the_url_host(monkeypatch, tmp_path, cert_host, ok):
    import socket as socket_mod

    import httpcore._backends.sync as sync_backend

    server, trust = _tls_server(tmp_path, cert_host)
    real_connect = socket_mod.create_connection
    dialed = []

    def connect(address, *args, **kwargs):
        dialed.append(address)
        # The pinned public IP stands in for the real host; route it to the local server.
        if address[0] == PUBLIC_IP:
            address = ("127.0.0.1", server.server_address[1])
        return real_connect(address, *args, **kwargs)

    real_client = httpx.Client
    monkeypatch.setattr(sync_backend.socket, "create_connection", connect)
    monkeypatch.setattr(url_guard.httpx, "Client", lambda **kw: real_client(verify=trust, **kw))
    try:
        if ok:
            body = url_guard.fetch_public_url_bytes("https://public.example.com/a.png", timeout=5, max_bytes=100)
            assert body == b"host=public.example.com"
        else:
            with pytest.raises(httpx.ConnectError, match="CERTIFICATE_VERIFY_FAILED|hostname"):
                url_guard.fetch_public_url_bytes("https://public.example.com/a.png", timeout=5, max_bytes=100)
    finally:
        server.shutdown()
    assert dialed and dialed[0][0] == PUBLIC_IP



@pytest.mark.parametrize("cert_host,ok", [("public.example.com", True), ("other.example.com", False)])
def test_async_https_image_download_through_httpcore(monkeypatch, tmp_path, cert_host, ok):
    import httpcore._backends.anyio as anyio_backend

    server, trust = _tls_server(tmp_path, cert_host)
    real_connect = anyio_backend.anyio.connect_tcp
    dialed = []

    async def connect(remote_host, remote_port, **kwargs):
        dialed.append(remote_host)
        if remote_host == PUBLIC_IP:
            remote_host, remote_port = "127.0.0.1", server.server_address[1]
        return await real_connect(remote_host=remote_host, remote_port=remote_port, **kwargs)

    real_client = httpx.AsyncClient
    monkeypatch.setattr(anyio_backend.anyio, "connect_tcp", connect)
    monkeypatch.setattr(utils.httpx, "AsyncClient", lambda **kw: real_client(verify=trust, **kw))
    target = str(tmp_path / "in" / "a")
    try:
        if ok:
            saved = asyncio.run(utils._save_url_image_to_path("https://public.example.com/a.png", target))
            assert open(saved, "rb").read() == b"host=public.example.com"
        else:
            with pytest.raises(Exception, match="CERTIFICATE_VERIFY_FAILED|hostname"):
                asyncio.run(utils._save_url_image_to_path("https://public.example.com/a.png", target))
    finally:
        server.shutdown()
    assert dialed and dialed[0] == PUBLIC_IP

def _server_args(monkeypatch, input_save_path):
    import sglang.multimodal_gen.runtime.server_args as server_args_pkg

    args = SimpleNamespace(input_save_path=input_save_path)
    monkeypatch.setattr(server_args_pkg, "get_global_server_args", lambda: args)
    return args


def test_edit_upload_saved_by_the_server_loads_without_opt_in(monkeypatch, tmp_path):
    from PIL import Image

    from sglang.multimodal_gen.runtime.server_warmup import prepare_warmup_image_path
    from sglang.srt.utils import common

    monkeypatch.delenv(common.LOCAL_MEDIA_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    args = _server_args(monkeypatch, "inputs/uploads")
    buf = BytesIO()
    Image.new("RGB", (3, 2)).save(buf, format="PNG")

    class Upload:
        filename = "cat.png"

        async def read(self):
            return buf.getvalue()

    # The image-edit route saves the multipart upload, then the pipeline loads that path.
    saved = asyncio.run(
        utils.save_image_to_path(
            Upload(), os.path.join("inputs/uploads", "rid_0_cat.png"), uploads_root="inputs/uploads"
        )
    )
    assert vision.load_image(saved).size == (3, 2)
    # Warmup saves its image the same way.
    assert vision.load_image(prepare_warmup_image_path(args)).size[0] > 0
    # With no save dir configured, uploads go to an sglang_* temp dir.
    _server_args(monkeypatch, None)
    with utils.temp_dir_if_disabled(None) as tmp_uploads:
        saved = asyncio.run(utils.save_image_to_path(Upload(), os.path.join(tmp_uploads, "x.png")))
        assert vision.load_image(saved).size == (3, 2)


def test_client_paths_outside_the_server_dirs_are_still_refused(monkeypatch, tmp_path):
    from sglang.srt.utils import common

    monkeypatch.delenv(common.LOCAL_MEDIA_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    _server_args(monkeypatch, "inputs/uploads")
    os.makedirs("inputs/uploads")
    (tmp_path / "secret.png").write_bytes(b"\x89PNG")
    os.symlink(tmp_path / "secret.png", "inputs/uploads/link.png")
    for ref in ("inputs/uploads/../secret.png", str(tmp_path / "secret.png"), "inputs/uploads/link.png"):
        assert not vision.server_input_path(ref)
        with pytest.raises(ValueError):
            vision.load_image(ref)

# --- diffusion-operator-only-request-fields --------------------------------------


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
