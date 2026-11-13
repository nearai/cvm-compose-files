"""nginx: public metrics and HTTP/2 keepalive contracts, on every nginx config text of the un-interpolated render.

nginx_test() runs `nginx -t` for each nginx service's default.conf in its digest-pinned image.
"""

import ipaddress
import os
import re
import subprocess
import tarfile
import tempfile
from io import BytesIO

from common import Finding

RULES = ["nginx.parse", "nginx.public-metrics", "nginx.keepalive"]
DIFF_RULES = []

SERVER_START = re.compile(r"\bserver\s*\{")
NGINX_COMMENT = re.compile(r"#.*$", re.M)
SSL_LISTEN = re.compile(r"\blisten\b[^;]*\bssl\b[^;]*;")
INFERENCE_PROXY_UPSTREAM = re.compile(
    r"\b(?:proxy_pass|set\s+\$\$?\w+)\s+(?:(?:https?)://)?(?:proxy-|vllm-proxy-)")
METRICS_DENY = re.compile(r"\blocation\s+\^~\s+/metrics\s*\{\s*return\s+404\s*;\s*\}")
OTHER_METRICS_LOCATION = re.compile(r"(?i)\blocation\s+(=|~\*?)\s*[^{]*metrics|\b(proxy_pass|rewrite|return)\b[^;]*/metrics\b")
BACKEND_METRICS_DENY = re.compile(r"\blocation\s+\^~\s+/v1/metrics\s*\{\s*return\s+404\s*;\s*\}")
HTTP2_LISTEN = re.compile(r"\blisten\b[^;]*\bhttp2\b[^;]*;")
HTTP2_ON = re.compile(r"\bhttp2\s+on\s*;")
KEEPALIVE_TIMEOUT = re.compile(r"\bkeepalive_timeout\s+[^;]+;")
KEEPALIVE_REQUESTS = re.compile(r"\bkeepalive_requests\s+\d+;")
PROXY_PASS = re.compile(r"\bproxy_pass\s+(?:(?:https?)://)?([A-Za-z0-9_.-]+)(?::\d+)?(?=[/;\s])")
SSL_CERT = re.compile(r"\bssl_certificate\s+([^;\s]+)\s*;")
SSL_KEY = re.compile(r"\bssl_certificate_key\s+([^;\s]+)\s*;")
PINNED_DIGEST = re.compile(r"@sha256:[0-9a-f]{64}$")
DEFAULT_CONF = "/etc/nginx/conf.d/default.conf"
DEPRECATED_HTTP2_WARNING = 'the "listen ... http2" directive is deprecated'


def texts(node, where=""):
    """Every string in the model with its location, so a config is found wherever it is embedded."""
    if isinstance(node, str):
        yield where, node
    elif isinstance(node, dict):
        for k, v in node.items():
            yield from texts(v, f"{where}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from texts(v, f"{where}[{i}]")


def parse_servers(text):
    """Top-level server blocks as (offset, text), the text outside them, and whether braces balance."""
    blocks, prelude, cursor = [], [], 0
    for match in SERVER_START.finditer(text):
        if match.start() < cursor:
            continue
        depth, end = 1, match.end()
        while end < len(text) and depth:
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
            end += 1
        if depth:
            return blocks, "".join(prelude), False
        prelude.append(text[cursor:match.start()])
        blocks.append((match.start(), text[match.start():end]))
        cursor = end
    prelude.append(text[cursor:])
    return blocks, "".join(prelude), True


def server_scope(block):
    """The directives of a server block, without what sits in nested blocks."""
    depth, chars = 0, []
    for ch in block:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        elif depth == 1:
            chars.append(ch)
    return "".join(chars)


def check(ctx):
    out = []
    tls_listeners = 0
    for path in ctx.head.deployables():
        model = ctx.raw(path)
        if model is None:
            continue
        for where, raw in texts(model):
            if "server" not in raw:
                continue
            text = NGINX_COMMENT.sub("", raw)
            blocks, prelude, balanced = parse_servers(text)
            if not balanced:
                out.append(Finding(path, 0, "nginx.parse", f"{path}: {where}: unbalanced nginx server block",
                                   key=where))
                continue
            for _, block in blocks:
                if not SSL_LISTEN.search(block) or not INFERENCE_PROXY_UPSTREAM.search(block):
                    continue
                tls_listeners += 1
                if not METRICS_DENY.search(block):
                    out.append(Finding(path, 0, "nginx.public-metrics",
                                       f"{path}: {where}: TLS inference listener exposes /metrics", key=where + "m"))
                # An exact location wins over the ^~ prefix, and any proxy_pass or rewrite can target a
                # metrics path from another location, so neither may name one.
                if OTHER_METRICS_LOCATION.search(block):
                    out.append(Finding(path, 0, "nginx.public-metrics",
                                       f"{path}: {where}: TLS inference listener routes a metrics path past the 404",
                                       key=where + "x"))
                if not BACKEND_METRICS_DENY.search(block):
                    out.append(Finding(path, 0, "nginx.public-metrics",
                                       f"{path}: {where}: TLS inference listener exposes /v1/metrics",
                                       key=where + "v1"))
            for _, block in blocks:
                scope = server_scope(block)
                local_http2 = HTTP2_LISTEN.search(scope) or HTTP2_ON.search(scope)
                is_http2 = bool(local_http2 or (HTTP2_ON.search(prelude) and SSL_LISTEN.search(scope)))
                if not is_http2:
                    continue
                has_keepalive = bool(KEEPALIVE_TIMEOUT.search(scope)) and bool(KEEPALIVE_REQUESTS.search(scope))
                for upstream in PROXY_PASS.findall(block):
                    if not upstream.startswith("proxy-") or has_keepalive:
                        continue
                    out.append(Finding(path, 0, "nginx.keepalive",
                                       f"{path}: {where}: HTTP/2 server for {upstream} lacks keepalive_timeout "
                                       "and keepalive_requests in server scope", key=f"{where}:{upstream}"))
    if tls_listeners == 0:
        out.append(Finding("prod/", 0, "nginx.public-metrics", "prod/: no TLS inference listeners found",
                           key="none"))
    return out


# ---------------------------------------------------------------- runtime nginx -t

def _openssl_pair(workdir):
    """A throwaway certificate and key, generated here so no key is committed."""
    key, cert = os.path.join(workdir, "key.pem"), os.path.join(workdir, "cert.pem")
    subprocess.run(["openssl", "ecparam", "-name", "prime256v1", "-genkey", "-noout", "-out", key],
                   check=True, capture_output=True)
    subprocess.run(["openssl", "req", "-x509", "-key", key, "-out", cert, "-days", "1", "-subj", "/CN=ci-test"],
                   check=True, capture_output=True)
    with open(cert, "rb") as f:
        cert_bytes = f.read()
    with open(key, "rb") as f:
        key_bytes = f.read()
    return cert_bytes, key_bytes


def _archive(files):
    """A tar of (path, bytes, mode) entries, extracted at / in the container."""
    with BytesIO() as buffer:
        with tarfile.open(fileobj=buffer, mode="w") as archive:
            for name, payload, mode in files:
                info = tarfile.TarInfo(name.lstrip("/"))
                info.size = len(payload)
                info.mode = mode
                archive.addfile(info, BytesIO(payload))
        return buffer.getvalue()


def _test_one(path, model, cert, key):
    services = model.get("services") or {}
    svc = services.get("nginx")
    if svc is None:
        return None
    source = None
    for entry in svc.get("configs") or []:
        if isinstance(entry, dict) and entry.get("target") == DEFAULT_CONF:
            source = entry.get("source")
    text = ((model.get("configs") or {}).get(str(source)) or {}).get("content") if source else None
    if not text:
        print(f"Skipping {path}: no nginx default.conf config")
        return None
    image = str(svc.get("image") or "")
    if not PINNED_DIGEST.search(image):
        print(f"Skipping {path}: no pinned nginx image")
        return None
    hosts = set()
    for host in PROXY_PASS.findall(text):
        try:
            ipaddress.ip_address(host)
        except ValueError:
            if host != "localhost":
                hosts.add(host)
    add_host = [opt for host in sorted(hosts) for opt in ("--add-host", f"{host}:127.0.0.1")]
    files = [("etc/nginx/conf.d/default.conf", text.replace("$$", "$").encode(), 0o644)]
    for target in sorted(set(SSL_CERT.findall(text))):
        if target.startswith("/"):
            files.append((target, cert, 0o644))
    for target in sorted(set(SSL_KEY.findall(text))):
        if target.startswith("/"):
            files.append((target, key, 0o600))
    cmd = ["docker", "run", "--rm", "-i", "--network", "none", *add_host, "--entrypoint", "/bin/sh", image,
           "-c", "tar -xC / && nginx -t"]
    try:
        result = subprocess.run(cmd, input=_archive(files), capture_output=True, check=False)
    except FileNotFoundError:
        return f"{path}: docker CLI not found; cannot run nginx -t with {image}"
    output = (result.stdout + result.stderr).decode(errors="replace").strip()
    if result.returncode:
        print(output)
        kept = "\n".join(line for line in output.splitlines() if DEPRECATED_HTTP2_WARNING not in line)
        return f"{path}: nginx -t failed with {image}\n{kept}"
    print(f"Validated {path} with {image}")
    print(output)
    return None


def nginx_test(models):
    """Run nginx -t for every model that has an nginx default.conf; returns the failures."""
    with tempfile.TemporaryDirectory() as workdir:
        cert, key = _openssl_pair(workdir)
        failures = []
        for path in sorted(models):
            failure = _test_one(path, models[path], cert, key)
            if failure:
                failures.append(failure)
    return failures


# ---------------------------------------------------------------- self-test fixtures

P = "prod/a.yaml"
FILES = {P: "services: {}\n"}
DENY = ("  location ^~ /metrics { return 404; }\n"
        "  location ^~ /v1/metrics { return 404; }\n")
TLS = "server {\n  listen 443 ssl;\n  location / { proxy_pass http://proxy-x:8000; }\n" + DENY + "}\n"


def _model(*contents):
    return {"configs": {f"c{i}": {"content": text} for i, text in enumerate(contents)}}


def _case(name, rule, model):
    return (name, rule, FILES, None, None, None, {"raw": {P: model}})


CASES = [
    ("green", None, FILES, None, None, None, {"raw": {P: _model(TLS)}}),
    _case("metrics listener exposed", "nginx.public-metrics",
          _model(TLS.replace("  location ^~ /metrics { return 404; }\n", ""))),
    _case("exact metrics location", "nginx.public-metrics",
          _model(TLS.replace("  location ^~ /metrics { return 404; }\n",
                             "  location ^~ /metrics { return 404; }\n  location = /metrics { proxy_pass http://x; }\n"))),
    _case("metrics via another location", "nginx.public-metrics",
          _model(TLS.replace("  location ^~ /metrics { return 404; }\n",
                             "  location ^~ /metrics { return 404; }\n  location /stats { proxy_pass http://e/metrics; }\n"))),
    _case("no TLS inference listener", "nginx.public-metrics", _model("server {\n  listen 80;\n}\n")),
    _case("unbalanced server block", "nginx.parse", _model("server {\n  listen 80;\n", TLS)),
    _case("http2 proxy without keepalive", "nginx.keepalive",
          _model(TLS.replace("listen 443 ssl;", "listen 443 ssl http2;"))),
]
