#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

# ─── How to run ───
# python3 scripts/validate_streaming_keepalive.py
# ──────────────────

"""Validate HTTP/2 streaming-chat nginx keepalive contracts."""

from __future__ import annotations

import re
import subprocess
import tarfile
from io import BytesIO
from ipaddress import ip_address
from pathlib import Path
from typing import Final


ROOT: Final = Path(__file__).resolve().parents[1]
HTTP2_LISTEN: Final = re.compile(r"\blisten\s+443\s+ssl\s+http2\s*;")
KEEPALIVE_TIMEOUT: Final = re.compile(r"\bkeepalive_timeout\s+[^;]+;")
KEEPALIVE_REQUESTS: Final = re.compile(r"\bkeepalive_requests\s+\d+;")
PROXY_PASS: Final = re.compile(
    r"\bproxy_pass\s+(?:(?:https?)://)?([A-Za-z0-9_.-]+)(?::\d+)?(?=[/;\s])"
)
SERVER_START: Final = re.compile(r"\bserver\s*\{")
NGINX_COMMENT: Final = re.compile(r"#.*$", re.MULTILINE)
IMAGE: Final = re.compile(r"(?m)^    image:\s*(\S+)\s*$")
DEFAULT_CONF_SOURCE: Final = re.compile(
    r"(?m)^      - source:\s*([A-Za-z0-9_.-]+)\s*$\n"
    r"^        target:\s*/etc/nginx/conf\.d/default\.conf\s*$"
)
CONFIG_CONTENT: Final = re.compile(
    r"(?m)^    content:\s*\|[-+]?\s*\n"
    r"(?P<content>(?:(?:^ {6}.*(?:\n|$))|(?:^[ \t]*\n))*)"
)
PINNED_DIGEST: Final = re.compile(r"@sha256:[0-9a-f]{64}$")
DEPRECATED_HTTP2_WARNING: Final = 'the "listen ... http2" directive is deprecated'
TEST_CERTIFICATE: Final = """-----BEGIN CERTIFICATE-----
MIIBfTCCASOgAwIBAgIUFuehj4tdhHr73zGi66lCRUlxIK4wCgYIKoZIzj0EAwIw
FDESMBAGA1UEAwwJbG9jYWxob3N0MB4XDTI2MDgyMDIzNTU1NloXDTM2MDgxNzIz
NTU1NlowFDESMBAGA1UEAwwJbG9jYWxob3N0MFkwEwYHKoZIzj0CAQYIKoZIzj0D
AQcDQgAEh9TJwJ9Gjl7FcD2fdZanUdtdL7oMzSI/rODLVhE3pXW0ZPpYUD0IIUbx
8SbwGkPh8qnwHVJmEgbXkskgPzpRAaNTMFEwHQYDVR0OBBYEFAFLMDVnWTyhBJ34
tos9PgcKsdTdMB8GA1UdIwQYMBaAFAFLMDVnWTyhBJ34tos9PgcKsdTdMA8GA1Ud
EwEB/wQFMAMBAf8wCgYIKoZIzj0EAwIDSAAwRQIgSwjEU1TRkPY7qeQ9eyzn82OD
N2Y9MHsuDaqCAflTB+YCIQDQ195Z2kYMZFwV3VJyh0hWxOAUH9kqTtnIsQeLidFI
Vg==
-----END CERTIFICATE-----
"""
TEST_PRIVATE_KEY: Final = """-----BEGIN PRIVATE KEY-----
MIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQgB2qmkfYJYE2+vdhs
YZjfdolZZvVmt4cw0SRJA/5gMzShRANCAASH1MnAn0aOXsVwPZ91lqdR210vugzN
Ij+s4MtWETeldbRk+lhQPQghRvHxJvAaQ+HyqfAdUmYSBteSySA/OlEB
-----END PRIVATE KEY-----
"""

NON_STREAMING_HTTP2_PROXIES: Final = frozenset(
    {
        ("prod/small-models.yaml", "proxy-flux2-klein-4b"),
        ("prod/small-models.yaml", "proxy-privacy-filter"),
        ("prod/small-models.yaml", "proxy-qwen3-embedding-0.6b"),
        ("prod/small-models.yaml", "proxy-qwen3-reranker-0.6b"),
        ("prod/small-models.yaml", "proxy-whisper-large-v3"),
    }
)


def server_blocks(content: str) -> tuple[tuple[str, ...], str, bool]:
    """Return nginx server blocks, shared http prelude, and balance state."""
    blocks: list[str] = []
    prelude_parts: list[str] = []
    cursor = 0
    for match in SERVER_START.finditer(content):
        if match.start() < cursor:
            continue
        depth = 1
        end = match.end()
        while end < len(content) and depth:
            if content[end] == "{":
                depth += 1
            elif content[end] == "}":
                depth -= 1
            end += 1
        if depth:
            return tuple(blocks), "".join(prelude_parts), False
        prelude_parts.append(content[cursor : match.start()])
        blocks.append(content[match.start() : end])
        cursor = end
    prelude_parts.append(content[cursor:])
    return tuple(blocks), "".join(prelude_parts), True


def mapping_body(content: str, key: str) -> str | None:
    """Return the indented body of a two-space YAML mapping entry."""
    match = re.search(
        rf"(?m)^  {re.escape(key)}:\s*\n"
        r"(?P<body>(?:(?:^ {4}.*(?:\n|$))|(?:^[ \t]*\n))*)",
        content,
    )
    return match.group("body") if match else None


def nginx_config(content: str) -> str | None:
    """Extract the Compose config mounted as nginx's default.conf."""
    service = mapping_body(content, "nginx")
    if service is None:
        return None
    source_match = DEFAULT_CONF_SOURCE.search(service)
    if source_match is None:
        return None
    config = mapping_body(content, source_match.group(1))
    if config is None:
        return None
    content_match = CONFIG_CONTENT.search(config)
    if content_match is None:
        return None
    rendered = "".join(
        line[6:] if line.startswith("      ") else line
        for line in content_match.group("content").splitlines(keepends=True)
    )
    return rendered.replace("$$", "$")


def pinned_nginx_image(content: str) -> str | None:
    """Return the digest-pinned image declared by the nginx service."""
    service = mapping_body(content, "nginx")
    if service is None:
        return None
    image_match = IMAGE.search(service)
    if image_match is None:
        return None
    image = image_match.group(1)
    return image if PINNED_DIGEST.search(image) else None


def nginx_test_archive(config: str) -> bytes:
    """Build the files needed to run nginx -t without deployment secrets."""
    files = (
        ("etc/nginx/conf.d/default.conf", config, 0o644),
        (
            "etc/letsencrypt/live/completions.near.ai/fullchain.pem",
            TEST_CERTIFICATE,
            0o644,
        ),
        (
            "etc/letsencrypt/live/completions.near.ai/privkey.pem",
            TEST_PRIVATE_KEY,
            0o600,
        ),
    )
    with BytesIO() as buffer:
        with tarfile.open(fileobj=buffer, mode="w") as archive:
            for name, text, mode in files:
                payload = text.encode()
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                info.mode = mode
                archive.addfile(info, BytesIO(payload))
        return buffer.getvalue()


def validate_nginx(path: Path) -> str | None:
    """Run nginx -t for one Compose file in its digest-pinned image."""
    compose = path.read_text()
    file_name = path.relative_to(ROOT).as_posix()
    config = nginx_config(compose)
    if config is None:
        print(f"Skipping {file_name}: no nginx default.conf config")
        return None
    image = pinned_nginx_image(compose)
    if image is None:
        print(f"Skipping {file_name}: no pinned nginx image")
        return None
    upstream_hosts: set[str] = set()
    for hostname in PROXY_PASS.findall(config):
        try:
            ip_address(hostname)
        except ValueError:
            if hostname != "localhost":
                upstream_hosts.add(hostname)
    add_host_options = [
        option
        for hostname in sorted(upstream_hosts)
        for option in ("--add-host", f"{hostname}:127.0.0.1")
    ]
    try:
        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "-i",
                *add_host_options,
                "--entrypoint",
                "/bin/sh",
                image,
                "-c",
                "tar -xC / && nginx -t",
            ],
            input=nginx_test_archive(config),
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        return f"{file_name}: docker CLI not found; cannot run nginx -t with {image}"
    output = (result.stdout + result.stderr).decode(errors="replace").strip()
    if result.returncode:
        print(output)
        annotation_output = "\n".join(
            line for line in output.splitlines() if DEPRECATED_HTTP2_WARNING not in line
        )
        return f"{file_name}: nginx -t failed with {image}\n{annotation_output}"
    print(f"Validated {file_name} with {image}")
    print(output)
    return None


def validate_compose(path: Path) -> list[str]:
    """Return violations for streaming-chat HTTP/2 server blocks in one compose file."""
    content = NGINX_COMMENT.sub("", path.read_text())
    file_name = path.relative_to(ROOT).as_posix()
    blocks, _prelude, is_balanced = server_blocks(content)
    errors: list[str] = []
    if not is_balanced:
        errors.append(f"{file_name}: unbalanced nginx server block")
        return errors
    for block in blocks:
        depth = 0
        scope_chars: list[str] = []
        for character in block:
            if character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
            elif depth == 1:
                scope_chars.append(character)
        server_scope = "".join(scope_chars)
        is_http2 = bool(HTTP2_LISTEN.search(server_scope))
        if not is_http2:
            continue
        has_keepalive = bool(KEEPALIVE_TIMEOUT.search(server_scope)) and bool(
            KEEPALIVE_REQUESTS.search(server_scope)
        )
        for service_name in PROXY_PASS.findall(block):
            key = (file_name, service_name)
            if not service_name.startswith("proxy-") or key in NON_STREAMING_HTTP2_PROXIES:
                continue
            if not has_keepalive:
                errors.append(
                    f"{file_name}: HTTP/2 server for {service_name} lacks keepalive_timeout "
                    "and keepalive_requests in server scope"
                )
    return errors


def main() -> int:
    """Print GitHub annotations for every contract violation."""
    paths = sorted((ROOT / "prod").glob("*.yaml"))
    errors = [
        error
        for path in paths
        for error in validate_compose(path)
    ]
    if not errors:
        errors = [error for path in paths if (error := validate_nginx(path)) is not None]
    if errors:
        for error in errors:
            file_name = error.split(":", 1)[0]
            print(f"::error file={file_name}::{error.replace(chr(10), '%0A')}")
        return 1
    print("Streaming HTTP/2 keepalive contract OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
