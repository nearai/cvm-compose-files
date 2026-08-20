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
from pathlib import Path
from typing import Final


ROOT: Final = Path(__file__).resolve().parents[1]
HTTP2_LISTEN: Final = re.compile(r"\blisten\b[^;]*\bhttp2\b[^;]*;")
HTTP2_ENABLED: Final = re.compile(r"\bhttp2\s+on\s*;")
KEEPALIVE_TIMEOUT: Final = re.compile(r"\bkeepalive_timeout\s+[^;]+;")
KEEPALIVE_REQUESTS: Final = re.compile(r"\bkeepalive_requests\s+\d+;")
PROXY_PASS: Final = re.compile(
    r"\bproxy_pass\s+(?:(?:https?)://)?([A-Za-z0-9_.-]+)(?::\d+)?(?=[/;\s])"
)
SERVER_START: Final = re.compile(r"\bserver\s*\{")
NGINX_COMMENT: Final = re.compile(r"#.*$", re.MULTILINE)

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


def validate_compose(path: Path) -> list[str]:
    """Return violations for streaming-chat HTTP/2 server blocks in one compose file."""
    content = NGINX_COMMENT.sub("", path.read_text())
    file_name = path.relative_to(ROOT).as_posix()
    blocks, prelude, is_balanced = server_blocks(content)
    errors: list[str] = []
    if not is_balanced:
        errors.append(f"{file_name}: unbalanced nginx server block")
        return errors
    inherited_keepalive = bool(KEEPALIVE_TIMEOUT.search(prelude)) and bool(
        KEEPALIVE_REQUESTS.search(prelude)
    )

    for block in blocks:
        is_http2 = bool(HTTP2_LISTEN.search(block) or HTTP2_ENABLED.search(block))
        if not is_http2:
            continue
        has_keepalive = inherited_keepalive or (
            bool(KEEPALIVE_TIMEOUT.search(block)) and bool(KEEPALIVE_REQUESTS.search(block))
        )
        for service_name in PROXY_PASS.findall(block):
            key = (file_name, service_name)
            if not service_name.startswith("proxy-") or key in NON_STREAMING_HTTP2_PROXIES:
                continue
            if not has_keepalive:
                errors.append(
                    f"{file_name}: HTTP/2 server for {service_name} lacks keepalive_timeout "
                    "and keepalive_requests in server scope or the shared http prelude"
                )
    return errors


def main() -> int:
    """Print GitHub annotations for every contract violation."""
    errors = [
        error
        for path in sorted((ROOT / "prod").glob("*.yaml"))
        for error in validate_compose(path)
    ]
    if errors:
        for error in errors:
            print(f"::error file={error.split(':', 1)[0]}::{error}")
        return 1
    print("Streaming HTTP/2 keepalive contract OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
