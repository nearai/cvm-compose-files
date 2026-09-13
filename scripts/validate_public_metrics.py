#!/usr/bin/env python3
"""Ensure public model TLS listeners never expose inference metrics."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVER_START = re.compile(r"\bserver\s*\{")
NGINX_COMMENT = re.compile(r"#.*$", re.MULTILINE)
SSL_LISTEN = re.compile(r"\blisten\b[^;]*\bssl\b[^;]*;")
INFERENCE_PROXY_UPSTREAM = re.compile(
    r"\b(?:proxy_pass|set\s+\$\$?\w+)\s+"
    r"(?:(?:https?)://)?(?:proxy-|vllm-proxy-)"
)
METRICS_DENY = re.compile(
    r"\blocation\s+\^~\s+/metrics\s*\{\s*return\s+404\s*;\s*\}"
)
BACKEND_METRICS_DENY = re.compile(
    r"\blocation\s+\^~\s+/v1/metrics\s*\{\s*return\s+404\s*;\s*\}"
)


def server_blocks(content: str) -> tuple[list[tuple[int, str]], bool]:
    """Return top-level nginx server blocks and whether all braces balanced."""
    blocks: list[tuple[int, str]] = []
    cursor = 0
    while match := SERVER_START.search(content, cursor):
        depth = 1
        end = match.end()
        while end < len(content) and depth:
            if content[end] == "{":
                depth += 1
            elif content[end] == "}":
                depth -= 1
            end += 1
        if depth:
            return blocks, False
        blocks.append((match.start(), content[match.start() : end]))
        cursor = end
    return blocks, True


def validate(path: Path) -> tuple[list[str], int]:
    """Return public metrics violations and protected TLS listener count."""
    content = NGINX_COMMENT.sub("", path.read_text())
    relative = path.relative_to(ROOT).as_posix()
    blocks, balanced = server_blocks(content)
    if not balanced:
        return [f"{relative}: unbalanced nginx server block"], 0

    errors: list[str] = []
    checked = 0
    for offset, block in blocks:
        if not SSL_LISTEN.search(block) or not INFERENCE_PROXY_UPSTREAM.search(block):
            continue
        checked += 1
        line = content.count("\n", 0, offset) + 1
        if not METRICS_DENY.search(block):
            errors.append(f"{relative}:{line}: TLS inference listener exposes /metrics")
        if not BACKEND_METRICS_DENY.search(block):
            errors.append(f"{relative}:{line}: TLS inference listener exposes /v1/metrics")
    return errors, checked


def main() -> int:
    """Print GitHub annotations for every public metrics exposure."""
    validations = [validate(path) for path in sorted((ROOT / "prod").glob("*.yaml"))]
    errors = [error for file_errors, _checked in validations for error in file_errors]
    checked = sum(file_checked for _errors, file_checked in validations)
    if checked == 0:
        errors.append("prod/: no TLS inference listeners found")
    if errors:
        for error in errors:
            file_name = error.split(":", 1)[0]
            print(f"::error file={file_name}::{error}")
        return 1
    print(f"Public metrics contract OK ({checked} TLS inference listeners checked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
