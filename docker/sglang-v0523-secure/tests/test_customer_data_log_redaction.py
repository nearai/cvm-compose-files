"""CPU checks for customer-data-log-redaction.diff (SGLang v0.5.23 port, NEAR AI).

Runs outside the image (stdlib + `requests`):
    python3 test_customer_data_log_redaction.py --root <.../python/sglang/srt>

* utils/log_redaction.py is loaded by path and exercised directly.
* utils/common.py download_remote_media / _redacted_request_exception are extracted with ast and
  run against a fake HTTP session that answers 404, times out, and fails to connect: the error text
  must keep inference-proxy's `NNN Client Error: ... for url: ` shape (src/proxy.rs
  message_is_client_fetch_4xx) while carrying only `<url host=...>`.
* Static checks that the audited log sites no longer print the raw values.
"""

import argparse
import ast
import importlib.util
import re
import sys
import types
from pathlib import Path

URL = "https://cdn.example.com/users/u-8812734/uploads/IMG_holiday.png?X-Amz-Signature=deadbeefcafebabe&X-Amz-Credential=AKIAEXAMPLE"
SECRET_BITS = ("u-8812734", "IMG_holiday", "X-Amz-Signature", "deadbeefcafebabe", "AKIAEXAMPLE", "/users/")
B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
DATA_URI = "data:image/png;base64," + B64

failures = []


def check(cond, what):
    print(("ok   " if cond else "FAIL ") + what)
    if not cond:
        failures.append(what)


def no_secret(text, what):
    leaked = [b for b in SECRET_BITS if b in text]
    check(not leaked, f"{what} (leaked: {leaked})" if leaked else what)


def load_redaction(root: Path):
    spec = importlib.util.spec_from_file_location("near_log_redaction", root / "utils/log_redaction.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# inference-proxy src/proxy.rs message_is_client_fetch_4xx, transcribed.
def proxy_maps_to_400(message: str) -> bool:
    lower = message.lower()
    if "url=" not in lower and "for url:" not in lower:
        return False
    return re.search(r"\b4\d\d, message=|status=4\d\d\b|\b4\d\d client error:", lower) is not None


def unit(r) -> None:
    check(r.redact_url(URL) == "<url host=cdn.example.com>", "redact_url keeps the host only")
    check(r.redact_url("https://user:pw@h.example.org:8443/p") == "<url host=h.example.org>", "redact_url drops userinfo and port")
    check(r.redact_media_ref(DATA_URI) == f"<base64 len={len(DATA_URI)}>", "data: URI -> <base64 len=N>")
    check(r.redact_media_ref(B64).startswith("<base64 len="), "bare base64 -> <base64 len=N>")
    check(r.redact_media_ref(b"\x89PNG....") == "<bytes len=8>", "bytes -> <bytes len=N>")
    check(r.redact_media_ref({"url": URL}) == "<dict url=<url host=cdn.example.com>>", "dict with url")
    fp = r.fingerprint('{"type":"object","properties":{"ssn":{"type":"string"}}}')
    check(re.fullmatch(r"<len=\d+ sha256=[0-9a-f]{8}>", fp) is not None and "ssn" not in fp, "fingerprint = len + 8-hex sha256")
    check(r.fingerprint("abc") == r.fingerprint("abc") != r.fingerprint("abd"), "fingerprint is stable and distinguishes values")

    text = f"404 Client Error: Not Found for url: {URL}"
    red = r.redact_text(text)
    check(red == "404 Client Error: Not Found for url: <url host=cdn.example.com>", f"requests HTTPError text keeps its shape: {red!r}")
    check(proxy_maps_to_400(red), "inference-proxy still maps the redacted text to 400")
    check(not proxy_maps_to_400(r.redact_text("500 Server Error: x for url: " + URL)), "a 5xx fetch is still not mapped")
    u3 = ("HTTPSConnectionPool(host='cdn.example.com', port=443): Max retries exceeded with url: "
          "/users/u-8812734/uploads/IMG_holiday.png?X-Amz-Signature=deadbeefcafebabe (Caused by NewConnectionError('x'))")
    red = r.redact_text(u3)
    no_secret(red, "urllib3 'with url: /path' form is redacted")
    check("host='cdn.example.com'" in red and "Caused by" in red, "urllib3 text keeps host and cause")
    red = r.redact_text(f"Error while loading data {DATA_URI}: bad")
    check(B64 not in red and "<base64 len=" in red, "data: URI inside text")
    path = "/sgl-workspace/sglang/python/sglang/srt/multimodal/processors/base_processor.py"
    check(path in r.redact_text(f'File "{path}", line 1'), "traceback source paths are left alone")
    check(r.redact_text("Qwen/Qwen3-8B model glm-5.3-flash") == "Qwen/Qwen3-8B model glm-5.3-flash", "ordinary text unchanged")

    try:
        try:
            raise ValueError(f"404 Client Error: Not Found for url: {URL}")
        except ValueError as inner:
            raise RuntimeError(f"wrap {inner}") from inner
    except RuntimeError as e:
        tb = r.redacted_traceback(e)
        no_secret(tb, "redacted_traceback covers the cause chain")
        check("RuntimeError" in tb and "ValueError" in tb, "redacted_traceback keeps types")
        ot = r.opaque_traceback(e)
        check("wrap 404" not in ot and "u-8812734" not in ot and "RuntimeError: <len=" in ot and "Traceback" in ot, "opaque_traceback hides the message")


def download(root: Path, r) -> None:
    import requests

    src = (root / "utils/common.py").read_text()
    tree = ast.parse(src)
    want = {"_redacted_request_exception", "download_remote_media", "_download_remote_media", "_assert_media_url_allowed"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in want]
    check({n.name for n in nodes} == want, "download helpers found in utils/common.py")
    if len(nodes) != len(want):
        return

    class FakeResp:
        def __init__(self, status, url):
            self.status_code, self.url, self.headers = status, url, {}
            self.reason = "Not Found"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def raise_for_status(self):
            real = requests.Response()
            real.status_code, real.url, real.reason = self.status_code, self.url, self.reason
            real.raise_for_status()

        def iter_content(self, chunk_size):
            yield b"x"

    class Session:
        mode = "404"

        def get(self, url, **kw):
            if self.mode == "404":
                return FakeResp(404, url)
            if self.mode == "timeout":
                raise requests.exceptions.ReadTimeout(f"HTTPSConnectionPool(host='cdn.example.com', port=443): Read timed out. (url: {url})")
            raise requests.exceptions.ConnectionError(
                "HTTPSConnectionPool(host='cdn.example.com', port=443): Max retries exceeded with url: "
                "/users/u-8812734/uploads/IMG_holiday.png?X-Amz-Signature=deadbeefcafebabe"
            )

    session = Session()
    import time
    from urllib.parse import urljoin, urlparse

    ns = {
        "requests": requests, "time": time, "urljoin": urljoin, "urlparse": urlparse,
        "redact_text": r.redact_text, "redact_url": r.redact_url,
        "get_mm_http_session": lambda: session,
        "_MAX_MEDIA_URL_REDIRECTS": 3, "_MEDIA_URL_REDIRECT_STATUS_CODES": {301, 302, 303, 307, 308},
        "_media_url_max_file_size_bytes": 0, "_allowed_media_domains": frozenset(),
        "_normalize_media_domain": lambda d: d.lower(),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<common>", "exec"), ns)
    for mode, cls in (("404", requests.exceptions.HTTPError), ("timeout", requests.exceptions.Timeout),
                      ("conn", requests.exceptions.ConnectionError)):
        session.mode = mode
        try:
            ns["download_remote_media"](URL, timeout=5)
            check(False, f"{mode}: raised")
            continue
        except requests.exceptions.RequestException as e:
            msg = str(e)
            check(isinstance(e, cls), f"{mode}: keeps its requests exception type ({type(e).__name__})")
            no_secret(msg, f"{mode}: message carries no path/query: {msg!r}")
            check(e.__suppress_context__, f"{mode}: raised `from None` (traceback hides the raw message)")
            if mode == "404":
                check(msg == "404 Client Error: Not Found for url: <url host=cdn.example.com>", f"404 wording kept: {msg!r}")
                check(proxy_maps_to_400(msg), "404: inference-proxy 500->400 mapping still matches")
                check(getattr(e, "response", None) is not None and e.response.status_code == 404, "404: .response kept")
    try:
        ns["_assert_media_url_allowed"]("gopher://cdn.example.com/users/u-8812734/x")
    except ValueError as e:
        no_secret(str(e), "Invalid media URL text is redacted")


def static(root: Path) -> None:
    bp = (root / "multimodal/processors/base_processor.py").read_text()
    check("data_str[:100]" not in bp, "base_processor: no data_str[:100] in error texts")
    check("redact_media_ref(data)" in bp, "base_processor: _load_single_item describes the item")
    check('"[load_mm_data(simple)] error loading %s data at index=%d",' not in bp and "redacted_traceback(e)" in bp,
          "base_processor: load_mm_data error is logged without the raw traceback")
    sb = (root / "entrypoints/openai/serving_base.py").read_text()
    check('logger.exception(f"Error in request: {e}")' not in sb, "serving_base: no raw 'Error in request: {e}'")
    for f in ("xgrammar_backend.py", "llguidance_backend.py", "outlines_backend.py"):
        t = (root / "constrained" / f).read_text()
        check("{key_string=}" not in t, f"{f}: no raw key_string in logs")
    for f in ("glm47_moe_detector.py", "glm4_moe_detector.py"):
        t = (root / "function_call" / f).read_text()
        check("Failed to parse '{value}'" not in t, f"{f}: number-parse warning does not quote the value")
        check(", exc_info=True)" not in t, f"{f}: no exc_info tracebacks with raw messages")
    common = (root / "utils/common.py").read_text()
    check("Timed out while downloading media URL: {url}" not in common, "common: timeout text redacted")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    args = ap.parse_args()
    r = load_redaction(args.root)
    unit(r)
    download(args.root, r)
    static(args.root)
    print(f"SUMMARY customer-data log redaction: {len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
