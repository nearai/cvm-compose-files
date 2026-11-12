"""CPU checks for customer-data-log-redaction.diff and log-format-followup.diff
(SGLang v0.5.23 port, NEAR AI).

Runs outside the image (stdlib + `requests`):
    python3 test_customer_data_log_redaction.py --root <.../python/sglang/srt>

* utils/log_redaction.py is loaded by path and exercised directly.
* utils/common.py download_remote_media / _redacted_request_exception are extracted with ast and
  run against a fake HTTP session that answers 404, times out, and fails to connect: the error text
  must keep inference-proxy's `NNN Client Error: ... for url: ` shape (src/proxy.rs
  message_is_client_fetch_4xx) while carrying only `<url scheme=... sha256=...>`.
* Functions touched by the follow-up are extracted with ast and run with a synthetic marker value.
* Static checks of the log format at the remaining sites.
"""

import argparse
import ast
import importlib.util
import json
import logging
import re
import sys
import types
import typing
from pathlib import Path

URL = "https://cdn.example.com/users/u-8812734/uploads/IMG_holiday.png?X-Amz-Signature=deadbeefcafebabe&X-Amz-Credential=AKIAEXAMPLE"
SECRET_BITS = ("u-8812734", "IMG_holiday", "X-Amz-Signature", "deadbeefcafebabe", "AKIAEXAMPLE", "/users/",
               "cdn.example.com")
# Synthetic value used by the follow-up checks.
MARKER = "zq-marker-5521"
URL_SHAPE = re.compile(r"<url scheme=https sha256=[0-9a-f]{8}>")
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
    check(URL_SHAPE.fullmatch(r.redact_url(URL)) is not None, f"url format: {r.redact_url(URL)!r}")
    no_secret(r.redact_url(URL), "url format: scheme and digest only")
    check(r.redact_url(URL) == r.redact_url(URL) != r.redact_url(URL + "x"), "url format: stable digest")
    red = r.redact_url("https://user:pw@h.example.org:8443/p")
    check(URL_SHAPE.fullmatch(red) is not None and "h.example.org" not in red and "pw" not in red, "url format with userinfo and port")
    check(r.redact_url("not a url").startswith("<url scheme=? sha256="), "url format without a scheme")
    check(r.redact_media_ref(DATA_URI) == f"<base64 len={len(DATA_URI)}>", "data: URI -> <base64 len=N>")
    check(r.redact_media_ref(B64).startswith("<base64 len="), "bare base64 -> <base64 len=N>")
    check(r.redact_media_ref(b"\x89PNG....") == "<bytes len=8>", "bytes -> <bytes len=N>")
    check(r.redact_media_ref({"url": URL}) == f"<dict url={r.redact_url(URL)}>", "dict with url")
    fp = r.fingerprint('{"type":"object","properties":{"ssn":{"type":"string"}}}')
    check(re.fullmatch(r"<len=\d+ sha256=[0-9a-f]{8}>", fp) is not None and "ssn" not in fp, "fingerprint = len + 8-hex sha256")
    check(r.fingerprint("abc") == r.fingerprint("abc") != r.fingerprint("abd"), "fingerprint is stable and distinguishes values")

    text = f"404 Client Error: Not Found for url: {URL}"
    red = r.redact_text(text)
    check(red == f"404 Client Error: Not Found for url: {r.redact_url(URL)}", f"requests HTTPError text keeps its shape: {red!r}")
    check(proxy_maps_to_400(red), "inference-proxy still maps the redacted text to 400")
    check(not proxy_maps_to_400(r.redact_text("500 Server Error: x for url: " + URL)), "a 5xx fetch is still not mapped")
    u3 = ("HTTPSConnectionPool(host='cdn.example.com', port=443): Max retries exceeded with url: "
          "/users/u-8812734/uploads/IMG_holiday.png?X-Amz-Signature=deadbeefcafebabe (Caused by NewConnectionError('x'))")
    red = r.redact_text(u3)
    no_secret(red, "urllib3 text format")
    check("host='<len=15 sha256=" in red and "Caused by" in red, "urllib3 text keeps its structure")
    for text in ("Failed to resolve 'cdn.example.com' ([Errno -2] Name or service not known)",
                 "Connection to cdn.example.com timed out. (connect timeout=10)"):
        red = r.redact_text(text)
        no_secret(red, f"urllib3 text format: {red!r}")
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
                check(msg == f"404 Client Error: Not Found for url: {r.redact_url(URL)}", f"404 wording kept: {msg!r}")
                check(proxy_maps_to_400(msg), "404: inference-proxy 500->400 mapping still matches")
                check(getattr(e, "response", None) is not None and e.response.status_code == 404, "404: .response kept")
    try:
        ns["_assert_media_url_allowed"]("gopher://cdn.example.com/users/u-8812734/x")
    except ValueError as e:
        no_secret(str(e), "Invalid media URL text is redacted")
    ns["_allowed_media_domains"] = frozenset({"media.example.net"})
    ns["fingerprint"] = r.fingerprint
    try:
        ns["_assert_media_url_allowed"](URL)
        check(False, "domain allowlist rejects the URL")
    except ValueError as e:
        check("Media URL domain is not allowed" in str(e), "domain allowlist wording kept")
        no_secret(str(e), "domain allowlist text format")


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
    sc = (root / "entrypoints/openai/serving_chat.py").read_text()
    for raw in ('logger.error(f"Reasoning parsing error: {e}")', 'logger.error(f"Tool call parsing error: {e}")', "text[:2000]"):
        check(raw not in sc, f"serving_chat: no {raw!r}")
    check(sc.count("parsing error: {type(e).__name__}") == 3, "serving_chat: parse errors log the exception type")
    q3 = (root / "function_call/qwen3_coder_detector.py").read_text()
    check("'{param_value}'" not in q3 and 'f"Error in detect_and_parse: {e}"' not in q3, "qwen3_coder: log format")
    bf = (root / "function_call/base_format_detector.py").read_text()
    check('f"Error in parse_streaming_increment: {e}"' not in bf, "base_format_detector: log format")
    gm = (root / "constrained/grammar_manager.py").read_text()
    check("grammar_key={req.grammar_key}" not in gm and "raised an exception: {e}" not in gm, "grammar_manager: log format")
    check(gm.count("set_finish_with_abort(error_msg, log_msg=log_msg)") == 3, "grammar_manager: the three grammar aborts pass log_msg")
    ic = (root / "managers/scheduler_components/invariant_checker.py").read_text()
    check("cur_batch_for_debug.reqs=}" in ic, "invariant_checker: dump uses Req.__repr__")
    mm = (root.parent / "multimodal_gen/runtime/entrypoints/openai/utils.py").read_text()
    check("redact_url(image_url)" in mm and "URL {image_url}" not in mm, "multimodal_gen: image download url format")
    check("redact_text(e)" in mm, "multimodal_gen: image download error format")


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


def _extract(path: Path, cls: str, names: set) -> dict:
    """Method sources of ``cls`` in ``path`` as top-level function nodes."""
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            return {n.name: n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in names}
    return {}


def _run_defs(nodes, ns):
    for n in nodes:
        n.decorator_list = []
    exec(compile(ast.Module(body=list(nodes), type_ignores=[]), "<extracted>", "exec"), ns)
    return ns


def followup(root: Path, r) -> None:
    cap = _Capture()
    log = logging.getLogger("near_followup_test")
    log.addHandler(cap)
    log.setLevel(logging.DEBUG)
    log.propagate = False

    # qwen3_coder warnings.
    defs = _extract(root / "function_call/qwen3_coder_detector.py", "Qwen3CoderDetector",
                    {"_convert_param_value", "_get_arguments_config"})
    check(set(defs) == {"_convert_param_value", "_get_arguments_config"}, "qwen3_coder methods found")
    if len(defs) == 2:
        def literal_eval(v):
            raise ValueError(v)
        ns = _run_defs(defs.values(), {"json": json, "logger": log, "fingerprint": r.fingerprint,
                                       "safe_literal_eval": literal_eval,
                                       "get_schema_properties": lambda p: p.get("properties", {}), "Any": typing.Any,
                                       "Optional": typing.Optional, "Tool": object})
        self_ = types.SimpleNamespace(_get_param_type=lambda schema: schema["type"])
        config = {t: {"type": t} for t in ("integer", "number", "boolean", "object", "custom")}
        for name in config:
            ns["_convert_param_value"](self_, MARKER, name, config, "lookup_tool")
        ns["_convert_param_value"](self_, "1", MARKER, config, "lookup_tool")
        ns["_get_arguments_config"](self_, MARKER, [])
        lines = list(cap.lines)
        check(len(lines) == 8, f"qwen3_coder: 8 warnings captured ({len(lines)})")
        check(all(MARKER not in m for m in lines), "qwen3_coder: warnings do not contain the marker")
        check(sum("Parsed value <len=" in m for m in lines) == 6, "qwen3_coder: value format")
        check(sum("'lookup_tool'" in m for m in lines) == 7, "qwen3_coder: declared tool name kept")
        check(any("of parameter 'integer'" in m for m in lines), "qwen3_coder: declared parameter name kept")
        cap.lines.clear()

    # base_format_detector warning.
    defs = _extract(root / "function_call/base_format_detector.py", "BaseFormatDetector", {"parse_base_json"})
    check("parse_base_json" in defs, "base_format_detector.parse_base_json found")
    if defs:
        env = types.SimpleNamespace(SGLANG_FORWARD_UNKNOWN_TOOLS=types.SimpleNamespace(get=lambda: False))
        ns = _run_defs(defs.values(), {"json": json, "logger": log, "fingerprint": r.fingerprint, "envs": env,
                                       "ToolCallItem": dict, "List": typing.List, "Any": typing.Any, "Tool": object})
        self_ = types.SimpleNamespace(_get_tool_indices=lambda tools: {"known_tool": 0})
        out = ns["parse_base_json"](self_, {"name": MARKER, "parameters": {}}, [])
        check(out == [] and len(cap.lines) == 1, "base_format_detector: unknown tool dropped and warned once")
        check(all(MARKER not in m for m in cap.lines) and "undefined function: <len=" in cap.lines[0],
              f"base_format_detector: warning format: {cap.lines!r}")
        cap.lines.clear()

    # Req.__repr__ and set_finish_with_abort.
    defs = _extract(root / "managers/schedule_batch.py", "Req", {"__repr__", "set_finish_with_abort"})
    check(set(defs) == {"__repr__", "set_finish_with_abort"}, "Req.__repr__ / set_finish_with_abort found")
    if len(defs) == 2:
        ns = _run_defs(defs.values(), {"logger": log, "HTTPStatus": types.SimpleNamespace(BAD_REQUEST=400),
                                       "Optional": typing.Optional, "array": lambda t, v: list(v),
                                       "get_parallel": lambda: types.SimpleNamespace(tp_rank=0),
                                       "FINISH_ABORT": lambda *a: a})
        req = types.SimpleNamespace(rid="rid-1", origin_input_ids=[101, 202, 303], output_ids=[404],
                                    grammar=types.SimpleNamespace(key=MARKER),
                                    sampling_params=types.SimpleNamespace(stop=[MARKER], json_schema=MARKER),
                                    release_mm_inputs_on_abort=lambda: None, multimodal_inputs=None,
                                    session=None)
        rep_ = ns["__repr__"](req)
        check(rep_ == "Req(rid=rid-1, input_len=3, output_len=1, has_grammar=True)", f"Req repr format: {rep_!r}")
        check("101" not in rep_ and MARKER not in rep_, "Req repr does not contain the marker")
        try:
            ns["set_finish_with_abort"](req, f"client text {MARKER}", log_msg="log text")
            check(cap.lines == ["log text, self.rid='rid-1'"], f"set_finish_with_abort logs log_msg: {cap.lines!r}")
            check(MARKER in req.to_finish[0], "set_finish_with_abort keeps error_msg")
        except TypeError as e:
            check(False, f"set_finish_with_abort takes log_msg ({type(e).__name__})")
        cap.lines.clear()
        ns["set_finish_with_abort"](req, "plain client text")
        check(cap.lines == ["plain client text, self.rid='rid-1'"], "set_finish_with_abort without log_msg is unchanged")
        cap.lines.clear()

    # summarize_grammar_error.
    pointer_err = f"JSON Schema at #/properties/{MARKER}/items is not supported by xgrammar: keyword(s) minContains would be ignored"
    s = r.summarize_grammar_error(pointer_err)
    check(s == "JSON Schema at <pointer> is not supported by xgrammar: keyword(s) minContains would be ignored",
          f"grammar error: unsupported-feature format: {s!r}")
    s = r.summarize_grammar_error(f"JSON Schema at #/properties/a is not supported by xgrammar: string format '{MARKER}' is not implemented")
    check(MARKER not in s and "string format <len=" in s, f"grammar error: format name: {s!r}")
    s = r.summarize_grammar_error(f"Invalid JSON schema near '{MARKER}'")
    check(re.fullmatch(r"<len=\d+ sha256=[0-9a-f]{8}>", s) is not None, "grammar error: other text")
    for fixed in ("Grammar preprocessing timed out", "Unknown grammar error", "Invalid json: NUL bytes (\\u0000) are not allowed"):
        check(r.summarize_grammar_error(fixed) == fixed, f"grammar error: fixed text kept: {fixed!r}")
    log.removeHandler(cap)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    args = ap.parse_args()
    r = load_redaction(args.root)
    unit(r)
    download(args.root, r)
    followup(args.root, r)
    static(args.root)
    print(f"SUMMARY customer-data log redaction: {len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
