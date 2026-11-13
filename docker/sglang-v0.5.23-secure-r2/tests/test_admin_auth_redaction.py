"""CPU checks for admin-auth-redaction.diff, ported to upstream SGLang v0.5.23.

v0.5.23 already carries upstream #41322 (admin auth on /set_trace_level and
/load_lora_adapter_from_tensors); the patch backports upstream #48518 (credential redaction).
In v0.5.23 /server_info, /get_server_info and the gRPC GetServerInfo build their record through
utils/common.py:build_server_info, the scheduler's internal_states come from
managers/scheduler_components/internal_state.py, and redact_server_args_secrets also applies the
watermark redaction (stubbed here; the real one imports torch).

With --admin-api-key set and no --api-key (the v13 launch form), the engine must:
  * reject /set_trace_level and /load_lora_adapter_from_tensors without the admin key (401),
  * keep /health, /health_generate, /metrics, /v1/loads, /v1/models, chat, completions,
    /server_info and /get_server_info open (inference-proxy and the scrapers send no key),
  * never return the key from /server_info or /get_server_info (top level or internal_states),
  * never log it (engine.py server_args= lines), dump it (tokenizer_manager crash/request dumps,
    launch_command) or print it (ServerArgs.__repr__).

Static mode (default) reads the source files and runs the changed functions on stubs, so it also runs
outside the image: python3 test_admin_auth_redaction.py --root <.../python/sglang/srt>.
--live additionally imports the installed sglang (inside the image, after maybe_stub_sgl_kernel) and
checks the real FastAPI app's route auth levels and a real /server_info call with a stubbed engine state.
Every check fails on the v12 (unpatched) sources.
"""

import argparse
import ast
import asyncio
import dataclasses
import importlib.util
import json
import sys
import types
from pathlib import Path

KEY = "test-admin-key-0123456789"  # a fixed dummy, never a real credential
API_KEY = "test-api-key-0123456789"
ADMIN_PATHS = (
    "/set_trace_level",
    "/load_lora_adapter_from_tensors",
    # Can switch on log_requests (level 3 logs full prompts/outputs): must not be open.
    "/configure_logging",
    "/set_internal_state",
)
# Already admin-gated in fc91d24; must stay so.
INHERITED_ADMIN_PATHS = ("/update_weights_from_tensor", "/update_weights_from_disk", "/flush_cache", "/abort_request")
NORMAL_PATHS = ("/health", "/health_generate", "/v1/models", "/v1/chat/completions", "/v1/completions", "/server_info", "/get_server_info")

failures: list[str] = []


def check(cond: bool, what: str) -> None:
    print(("ok   " if cond else "FAIL ") + what)
    if not cond:
        failures.append(what)


def load_auth(root: Path):
    spec = importlib.util.spec_from_file_location("near_test_auth", root / "utils/auth.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclass(frozen=True) looks its module up
    spec.loader.exec_module(mod)  # stdlib-only module
    return mod


def route_levels(tree: ast.Module, router_names=("app", "router")) -> dict[str, str]:
    """Route path -> auth level name from the decorators (NORMAL when none)."""
    levels: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        paths, level = [], "NORMAL"
        for dec in node.decorator_list:
            if not isinstance(dec, ast.Call):
                continue
            f = dec.func
            if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id in router_names:
                if dec.args and isinstance(dec.args[0], ast.Constant) and isinstance(dec.args[0].value, str):
                    paths.append(dec.args[0].value)
            if isinstance(f, ast.Name) and f.id == "auth_level" and dec.args:
                arg = dec.args[0]
                if isinstance(arg, ast.Attribute):
                    level = arg.attr
        for p in paths:
            levels[p] = level
    return levels


def extract(tree: ast.Module, names: set[str], cls: str | None = None) -> list[ast.stmt]:
    body = tree.body
    if cls is not None:
        body = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls).body
    out = []
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            node.decorator_list = []
            out.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets):
            out.append(node)
    return out


def exec_nodes(nodes: list[ast.stmt], ns: dict) -> dict:
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<extracted>", "exec"), ns)
    return ns


def wrapped_in_redact(tree: ast.Module) -> tuple[int, int]:
    """(ServerArgs.resolved_dict() calls, of which inside redact_server_args_secrets(...)). The scheduler's
    resolved_server_args_dict() (internal_states) is redacted by its readers, checked via /server_info."""
    parents = {}
    for p in ast.walk(tree):
        for c in ast.iter_child_nodes(p):
            parents[c] = p
    total = wrapped = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "resolved_dict":
            total += 1
            cur = node
            while cur in parents:
                cur = parents[cur]
                if isinstance(cur, ast.Call) and isinstance(cur.func, ast.Name) and cur.func.id == "redact_server_args_secrets":
                    wrapped += 1
                    break
                if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    break
    return total, wrapped


def _stub_watermark_module() -> None:
    """redact_server_args_secrets lazily imports the watermark redaction (torch); identity stub."""
    if "sglang.srt.sampling.watermarking.core" in sys.modules:
        return
    for name in ("sglang", "sglang.srt", "sglang.srt.sampling", "sglang.srt.sampling.watermarking"):
        sys.modules.setdefault(name, types.ModuleType(name))
    core = types.ModuleType("sglang.srt.sampling.watermarking.core")
    core.redact_watermark_secrets = lambda value, **_: value
    core.redact_watermark_command_line = lambda argv, cfg=None: " ".join(argv)
    sys.modules[core.__name__] = core


def _small_parser():
    import argparse as _ap

    p = _ap.ArgumentParser(prog="sglang serve")
    for flag in ("--api-key", "--admin-api-key", "--ssl-keyfile-password", "--ssl-keyfile", "--port", "--model-path"):
        p.add_argument(flag)
    return p


def redaction_helpers(sa_tree: ast.Module) -> dict:
    nodes = extract(sa_tree, {"SERVER_ARGS_SECRET_FIELD_NAMES", "_REDACTED", "redact_server_args_secrets", "redact_secret_cli_args"})
    _stub_watermark_module()
    import argparse as _ap

    ns = {"Any": object, "Dict": dict, "List": list, "argparse": _ap, "functools": __import__("functools")}
    ns = exec_nodes(nodes, ns)
    full = ns["redact_secret_cli_args"]
    ns["redact_secret_cli_args"] = lambda argv, parser=None: full(argv, parser or _small_parser())
    return ns


def static(root: Path) -> None:
    auth = load_auth(root)
    hs_tree = ast.parse((root / "entrypoints/http_server.py").read_text())
    levels = route_levels(hs_tree)
    loads_levels = route_levels(ast.parse((root / "entrypoints/v1_loads.py").read_text()), ("router",))
    check(loads_levels.get("/v1/loads") == "NORMAL", "/v1/loads stays NORMAL (inference-proxy replica_state reads it without a key)")

    def decide(path: str, header: str | None, level: str):
        return auth.decide_request_auth(
            method="POST", path=path, authorization_header=header,
            api_key=None, admin_api_key=KEY, auth_level=auth.AuthLevel[level],
        )

    for p in ADMIN_PATHS + INHERITED_ADMIN_PATHS:
        lvl = levels.get(p, "MISSING")
        check(lvl in ("ADMIN_OPTIONAL", "ADMIN_FORCE"), f"{p} is admin-gated (level {lvl})")
        if lvl in ("ADMIN_OPTIONAL", "ADMIN_FORCE"):
            no, wrong, ok = decide(p, None, lvl), decide(p, "Bearer nope", lvl), decide(p, f"Bearer {KEY}", lvl)
            check(not no.allowed and no.error_status_code == 401, f"{p} without the admin key -> 401")
            check(not wrong.allowed, f"{p} with a wrong key -> rejected")
            check(ok.allowed, f"{p} with the admin key -> allowed")
    for p in NORMAL_PATHS:
        lvl = levels.get(p, "MISSING")
        check(lvl == "NORMAL", f"{p} stays NORMAL (level {lvl})")
        check(decide(p, None, "NORMAL").allowed, f"{p} open without a key when only --admin-api-key is set")
    check(decide("/metrics", None, "NORMAL").allowed, "/metrics open (prefix rule)")

    # Redaction helpers (server_args.py).
    sa_tree = ast.parse((root / "server_args.py").read_text())
    try:
        h = redaction_helpers(sa_tree)
        red = h["redact_server_args_secrets"]({"admin_api_key": KEY, "api_key": API_KEY, "ssl_keyfile_password": "pw", "model_path": "m", "port": 8000})
        check(KEY not in json.dumps(red) and API_KEY not in json.dumps(red) and "pw" not in red.values(), "redact_server_args_secrets hides all three credentials")
        check(red["model_path"] == "m" and red["port"] == 8000, "redact_server_args_secrets keeps the other fields")
        check(h["redact_server_args_secrets"]({"admin_api_key": None})["admin_api_key"] is None, "an unset credential stays None")
        argv = ["sglang", "serve", "--admin-api-key", KEY, f"--api-key={API_KEY}", "--ssl-keyfile-pass", "pw", "--port", "8000"]
        out = h["redact_secret_cli_args"](argv)
        check(
            KEY not in out and all(API_KEY not in a for a in out) and "pw" not in out and out[-2:] == ["--port", "8000"],
            "redact_secret_cli_args hides --flag value, --flag=value and an abbreviated flag",
        )
    except (StopIteration, KeyError) as e:
        check(False, f"redaction helpers present in server_args.py ({e!r})")
        h = None

    # ServerArgs.__repr__ (any f-string, logger or exception that prints the record).
    repr_nodes = extract(sa_tree, {"__repr__"}, cls="ServerArgs")
    check(bool(repr_nodes), "ServerArgs defines its own __repr__ (dataclass would print the keys)")
    if repr_nodes and h is not None:
        ns = exec_nodes(repr_nodes, {**h, "dataclasses": dataclasses, "record_fields": dataclasses.fields})

        @dataclasses.dataclass
        class Fake:
            model_path: str = "m"
            api_key: str | None = None
            admin_api_key: str | None = None
            ssl_keyfile_password: str | None = None

        Fake.__repr__ = ns["__repr__"]
        r = repr(Fake(admin_api_key=KEY, api_key=API_KEY))
        check(KEY not in r and API_KEY not in r and "model_path='m'" in r, "ServerArgs repr redacts the keys and keeps the rest")

    # /server_info and /get_server_info on a stubbed engine state.
    nodes = extract(hs_tree, {"server_info", "get_server_info"})
    check(len(nodes) == 2, "server_info and get_server_info found")
    common_tree = ast.parse((root / "utils/common.py").read_text())
    bsi = extract(common_tree, {"build_server_info"})
    check(len(bsi) == 1, "utils/common.py build_server_info found")
    if len(nodes) == 2 and len(bsi) == 1 and h is not None:
        ns = run_server_info_ns(h)
        # build_server_info imports redact_server_args_secrets from sglang.srt.server_args lazily.
        fake_sa = types.ModuleType("sglang.srt.server_args")
        fake_sa.redact_server_args_secrets = h["redact_server_args_secrets"]
        saved_sa = sys.modules.get("sglang.srt.server_args")
        sys.modules["sglang.srt.server_args"] = fake_sa
        exec_nodes(bsi, ns)
        exec_nodes(nodes, ns)
        for name in ("server_info", "get_server_info"):
            body = json.dumps(asyncio.run(ns[name]()), default=str)
            check(KEY not in body and API_KEY not in body, f"/{name} does not return the keys (top level or internal_states)")
            check('"model_path": "m"' in body, f"/{name} still returns the configuration")
        if saved_sa is not None:
            sys.modules["sglang.srt.server_args"] = saved_sa
        else:
            del sys.modules["sglang.srt.server_args"]

    # The scheduler readback (internal_states) is redacted at its source as well.
    ist = (root / "managers/scheduler_components/internal_state.py").read_text()
    check(
        "redact_server_args_secrets(get_context().resolved_server_args_dict())" in ist,
        "scheduler internal_state readback is redacted at the source",
    )

    # Every whole-record readback / log / dump goes through the redaction.
    # engine / tokenizer_manager / build_server_info call resolved_dict() themselves; http_server and
    # grpc_bridge go through build_server_info in v0.5.23 (0 direct calls is fine there).
    for rel, need in (
        ("entrypoints/engine.py", True),
        ("managers/tokenizer_manager.py", True),
        ("utils/common.py", True),
        ("entrypoints/http_server.py", False),
        ("entrypoints/grpc_bridge.py", False),
    ):
        src = (root / rel).read_text()
        total, wrapped = wrapped_in_redact(ast.parse(src))
        check((total > 0 or not need) and total == wrapped, f"{rel}: {wrapped}/{total} resolved_dict() readbacks are redacted")
        if not need:
            check("build_server_info(" in src, f"{rel}: server info goes through build_server_info")
    tm = (root / "managers/tokenizer_manager.py").read_text()
    check('"server_args": self.server_args,' not in tm, "tokenizer_manager dumps no raw ServerArgs object")
    check("return self.server_args\n" not in tm, "_server_args_for_dump never returns the raw ServerArgs (watermark off path)")
    check('" ".join(sys.argv)' not in tm and "redact_secret_cli_args(sys.argv)" in tm, "crash dump launch_command is redacted")


class _Logger:
    def warning(self, *a, **k):
        pass


def run_server_info_ns(h: dict | None) -> dict:
    record = {"model_path": "m", "admin_api_key": KEY, "api_key": API_KEY, "port": 8000}

    class SA:
        launch_command = "sglang serve --admin-api-key <redacted>"

        def resolved_dict(self):
            return dict(record)

    class TM:
        server_args = SA()
        startup_time = 0.0

        async def get_internal_state(self):
            return [dict(record, world_size=2)]

    ns = {
        "_global_state": types.SimpleNamespace(tokenizer_manager=TM(), scheduler_info={"max_total_num_tokens": 1}),
        "msgspec_to_builtins": lambda x: x,
        "describe_kv_events_publisher": lambda sa: None,
        "redact_watermark_secrets": lambda value, **_: value,
        "__version__": "test",
        "logger": _Logger(),
        "List": list, "Dict": dict, "Any": object,
    }
    if h is not None:
        ns["redact_server_args_secrets"] = h["redact_server_args_secrets"]
    return ns


def live() -> None:
    # The static phase registers empty stand-in modules for the sglang packages; drop them so
    # the real, installed package is imported below.
    for name in [n for n in sys.modules if n == "sglang" or n.startswith("sglang.")]:
        if getattr(sys.modules[name], "__file__", None) is None or name.endswith("watermarking.core"):
            del sys.modules[name]
    from sglang.test.test_utils import maybe_stub_sgl_kernel

    maybe_stub_sgl_kernel()
    import sglang.srt.entrypoints.http_server as hs
    from sglang.srt.server_args import ServerArgs, redact_secret_cli_args, redact_server_args_secrets
    from sglang.srt.utils.auth import _get_auth_level_from_app_and_scope, decide_request_auth

    def decide(path: str, header: str | None):
        scope = {"type": "http", "method": "POST", "path": path, "root_path": ""}
        return decide_request_auth(
            method="POST", path=path, authorization_header=header, api_key=None, admin_api_key=KEY,
            auth_level=_get_auth_level_from_app_and_scope(hs.app, scope),
        )

    for p in ADMIN_PATHS + INHERITED_ADMIN_PATHS:
        check(not decide(p, None).allowed and decide(p, f"Bearer {KEY}").allowed, f"live app: {p} needs the admin key")
    for p in NORMAL_PATHS + ("/v1/loads", "/metrics"):
        check(decide(p, None).allowed, f"live app: {p} open without a key")

    # v0.5.23 ServerArgs is a record with a guarded constructor: build a real one.
    sa = ServerArgs(model_path="dummy", admin_api_key=KEY, api_key=API_KEY)
    r = repr(sa)
    check(KEY not in r and API_KEY not in r, "live ServerArgs repr redacts the keys")
    check(KEY not in " ".join(redact_secret_cli_args(["x", "--admin-api-key", KEY])), "live redact_secret_cli_args")
    check(redact_server_args_secrets({"admin_api_key": KEY})["admin_api_key"] != KEY, "live redact_server_args_secrets")

    ns = run_server_info_ns(None)
    from sglang.srt.utils import common as _common

    saved = {"_global_state": hs._global_state}
    saved_desc = _common.describe_kv_events_publisher
    try:
        hs._global_state = ns["_global_state"]
        _common.describe_kv_events_publisher = ns["describe_kv_events_publisher"]
        for fn in (hs.server_info, hs.get_server_info):
            body = json.dumps(asyncio.run(fn()), default=str)
            check(KEY not in body and API_KEY not in body, f"live /{fn.__name__} does not return the keys")
    finally:
        _common.describe_kv_events_publisher = saved_desc
        for k, v in saved.items():
            setattr(hs, k, v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path, help=".../python/sglang/srt")
    ap.add_argument("--live", action="store_true")
    args = ap.parse_args()
    static(args.root)
    if args.live:
        live()
    print(f"SUMMARY admin auth + redaction: {len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
