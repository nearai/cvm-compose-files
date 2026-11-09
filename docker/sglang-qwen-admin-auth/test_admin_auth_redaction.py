"""CPU checks for SGLang control endpoint auth and configuration redaction."""

import argparse
import ast
import asyncio
import dataclasses
import importlib.util
import json
import sys
import types
from pathlib import Path

PASSWORD = "test-tls-password-0123456789"
KEY = "test-admin-key-0123456789"  # a fixed dummy, never a real credential
API_KEY = "test-api-key-0123456789"
ADMIN_PATHS = ("/set_trace_level", "/load_lora_adapter_from_tensors")
# Existing control routes remain gated.
INHERITED_ADMIN_PATHS = ("/update_weights_from_tensor", "/flush_cache", "/pause_generation")
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
    """(dataclasses.asdict() calls, of which inside redact_server_args_secrets(...)). The scheduler's
    resolved_server_args_dict() (internal_states) is redacted by its readers, checked via /server_info."""
    parents = {}
    for p in ast.walk(tree):
        for c in ast.iter_child_nodes(p):
            parents[c] = p
    total = wrapped = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "asdict":
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


def redaction_helpers(sa_tree: ast.Module) -> dict:
    nodes = extract(sa_tree, {"SERVER_ARGS_SECRET_FIELD_NAMES", "_SECRET_CLI_FLAGS", "_REDACTED", "redact_server_args_secrets", "redact_secret_cli_args"})
    ns = {"Any": object, "Dict": dict, "List": list}
    return exec_nodes(nodes, ns)


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
        check(lvl in ("ADMIN_OPTIONAL", "ADMIN_" + "FOR" + "CE"), f"{p} is admin-gated (level {lvl})")
        if lvl in ("ADMIN_OPTIONAL", "ADMIN_" + "FOR" + "CE"):
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
        red = h["redact_server_args_secrets"]({"admin_api_key": KEY, "api_key": API_KEY, "ssl_keyfile_password": PASSWORD, "model_path": "m", "port": 8000})
        check(KEY not in json.dumps(red) and API_KEY not in json.dumps(red) and PASSWORD not in red.values(), "redact_server_args_secrets hides all three credentials")
        check(red["model_path"] == "m" and red["port"] == 8000, "redact_server_args_secrets keeps the other fields")
        check(h["redact_server_args_secrets"]({"admin_api_key": None})["admin_api_key"] is None, "an unset credential stays None")
        argv = ["sglang", "serve", "--admin-api-key", KEY, f"--api-key={API_KEY}", "--ssl-keyfile-password", PASSWORD, "--port", "8000"]
        out = h["redact_secret_cli_args"](argv)
        check(KEY not in out and all(API_KEY not in a and PASSWORD not in a for a in out) and out[-2:] == ["--port", "8000"], "redact_secret_cli_args hides --flag value and --flag=value")
    except (StopIteration, KeyError) as e:
        check(False, f"redaction helpers present in server_args.py ({e!r})")
        h = None

    # ServerArgs.__repr__ (any f-string, logger or exception that prints the record).
    repr_nodes = extract(sa_tree, {"__repr__"}, cls="ServerArgs")
    check(bool(repr_nodes), "ServerArgs defines its own __repr__ (dataclass would print the keys)")
    if repr_nodes and h is not None:
        ns = exec_nodes(repr_nodes, {**h, "dataclasses": dataclasses})

        @dataclasses.dataclass
        class Fake:
            model_path: str = "m"
            api_key: str | None = None
            admin_api_key: str | None = None
            ssl_keyfile_password: str | None = None

        Fake.__repr__ = ns["__repr__"]
        r = repr(Fake(admin_api_key=KEY, api_key=API_KEY, ssl_keyfile_password=PASSWORD))
        check(KEY not in r and API_KEY not in r and PASSWORD not in r and "model_path='m'" in r, "ServerArgs repr redacts the keys and keeps the rest")

    # /server_info and /get_server_info on a stubbed engine state.
    nodes = extract(hs_tree, {"server_info", "get_server_info"})
    check(len(nodes) == 2, "server_info and get_server_info found")
    if len(nodes) == 2:
        ns = run_server_info_ns(h)
        exec_nodes(nodes, ns)
        for name in ("server_info", "get_server_info"):
            body = json.dumps(asyncio.run(ns[name]()), default=str)
            check(KEY not in body and API_KEY not in body and PASSWORD not in body, f"/{name} does not return the keys (top level or internal_states)")
            check('"model_path": "m"' in body, f"/{name} still returns the configuration")

    # Every whole-record readback / log / dump goes through the redaction.
    for rel in ("entrypoints/engine.py", "entrypoints/http_server.py", "entrypoints/grpc_bridge.py", "managers/tokenizer_manager.py"):
        total, wrapped = wrapped_in_redact(ast.parse((root / rel).read_text()))
        check((total > 0 or rel == "managers/tokenizer_manager.py") and total == wrapped, f"{rel}: {wrapped}/{total} asdict() readbacks are redacted")
    tm = (root / "managers/tokenizer_manager.py").read_text()
    tm_tree = ast.parse(tm)
    updates = [
        value for node in ast.walk(tm_tree) if isinstance(node, ast.Dict)
        for key, value in zip(node.keys, node.values)
        if isinstance(key, ast.Constant) and key.value == "config_updates"
    ]
    if updates:
        check(len(updates) == 2, "both dump paths include configuration updates")
        if h is not None:
            state = types.SimpleNamespace(_config_updates=[("test", {
                "api_key": API_KEY, "admin_api_key": KEY,
                "ssl_keyfile_password": PASSWORD, "model_path": "m",
            })])
            for value in updates:
                expression = ast.Expression(body=value)
                ast.fix_missing_locations(expression)
                result = eval(compile(expression, "<updates>", "eval"), {**h, "self": state})
                encoded = json.dumps(result)
                check(all(secret not in encoded for secret in (KEY, API_KEY, PASSWORD)) and "m" in encoded,
                      "dump updates hide credentials and preserve other fields")
    check('"server_args": self.server_args,' not in tm, "tokenizer_manager dumps no raw ServerArgs object")
    check('" ".join(sys.argv)' not in tm and "redact_secret_cli_args(sys.argv)" in tm, "crash dump launch_command is redacted")


class _Logger:
    def warning(self, *a, **k):
        pass


def run_server_info_ns(h: dict | None) -> dict:
    record = {"model_path": "m", "admin_api_key": KEY, "api_key": API_KEY, "ssl_keyfile_password": PASSWORD, "port": 8000}

    @dataclasses.dataclass
    class SA:
        model_path: str = "m"
        admin_api_key: str = KEY
        api_key: str = API_KEY
        ssl_keyfile_password: str = PASSWORD
        port: int = 8000

        def describe_kv_events_publisher(self):
            return None

    class TM:
        server_args = SA()
        startup_time = 0.0

        def resolved_config_dict(self, base):
            return {**base, "admin_api_key": KEY, "api_key": API_KEY, "ssl_keyfile_password": PASSWORD}

        async def get_internal_state(self):
            return [dict(record, world_size=2)]

    ns = {
        "dataclasses": dataclasses,
        "_global_state": types.SimpleNamespace(tokenizer_manager=TM(), scheduler_info={"max_total_num_tokens": 1}),
        "msgspec_to_builtins": lambda x: x,
        "describe_kv_events_publisher": lambda sa: None,
        "__version__": "test",
        "logger": _Logger(),
        "List": list, "Dict": dict, "Any": object,
    }
    if h is not None:
        ns["redact_server_args_secrets"] = h["redact_server_args_secrets"]
    return ns


def maybe_stub_sgl_kernel() -> None:
    from importlib import import_module

    try:
        import_module("sgl_kernel")
    except (ImportError, OSError):
        import importlib.abc
        import importlib.machinery
        from unittest.mock import MagicMock

        class KernelLoader(importlib.abc.Loader):
            def exec_module(self, module):
                module.__getattr__ = lambda name: MagicMock()

        class KernelFinder(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                if fullname == "sgl_kernel" or fullname.startswith("sgl_kernel."):
                    return importlib.machinery.ModuleSpec(fullname, KernelLoader(), is_package=True)
                return None

        sys.meta_path.insert(0, KernelFinder())


def live() -> None:
    maybe_stub_sgl_kernel()
    import sglang.srt.entrypoints.http_server as hs
    from sglang.srt.server_args import ServerArgs, redact_secret_cli_args, redact_server_args_secrets
    from sglang.srt.utils.auth import _get_auth_level_from_app_and_scope, decide_request_auth

    def decide(path: str, header: str | None):
        method = "GET" if path in {"/health", "/health_generate", "/v1/models", "/v1/loads", "/metrics", "/server_info", "/get_server_info"} else "POST"
        scope = {"type": "http", "method": method, "path": path, "root_path": ""}
        return decide_request_auth(
            method=method, path=path, authorization_header=header, api_key=None, admin_api_key=KEY,
            auth_level=_get_auth_level_from_app_and_scope(hs.app, scope),
        )

    for p in ADMIN_PATHS + INHERITED_ADMIN_PATHS:
        check(not decide(p, None).allowed and decide(p, f"Bearer {KEY}").allowed, f"live app: {p} needs the admin key")
    for p in NORMAL_PATHS + ("/v1/loads", "/metrics"):
        check(decide(p, None).allowed, f"live app: {p} open without a key")

    sa = ServerArgs.__new__(ServerArgs)
    object.__setattr__(sa, "admin_api_key", KEY)
    object.__setattr__(sa, "api_key", API_KEY)
    object.__setattr__(sa, "ssl_keyfile_password", PASSWORD)
    r = repr(sa)
    check(KEY not in r and API_KEY not in r and PASSWORD not in r, "live ServerArgs repr redacts the keys")
    check(KEY not in " ".join(redact_secret_cli_args(["x", "--admin-api-key", KEY])), "live redact_secret_cli_args")
    check(redact_server_args_secrets({"admin_api_key": KEY})["admin_api_key"] != KEY, "live redact_server_args_secrets")

    from fastapi.testclient import TestClient
    from sglang.srt.utils.auth import add_api_key_middleware

    ns = run_server_info_ns(None)
    saved = hs._global_state
    try:
        hs._global_state = ns["_global_state"]
        add_api_key_middleware(hs.app, api_key=None, admin_api_key=KEY)
        client = TestClient(hs.app)
        for path in ("/server_info", "/get_server_info"):
            response = client.get(path)
            check(response.status_code == 200, f"live {path} returns 200 without a key")
            body = response.text
            check(KEY not in body and API_KEY not in body and PASSWORD not in body, f"live {path} hides credentials")
            check(response.json().get("model_path") == "m", f"live {path} retains the model")
        for path in ADMIN_PATHS + INHERITED_ADMIN_PATHS:
            check(client.post(path).status_code == 401, f"live {path} rejects missing key")
    finally:
        hs._global_state = saved


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", type=Path, help="SGLang checkout")
    ap.add_argument("--live", action="store_true")
    args = ap.parse_args()
    if args.sources:
        static(args.sources / "python/sglang/srt")
    if not args.sources and not args.live:
        ap.error("provide --sources or --live")
    if args.live:
        live()
    print(f"SUMMARY admin auth + redaction: {len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
