"""privacy: engine logging flags and env, host access, deploy-time variables."""

import bisect
import re

from common import (
    ENGINE_KEY, SECRET_SUFFIX, SERVICES, Finding, engine_kind, engines, find_line, flag_values, has_gpu, key_paths,
    service_tokens,
)

RULES = [
    "privacy.no-log-flag", "privacy.log-enabled", "privacy.env", "privacy.role", "privacy.host-access",
    "privacy.var", "privacy.secret", "privacy.default", "privacy.egress", "privacy.dummy", "privacy.tmpfs",
    "privacy.maintenance",
]
DIFF_RULES = []

VLLM_NO_LOG = "--no-enable-log-requests"
BANNED_FLAG = re.compile(
    r"^--(log-requests|enable-log-requests|enable-log-outputs|config|[\w-]*(dump|trace|profil)[\w-]*)(=.*)?$")
BANNED_ENV = re.compile(r"LOG_REQUEST|DUMP|PROFILE|TRACE|^NEAR_SELF_PROFILE")
ENGINE_IMAGE = re.compile(r"^(sglang|vllm|vllm-openai)$")
UPPER_SNAKE = re.compile(r"^[A-Z][A-Z0-9_]*$")
DUMMY = re.compile(r"^(ci-[a-z0-9-]+|\d{1,5})$")
VAR = re.compile(r"\$(\$|\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}|([A-Za-z_][A-Za-z0-9_]*))")
OPS = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)(?:(:?[-?+])(.*))?$", re.S)


def host_access(svc, volumes):
    items = set()
    if svc.get("privileged"):
        items.add("privileged")
    for d in svc.get("devices") or []:
        target = d.get("target") or d.get("source") if isinstance(d, dict) else str(d).split(":")[-1 if ":" in str(d)
                                                                                                   else 0]
        items.add(f"device:{target}")
    for c in svc.get("cap_add") or []:
        items.add(f"cap_add:{c}")
    for k, name in (("network_mode", "network"), ("pid", "pid"), ("ipc", "ipc")):
        if svc.get(k) == "host":
            items.add(f"{name}:host")
    for s in svc.get("security_opt") or []:
        items.add(f"security_opt:{s}")
    for v in svc.get("volumes") or []:
        if not isinstance(v, dict):
            continue
        if v.get("type") == "bind":
            src = str(v.get("source", ""))
            items.add(f"{'sock' if src.endswith('.sock') else 'bind'}:{v.get('target')}")
        elif v.get("type") == "volume" and v.get("source") in volumes:
            opts = (volumes[v["source"]] or {}).get("driver_opts") or {}
            if "bind" in str(opts.get("o", "")) or opts.get("type") == "none":
                items.add(f"bind-volume:{v['source']}")
    for s in svc.get("volumes_from") or []:
        items.add(f"volumes_from:{s}")
    for p in svc.get("ports") or []:
        if isinstance(p, dict) and p.get("published"):
            items.add(f"port:{p['published']}/{p.get('protocol', 'tcp')}")
    return items


def check(ctx):
    out, head, sa = [], ctx.head, ctx.services_allow
    allow = {(s["file"], s["service"]): s for s in sa.get("services", [])}
    env = {e["name"]: e for e in sa.get("env", [])}
    used = set()
    for p in head.deployables() + list(ctx.templates):
        r = ctx.rendered(p)
        text = head.text(p)
        out += check_vars(p, text, env)
        if r is None:
            continue
        services, volumes = r.get("services", {}), r.get("volumes") or {}
        eng = set(engines(p, r, sa))
        for key, svc in sorted(services.items()):
            line = find_line(text, rf"^  {re.escape(key)}:")
            entry = allow.get((p, key), {})
            image = svc.get("image") or ""
            repo = image.split("@")[0].rsplit("/", 1)[-1].split(":")[0]
            if key not in eng and not ENGINE_KEY.match(key):
                if (has_gpu(svc) or ENGINE_IMAGE.match(repo)) and entry.get("role") not in ("engine", "aux"):
                    out.append(Finding(p, line, "privacy.role", f"service {key} needs role engine or aux", key=key))
            found = host_access(svc, volumes)
            want = set(entry.get("items", []))
            if (p, key) in allow:
                used.add((p, key))
            for item in sorted(found - want):
                out.append(Finding(p, line, "privacy.host-access", f"service {key}: {item} is not listed in {SERVICES}",
                                   key=f"{key}:{item}"))
            for item in sorted(want - found):
                out.append(Finding(SERVICES, 0, "privacy.host-access", f"{p} {key}: listed {item} is not used",
                                   key=f"{p}:{key}:{item}"))
            # Any service classified SGLang/vLLM gets the logging checks, whatever its role.
            if key in eng or ENGINE_IMAGE.match(repo) or engine_kind(key, svc) != "unknown":
                out += check_engine(p, line, key, svc, key in eng)
        for vname, vol in volumes.items():
            if "ghost" in vname:
                opts = (vol or {}).get("driver_opts") or {}
                if opts.get("type") != "tmpfs" or opts.get("device") != "tmpfs":
                    out.append(Finding(p, find_line(text, rf"^  {re.escape(vname)}:"), "privacy.tmpfs",
                                       f"volume {vname} must be tmpfs", key=vname))
        if "hf-cleanup" in services and len(services) > 1 and services["hf-cleanup"].get("profiles") != ["maintenance"]:
            out.append(Finding(p, find_line(text, r"^  hf-cleanup:"), "privacy.maintenance",
                               "hf-cleanup must sit behind the maintenance profile", key="hf-cleanup"))
    rendered_ok = {p for p in head.deployables() + list(ctx.templates) if ctx.rendered(p) is not None}
    for (f, s), entry in sorted(allow.items()):
        if (f, s) not in used and (f in rendered_ok or not head.has(f)):
            out.append(Finding(SERVICES, 0, "privacy.host-access", f"entry for {f} {s} matches no service",
                               key=f"{f}:{s}"))
    for name, e in sorted(env.items()):
        if not UPPER_SNAKE.match(name):
            out.append(Finding(SERVICES, 0, "privacy.var", f"env name {name} must be UPPER_SNAKE", key=name))
        # Every deploy-time variable states whether it can send data out; no name pattern decides it.
        if not isinstance(e.get("egress"), bool):
            out.append(Finding(SERVICES, 0, "privacy.egress", f"env {name}: set egress to true or false", key=name))
        if "dummy" in e and not DUMMY.match(str(e["dummy"])):
            out.append(Finding(SERVICES, 0, "privacy.dummy", f"env {name}: dummy must match {DUMMY.pattern}",
                               key=name))
    return out


def check_engine(p, line, key, svc, serving):
    """Logging checks; `serving` engines must also carry the no-log flag."""
    out, toks = [], service_tokens(svc)
    kind = engine_kind(key, svc)
    levels = flag_values(toks, "--log-requests-level")
    if any(v != "0" for v in levels):
        out.append(Finding(p, line, "privacy.log-enabled", f"engine {key}: request logging level must be 0",
                           key=f"{key}:--log-requests-level={','.join(levels)}"))
    elif serving and kind == "sglang" and len(levels) != 1:
        out.append(Finding(p, line, "privacy.no-log-flag", f"engine {key}: set --log-requests-level 0 exactly once",
                           key=f"{key}:{svc.get('image')}"))
    # Exception keys bind the image: the logging default is a property of the pinned engine.
    if serving and kind == "vllm" and toks.count(VLLM_NO_LOG) != 1:
        out.append(Finding(p, line, "privacy.no-log-flag", f"engine {key}: set {VLLM_NO_LOG} exactly once",
                           key=f"{key}:{svc.get('image')}"))
    for i, t in enumerate(toks):
        m = BANNED_FLAG.match(t)
        if m:
            out.append(Finding(p, line, "privacy.log-enabled", f"engine {key}: flag {m.group(1)} is not allowed",
                               key=f"{key}:{t}"))
    for v in flag_values(toks, "--log-level"):
        if v.lower() == "debug":
            out.append(Finding(p, line, "privacy.log-enabled", f"engine {key}: debug logging is not allowed",
                               key=f"{key}:--log-level={v}"))
    envd = svc.get("environment") or {}
    for name, val in sorted(envd.items()):
        if BANNED_ENV.search(name) or (name.endswith("_LOG_LEVEL") and str(val).upper() == "DEBUG"):
            out.append(Finding(p, line, "privacy.env", f"engine {key}: env {name} is not allowed",
                               key=f"{key}:{name}={val}"))
    return out


def var_refs(text):
    """(offset, name, op, rest) for each compose interpolation; `$$` is a literal dollar."""
    out = []
    for m in VAR.finditer(text):
        if m.group(1) == "$":
            continue
        body = m.group(2) if m.group(2) is not None else m.group(3)
        om = OPS.match(body)
        if not om:
            out.append((m.start(), body, "bad", ""))
            continue
        out.append((m.start(), om.group(1), om.group(2) or "", om.group(3) or ""))
        if om.group(3):
            out += [(m.start(), n, o, r) for _, n, o, r in var_refs(om.group(3))]
    return out


def check_vars(p, text, env):
    out = []
    lines, kinds, paths = key_paths(text)
    offs, total = [], 0
    for line in lines:
        offs.append(total)
        total += len(line) + 1
    for off, name, op, rest in var_refs(text):
        n = bisect.bisect_right(offs, off)
        # Exception keys bind the variable to the line that uses it, default included.
        bound = f"{name}:{lines[n - 1].strip()}"
        if op == "bad" or not UPPER_SNAKE.match(name):
            out.append(Finding(p, n, "privacy.var", "variable name must be UPPER_SNAKE", key=bound))
            continue
        if name not in env:
            out.append(Finding(p, n, "privacy.var", f"${{{name}}} is not listed under env in {SERVICES}", key=bound))
            continue
        if SECRET_SUFFIX.search(name):
            if op not in ("", ":?", "?"):
                out.append(Finding(p, n, "privacy.secret", f"secret {name} must not have a default", key=bound))
            elif "environment" not in paths[n - 1] or kinds[n - 1] == "scalar":
                out.append(Finding(p, n, "privacy.secret", f"secret {name} is only allowed under environment:",
                                   key=bound))
        elif op in (":-", "-", ":+", "+") and not env[name].get("default_ok"):
            out.append(Finding(p, n, "privacy.default", f"{name} has a default; set default_ok", key=bound))
    return out


P = "prod/m-sglang-tp1.yaml"
V = "prod/m-vllm-tp1.yaml"
SG = {"container_name": "model-sg-a", "command": ["sglang", "serve", "--log-requests-level", "0"],
      "ipc": "host"}
VL = {"container_name": "model-vllm-a", "command": ["org/m", VLLM_NO_LOG]}
ALLOW = ('{"services": [{"file": "' + P + '", "service": "model-sg-a", "items": ["ipc:host"]}, '
         '{"file": "' + P + '", "service": "aux-a", "role": "aux", "items": []}], "env": ['
         '{"name": "HF_TOKEN", "egress": false}, {"name": "PORT", "egress": false, "default_ok": true, "dummy": "8000"}, '
         '{"name": "UP_URL", "egress": true}]}')
AUX = {"container_name": "aux-a", "image": "nearaidev/sglang@sha256:" + "0" * 64, "command": ["python3", "-m", "x"]}
RAW = ("services:\n  model-sg-a:\n    environment:\n      HF_TOKEN: ${HF_TOKEN:?set}\n"
       "      UP: ${UP_URL}\n    command: sh -c 'echo $${HOME} ${PORT:-8000}'\n")
FILES = {P: RAW, V: "services:\n  model-vllm-a:\n    image: x\n", "ci/services.allow.json": ALLOW}
REND = {P: {"services": {"model-sg-a": SG, "aux-a": AUX}}, V: {"services": {"model-vllm-a": VL}}}


def _sg(**kw):
    return dict(REND, **{P: {"services": {"model-sg-a": dict(SG, **kw), "aux-a": AUX}}})


def _aux(**kw):
    return dict(REND, **{P: {"services": {"model-sg-a": SG, "aux-a": dict(AUX, **kw)}}})


SENTINEL = "ghp_" + "Q7" * 18
CASES = [
    ("engines with their no-log flags", None, FILES, FILES, REND, None),
    ("SGLang without the flag", "privacy.no-log-flag", FILES, FILES, _sg(command=["sglang", "serve"]), None),
    ("vLLM without the flag", "privacy.no-log-flag", FILES, FILES,
     dict(REND, **{V: {"services": {"model-vllm-a": dict(VL, command=["org/m"])}}}), None),
    ("log level 2", "privacy.log-enabled", FILES, FILES,
     _sg(command=["sglang", "serve", "--log-requests-level", "0", "--log-requests-level", "2"]), None),
    ("trace flag in sh -c", "privacy.log-enabled", FILES, FILES,
     _sg(entrypoint=["sh", "-c"], command=["exec sglang serve --log-requests-level 0 --enable-trace"]), None),
    ("dump env", "privacy.env", FILES, FILES, _sg(environment={"SGLANG_DUMP_REQUESTS": "1"}), None),
    ("aux on an engine image with log level 2", "privacy.log-enabled", FILES, FILES,
     _aux(command=["python3", "-m", "sglang.launch_server", "--log-requests-level", "2"]), None),
    ("aux on an engine image with a dump env", "privacy.env", FILES, FILES,
     _aux(environment={"SGLANG_DUMP_REQUESTS": "1"}), None),
    ("GPU service without role", "privacy.role", FILES, FILES,
     dict(REND, **{P: {"services": {"model-sg-a": SG, "aux-a": AUX, "x": {
         "container_name": "x", "deploy": {"resources": {"reservations": {"devices": [{"capabilities": ["gpu"]}]}}}}}}}),
     None),
    ("privileged service not listed", "privacy.host-access", FILES, FILES,
     dict(REND, **{P: {"services": {"model-sg-a": SG, "aux-a": AUX, "x": {"container_name": "x", "privileged": True}}}}), None),
    ("listed access unused", "privacy.host-access", FILES, FILES, _sg(ipc=None), None),
    ("unlisted variable", "privacy.var", dict(FILES, **{P: RAW + "    labels:\n      a: ${FOO_URL:-}\n"}), FILES,
     REND, None),
    ("secret with a default", "privacy.secret",
     dict(FILES, **{P: RAW.replace("${HF_TOKEN:?set}", "${HF_TOKEN:-" + SENTINEL + "}")}), FILES, REND, None),
    ("secret in a command", "privacy.secret", dict(FILES, **{P: RAW + "    entrypoint: echo ${HF_TOKEN}\n"}), FILES,
     REND, None),
    ("default without default_ok", "privacy.default", dict(FILES, **{P: RAW.replace("${UP_URL}", "${UP_URL:-x}")}),
     FILES, REND, None),
    ("variable without an egress flag", "privacy.egress",
     dict(FILES, **{"ci/services.allow.json": ALLOW.replace('"name": "HF_TOKEN", "egress": false', '"name": "HF_TOKEN"')}),
     FILES, REND, None),
    ("bad dummy", "privacy.dummy", dict(FILES, **{"ci/services.allow.json": ALLOW.replace('"8000"', '"x y"')}), FILES,
     REND, None),
    ("ghost key volume on disk", "privacy.tmpfs", FILES, FILES,
     dict(REND, **{P: {"services": {"model-sg-a": SG, "aux-a": AUX}, "volumes": {"ghost": {"driver_opts": {}}}}}), None),
    ("hf-cleanup without profile", "privacy.maintenance", FILES, FILES,
     dict(REND, **{P: {"services": {"model-sg-a": SG, "aux-a": AUX, "hf-cleanup": {"container_name": "hf-cleanup"}}}}), None),
]
