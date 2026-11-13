"""names: a deployable's path follows from what it runs.

Serving file  prod/<served repo>-<engine>-<topology>[-<tier>].yaml   one served model, one engine kind
Pack          prod/pack-<purpose>.yaml                               several served models or engine kinds
Utility       prod/util-<verb>-<object>.yaml                         no engine
Paths in ci/data/legacy.txt keep their names and labels until a standard-named file replaces them.
"""

import json
import re

from common import Finding, engine_kind, engines, flag_values, legacy_lines, served_names, service_tokens, under_legacy

RULES = ["names.kind", "names.serving", "names.tier", "names.labels"]
DIFF_RULES = []

TOPO_ORDER = re.compile(r"^(dep|dp|tp)(\d+)(?:x(\d+))?$")
TAG = re.compile(r'"deployment:([^"]*)"')


def _int(toks, flags):
    for f in flags:
        v = flag_values(toks, f)
        if v and v[0].isdigit():
            return int(v[0])
    return 1


def shape(path, rendered, services_allow):
    """(served repos, engine kinds, topologies) of a rendered deployable."""
    groups = {}
    for key in engines(path, rendered, services_allow):
        svc = rendered["services"][key]
        kind, toks = engine_kind(key, svc), service_tokens(svc)
        tp = _int(toks, ("--tp-size", "--tp", "--tensor-parallel-size"))
        dp = _int(toks, ("--dp-size", "--dp", "--data-parallel-size"))
        ep = "--enable-expert-parallel" in toks or "--enable-dp-attention" in toks
        unit = (f"dep{dp}" if ep else f"dp{dp}") if dp > 1 else f"tp{tp}"
        span = dp if flag_values(toks, "--data-parallel-rank") else 1
        g = groups.setdefault((kind, unit, tuple(sorted(served_names(kind, toks)))), [0, span])
        g[0] += 1
    repos, kinds, topos = set(), set(), set()
    for (kind, unit, served), (n, span) in groups.items():
        reps = max(n // span, 1)
        topos.add(unit + (f"x{reps}" if reps > 1 else ""))
        kinds.add(kind)
        repos.update(s.split("/")[-1].lower() for s in served)
    return repos, kinds, topos


def topo_key(t):
    m = TOPO_ORDER.match(t)
    return (int(m.group(2)), int(m.group(3) or 1), m.group(1)) if m else (0, 0, t)


def prefix(repos, kinds, topos):
    return f"{next(iter(repos))}-{next(iter(kinds))}-" + "_".join(sorted(topos, key=topo_key))


def label_values(rendered):
    """Every deployment name a label or log tag in the file uses."""
    out = set()
    for svc in rendered.get("services", {}).values():
        labels = svc.get("labels") or {}
        if isinstance(labels, list):
            labels = dict(x.split("=", 1) for x in labels if isinstance(x, str) and "=" in x)
        for k, v in labels.items():
            if k in ("deployment", "nearai.otel.deployment"):
                out.add(str(v))
            out.update(TAG.findall(json.dumps(v)) if not isinstance(v, str) else TAG.findall(v))
    return out


def check(ctx):
    out, head = [], ctx.head
    legacy = legacy_lines(head)
    prefixes = {}
    for p in head.deployables():
        if under_legacy(p, legacy):
            continue
        r = ctx.rendered(p)
        if r is None:
            continue
        stem = p[len("prod/"):-len(".yaml")]
        repos, kinds, topos = shape(p, r, ctx.services_allow)
        single = len(repos) == 1 and len(kinds) == 1
        if not kinds:
            if not stem.startswith("util-"):
                out.append(Finding(p, 0, "names.kind", "a file without an engine is prod/util-<verb>-<object>.yaml"))
        elif not single:
            if not stem.startswith("pack-"):
                out.append(Finding(p, 0, "names.kind", "a file with several served models or engines is prod/pack-<purpose>.yaml"))
        else:
            want = prefix(repos, kinds, topos)
            if stem.startswith(("pack-", "util-")):
                out.append(Finding(p, 0, "names.kind", f"one served model on one engine kind: name it prod/{want}.yaml"))
            elif stem != want and not stem.startswith(want + "-"):
                out.append(Finding(p, 0, "names.serving", f"name must be prod/{want}[-<tier>].yaml"))
            else:
                prefixes.setdefault(want, []).append((p, stem[len(want) + 1:] if stem != want else ""))
        top = re.search(r"(?m)^name:\s*(\S+)", head.text(p))
        if stem.startswith("util-") and (not top or top.group(1).strip("'\"") != stem):
            out.append(Finding(p, 1, "names.labels", "a utility file sets top-level name: to its file stem"))
        if not stem.startswith("util-") and top:
            out.append(Finding(p, 1, "names.labels", "serving and pack files set no top-level name:"))
        wrong = sorted(v for v in label_values(r) if v != stem)
        if wrong:
            out.append(Finding(p, 0, "names.labels", "deployment labels and log tags must equal the file stem"))
    for want, members in prefixes.items():
        tiers = [t for _, t in members]
        if len(members) == 1 and tiers[0]:
            out.append(Finding(members[0][0], 0, "names.tier", "a tier only tells apart files with the same name"))
        elif len(members) > 1 and (not all(tiers) or len(set(tiers)) != len(tiers)):
            out.append(Finding(members[0][0], 0, "names.tier", f"files named {want} each need a distinct tier"))
    return out


def _svc(served="z-ai/glm-5.3-flash", kind="sglang", tp=2, dep="glm-5.3-flash-sglang-tp2x2"):
    cmd = ["sglang", "serve", "--tp", str(tp), "--served-model-name", served] if kind == "sglang" else \
        ["vllm", "serve", "--tensor-parallel-size", str(tp), "--served-model-name", served]
    return {"command": cmd, "labels": {"deployment": dep}}


def _r(**svcs):
    return {"services": svcs}


S = "prod/glm-5.3-flash-sglang-tp2x2.yaml"
GOOD = {S: "services:\n"}
REND = {S: _r(**{"model-sg-a": _svc(), "model-sg-b": _svc()})}
CASES = [
    ("serving name from contents", None, GOOD, None, REND),
    ("legacy file keeps its name", None, {"prod/Old.yaml": "services:\n", "ci/data/legacy.txt": "prod/Old.yaml\n"},
     None, {"prod/Old.yaml": _r(**{"model-sg-a": _svc(dep="old")})}),
    ("pack with two models", None, {"prod/pack-x.yaml": "services:\n"}, None,
     {"prod/pack-x.yaml": _r(**{"model-sg-a": _svc(dep="pack-x"), "model-vllm-b": _svc("Qwen/Q", "vllm", 1, "pack-x")})}),
    ("utility", None, {"prod/util-do-x.yaml": "name: util-do-x\nservices:\n"}, None,
     {"prod/util-do-x.yaml": _r(**{"x": {"labels": {}}})}),
    ("two tiers", None, {"prod/glm-5.3-flash-sglang-tp2x2-base.yaml": "s\n", "prod/glm-5.3-flash-sglang-tp2x2-long.yaml": "s\n"},
     None, {"prod/glm-5.3-flash-sglang-tp2x2-base.yaml": _r(**{"model-sg-a": _svc(dep="glm-5.3-flash-sglang-tp2x2-base"),
                                                               "model-sg-b": _svc(dep="glm-5.3-flash-sglang-tp2x2-base")}),
            "prod/glm-5.3-flash-sglang-tp2x2-long.yaml": _r(**{"model-sg-a": _svc(dep="glm-5.3-flash-sglang-tp2x2-long"),
                                                               "model-sg-b": _svc(dep="glm-5.3-flash-sglang-tp2x2-long")})}),
    ("wrong topology in name", "names.serving", {"prod/glm-5.3-flash-sglang-tp4.yaml": "s\n"}, None,
     {"prod/glm-5.3-flash-sglang-tp4.yaml": _r(**{"model-sg-a": _svc(dep="glm-5.3-flash-sglang-tp4"),
                                                  "model-sg-b": _svc(dep="glm-5.3-flash-sglang-tp4")})}),
    ("single model named as a pack", "names.kind", {"prod/pack-glm.yaml": "s\n"}, None,
     {"prod/pack-glm.yaml": _r(**{"model-sg-a": _svc(dep="pack-glm")})}),
    ("engine in a utility file", "names.kind", {"prod/util-do-x.yaml": "name: util-do-x\n"}, None,
     {"prod/util-do-x.yaml": _r(**{"model-sg-a": _svc(dep="util-do-x")})}),
    ("lone tier", "names.tier", {"prod/glm-5.3-flash-sglang-tp2x2-long.yaml": "s\n"}, None,
     {"prod/glm-5.3-flash-sglang-tp2x2-long.yaml": _r(**{"model-sg-a": _svc(dep="glm-5.3-flash-sglang-tp2x2-long"),
                                                         "model-sg-b": _svc(dep="glm-5.3-flash-sglang-tp2x2-long")})}),
    ("label differs from stem", "names.labels", GOOD, None, {S: _r(**{"model-sg-a": _svc(), "model-sg-b": _svc(dep="x")})}),
    ("pack sets a project name", "names.labels", {"prod/pack-x.yaml": "name: x\n"}, None,
     {"prod/pack-x.yaml": _r(**{"model-sg-a": _svc(dep="pack-x"), "model-vllm-b": _svc("Qwen/Q", "vllm", 1, "pack-x")})}),
]
