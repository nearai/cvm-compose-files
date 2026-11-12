"""frozen: published build dirs, adapters and existing publish workflows are only ever deleted."""

import re

from common import Finding

RULES = ["frozen.edit"]
DIFF_RULES = ["frozen.edit"]

PUBLISH = re.compile(r"^\.github/workflows/publish-[^/]+\.yaml$")
BYTECODE = re.compile(r"(\.pyc$|(^|/)__pycache__/)")


def frozen_unit(path, base):
    """The frozen unit a base path belongs to, or None."""
    m = re.match(r"^(docker/[^/]+/)", path)
    if m and any(p.startswith(m.group(1)) for p in base.paths):
        return m.group(1)
    if path.startswith("adapters/"):
        return "adapters/"
    if PUBLISH.match(path) and base.has(path):
        return path
    return None


def check(ctx):
    out, head, base = [], ctx.head, ctx.base
    if base is None:
        return out
    for entry in ctx.name_status:
        status, paths = entry[0], entry[1:]
        for p in paths:
            unit = frozen_unit(p, base)
            if unit is None:
                continue
            if status == "D":
                if BYTECODE.search(p) or unit == p:
                    continue
                if any(h.startswith(unit) for h in head.paths):
                    out.append(Finding(p, 0, "frozen.edit", f"delete {unit} whole or not at all"))
                continue
            if status in ("M", "T") and ctx.bootstrap and unit == p and PUBLISH.match(p):
                continue
            out.append(Finding(p, 0, "frozen.edit", f"{unit} is frozen: only whole deletion is allowed"))
    return out


BASE = {"docker/sglang-x-r1/Dockerfile": "FROM a\n", "docker/sglang-x-r1/p.diff": "x\n",
        ".github/workflows/publish-sglang-x-r1.yaml": "x\n", "adapters/a/w.json": "{}\n",
        "docker/sglang-x-r1/__pycache__/m.pyc": "x"}
NOCOMMON = dict(BASE)
WITHCOMMON = dict(BASE, **{"ci/lib/common.py": "x\n"})


def _head(*drop, **add):
    files = {k: v for k, v in WITHCOMMON.items() if k not in drop}
    files.update(add)
    return files


CASES = [
    ("whole dir deleted", None, _head("docker/sglang-x-r1/Dockerfile", "docker/sglang-x-r1/p.diff",
                                      "docker/sglang-x-r1/__pycache__/m.pyc"), WITHCOMMON, None, None,
     {"name_status": [("D", "docker/sglang-x-r1/Dockerfile"), ("D", "docker/sglang-x-r1/p.diff"),
                      ("D", "docker/sglang-x-r1/__pycache__/m.pyc")]}),
    ("bytecode deleted alone", None, _head("docker/sglang-x-r1/__pycache__/m.pyc"), WITHCOMMON, None, None,
     {"name_status": [("D", "docker/sglang-x-r1/__pycache__/m.pyc")]}),
    ("publish workflow edited in bootstrap mode", None, BASE, NOCOMMON, None, None,
     {"name_status": [("M", ".github/workflows/publish-sglang-x-r1.yaml")]}),
    ("diff rule skipped without a base", None, BASE, None, None, None, {"name_status": []}),
    ("build dir edited", "frozen.edit", WITHCOMMON, WITHCOMMON, None, None,
     {"name_status": [("M", "docker/sglang-x-r1/p.diff")]}),
    ("file added to a build dir", "frozen.edit", _head(**{"docker/sglang-x-r1/q.diff": "y\n"}), WITHCOMMON, None,
     None, {"name_status": [("A", "docker/sglang-x-r1/q.diff")]}),
    ("partial delete", "frozen.edit", _head("docker/sglang-x-r1/p.diff"), WITHCOMMON, None, None,
     {"name_status": [("D", "docker/sglang-x-r1/p.diff")]}),
    ("publish workflow edited after bootstrap", "frozen.edit", WITHCOMMON, WITHCOMMON, None, None,
     {"name_status": [("M", ".github/workflows/publish-sglang-x-r1.yaml")]}),
    ("adapter edited", "frozen.edit", WITHCOMMON, WITHCOMMON, None, None,
     {"name_status": [("M", "adapters/a/w.json")]}),
]
