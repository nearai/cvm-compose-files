"""frozen: build inputs of published dirs, adapters and existing publish workflows are only ever deleted.

A Markdown file in a build dir that neither the build nor a publish workflow reads may be edited
(`editable_doc`); the public-text rules scan it.
"""

import re

from common import Finding, editable_doc

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
            if status == "M" and editable_doc(base, p) and editable_doc(head, p):
                continue
            out.append(Finding(p, 0, "frozen.edit", f"{unit} is frozen: only whole deletion is allowed"))
    return out


BASE = {"docker/sglang-x-r1/Dockerfile": "FROM a\nCOPY p.diff /tmp/\n", "docker/sglang-x-r1/p.diff": "x\n",
        "docker/sglang-x-r1/README.md": "x\n", "docker/sglang-x-r1/NOTES.md": "x\n",
        "docker/sglang-x-r1/SHA256SUMS": "0" * 64 + "  NOTES.md\n",
        ".github/workflows/publish-sglang-x-r1.yaml": "env:\n  CTX: docker/sglang-x-r1\n", "adapters/a/w.json": "{}\n",
        "docker/sglang-x-r1/__pycache__/m.pyc": "x"}
NOCOMMON = dict(BASE)
WITHCOMMON = dict(BASE, **{"ci/lib/common.py": "x\n"})


def _head(*drop, **add):
    files = {k: v for k, v in WITHCOMMON.items() if k not in drop}
    files.update(add)
    return files


X = "docker/sglang-x-r1/"
WF = ".github/workflows/publish-sglang-x-r1.yaml"
CASES = [
    ("whole dir deleted", None, {k: v for k, v in WITHCOMMON.items() if not k.startswith(X)}, WITHCOMMON, None, None,
     {"name_status": [("D", k) for k in WITHCOMMON if k.startswith(X)]}),
    ("unread doc edited", None, _head(**{X + "README.md": "y\n"}), WITHCOMMON, None, None,
     {"name_status": [("M", X + "README.md")]}),
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
    ("doc listed in SHA256SUMS edited", "frozen.edit", _head(**{X + "NOTES.md": "y\n"}), WITHCOMMON, None, None,
     {"name_status": [("M", X + "NOTES.md")]}),
    ("doc under COPY . edited", "frozen.edit",
     _head(**{X + "Dockerfile": "FROM a\nCOPY . /tmp/\n", X + "README.md": "y\n"}),
     dict(WITHCOMMON, **{X + "Dockerfile": "FROM a\nCOPY . /tmp/\n"}), None, None,
     {"name_status": [("M", X + "README.md")]}),
    ("doc named by the publish workflow edited", "frozen.edit",
     _head(**{WF: WITHCOMMON[WF] + "  DOC: README.md\n", X + "README.md": "y\n"}),
     dict(WITHCOMMON, **{WF: WITHCOMMON[WF] + "  DOC: README.md\n"}), None, None,
     {"name_status": [("M", X + "README.md")]}),
    ("doc made a build input in the same change", "frozen.edit",
     _head(**{X + "Dockerfile": "FROM a\nCOPY . /tmp/\n", X + "README.md": "y\n"}), WITHCOMMON, None, None,
     {"name_status": [("M", X + "Dockerfile"), ("M", X + "README.md")]}),
    ("adapter edited", "frozen.edit", WITHCOMMON, WITHCOMMON, None, None,
     {"name_status": [("M", "adapters/a/w.json")]}),
]
