#!/usr/bin/env python3
"""Run the ci/ guards.

  run.py [--base <rev>]   all guards; diff rules need a base
  run.py --self-test      every rule against built-in red and green inputs
  run.py --render <file>  rendered compose JSON of one file
  run.py --noop <rev>     rendered diff of every deployable between <rev> and the working tree
"""

import argparse
import difflib
import json
import re
import sys

import check_frozen
import check_layout
import check_pins
import check_privacy
import check_public
import check_status
from common import (
    EXCEPTIONS, ROOT, SERVICES, TEMPLATES, Finding, RenderError, Renderer, Tree, env_lines, git, load_json,
    render_dir, sha256,
)

GUARDS = [check_layout, check_status, check_pins, check_privacy, check_public, check_frozen]
RULES = ["run.render", "exceptions.format", "exceptions.unused"]
ALL_RULES = RULES + [r for g in GUARDS for r in g.RULES]
ENTRY_KEYS = {"rule", "path", "sha256", "reason", "remove_when"}


class Context:
    def __init__(self, head, base=None, rendered=None, rendered_base=None, name_status=(), commits=(), pr=None):
        self.head, self.base = head, base
        self.bootstrap = base is not None and not base.has("ci/common.py")
        self.services_allow = load_json(head, SERVICES, {})
        self.services_allow_base = load_json(base, SERVICES, {}) if base else {}
        self.templates = [t for t in TEMPLATES if head.has(t)]
        self._rendered, self._rendered_base = rendered, rendered_base
        self.name_status, self.commits, self.pr = list(name_status), list(commits), pr or {}
        self.render_errors = []

    def _get(self, fn, path):
        try:
            return fn(path)
        except RenderError as e:
            self.render_errors.append(Finding(path, 0, "run.render", "compose render failed: " + str(e)[:200]))
            return None

    def rendered(self, path):
        return self._get(self._rendered, path) if self._rendered else None

    def rendered_base(self, path):
        return self._get(self._rendered_base, path) if self._rendered_base else None


def apply_exceptions(ctx, findings):
    out = []
    try:
        entries = load_json(ctx.head, EXCEPTIONS, [])
    except ValueError:
        return findings + [Finding(EXCEPTIONS, 0, "exceptions.format", "not valid JSON")]
    if not isinstance(entries, list):
        return findings + [Finding(EXCEPTIONS, 0, "exceptions.format", "must be a JSON list")]
    valid, seen = {}, set()
    for i, e in enumerate(entries):
        where = f"entry {i + 1}"
        if not isinstance(e, dict) or set(e) != ENTRY_KEYS or not all(isinstance(e[k], str) and e[k] for k in e):
            out.append(Finding(EXCEPTIONS, 0, "exceptions.format", f"{where}: needs exactly {sorted(ENTRY_KEYS)}"))
            continue
        if e["rule"] not in ALL_RULES or e["rule"].startswith("exceptions."):
            out.append(Finding(EXCEPTIONS, 0, "exceptions.format", f"{where}: unknown rule"))
            continue
        if re.search(r"[*?\[\]]", e["path"]) or not ctx.head.has(e["path"]):
            out.append(Finding(EXCEPTIONS, 0, "exceptions.format", f"{where}: path must be one existing file"))
            continue
        if not re.fullmatch(r"[0-9a-f]{64}", e["sha256"]):
            out.append(Finding(EXCEPTIONS, 0, "exceptions.format", f"{where}: sha256 must be 64 hex"))
            continue
        k = (e["rule"], e["path"], e["sha256"])
        if k in seen:
            out.append(Finding(EXCEPTIONS, 0, "exceptions.format", f"{where}: duplicate entry"))
            continue
        seen.add(k)
        valid[k] = i
    used = set()
    for f in findings:
        k = (f.rule, f.path, sha256(f.key))
        if k in valid:
            used.add(k)
        else:
            out.append(f)
    unrendered = {f.path for f in ctx.render_errors}
    for k, i in sorted(valid.items(), key=lambda x: x[1]):
        if k not in used and k[1] not in unrendered:
            out.append(Finding(EXCEPTIONS, 0, "exceptions.unused", f"entry {i + 1} ({k[0]} {k[1]}) matches nothing"))
    return out


def run_guards(ctx, guards=GUARDS):
    findings = []
    for g in guards:
        findings += g.check(ctx)
    seen = set()
    for f in ctx.render_errors:
        if (f.path, f.rule) not in seen:
            seen.add((f.path, f.rule))
            findings.append(f)
    return apply_exceptions(ctx, findings)


# ---------------------------------------------------------------- self-test

def _tree(files):
    return None if files is None else Tree(files=files)


def _fake(rendered):
    if rendered is None:
        return None

    def get(path):
        if path not in rendered:
            return None
        if rendered[path] == "ERROR":
            raise RenderError(path, ["invalid compose file"])
        return rendered[path]
    return get


def case_ctx(case):
    name, rule, files, base = case[:4]
    rendered = case[4] if len(case) > 4 else None
    rendered_base = case[5] if len(case) > 5 else None
    extra = case[6] if len(case) > 6 else {}
    return Context(_tree(files), _tree(base), _fake(rendered), _fake(rendered_base),
                   extra.get("name_status", ()), extra.get("commits", ()), extra.get("pr"))


GOOD_EX = [{"rule": "layout.ext", "path": "prod/a.bin", "sha256": sha256("prod/a.bin"), "reason": "r",
            "remove_when": "w"}]
RUN_CASES = [
    ("exception matches", None, {"prod/a.bin": "x", "ci/exceptions.json": json.dumps(GOOD_EX)}, None),
    ("render failure", "run.render", {"prod/a.yaml": "x"}, None, {"prod/a.yaml": "ERROR"}),
    ("exception with a glob", "exceptions.format",
     {"prod/a.bin": "x", "ci/exceptions.json": json.dumps(GOOD_EX + [dict(GOOD_EX[0], path="prod/*.bin")])}, None),
    ("unused exception", "exceptions.unused",
     {"prod/a.yaml": "x", "ci/exceptions.json": json.dumps([dict(GOOD_EX[0], path="prod/a.yaml")])}, None),
]


class RunGuard:
    RULES = RULES
    DIFF_RULES = []
    CASES = RUN_CASES

    @staticmethod
    def check(ctx):
        out = [Finding(p, 0, "layout.ext", "x") for p in ctx.head.paths if p.endswith(".bin")]
        for p in ctx.head.deployables():
            ctx.rendered(p)
        return out


def self_test():
    sentinel = check_privacy.SENTINEL
    total, failures, red = 0, [], set()
    for g in GUARDS + [RunGuard]:
        greens = 0
        for case in g.CASES:
            name, rule = case[0], case[1]
            ctx = case_ctx(case)
            findings = run_guards(ctx, [g])
            text = "\n".join(str(f) for f in findings)
            total += 1
            got = {f.rule for f in findings}
            if sentinel in text:
                failures.append(f"{g.__name__}: {name}: sentinel printed")
            if rule is None:
                greens += 1
                if findings:
                    failures.append(f"{g.__name__}: {name}: expected green, got:\n    " + text.replace("\n", "\n    "))
            elif got != {rule}:
                failures.append(f"{g.__name__}: {name}: expected only {rule}, got:\n    " +
                                (text.replace("\n", "\n    ") or "nothing"))
            else:
                red.add(rule)
        if not greens:
            failures.append(f"{g.__name__}: no green case")
        for dr in g.DIFF_RULES:
            if not any(c[1] is None and (c[3] is None or not any(k.startswith("ci/") or k == "README.md"
                                                                  for k in c[3])) for c in g.CASES):
                failures.append(f"{g.__name__}: {dr} has no green case without base data")
    for r in ALL_RULES:
        if r not in red:
            failures.append(f"rule {r} has no red case that fires")
    for f in failures:
        print("FAIL", f)
    if failures or total == 0:
        return 1
    print(f"OK ({total})")
    return 0


# ---------------------------------------------------------------- real runs

def resolve(rev):
    return git("rev-parse", "--verify", f"{rev}^{{commit}}").decode().strip()


def real_ctx(base_rev):
    from check_public import env_pr
    head = Tree()
    sa = load_json(head, SERVICES, {})
    base, name_status, commits = None, [], []
    if base_rev:
        base_rev = resolve(base_rev)
        base = Tree(base_rev)
        raw = git("diff", "--name-status", "-M", "-z", base_rev, "HEAD").decode().split("\0")
        i = 0
        while i < len(raw) - 1:
            st = raw[i]
            n = 2 if st[0] in "RC" else 1
            name_status.append((st[0], *raw[i + 1:i + 1 + n]))
            i += 1 + n
        log = git("log", "--format=%H%x00%B%x01", f"{base_rev}..HEAD").decode()
        commits = [tuple(x.strip("\n").split("\0", 1)) for x in log.split("\x01") if x.strip()]
    ctx = Context(head, base, Renderer(head, sa), Renderer(base, load_json(base, SERVICES, {}) or sa) if base else None,
                  name_status, commits, env_pr())
    return ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--base")
    ap.add_argument("--render")
    ap.add_argument("--noop")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if a.render:
        sa = load_json(Tree(), SERVICES, {})
        print(json.dumps(render_dir(ROOT, a.render, env_lines(sa)), indent=2, sort_keys=True))
        return 0
    if a.noop:
        return noop(a.noop)
    ctx = real_ctx(a.base)
    if ctx.base is None:
        for g in GUARDS:
            for r in g.DIFF_RULES:
                print(f"SKIP (no base): {r}")
    elif ctx.bootstrap:
        print("bootstrap mode: ci/common.py is absent at base")
    findings = run_guards(ctx)
    for f in sorted(findings, key=lambda f: (f.path, f.line, f.rule)):
        print(f)
    if findings:
        print(f"FAIL ({len(findings)} findings)")
        return 1
    print("OK")
    return 0


def noop(base_rev):
    head = Tree()
    base = Tree(resolve(base_rev))
    sa = load_json(head, SERVICES, {})
    rh, rb = Renderer(head, sa), Renderer(base, sa)
    paths = sorted(set(head.deployables()) | set(base.deployables()))
    changed = 0
    for p in paths:
        if not head.has(p) or not base.has(p):
            print(f"only in {'head' if head.has(p) else 'base'}: {p}")
            changed += 1
            continue
        a = json.dumps(rb(p), indent=1, sort_keys=True).splitlines()
        b = json.dumps(rh(p), indent=1, sort_keys=True).splitlines()
        diff = list(difflib.unified_diff(a, b, f"base/{p}", f"head/{p}", lineterm=""))
        if diff:
            changed += 1
            print("\n".join(diff))
    return 1 if changed else 0


if __name__ == "__main__":
    sys.exit(main())
