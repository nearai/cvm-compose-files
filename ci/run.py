#!/usr/bin/env python3
"""Run the ci/ guards.

  run.py [--base <rev>]   all guards; diff rules need a base
  run.py --self-test      every rule against built-in red and green inputs
  run.py --render <file>  rendered compose JSON of one file
  run.py --list-deployables  every deployable compose file, one per line
  run.py --noop <rev>     rendered diff of every deployable between <rev> and the working tree
  run.py --write-expected rewrite ci/data/expected.json from the working tree

Supported Python: 3.12, the python3 of the ubuntu-24.04 runner; CI fails on any other version. Stdlib only.
"""

import argparse
import difflib
import json
import os
import re
import sys

sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), d) for d in ("lib", "checks")]

import claude
import frozen
import layout
import pins
import privacy
import public
import status
from common import (
    EXCEPTIONS, EXPECTED, IMAGES, ROOT, SERVICES, TEMPLATES, Finding, RenderError, Renderer, Tree, env_lines, git,
    load_json, render_dir, sha256,
)

GUARDS = [layout, status, pins, privacy, public, frozen, claude]
RULES = ["run.render", "exceptions.format", "exceptions.unused"]
ALL_RULES = RULES + [r for g in GUARDS for r in g.RULES]
ENTRY_KEYS = {"rule", "path", "sha256", "reason", "remove_when"}
# Rules that guard the agent setup and the waivers themselves.
UNWAIVABLE_PREFIX = ("exceptions.", "claude.")
UNWAIVABLE = {"layout.claude-md", "layout.agent-file"}


class Context:
    def __init__(self, head, base=None, rendered=None, rendered_base=None, name_status=(), commits=(), pr=None):
        self.head, self.base = head, base
        self.bootstrap = base is not None and not base.has("ci/lib/common.py")
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
        if e["rule"] not in ALL_RULES:
            out.append(Finding(EXCEPTIONS, 0, "exceptions.format", f"{where}: unknown rule"))
            continue
        if e["rule"].startswith(UNWAIVABLE_PREFIX) or e["rule"] in UNWAIVABLE:
            out.append(Finding(EXCEPTIONS, 0, "exceptions.format", f"{where}: rule cannot be waived"))
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


# ---------------------------------------------------------------- expected findings

def _command(svc):
    cmd = svc.get("command")
    return list(cmd) if isinstance(cmd, list) else [] if cmd is None else [cmd]


# Each probe breaks one service; the rules that fire for it are recorded per service.
PROBES = {
    "log": lambda s: dict(s, command=_command(s) + ["--log-requests-level", "2", "--enable-log-requests"]),
    "env": lambda s: dict(s, environment=dict(s.get("environment") or {}, SGLANG_DUMP_REQUESTS="1")),
    "privileged": lambda s: dict(s, privileged=True),
    "tag": lambda s: dict(s, image="nginx:latest"),
}
PROBE_GUARDS = [privacy, pins]


def _keys(findings):
    return {(f.rule, f.path, f.key) for f in findings}


def probe_lines(head, rendered):
    """`path service-hash probe: rules` for every service of every rendered file."""
    out = []
    for p in head.deployables() + [t for t in TEMPLATES if head.has(t)]:
        try:
            doc = rendered(p)
        except RenderError:
            continue
        files = {x: head.data(x) for x in (p, IMAGES) if head.has(x)}
        sa = load_json(head, SERVICES, {})
        files[SERVICES] = json.dumps(dict(sa, services=[e for e in sa.get("services", []) if e.get("file") == p]))

        def run(d):
            ctx = Context(Tree(files=files), None, lambda path: d if path == p else None)
            return _keys(f for g in PROBE_GUARDS for f in g.check(ctx))
        clean = run(doc)
        for key, svc in sorted(doc.get("services", {}).items()):
            for name, probe in PROBES.items():
                d = dict(doc, services=dict(doc["services"], **{key: probe(svc)}))
                rules = sorted({r for r, _, _ in run(d) - clean})
                out.append(f"{p} {sha256(key)[:16]} {name}: {' '.join(rules) or '-'}")
    return sorted(out)


def expected_doc(head, rendered):
    """Raw findings (before exceptions) of a head-only run, plus the probe results."""
    ctx = Context(head, None, rendered)
    findings = [f for g in GUARDS for f in g.check(ctx)]
    lines = sorted({f"{f.rule} {f.path} {sha256(f.key)}" for f in findings})
    return {"findings": lines, "probes": probe_lines(head, rendered)}


def check_expected(head, doc):
    """Findings for every line that differs from ci/data/expected.json. Exceptions never apply."""
    want = load_json(head, EXPECTED, None)
    if not isinstance(want, dict):
        return [Finding(EXPECTED, 0, "run.expected", "missing or not a JSON object; run --write-expected")]
    out = []
    for part in ("findings", "probes"):
        have, exp = set(doc[part]), set(want.get(part) or [])
        for line in sorted(have - exp):
            out.append(Finding(EXPECTED, 0, "run.expected", f"{part}: new: {line}"))
        for line in sorted(exp - have):
            out.append(Finding(EXPECTED, 0, "run.expected", f"{part}: gone: {line}"))
    return out


def write_expected(doc):
    with open(f"{ROOT}/{EXPECTED}", "w") as f:
        f.write(json.dumps(doc, indent=2) + "\n")


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
    ("exception matches", None, {"prod/a.bin": "x", "ci/data/exceptions.json": json.dumps(GOOD_EX)}, None),
    ("render failure", "run.render", {"prod/a.yaml": "x"}, None, {"prod/a.yaml": "ERROR"}),
    ("exception with a glob", "exceptions.format",
     {"prod/a.bin": "x", "ci/data/exceptions.json": json.dumps(GOOD_EX + [dict(GOOD_EX[0], path="prod/*.bin")])}, None),
    ("waiver of a claude rule", "exceptions.format",
     {"prod/a.bin": "x", ".claude/settings.json": "{}", "ci/data/exceptions.json": json.dumps(
         GOOD_EX + [dict(GOOD_EX[0], rule="claude.settings", path=".claude/settings.json",
                         sha256=sha256(".claude/settings.json"))])}, None),
    ("waiver of layout.claude-md", "exceptions.format",
     {"prod/a.bin": "x", "CLAUDE.md": "x", "ci/data/exceptions.json": json.dumps(
         GOOD_EX + [dict(GOOD_EX[0], rule="layout.claude-md", path="CLAUDE.md", sha256=sha256("CLAUDE.md"))])}, None),
    ("unused exception", "exceptions.unused",
     {"prod/a.yaml": "x", "ci/data/exceptions.json": json.dumps([dict(GOOD_EX[0], path="prod/a.yaml")])}, None),
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
    sentinel = privacy.SENTINEL
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
    failures += expected_self_test()
    total += 3
    for f in failures:
        print("FAIL", f)
    if failures or total == 0:
        return 1
    print(f"OK ({total})")
    return 0


def expected_self_test():
    p = "prod/m-sglang-tp1.yaml"
    svc = {"container_name": "model-sg-a", "image": "x@sha256:" + "0" * 64,
           "command": ["sglang", "serve", "--log-requests-level", "0"]}
    files = {p: "services:\n  model-sg-a:\n    image: x\n", SERVICES: '{"services": [], "env": []}'}
    head = Tree(files=files)
    doc = expected_doc(head, lambda path: {"services": {"model-sg-a": svc}} if path == p else None)
    fails = []
    if not any(l.endswith(" log: privacy.log-enabled") for l in doc["probes"]):
        fails.append("expected: the log probe does not fire privacy.log-enabled")
    good = Tree(files=dict(files, **{EXPECTED: json.dumps(doc)}))
    if check_expected(good, doc):
        fails.append("expected: matching file is not green")
    bad = dict(doc, probes=doc["probes"][1:])
    if {f.rule for f in check_expected(Tree(files=dict(files, **{EXPECTED: json.dumps(bad)})), doc)} != {"run.expected"}:
        fails.append("expected: a missing probe line is not red")
    return fails


# ---------------------------------------------------------------- real runs

def resolve(rev):
    return git("rev-parse", "--verify", f"{rev}^{{commit}}").decode().strip()


def real_ctx(base_rev):
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
        log = git("log", "--format=%H%x00%P%x00%B%x01", f"{base_rev}..HEAD").decode()
        for x in log.split("\x01"):
            if x.strip():
                sha, parents, msg = x.strip("\n").split("\0", 2)
                commits.append((sha, msg, len(parents.split()) > 1))
    ctx = Context(head, base, Renderer(head, sa), Renderer(base, load_json(base, SERVICES, {}) or sa) if base else None,
                  name_status, commits, public.env_pr())
    return ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--base")
    ap.add_argument("--render")
    ap.add_argument("--list-deployables", action="store_true")
    ap.add_argument("--noop")
    ap.add_argument("--write-expected", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if a.list_deployables:
        print("\n".join(Tree().deployables()))
        return 0
    if a.render:
        sa = load_json(Tree(), SERVICES, {})
        print(json.dumps(render_dir(ROOT, a.render, env_lines(sa)), indent=2, sort_keys=True))
        return 0
    if a.noop:
        return noop(a.noop)
    if a.write_expected:
        head = Tree()
        write_expected(expected_doc(head, Renderer(head, load_json(head, SERVICES, {}))))
        return 0
    ctx = real_ctx(a.base)
    if ctx.base is None:
        for g in GUARDS:
            for r in g.DIFF_RULES:
                print(f"SKIP (no base): {r}")
    elif ctx.bootstrap:
        print("bootstrap mode: ci/lib/common.py is absent at base")
    findings = run_guards(ctx)
    if not ctx.render_errors:
        findings += check_expected(ctx.head, expected_doc(ctx.head, ctx._rendered))
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
