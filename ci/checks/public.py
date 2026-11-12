"""public: nothing internal in files, paths, commit messages or PR text."""

import ipaddress
import json
import os
import re

from common import (
    IP_ALLOW, IP_ALLOW_PREFIX, LIFECYCLE, ORG, ORG_PUBLIC_REPOS, PUBLIC_HOSTS, PUBLIC_OWNERS, RUNNERS, Finding,
    YamlError, is_text, parse_yaml, sha256, yaml_comments,
)

RULES = [
    "public.ip", "public.domain", "public.runner", "public.repo", "public.host", "public.lifecycle", "public.ignore",
    "public.title", "public.comment",
]
DIFF_RULES = []

IPV4 = re.compile(r"(?<![\w.])(\d{1,3}(?:\.\d{1,3}){3})(?![\w.]*\d)")
URL_HOST = re.compile(r"\b[a-z][a-z0-9+.-]*://(?:[^/\s:@'\"]+@)?([A-Za-z0-9.-]+)")
USER_HOST = re.compile(r"\b[\w.+-]+@([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+)")
HOST_PORT = re.compile(r"(?<![\w./-])([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+):\d{1,5}\b")
INTERNAL = re.compile(r"\b[\w-]+(?:\.[\w-]+)*\.(?:internal|local|lan|corp|intra|home\.arpa|ts\.net)\b", re.I)
HOST = re.compile(r"(?i)\b(gpu|cpu)[\s_-]?\d+\b")
GH_REF = re.compile(
    r"(?:github\.com[/:]|raw\.githubusercontent\.com/)([\w.-]+)/([\w.-]+)"
    r"|\buses:\s*['\"]?([\w.-]+)/([\w.-]+)"
    r"|\brepository:\s*['\"]?([\w.-]+)/([\w.-]+)"
    r"|(?<![\w./-])(" + re.escape(ORG) + r")/([\w.-]+)"
)
IGNORE = re.compile("|".join(re.escape(t) for t in (
    "gitleaks" + ":allow", "zizmor" + ": ignore", "yamllint" + " disable", "no" + "qa")), re.I)
TRAILER = re.compile(r"^[A-Za-z]+(?:-[A-Za-z]+)*-by: ", re.I)
TITLE = re.compile(r"^(prod|docker|ci|docs|repo): \S.*$")
COMMENT_BANNED = re.compile(
    r"\b(TODO|FIXME|HACK|XXX)\b|\b\d{4}-\d{2}-\d{2}\b|(?i:\b(tried|previously|used to|canary|rollout|incident)\b)"
    r"|#\d+|\b\d{9,}\b|(?i:generated.*do not)")
COMMENTED_OUT = re.compile(r"^(?:[a-z_][\w.-]*:(?:\s+\S+)?\s*$|-\s+[\w.-]+:|--[a-z][\w-]*)")


def public_host(h):
    h = h.lower().rstrip(".")
    return any(h == p or h.endswith("." + p) for p in PUBLIC_HOSTS)


def text_findings(path, text, rules=RULES):
    out = []
    for n, line in enumerate(text.split("\n"), 1):
        if "public.ip" in rules:
            for ip in IPV4.findall(line):
                try:
                    ipaddress.IPv4Address(ip)
                except ValueError:
                    continue
                if ip not in IP_ALLOW and not ip.startswith(IP_ALLOW_PREFIX):
                    out.append(Finding(path, n, "public.ip", "IP address", key=ip))
        if "public.domain" in rules:
            hosts = set(URL_HOST.findall(line)) | set(USER_HOST.findall(line)) | set(HOST_PORT.findall(line))
            for h in sorted(hosts):
                if "." not in h or not re.search(r"\.[A-Za-z]{2,}$", h) or IPV4.fullmatch(h):
                    continue
                if not public_host(h):
                    out.append(Finding(path, n, "public.domain", "host not in PUBLIC_HOSTS", key=h))
            for m in INTERNAL.finditer(line):
                if m.group(0) not in hosts:
                    out.append(Finding(path, n, "public.domain", "internal domain", key=m.group(0)))
        if "public.host" in rules:
            # Exception keys bind the host to its line, so a waived host cannot appear in new text.
            for m in HOST.finditer(line):
                out.append(Finding(path, n, "public.host", "host name", key=f"{m.group(0)}:{line.strip()}"))
        if "public.repo" in rules:
            for m in GH_REF.finditer(line):
                g = [x for x in m.groups() if x is not None]
                owner, repo = g[0], re.sub(r"\.git$", "", g[1])
                if owner.lower() == ORG.lower():
                    if repo not in ORG_PUBLIC_REPOS:
                        out.append(Finding(path, n, "public.repo", "repository not in ORG_PUBLIC_REPOS",
                                           key=f"{owner}/{repo}"))
                elif owner.lower() not in {o.lower() for o in PUBLIC_OWNERS}:
                    out.append(Finding(path, n, "public.repo", "repository owner not in PUBLIC_OWNERS",
                                       key=f"{owner}/{repo}"))
        if "public.ignore" in rules:
            for m in IGNORE.finditer(line):
                out.append(Finding(path, n, "public.ignore", "inline linter ignore", key=m.group(0).lower()))
    return out


def comment_findings(path, text):
    out, run = [], []
    comments = yaml_comments(text)
    for n, c, full in comments:
        body = c.lstrip("#").strip()
        if COMMENT_BANNED.search(body):
            out.append(Finding(path, n, "public.comment", "comment has history, plan, date or ref words", key=c))
        if COMMENTED_OUT.match(body):
            out.append(Finding(path, n, "public.comment", "commented-out config", key=c))
    prev, length, start = None, 0, 0
    for n, c, full in comments + [(None, "", False)]:
        if full and prev is not None and n == prev + 1:
            length += 1
        else:
            if length > 2:
                out.append(Finding(path, start, "public.comment", f"{length} comment lines in a row; at most 2",
                                   key=f"block:{start}"))
            length, start = (1, n) if full else (0, n)
        prev = n if full else None
    return out


def workflow_findings(path, text):
    out = []
    try:
        doc = parse_yaml(text)
    except YamlError:
        return out
    if not isinstance(doc, dict):
        return out
    names = [("workflow", doc.get("name"))]
    for jk, job in (doc.get("jobs") or {}).items():
        if not isinstance(job, dict):
            continue
        names.append(("job", job.get("name")))
        ro = job.get("runs-on")
        labels = ro if isinstance(ro, list) else [ro]
        if not labels or any(str(l) not in RUNNERS for l in labels):
            raw = "[" + ", ".join(map(str, labels)) + "]" if isinstance(ro, list) else str(ro)
            out.append(Finding(path, line_of(text, "runs-on:", jk), "public.runner",
                               "runs-on must be a GitHub-hosted runner in RUNNERS", key=raw))
        for step in job.get("steps") or []:
            if isinstance(step, dict):
                names.append(("step", step.get("name")))
    for kind, name in names:
        if name and LIFECYCLE.search(str(name)):
            out.append(Finding(path, line_of(text, str(name)), "public.lifecycle", f"{kind} name has a lifecycle or "
                               "version token", key=str(name)))
    for n, line in enumerate(text.split("\n"), 1):
        if "self-hosted" in line and not line.strip().startswith("runs-on:"):
            out.append(Finding(path, n, "public.runner", "self-hosted runner reference", key="self-hosted"))
    return out


def line_of(text, needle, after=None):
    lines = text.split("\n")
    start = 0
    if after:
        for i, l in enumerate(lines):
            if l.strip().startswith(after + ":"):
                start = i
                break
    for i in range(start, len(lines)):
        if needle in lines[i]:
            return i + 1
    return 0


def check(ctx):
    out, head = [], ctx.head
    base_build = None
    if ctx.base is not None:
        base_build = {p.split("/")[1] for p in ctx.base.paths if p.startswith("docker/") and p.count("/") >= 2}
    for p in head.paths:
        out += [f for f in text_findings(p, p, ("public.ip", "public.domain", "public.host"))]
        if p.startswith("docker/") and (base_build is None or p.split("/")[1] in base_build):
            continue
        data = head.data(p)
        if not is_text(data):
            continue
        text = data.decode("utf-8")
        out += text_findings(p, text)
        if re.search(r"\.ya?ml$", p):
            out += comment_findings(p, text)
        if p.startswith(".github/workflows/"):
            out += workflow_findings(p, text)
    for sha, msg, *merge in ctx.commits:
        where = f"commit {sha[:12]}"
        # Git trailers (Co-Authored-By and the like) carry identities by design.
        body = "\n".join("" if TRAILER.match(l) else l for l in msg.split("\n"))
        out += text_findings(where, body)
        subject = msg.split("\n", 1)[0]
        # Merge commits from "Update branch" keep their generated subject; their bodies are still scanned.
        if merge and merge[0]:
            continue
        if not TITLE.match(subject) or len(subject) > 72:
            out.append(Finding(where, 1, "public.title", "subject must be `<prod|docker|ci|docs|repo>: <summary>`, "
                               "at most 72 chars"))
    pr = ctx.pr
    for field in ("title", "body", "branch"):
        if pr.get(field):
            out += text_findings(f"pr {field}", pr[field])
    # The squash commit subject is the title plus ` (#N)`; it has to fit in 72 chars too.
    cap = 72 - len(f" (#{pr.get('number') or '99999'})")
    if pr.get("title") and (not TITLE.match(pr["title"]) or len(pr["title"]) > cap):
        out.append(Finding("pr title", 1, "public.title", "title must be `<prod|docker|ci|docs|repo>: <summary>`, "
                           f"at most {cap} chars"))
    return out


def env_pr():
    return {k: os.environ.get(v, "") for k, v in (("title", "PR_TITLE"), ("body", "PR_BODY"), ("branch", "PR_BRANCH"),
                                                  ("number", "PR_NUMBER"))}


# Self-test inputs are assembled at run time so this file never holds the strings it bans.
_IP = ".".join(["10", "0", "0", "5"])
_DOM = "build." + "corp" + "-example." + "net"
_HOST = "g" + "pu12"
P = "prod/m-sglang-tp1.yaml"
WF = (".github/workflows/ci.yaml", "name: ci\non:\n  pull_request:\njobs:\n  a:\n    runs-on: ubuntu-latest\n"
      "    steps:\n      - name: check\n        run: \"true\"\n")
FILES = {P: "services:\n  a:\n    image: x\n# Why this file exists.\n", WF[0]: WF[1], "README.md": "See docs.\n"}
OK_COMMIT = [("a" * 40, "ci: add guards\n\nBody text.\n\nCo-Authored-By: A <a@b." + "dev>\n")]
CASES = [
    ("clean files, commit and PR", None, FILES, None, None, None,
     {"commits": OK_COMMIT, "pr": {"title": "repo: tidy", "body": "x", "branch": "chore/x"}}),
    ("IP in a file", "public.ip", dict(FILES, **{"README.md": f"host {_IP}\n"}), None, None, None, {}),
    ("internal domain", "public.domain", dict(FILES, **{"README.md": f"https://{_DOM}/x\n"}), None, None, None, {}),
    ("self-hosted runner", "public.runner", dict(FILES, **{WF[0]: WF[1].replace("ubuntu-latest", "[self-" + "hosted]")}),
     None, None, None, {}),
    ("private repo", "public.repo", dict(FILES, **{"README.md": "https://github." + "com/someone/fork\n"}), None, None, None,
     {}),
    ("host name in a comment", "public.host", dict(FILES, **{P: FILES[P] + f"# Runs on {_HOST} only.\n"}), None, None,
     None, {}),
    ("waived host on a new line", "public.host",
     dict(FILES, **{P: FILES[P] + f"# Runs on {_HOST} only.\nx-host: {_HOST}\n",
                    "ci/data/exceptions.json": json.dumps([{"rule": "public.host", "path": P,
                                                            "sha256": sha256(f"{_HOST}:# Runs on {_HOST} only."),
                                                            "reason": "r", "remove_when": "w"}])}),
     None, None, None, {}),
    ("lifecycle step name", "public.lifecycle", dict(FILES, **{WF[0]: WF[1].replace("name: check", "name: canary check")}),
     None, None, None, {}),
    ("inline ignore", "public.ignore", dict(FILES, **{"README.md": "x  # " + "no" + "qa\n"}), None, None, None, {}),
    ("domain in a commit body", "public.domain", FILES, None, None, None,
     {"commits": [("c" * 40, f"ci: x\n\nsee https://{_DOM}/x\n")]}),
    ("bad PR title", "public.title", FILES, None, None, None, {"pr": {"title": "Fix stuff"}}),
    ("bad commit subject", "public.title", FILES, None, None, None, {"commits": [("b" * 40, "wip\n")]}),
    ("merge commit subject", None, FILES, None, None, None,
     {"commits": [("d" * 40, "Merge branch 'main' into chore/x\n", True)]}),
    ("domain in a merge commit body", "public.domain", FILES, None, None, None,
     {"commits": [("e" * 40, f"Merge branch 'main'\n\nsee https://{_DOM}/x\n", True)]}),
    ("PR title too long once squashed", "public.title", FILES, None, None, None,
     {"pr": {"title": "repo: " + "x" * 62, "number": "1234"}}),
    ("three comment lines", "public.comment", dict(FILES, **{P: FILES[P] + "# a\n# b\n"}), None, None, None, {}),
    ("commented-out key", "public.comment", dict(FILES, **{P: FILES[P] + "# image: nginx\n"}), None, None, None, {}),
    ("history in a comment", "public.comment", dict(FILES, **{P: FILES[P] + "# We previously used x.\n"}), None, None,
     None, {}),
]
