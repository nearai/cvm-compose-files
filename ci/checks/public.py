"""public: nothing internal in files, paths, commit messages or PR text."""

import difflib
import ipaddress
import json
import os
import re
import unicodedata

import privacy
from common import (
    EXCEPTIONS, EXPECTED, IP_ALLOW, IP_ALLOW_PREFIX, LIFECYCLE, ORG, ORG_PUBLIC_REPOS, PUBLIC_HOSTS, PUBLIC_OWNERS,
    RUNNERS, Finding, YamlError, editable_doc, is_text, parse_yaml, sha256, yaml_comments,
)

RULES = [
    "public.ip", "public.domain", "public.runner", "public.repo", "public.host", "public.lifecycle", "public.ignore",
    "public.title", "public.comment", "public.disclosure", "public.disclosure-added", "public.branch",
]
DIFF_RULES = ["public.disclosure-added"]

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


# Wording tripwire. Categories D1..D7 name groups of narrative phrases; messages carry the category only.
# Matching runs on NFKC text, case-insensitive, over a paragraph with its line breaks joined by a space.
DISCLOSURE = [
    ("D1", r"\bCVE-\d{4}-\d+|\bGHSA(?:-[0-9a-z]{4}){3}\b"),
    ("D2", r"\bvulnerab\w*|\bexploit\w*|\bRCE\b|\bSSRF\b|\bzero[- ]?day\b|\battacker\w*"),
    ("D3", r"\bbefore this (?:change|fix|pr|patch)\b|\b(?:was|were) (?:exposed|exploitable|leaking|vulnerable|"
           r"reachable by)\b|\bno longer (?:exposed|leaks?|leaking|reachable|vulnerable|possible)\b|"
           r"(?<![\w-])security (?:fix|hole|issue|bug|flaw|gap)"),
    ("D4", r"\b(?:incident|post-?mortem|pentest|penetration test|bug bounty)\b|\breported by\b"),
    ("D5", r"\bCVSS\b|\bhighest[- ]impact\b|\b(?:critical|high)[- ](?:severity|risk)\b"),
    ("D6", r"\bstill (?:unfixed|unpatched|vulnerable|exposed)\b|\bunfixed\b|"
           r"\bnot yet (?:fixed|patched|deployed|rolled out)\b"),
    ("D7", r"\baffect(?:s|ed|ing)? (?:all|every|both|\d+) (?:users?|customers?)\b"),
]
DISCLOSURE_RE = [(cat, re.compile(rx, re.I)) for cat, rx in DISCLOSURE]
# D2 is not applied to the policy file that has to name the subject.
D2_EXEMPT = {".github/SECURITY.md"}
EMPHASIS = re.compile(r"(?<=[^\W_])[*_`]+(?=[^\W_])")
COMMENT_TEXT = re.compile(r"(?:^|\s)#(?!!)(.*)$")
SPACES = re.compile(r"\s+")
BREAK = re.compile(r"^(?:\||#|```|[-*+] |\d+[.)] )")
BRANCH_PREFIX = ("vuln", "exploit", "cve", "ghsa", "incident", "breach", "attack", "0day")
BRANCH_EXACT = {"rce", "ssrf", "xss"}
CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
PATCHES = re.compile(r"^docker/[^/]+/patches/.*\.(?:diff|patch)$")


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


def norm(line):
    s = unicodedata.normalize("NFKC", line)
    s = "".join("-" if unicodedata.category(c) == "Pd" or c == "\u2212" else c
                for c in s if unicodedata.category(c) != "Cf")
    return SPACES.sub(" ", EMPHASIS.sub("", s)).strip()


def scan_paragraph(path, entries, rule="public.disclosure", d2=True):
    """Findings for one paragraph given as (line number, text) pairs; the line is where the match starts."""
    lines = [(n, norm(t)) for n, t in entries]
    joined, starts = "", []
    for n, t in lines:
        starts.append((len(joined), n))
        joined += t + " "
    out, seen = [], set()
    for cat, rx in DISCLOSURE_RE:
        if cat == "D2" and not d2:
            continue
        for m in rx.finditer(joined):
            n = [ln for off, ln in starts if off <= m.start()][-1]
            if (n, cat) not in seen:
                seen.add((n, cat))
                out.append(Finding(path, n, rule, f"wording category {cat}: describe what the config does now, "
                                   "not a weakness, its history or its scope", key=f"{cat}:{path}:{n}"))
    return out


def md_paragraphs(text):
    """Runs of non-blank lines; table rows, headings, fences and list items start their own."""
    para, out = [], []
    for n, line in enumerate(text.split("\n"), 1):
        s = line.strip()
        if not s:
            para, out = [], out + ([para] if para else [])
            continue
        if para and BREAK.match(s):
            out.append(para)
            para = []
        para.append((n, line))
    return out + ([para] if para else [])


def comment_paragraphs(text):
    """Consecutive full-line comments form one paragraph; a trailing comment stands alone."""
    out, run, prev = [], [], None
    for n, c, full in yaml_comments(text):
        body = c.lstrip("#")
        if full and prev is not None and n == prev + 1:
            run.append((n, body))
        else:
            if run:
                out.append(run)
            run = [(n, body)]
        prev = n if full else None
    return out + ([run] if run else [])


def scan_text(path, text, d2=True):
    out = []
    for para in md_paragraphs(text):
        out += scan_paragraph(path, para, d2=d2)
    return out


def frozen_build_path(ctx, base_build, p):
    """Build inputs of merged dirs are frozen and predate the guard; docs nothing reads are not."""
    head = ctx.head
    return p.startswith("docker/") and (base_build is None or p.split("/")[1] in base_build) \
        and not editable_doc(head, p)


def exception_values(text):
    try:
        entries = json.loads(text)
    except ValueError:
        return []
    out = []
    for e in entries if isinstance(entries, list) else []:
        for k in ("reason", "remove_when"):
            if isinstance(e, dict) and isinstance(e.get(k), str):
                out.append((e[k], e))
    return out


def disclosure_findings(ctx, base_build):
    out, head = [], ctx.head
    for p in head.paths:
        if frozen_build_path(ctx, base_build, p) or PATCHES.search(p):
            continue
        is_md, is_yaml = p.endswith(".md"), bool(re.search(r"\.ya?ml$", p))
        if not (is_md or is_yaml or p == EXCEPTIONS):
            continue
        data = head.data(p)
        if not is_text(data):
            if is_md or is_yaml:
                out.append(Finding(p, 0, "public.disclosure", "file is not readable as UTF-8 text; the wording "
                                   "check cannot scan it", key=f"unreadable:{p}"))
            continue
        text = data.decode("utf-8")
        if is_md:
            out += scan_text(p, text, d2=p not in D2_EXEMPT)
        elif is_yaml:
            for para in comment_paragraphs(text):
                out += scan_paragraph(p, para)
        else:
            for v, e in exception_values(text):
                needle = json.dumps(v, ensure_ascii=False)[1:41]
                # \uXXXX escapes in the file hide the decoded text; fall back to the entry's own lines.
                n = line_of(text, needle) or line_of(text, json.dumps(v)[1:41])
                for k in ("sha256", "path"):
                    if not n and isinstance(e.get(k), str) and e[k]:
                        n = line_of(text, json.dumps(e[k])[1:-1])
                out += scan_paragraph(p, [(n, v)])
    for sha, msg, *merge in ctx.commits:
        body = "\n".join("" if TRAILER.match(l) else l for l in msg.split("\n"))
        out += scan_text(f"commit {sha[:12]}", body)
    pr = ctx.pr
    for field in ("title", "body", "branch"):
        if pr.get(field):
            out += scan_text(f"pr {field}", pr[field])
    return out


def added_lines(old, new):
    """1-based numbers of lines in `new` that are not in `old`."""
    sm = difflib.SequenceMatcher(None, old, new, autojunk=False)
    return [j + 1 for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag in ("insert", "replace") for j in range(j1, j2)]


def diff_findings(ctx, base_build):
    """Added lines of text files that `disclosure_findings` does not read."""
    out = []
    if ctx.base is None:
        return out
    for entry in ctx.name_status:
        status, paths = entry[0], entry[1:]
        p = paths[-1]
        if status == "D" or not ctx.head.has(p):
            continue
        if p.endswith(".md") or p in (EXCEPTIONS, EXPECTED) or PATCHES.search(p):
            continue
        if frozen_build_path(ctx, base_build, p):
            continue
        data = ctx.head.data(p)
        if not is_text(data):
            continue
        old = paths[0] if status in ("R", "C") else p
        before = ctx.base.text(old).split("\n") if ctx.base.has(old) else []
        text = data.decode("utf-8")
        lines = text.split("\n")
        skip = {n for n, _, _ in yaml_comments(text)} if re.search(r"\.ya?ml$", p) else set()
        run = []
        # Only comment text is read; commands, flags and field names are not wording.
        for n in added_lines(before, lines) + [None]:
            m = COMMENT_TEXT.search(lines[n - 1]) if n is not None and n not in skip else None
            if m and run and n == run[-1][0] + 1:
                run.append((n, m.group(1)))
                continue
            if run:
                out += scan_paragraph(p, run, "public.disclosure-added")
            run = [(n, m.group(1))] if m else []
    return out


def branch_findings(branch):
    tokens = [t for t in re.split(r"[/._\s-]+", CAMEL.sub(" ", branch).lower()) if t]
    hit = any(t in BRANCH_EXACT or t.startswith(BRANCH_PREFIX) for t in tokens) or \
        any(a == "hotfix" and b == "security" for a, b in zip(tokens, tokens[1:]))
    return [Finding("pr branch", 1, "public.branch", "branch name has a weakness, advisory or operational-event "
                    "token; rename the branch", key=branch)] if hit else []


def check(ctx):
    out, head = [], ctx.head
    base_build = None
    if ctx.base is not None:
        base_build = {p.split("/")[1] for p in ctx.base.paths if p.startswith("docker/") and p.count("/") >= 2}
    out += disclosure_findings(ctx, base_build) + diff_findings(ctx, base_build)
    out += branch_findings(ctx.pr.get("branch") or "")
    for p in head.paths:
        out += [f for f in text_findings(p, p, ("public.ip", "public.domain", "public.host"))]
        # Build inputs of merged dirs are frozen and predate the guard; docs nothing reads are scanned.
        if p.startswith("docker/") and (base_build is None or p.split("/")[1] in base_build) \
                and not editable_doc(head, p):
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
    ("host name in an unread build-dir doc", "public.host",
     dict(FILES, **{"docker/sglang-x-r1/Dockerfile": "FROM a\n", "docker/sglang-x-r1/README.md": f"Built on {_HOST}.\n",
                    ".github/workflows/publish-sglang-x-r1.yaml": "env:\n  CTX: docker/sglang-x-r1\n"}),
     {"docker/sglang-x-r1/Dockerfile": "FROM a\n"}, None, None, {}),
    ("history in a comment", "public.comment", dict(FILES, **{P: FILES[P] + "# We previously used x.\n"}), None, None,
     None, {}),
]


# Wording fixtures are joined from pieces for the same reason. Red ones carry the privacy sentinel in the
# matched paragraph; the self-test fails if it ever shows up in output.
def j(*parts):
    return "".join(parts)


S = privacy.SENTINEL
W_ID, W_VOCAB, W_DELTA = j("CV", "E-2026-1"), j("vulner", "able"), j("was ex", "posed to callers")
W_PAST, W_HIST, W_RATE = j("no lon", "ger exposed"), j("inci", "dent"), j("high", "est-impact")
W_OPEN, W_SCOPE = j("still un", "patched"), j("affects ev", "ery\ncustomer")
LEGIT = ("The cache directory is read-only; `security_opt` stays; image sglang-v0523-secure; "
         "`--trust-remote-code`; `privacy.secret`; set HF_TOKEN and ADMIN_API_KEY at deploy.\n")
XB = "docker/sglang-x-r1/"
XWF = ".github/workflows/publish-sglang-x-r1.yaml"
XFILES = {XB + "Dockerfile": "FROM a\nCOPY NOTES.md /n\n", XB + "NOTES.md": "ok\n", XB + "README.md": "ok\n",
          XWF: "env:\n  CTX: docker/sglang-x-r1\n"}
SH = "tools/check.sh"
SH_OLD = "#!/bin/sh\n# Keep the probe short.\n"
SH_FILES = dict(FILES, **{SH: SH_OLD})
EX = [{"rule": "public.ip", "path": "README.md", "sha256": sha256(_IP), "reason": f"{S} {W_OPEN} upstream",
       "remove_when": "w"}]
CASES += [
    ("legit config words pass", None, dict(FILES, **{"README.md": LEGIT, P: FILES[P] + "# " + LEGIT.split(";")[1] + "\n"}),
     None, None, None, {"pr": {"title": "repo: tidy", "body": LEGIT, "branch": "prod/engine-cache-warmup"},
                        "commits": [("a" * 40, "ci: x\n\n" + LEGIT)]}),
    ("advisory id in a doc", "public.disclosure", dict(FILES, **{"README.md": f"{S} {W_ID}\n"}), None, None, None, {}),
    ("phrase split across wrapped lines", "public.disclosure", dict(FILES, **{"README.md": f"{S} It {W_SCOPE} now.\n"}),
     None, None, None, {}),
    ("full-width letters", "public.disclosure",
     dict(FILES, **{"README.md": f"{S} " + "\uff23\uff36\uff25" + "-2026-1\n"}),
     None, None, None, {}),
    ("double space inside a phrase", "public.disclosure",
     dict(FILES, **{"README.md": f"{S} It {W_PAST[:2]}  {W_PAST[3:]} now.\n"}), None, None, None, {}),
    ("non-ASCII hyphen in an id", "public.disclosure",
     dict(FILES, **{"README.md": f"{S} {W_ID[:3]}\u2011{W_ID[4:]}\n"}), None, None, None, {}),
    ("zero-width split", "public.disclosure", dict(FILES, **{"README.md": f"{S} {W_VOCAB[:4]}​{W_VOCAB[4:]}\n"}),
     None, None, None, {}),
    ("past-tense delta in a doc", "public.disclosure", dict(FILES, **{"docs/a.md": f"{S} It {W_PAST} now.\n"}), None, None,
     None, {}),
    ("rating word in a comment", "public.disclosure", dict(FILES, **{P: FILES[P] + f"# {S} Known {W_RATE} gap.\n"}),
     None, None, None, {}),
    ("open-state wording in a waiver reason", "public.disclosure",
     dict(FILES, **{"README.md": f"host {_IP}\n", "ci/data/exceptions.json": json.dumps(EX, indent=2) + "\n"}),
     None, None, None, {}),
    ("history wording in a commit body", "public.disclosure", FILES, None, None, None,
     {"commits": [("f" * 40, f"ci: x\n\n{S} after the {W_HIST}.\n")]}),
    ("trailers are not scanned", None, FILES, None, None, None,
     {"commits": [("f" * 40, "ci: x\n\nBody.\n\nReported-by: A <a@b." + "dev>\n")]}),
    ("delta wording in the PR body", "public.disclosure", FILES, None, None, None,
     {"pr": {"title": "repo: tidy", "body": f"{S} It {W_DELTA}."}}),
    ("vocabulary in the policy file", None, dict(FILES, **{".github/SECURITY.md": f"Report a {W_VOCAB} privately.\n"}),
     None, None, None, {}),
    ("advisory id in the policy file", "public.disclosure", dict(FILES, **{".github/SECURITY.md": f"{S} {W_ID}\n"}),
     None, None, None, {}),
    ("phrase in a frozen build input", None,
     dict(FILES, **dict(XFILES, **{XB + "NOTES.md": f"{W_VOCAB}\n"})), dict(XFILES), None, None, {}),
    ("phrase in an editable build-dir doc", "public.disclosure",
     dict(FILES, **dict(XFILES, **{XB + "README.md": f"{S} {W_VOCAB}\n"})), dict(XFILES), None, None, {}),
    ("phrase in a patch body", None, dict(FILES, **{"docker/a/patches/p.diff": f"+ if {W_VOCAB}:\n", "x.patch": f"{W_VOCAB}\n"}),
     None, None, None, {}),
    ("added line, new file", "public.disclosure-added", dict(FILES, **{"tools/new.sh": f"#!/bin/sh\n# {S} {W_DELTA}\n"}),
     FILES, None, None, {"name_status": [("A", "tools/new.sh")]}),
    ("added line, modified file", "public.disclosure-added", dict(SH_FILES, **{SH: SH_OLD + f"# {S} {W_DELTA}\n"}),
     SH_FILES, None, None, {"name_status": [("M", SH)]}),
    ("clean added line", None, dict(SH_FILES, **{SH: SH_OLD + "# Pin the base by digest.\n"}), SH_FILES, None, None,
     {"name_status": [("M", SH)]}),
    ("old wording on an untouched line", None, dict(SH_FILES, **{SH: f"#!/bin/sh\n# {W_DELTA}\nx\n"}),
     dict(SH_FILES, **{SH: f"#!/bin/sh\n# {W_DELTA}\n"}), None, None, {"name_status": [("M", SH)]}),
    ("renamed file keeps old wording", None,
     dict(SH_FILES, **{"tools/b.sh": f"# {W_DELTA}\nnew\n"}), dict(SH_FILES, **{"tools/a.sh": f"# {W_DELTA}\n"}), None, None,
     {"name_status": [("R", "tools/a.sh", "tools/b.sh")]}),
    ("renamed file with an added phrase", "public.disclosure-added",
     dict(SH_FILES, **{"tools/b.sh": f"# a\n# {S} {W_DELTA}\n"}), dict(SH_FILES, **{"tools/a.sh": "# a\n"}), None, None,
     {"name_status": [("R", "tools/a.sh", "tools/b.sh")]}),
    ("added line in a block scalar", "public.disclosure-added",
     dict(FILES, **{P: FILES[P] + f"x-conf: |\n  # {S} {W_DELTA}\n"}), FILES, None, None, {"name_status": [("M", P)]}),
    ("added YAML comment is read once", "public.disclosure",
     dict(FILES, **{P: FILES[P] + f"# {S} {W_DELTA}\n"}), FILES, None, None, {"name_status": [("M", P)]}),
    ("added line in a patch file", None, dict(FILES, **{"docker/a/patches/p.diff": f"+ {W_VOCAB}\n"}), FILES, None, None,
     {"name_status": [("A", "docker/a/patches/p.diff")]}),
    ("branch with a weakness token", "public.branch", FILES, None, None, None, {"pr": {"branch": "fix/key-" + j("vul", "n") + "-" + S}}),
    ("branch with an adjacent pair", "public.branch", FILES, None, None, None, {"pr": {"branch": "hotfix-security-x"}}),
    ("branch words that only contain a token", None, FILES, None, None, None,
     {"pr": {"branch": "chore/resource-force-patch-1"}}),
    ("branch with a camelCase token", "public.branch", FILES, None, None, None, {"pr": {"branch": "fix/keyExploit_" + S}}),
    ("branch with an xss token", "public.branch", FILES, None, None, None, {"pr": {"branch": "fix/xss_" + S}}),
    ("branch about a memory leak", None, FILES, None, None, None, {"pr": {"branch": "fix/memory-leak"}}),
    ("scanner step in a workflow", None,
     dict(FILES, **{WF[0]: WF[1] + "      - name: Scan image\n        run: |\n          scan --" + j("sever", "ity") +
                    " HIGH --ignore-" + j("un", "fixed") + " | jq '.[] | " + j(".vulner", "abilities") + "'\n"}),
     FILES, None, None, {"name_status": [("M", WF[0])]}),
    ("ordinary ops wording", None,
     dict(FILES, **{"README.md": "Discovered by the proxy; log " + j("sever", "ity") + "; SLO " + j("brea", "ch") +
                    "; no " + j("out", "age") + "; connection still open; " + j("un", "patched") +
                    " source; affects every replica.\n"}), None, None, None, {}),
    ("suite name before a word", None,
     dict(SH_FILES, **{SH: SH_OLD + "# Upgrade to the noble-" + j("secur", "ity fix") + " pocket.\n"}), SH_FILES, None, None,
     {"name_status": [("M", SH)]}),
    ("unicode minus in a phrase", "public.disclosure",
     dict(FILES, **{"README.md": f"{S} {W_RATE.replace('-', chr(0x2212))}\n"}), None, None, None, {}),
    ("emphasis inside a word", "public.disclosure",
     dict(FILES, **{"README.md": f"{S} {W_VOCAB[:4]}**{W_VOCAB[4:]}**\n"}), None, None, None, {}),
    ("invisible format character", "public.disclosure",
     dict(FILES, **{"README.md": f"{S} {W_VOCAB[:4]}\u2064{W_VOCAB[4:]}\n"}), None, None, None, {}),
    ("unreadable markdown file", "public.disclosure", dict(FILES, **{"docs/b.md": b"\xff\xfe" + S.encode()}), None, None,
     None, {}),
    ("added line in a non-comment shell command", None, dict(SH_FILES, **{SH: SH_OLD + "echo " + W_DELTA + "\n"}), SH_FILES,
     None, None, {"name_status": [("M", SH)]}),
]
