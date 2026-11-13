"""layout: tracked paths, banned files, legacy list, workflow set and publish workflow shape."""

import re

from common import (
    LEGACY, LIFECYCLE, PUBLISH_ENV, Finding, YamlError, find_line, is_text, legacy_lines, parse_yaml,
)

RULES = [
    "layout.path", "layout.ext", "layout.secret-file", "layout.size", "layout.services", "layout.workflow-set",
    "layout.legacy", "layout.yaml", "layout.w1", "layout.w2", "layout.w3", "layout.all-green", "layout.claude-md",
    "layout.agent-file", "layout.build-dir",
]
DIFF_RULES = ["layout.legacy", "layout.build-dir"]

SERVING = re.compile(
    r"^prod/(?P<ckpt>[a-z0-9]+(?:[.-][a-z0-9]+)*?)-(?P<engine>sglang|vllm)-"
    r"(?P<topo>(?:tp|dep|dp)\d+(?:x\d+)?(?:_(?:tp|dep|dp)\d+(?:x\d+)?)*)(?:-(?P<tier>[a-z]{2,12}))?\.yaml$"
)
PACK = re.compile(r"^prod/pack-[a-z0-9]+(?:-[a-z0-9]+)*\.yaml$")
UTIL = re.compile(r"^prod/util-[a-z0-9]+(?:-[a-z0-9]+)*\.yaml$")
DOCKER = re.compile(r"^docker/([a-z0-9][a-z0-9.-]*)/.+$")
# New build dirs only; dirs present at the base keep their names.
BUILD_DIR = re.compile(r"(sglang|vllm)-v[0-9.]+-[a-z0-9-]+-r[0-9]+")
PUBLISH = re.compile(r"^\.github/workflows/publish-(?P<dir>[a-z0-9][a-z0-9.-]*)\.yaml$")
DOCS = re.compile(r"^(README|AGENTS|CLAUDE)\.md$|^docs/[a-z0-9._-]+\.md$")
# Shared Claude Code files only; ci/checks/claude.py checks their content.
CLAUDE_DIR = re.compile(r"^\.claude/(settings\.json|(rules|agents)/[a-z][a-z0-9-]*\.md)$")
# Files coding agents load as instructions or config. Only the root CLAUDE.md and AGENTS.md and the
# CLAUDE_DIR paths may exist; anywhere else an agent working there would pick them up.
AGENT_FILE = re.compile(
    r"(?i)(^|/)((CLAUDE|AGENTS|GEMINI)[^/]*\.md|\.mcp\.json|\.cursorrules|\.windsurfrules|\.clinerules"
    r"|\.aider[^/]*|copilot-instructions\.md)$"
    r"|(^|/)\.(claude|cursor|gemini|codex|windsurf|clinerules|continue|roo|kiro|junie)/"
    r"|(^|/)\.github/(instructions|prompts|chatmodes)/"
)
FIXED = {
    ".gitignore", ".gitattributes", ".github/pull_request_template.md", ".github/SECURITY.md",
    ".github/workflows/ci.yaml", ".github/workflows/auto-tag.yaml",
    "ci/run.py", "ci/lib/common.py",
    "ci/checks/layout.py", "ci/checks/status.py", "ci/checks/pins.py", "ci/checks/privacy.py", "ci/checks/public.py",
    "ci/checks/frozen.py", "ci/checks/claude.py",
    "ci/data/legacy.txt", "ci/data/images.allow", "ci/data/services.allow.json", "ci/data/public.json",
    "ci/data/exceptions.json", "ci/data/expected.json",
    "ci/config/yamllint.yaml", "ci/config/gitleaks.toml",
    "ci/templates/sglang.yaml", "ci/templates/vllm.yaml",
}
BANNED_EXT = re.compile(r"\.(pyc|safetensors|bin|pt|gguf|onnx|ipynb|log|zip|tar(\.[a-z0-9]+)?)$|(^|/)__pycache__/", re.I)
SECRET_FILE = re.compile(
    r"(^|/)(\.env[^/]*|[^/]*\.pem|[^/]*\.key|[^/]*\.p12|[^/]*\.pfx|id_[^/]*|\.netrc|\.npmrc|[^/]*kubeconfig[^/]*)$"
    r"|(^|/)\.docker/config\.json$",
    re.I,
)
MAX_SIZE = 256 * 1024
CI_JOBS = {"base", "compose", "guards", "lint", "secrets"}
NEEDS_JSON = "${{ toJSON(needs) }}"
GUARD_CALLS = ("python3 ci/run.py --self-test", 'python3 ci/run.py --base "$BASE"')
NEW_PATH = re.compile(r"^[a-z0-9._-]+(/[a-z0-9._-]+){0,2}$")


def conforms(path, tree):
    """True when the path satisfies the naming rules without a legacy line."""
    if path in FIXED or DOCS.match(path) or CLAUDE_DIR.match(path):
        return True
    if path.startswith("prod/"):
        if not (SERVING.match(path) or PACK.match(path) or UTIL.match(path)):
            return False
        return not LIFECYCLE.search(path[len("prod/"):])
    m = PUBLISH.match(path)
    if m:
        return any(p.startswith(f"docker/{m.group('dir')}/") for p in tree.paths)
    return bool(DOCKER.match(path))


def check(ctx):
    out, head = [], ctx.head
    legacy = legacy_lines(head) or []
    for p in head.paths:
        data = head.data(p)
        if AGENT_FILE.search(p) and p not in ("CLAUDE.md", "AGENTS.md") and not CLAUDE_DIR.match(p):
            out.append(Finding(p, 0, "layout.agent-file", "agent instruction or config file outside the root "
                               "CLAUDE.md, AGENTS.md and .claude/ rules, agents and settings.json"))
        elif not conforms(p, head) and not any(p == e or (e.endswith("/") and p.startswith(e)) for e in legacy):
            rule = "layout.workflow-set" if p.startswith(".github/workflows/") else "layout.path"
            out.append(Finding(p, 0, rule, "path does not follow the layout rules and is not in ci/data/legacy.txt"))
        elif p.startswith("docs/") and LIFECYCLE.search(p[len("docs/"):]):
            out.append(Finding(p, 0, "layout.path", "lifecycle or host token in a docs path"))
        if not DOCKER.match(p) and not NEW_PATH.match(p) and p not in FIXED and not DOCS.match(p) \
                and not AGENT_FILE.search(p) and not any(p == e or (e.endswith("/") and p.startswith(e)) for e in legacy):
            out.append(Finding(p, 0, "layout.path", "path segments must be lowercase [a-z0-9._-], at most 3 levels"))
        if len(p) > 100 and not DOCKER.match(p):
            out.append(Finding(p, 0, "layout.path", "path longer than 100 characters"))
        if BANNED_EXT.search(p):
            out.append(Finding(p, 0, "layout.ext", "file type not allowed in this repo"))
        if SECRET_FILE.search(p):
            out.append(Finding(p, 0, "layout.secret-file", "secret-like file name"))
        if len(data) > MAX_SIZE:
            out.append(Finding(p, 0, "layout.size", f"file larger than {MAX_SIZE // 1024} KiB"))
        if re.search(r"\.ya?ml$", p) and is_text(data) and p not in head.deployables() and p not in ctx.templates:
            n = find_line(data.decode(), r"^services:\s*(#.*)?$")
            if n:
                out.append(Finding(p, n, "layout.services", "compose services outside prod/ or the ci templates"))
    if head.has("CLAUDE.md") and head.data("CLAUDE.md") != b"@AGENTS.md\n":
        out.append(Finding("CLAUDE.md", 1, "layout.claude-md", "CLAUDE.md must be exactly @AGENTS.md"))
    if ctx.base is not None:
        old = {m.group(1) for m in map(DOCKER.match, ctx.base.paths) if m}
        for d in sorted({m.group(1) for m in map(DOCKER.match, head.paths) if m} - old):
            if not BUILD_DIR.fullmatch(d):
                out.append(Finding(f"docker/{d}/", 0, "layout.build-dir",
                                   "new build dir must be named <engine>-v<upstream>-<scope>-r<N>"))
    out += check_legacy(ctx, legacy)
    out += check_workflows(ctx)
    return out


def check_legacy(ctx, legacy):
    out, head = [], ctx.head
    if not head.has(LEGACY):
        return out
    if legacy != sorted(set(legacy)):
        out.append(Finding(LEGACY, 0, "layout.legacy", "entries must be sorted and unique"))
    for n, e in enumerate(legacy, 1):
        if not e or e.startswith("#") or re.search(r"[*?\[\]\s]", e):
            out.append(Finding(LEGACY, n, "layout.legacy", "entry must be one exact path, no comments or globs"))
            continue
        files = [p for p in head.paths if p == e or (e.endswith("/") and p.startswith(e))]
        if not files:
            out.append(Finding(LEGACY, n, "layout.legacy", "entry does not exist"))
        elif all(conforms(p, head) for p in files):
            out.append(Finding(LEGACY, n, "layout.legacy", "entry already follows the layout rules; remove it"))
    base = legacy_lines(ctx.base) if ctx.base else None
    if base is not None:
        for n, e in enumerate(legacy, 1):
            if e not in base:
                out.append(Finding(LEGACY, n, "layout.legacy", "ci/data/legacy.txt only shrinks; entry not in base"))
    return out


def _uses_refs(doc):
    for jname, job in (doc.get("jobs") or {}).items():
        if not isinstance(job, dict):
            continue
        if job.get("uses"):
            yield job["uses"]
        for step in job.get("steps") or []:
            if isinstance(step, dict) and step.get("uses"):
                yield step["uses"]


def check_workflows(ctx):
    out, head = [], ctx.head
    for p in head.paths:
        if not p.startswith(".github/workflows/"):
            continue
        text = head.text(p)
        try:
            doc = parse_yaml(text)
        except YamlError as e:
            out.append(Finding(p, e.line, "layout.yaml", f"unsupported or invalid YAML: {e}"))
            continue
        if not isinstance(doc, dict) or not isinstance(doc.get("jobs"), dict):
            out.append(Finding(p, 0, "layout.yaml", "workflow has no jobs mapping"))
            continue
        for ref in _uses_refs(doc):
            ref = str(ref)
            if ref.startswith("./"):
                continue
            ok = re.fullmatch(r"docker://[^@\s]+@sha256:[0-9a-f]{64}", ref) or \
                re.fullmatch(r"[\w.-]+/[\w./-]+@[0-9a-f]{40}", ref)
            if not ok:
                out.append(Finding(p, find_line(text, re.escape(ref)), "layout.w3", "uses: must pin a 40-hex commit",
                                   key=ref))
        if PUBLISH.match(p):
            for jname, job in doc["jobs"].items():
                env = job.get("environment") if isinstance(job, dict) else None
                if isinstance(env, dict):
                    env = env.get("name")
                if env != PUBLISH_ENV:
                    out.append(Finding(p, find_line(text, rf"^  {re.escape(jname)}:"), "layout.w1",
                                       f"job must set environment: {PUBLISH_ENV}"))
            on = doc.get("on", doc.get(True))
            events = list(on) if isinstance(on, (dict, list)) else [on]
            for ev in events:
                if str(ev).startswith("pull_request") or ev == "workflow_run":
                    out.append(Finding(p, find_line(text, rf"\b{ev}\b"), "layout.w2",
                                       "publish workflows never run on pull requests or workflow_run"))
        if p == ".github/workflows/ci.yaml":
            jobs = doc["jobs"]
            ag = jobs.get("all-green")
            if not isinstance(ag, dict):
                out.append(Finding(p, 0, "layout.all-green", "ci.yaml must have an all-green job"))
                continue
            needs = ag.get("needs") or []
            needs = [needs] if isinstance(needs, str) else needs
            others = sorted(k for k in jobs if k != "all-green")
            if others != sorted(CI_JOBS):
                out.append(Finding(p, find_line(text, r"^jobs:"), "layout.all-green",
                                   f"ci.yaml jobs must be exactly {', '.join(sorted(CI_JOBS))} and all-green"))
            if str(ag.get("if", "")).strip() != "always()":
                out.append(Finding(p, find_line(text, r"^  all-green:"), "layout.all-green",
                                   "all-green must run if: always()"))
            # The result check reads every needed job from `needs`, so no job count is written down.
            if not any(isinstance(st, dict) and NEEDS_JSON in (st.get("env") or {}).values()
                       for st in ag.get("steps") or []):
                out.append(Finding(p, find_line(text, r"^  all-green:"), "layout.all-green",
                                   f"all-green must check the results in {NEEDS_JSON}"))
            guards = jobs.get("guards") if isinstance(jobs.get("guards"), dict) else {}
            runs = "\n".join(str(st.get("run", "")) for st in guards.get("steps") or [] if isinstance(st, dict))
            for call in GUARD_CALLS:
                if not re.search(rf"(?m)^\s*{re.escape(call)}(\s|$)", runs):
                    out.append(Finding(p, find_line(text, r"^  guards:"), "layout.all-green",
                                       f"the guards job must run `{call}`"))
            if sorted(needs) != others:
                out.append(Finding(p, find_line(text, r"^  all-green:"), "layout.all-green",
                                   "all-green.needs must list every other job"))
            if ag.get("name", "all-green") != "all-green":
                out.append(Finding(p, find_line(text, r"^  all-green:"), "layout.all-green",
                                   "all-green must keep its name"))
            for k in others:
                job = jobs[k] if isinstance(jobs[k], dict) else {}
                if str(job.get("name", k)).strip().lower() == "all-green":
                    out.append(Finding(p, find_line(text, rf"^  {re.escape(k)}:"), "layout.all-green",
                                       "only the all-green job may be named all-green"))
    return out


# Self-test fixtures: (description, expected rule or None, files, base files)
def _job(name, steps='      - run: "true"\n'):
    return f"  {name}:\n    runs-on: ubuntu-latest\n    steps:\n{steps}"


ALL_GREEN = ("  all-green:\n    if: always()\n    needs: [base, compose, guards, lint, secrets]\n"
             "    runs-on: ubuntu-latest\n    steps:\n      - env:\n          NEEDS: " + NEEDS_JSON + "\n"
             "        run: jq -e 'all(.[]; .result == \"success\")' <<< \"$NEEDS\"\n")
WF_OK = (
    "name: ci\non:\n  pull_request:\npermissions:\n  contents: read\njobs:\n" + _job("base")
    + _job("guards", "      - uses: actions/checkout@" + "a" * 40 + "\n      - run: python3 ci/run.py --self-test\n"
           "      - run: |\n          python3 ci/run.py --base \"$BASE\"\n")
    + _job("compose") + _job("lint") + _job("secrets") + ALL_GREEN
)
PUB_OK = (
    "name: Publish x\non:\n  workflow_dispatch:\njobs:\n  publish:\n    environment: " + PUBLISH_ENV + "\n"
    "    runs-on: ubuntu-latest\n    steps:\n      - uses: actions/checkout@" + "b" * 40 + "\n"
)
BASE_FILES = {
    "README.md": "x\n",
    "ci/data/legacy.txt": "prod/Old-File.yaml\n",
    "prod/Old-File.yaml": "services: {}\n",
    "prod/glm-5.3-sglang-tp4.yaml": "services: {}\n",
    "docker/sglang-v1-x-r1/Dockerfile": "FROM a\n",
    ".github/workflows/ci.yaml": WF_OK,
    ".github/workflows/publish-sglang-v1-x-r1.yaml": PUB_OK,
}


def _with(**changes):
    files = dict(BASE_FILES)
    for k, v in changes.items():
        k = k.replace("__", "/").replace("_DOT_", ".")
        if v is None:
            files.pop(k, None)
        else:
            files[k] = v
    return files


CASES = [
    ("conforming tree", None, BASE_FILES, BASE_FILES),
    ("diff rule green without base data file", None, BASE_FILES, {"docs/x.md": "x\n"}),
    ("new build dir with a standard name", None,
     _with(**{"docker/vllm-v0.33.0-dsv41-r1/Dockerfile": "FROM a\n"}), BASE_FILES),
    ("new build dir with a legacy-style name", "layout.build-dir",
     _with(**{"docker/sglang-glm53-r1/Dockerfile": "FROM a\n"}), BASE_FILES),
    ("unlisted bad path", "layout.path", dict(BASE_FILES, **{"prod/My-Config.yaml": "services: {}\n"}), BASE_FILES),
    ("lifecycle token in a serving path", "layout.path",
     dict(BASE_FILES, **{"prod/glm-5.3-sglang-tp4-test.yaml": "services: {}\n"}), BASE_FILES),
    ("bytecode", "layout.ext", dict(BASE_FILES, **{"docker/sglang-v1-x-r1/a.pyc": "x"}), BASE_FILES),
    ("secret file", "layout.secret-file", dict(BASE_FILES, **{"docker/sglang-v1-x-r1/.env": "A=1\n"}), BASE_FILES),
    ("oversized file", "layout.size", dict(BASE_FILES, **{"docs/big.md": "x" * (MAX_SIZE + 1)}), BASE_FILES),
    ("compose outside prod", "layout.services", dict(BASE_FILES, **{"docker/sglang-v1-x-r1/c.yaml": "services:\n"}),
     BASE_FILES),
    ("unknown workflow", "layout.workflow-set", dict(BASE_FILES, **{".github/workflows/extra.yaml": WF_OK}),
     BASE_FILES),
    ("legacy grows", "layout.legacy",
     dict(BASE_FILES, **{"ci/data/legacy.txt": "prod/Another.yaml\nprod/Old-File.yaml\n", "prod/Another.yaml": "x\n"}),
     BASE_FILES),
    ("legacy glob", "layout.legacy", dict(BASE_FILES, **{"ci/data/legacy.txt": "prod/*.yaml\nprod/Old-File.yaml\n"}),
     dict(BASE_FILES, **{"ci/data/legacy.txt": "prod/*.yaml\nprod/Old-File.yaml\n"})),
    ("workflow with an anchor", "layout.yaml",
     dict(BASE_FILES, **{".github/workflows/publish-sglang-v1-x-r1.yaml": PUB_OK.replace("environment: " + PUBLISH_ENV,
                                                                                      "environment: &e " + PUBLISH_ENV)}),
     BASE_FILES),
    ("publish job without environment", "layout.w1",
     dict(BASE_FILES, **{".github/workflows/publish-sglang-v1-x-r1.yaml": PUB_OK.replace(
         "    environment: " + PUBLISH_ENV + "\n", "")}), BASE_FILES),
    ("publish on pull_request", "layout.w2",
     dict(BASE_FILES, **{".github/workflows/publish-sglang-v1-x-r1.yaml": PUB_OK.replace(
         "  workflow_dispatch:\n", "  workflow_dispatch:\n  pull_request_target:\n")}), BASE_FILES),
    ("unpinned action", "layout.w3",
     dict(BASE_FILES, **{".github/workflows/ci.yaml": WF_OK.replace("@" + "a" * 40, "@v4")}), BASE_FILES),
    ("second job named all-green", "layout.all-green",
     dict(BASE_FILES, **{".github/workflows/ci.yaml": WF_OK.replace(
         "  all-green:\n", "  spoof:\n    name: all-green\n    runs-on: ubuntu-latest\n    steps:\n"
         "      - run: \"true\"\n  all-green:\n").replace("needs: [base,", "needs: [spoof, base,")}), BASE_FILES),
    ("CLAUDE.md imports AGENTS.md", None, dict(BASE_FILES, **{"CLAUDE.md": "@AGENTS.md\n"}), BASE_FILES),
    ("CLAUDE.md with extra text", "layout.claude-md", dict(BASE_FILES, **{"CLAUDE.md": "@AGENTS.md\nAlso: x\n"}),
     BASE_FILES),
    ("claude dir shapes", None, dict(BASE_FILES, **{".claude/rules/prod.md": "x\n", ".claude/agents/a.md": "x\n",
                                                    ".claude/settings.json": "{}\n"}), BASE_FILES),
    ("claude command file", "layout.agent-file", dict(BASE_FILES, **{".claude/commands/x.md": "x\n"}), BASE_FILES),
    ("claude local settings", "layout.agent-file", dict(BASE_FILES, **{".claude/settings." + "local.json": "{}\n"}),
     BASE_FILES),
    ("CLAUDE.md in a new build dir", "layout.agent-file",
     dict(BASE_FILES, **{"docker/sglang-v1-y-r1/Dockerfile": "FROM a\n", "docker/sglang-v1-y-r1/CLAUDE.md": "x\n"}),
     BASE_FILES),
    ("AGENTS.override.md in prod", "layout.agent-file", dict(BASE_FILES, **{"prod/AGENTS.override.md": "x\n"}),
     BASE_FILES),
    ("rules dir in a build dir", "layout.agent-file",
     dict(BASE_FILES, **{"docker/sglang-v1-x-r1/.claude/rules/x.md": "x\n"}), BASE_FILES),
    ("MCP config", "layout.agent-file", dict(BASE_FILES, **{".mcp.json": "{}\n"}), BASE_FILES),
    ("cursor rules", "layout.agent-file", dict(BASE_FILES, **{"docker/sglang-v1-x-r1/.cursorrules": "x\n"}), BASE_FILES),
    ("all-green misses a job", "layout.all-green",
     dict(BASE_FILES, **{".github/workflows/ci.yaml": WF_OK.replace("guards, lint, secrets]", "guards, secrets]")}),
     BASE_FILES),
    ("ci.yaml drops a job", "layout.all-green",
     dict(BASE_FILES, **{".github/workflows/ci.yaml": WF_OK.replace(_job("lint"), "").replace(
         "guards, lint, secrets]", "guards, secrets]")}), BASE_FILES),
    ("all-green without always()", "layout.all-green",
     dict(BASE_FILES, **{".github/workflows/ci.yaml": WF_OK.replace("    if: always()\n", "")}), BASE_FILES),
    ("all-green counts results itself", "layout.all-green",
     dict(BASE_FILES, **{".github/workflows/ci.yaml": WF_OK.replace(NEEDS_JSON, "${{ join(needs.*.result, ' ') }}")}),
     BASE_FILES),
    ("guards job skips the self-test", "layout.all-green",
     dict(BASE_FILES, **{".github/workflows/ci.yaml": WF_OK.replace("run: python3 ci/run.py --self-test",
                                                                    "run: python3 ci/run.py --help")}), BASE_FILES),
    ("guards job runs without a base", "layout.all-green",
     dict(BASE_FILES, **{".github/workflows/ci.yaml": WF_OK.replace(' --base \"$BASE\"', "")}), BASE_FILES),
]
