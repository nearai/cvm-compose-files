"""claude: AGENTS.md size, .claude/ rule and agent frontmatter, settings shape, path references, repeats."""

import json
import posixpath
import re

from common import Finding, find_line

RULES = ["claude.cap", "claude.rule", "claude.glob", "claude.agent", "claude.settings", "claude.ref", "claude.dup"]
DIFF_RULES = []

AGENTS, SETTINGS = "AGENTS.md", ".claude/settings.json"
AGENTS_CAP, LOADED_CAP = 140, 200
# The only rule file without paths: frontmatter; Claude Code loads it in every session.
UNSCOPED = ".claude/rules/review.md"
RULE = re.compile(r"^\.claude/rules/[a-z][a-z0-9-]*\.md$")
AGENT = re.compile(r"^\.claude/agents/([a-z][a-z0-9-]*)\.md$")
RULE_FM = re.compile(r'\A---\npaths:\n((?:  - "[^"\n]+"\n)+)---\n')
GLOB = re.compile(r"^[A-Za-z0-9._-]+(?:/[A-Za-z0-9._*-]+)*$")
# A missing tools: line inherits every tool, so the field is required and read-only.
AGENT_FM = re.compile(r"\A---\nname: ([^\n]*)\ndescription: ([^\n]*)\ntools: ([^\n]*)\nmodel: ([^\n]*)\n---\n")
READ_ONLY = {"Read", "Grep", "Glob"}
MODELS = {"inherit", "sonnet", "opus", "haiku"}
DENY = re.compile(r"^(Read|Edit|Bash|WebFetch)\([^()\n]+\)$|^WebFetch$")
# Secret reads, outward actions and history rewrites stay denied; dropping one is a gate change.
# Bash patterns are a guardrail, not a boundary; branch and tag protection live in GitHub rulesets.
SECRET_DENY = {
    "Read(**/.env*)", "Edit(**/.env*)", "Read(**/*.pem)", "Edit(**/*.pem)", "Read(**/*.key)", "Edit(**/*.key)",
    "Read(**/*.p12)", "Read(**/*.pfx)", "Read(**/id_rsa*)", "Read(**/id_ecdsa*)", "Read(**/id_ed25519*)",
    "Read(**/.netrc)", "Read(**/*kubeconfig*)", "Read(**/.docker/config.json)", "Read(~/.ssh/**)",
    "Read(~/.aws/**)", "Read(~/.kube/**)", "Read(~/.docker/config.json)", "Read(~/.cache/huggingface/token)",
    "Read(~/.config/gh/hosts.yml)",
    "Bash(*.env*)", "Bash(*.pem*)", "Bash(*.key)", "Bash(*.key *)", "Bash(*.p12*)", "Bash(*.pfx*)",
    "Bash(*id_rsa*)", "Bash(*id_ecdsa*)", "Bash(*id_ed25519*)", "Bash(*.netrc*)", "Bash(*kubeconfig*)",
    "Bash(*.ssh/*)", "Bash(*.aws/*)", "Bash(*.kube/*)", "Bash(*.docker/config.json*)", "Bash(*huggingface/token*)",
    "Bash(*gh/hosts.yml*)", "Bash(gh auth token*)", "Bash(gh auth status --show-token*)",
    "Bash(gh auth status * --show-token*)", "Bash(gh auth status -t*)", "Bash(gh auth status * -t*)",
    "Bash(printenv*)", "Bash(env)",
}
OUTWARD_DENY = {
    "Bash(gh pr review*)", "Bash(gh pr merge*)", "Bash(gh pr comment*)", "Bash(gh issue*)", "Bash(gh gist*)",
    "Bash(gh api*)", "Bash(curl*)", "Bash(wget*)", "WebFetch",
}
HISTORY_DENY = {
    "Bash(git push --force*)", "Bash(git push * --force*)", "Bash(git push -f*)", "Bash(git push * -f*)",
    "Bash(git push +*)", "Bash(git push * +*)", "Bash(git push --delete*)", "Bash(git push * --delete*)",
    "Bash(git push :*)", "Bash(git push * :*)", "Bash(git -C * push*)", "Bash(git -c * push*)",
    "Bash(git tag -d*)", "Bash(git tag * -d*)", "Bash(git tag -f*)", "Bash(git tag * -f*)",
    "Bash(git tag --delete*)", "Bash(git tag --force*)",
}
REQUIRED_DENY = SECRET_DENY | OUTWARD_DENY | HISTORY_DENY
SPAN = re.compile(r"`([^`\n]+)`|\]\(([^)\s]+)\)")
REPO_PATH = re.compile(r"^(?:prod|docker|ci|docs|scripts|adapters|\.github|\.claude)/|^[A-Z]+\.md$")
DUP_MIN = 40


def glob_rx(g):
    out, i = "", 0
    while i < len(g):
        if g.startswith("**/", i):
            out, i = out + "(?:[^/]+/)*", i + 3
        elif g.startswith("**", i):
            out, i = out + ".*", i + 2
        elif g[i] == "*":
            out, i = out + "[^/]*", i + 1
        else:
            out, i = out + re.escape(g[i]), i + 1
    return re.compile(out)


def norm(line):
    s = re.sub(r"^\s*(?:[-*]\s+|\d+\.\s+|#+\s+)?(?:\[[ x]\]\s+)?", "", line)
    return re.sub(r"\s+", " ", s).strip().lower()


def exists(head, p):
    p = p.rstrip("/")
    return head.has(p) or any(x.startswith(p + "/") for x in head.paths)


def refs(path, text):
    """(line, repo path) for link targets and repo paths inside code spans."""
    for n, line in enumerate(text.split("\n"), 1):
        for m in SPAN.finditer(line):
            if m.group(2):
                target = m.group(2).split("#")[0]
                if target and not re.match(r"[a-z][a-z0-9+.-]*:", target):
                    yield n, posixpath.normpath(posixpath.join(posixpath.dirname(path), target))
                continue
            for t in m.group(1).split():
                t = t.rstrip(".,;:")
                if REPO_PATH.search(t) and not re.search(r"[<>*{}\[\]$]", t):
                    yield n, t


def check_rule(head, p):
    text, out = head.text(p), []
    if re.search(r"(?m)^@\S", text):
        out.append(Finding(p, find_line(text, r"^@\S"), "claude.rule", "no @ imports in rule files"))
    m = RULE_FM.match(text)
    if not m:
        if text.startswith("---") or p != UNSCOPED:
            out.append(Finding(p, 1, "claude.rule",
                               f"start with a paths: list of double-quoted globs; only {UNSCOPED} has none"))
        return out
    for n, line in enumerate(m.group(1).split("\n")[:-1], 3):
        g = line[len('  - "'):-1]
        if not GLOB.match(g):
            out.append(Finding(p, n, "claude.rule", "glob must be repo-relative with * or ** only", key=g))
        elif not any(glob_rx(g).fullmatch(x) for x in head.paths):
            out.append(Finding(p, n, "claude.glob", "glob matches no file", key=g))
    return out


def check_agent(head, p):
    text, out = head.text(p), []
    m = AGENT_FM.match(text)
    if not m:
        return [Finding(p, 1, "claude.agent", "frontmatter must be exactly name, description, tools, model")]
    name, desc, tools, model = m.groups()
    if name != AGENT.match(p).group(1):
        out.append(Finding(p, 2, "claude.agent", "name must equal the file stem"))
    if not desc or len(desc) > 300 or re.search(r": | #|^[\"'&*!|>%@`\[{]", desc):
        out.append(Finding(p, 3, "claude.agent", "description must be one plain line of at most 300 chars"))
    if not {t.strip() for t in tools.split(",")} <= READ_ONLY:
        out.append(Finding(p, 4, "claude.agent", "tools must be a subset of Read, Grep, Glob"))
    if model not in MODELS:
        out.append(Finding(p, 5, "claude.agent", f"model must be one of {', '.join(sorted(MODELS))}"))
    if "\n## Output\n" not in text[m.end():]:
        out.append(Finding(p, 0, "claude.agent", "body needs an ## Output section"))
    if re.search(r"(?m)^@\S", text):
        out.append(Finding(p, find_line(text, r"^@\S"), "claude.agent", "no @ imports in agent files"))
    return out


def check_settings(text):
    try:
        doc = json.loads(text)
    except ValueError:
        return [Finding(SETTINGS, 0, "claude.settings", "not valid JSON")]
    perms = doc.get("permissions") if isinstance(doc, dict) else None
    if not isinstance(perms, dict) or set(doc) != {"permissions"} or set(perms) != {"deny"} \
            or not isinstance(perms["deny"], list):
        return [Finding(SETTINGS, 0, "claude.settings", 'must be exactly {"permissions": {"deny": [...]}}')]
    deny, out = [str(e) for e in perms["deny"]], []
    for e in deny:
        if not DENY.match(e):
            out.append(Finding(SETTINGS, find_line(text, re.escape(json.dumps(e))), "claude.settings",
                               "deny entry is not Read(), Edit(), Bash() or WebFetch", key=e))
    if len(set(deny)) != len(deny):
        out.append(Finding(SETTINGS, 0, "claude.settings", "duplicate deny entry"))
    for e in sorted(REQUIRED_DENY - set(deny)):
        out.append(Finding(SETTINGS, 0, "claude.settings", f"missing required deny entry {e}", key=e))
    return out


def check(ctx):
    out, head = [], ctx.head
    rules = [p for p in head.paths if RULE.match(p)]
    agents = [p for p in head.paths if AGENT.match(p)]
    agents_text = head.text(AGENTS) if head.has(AGENTS) else ""
    if agents_text.count("\n") > AGENTS_CAP:
        out.append(Finding(AGENTS, AGENTS_CAP + 1, "claude.cap", f"longer than {AGENTS_CAP} lines"))
    loaded = agents_text.count("\n") + (head.text(UNSCOPED).count("\n") if head.has(UNSCOPED) else 0)
    if loaded > LOADED_CAP:
        out.append(Finding(UNSCOPED, 0, "claude.cap", f"AGENTS.md plus this file exceed {LOADED_CAP} lines"))
    for p in rules:
        out += check_rule(head, p)
        if f"]({p})" not in agents_text:
            out.append(Finding(p, 0, "claude.ref", "rule file not linked from AGENTS.md"))
    for p in agents:
        out += check_agent(head, p)
    if head.has(SETTINGS):
        out += check_settings(head.text(SETTINGS))
    for p in ([AGENTS] if head.has(AGENTS) else []) + rules + agents:
        for n, ref in refs(p, head.text(p)):
            if not exists(head, ref):
                out.append(Finding(p, n, "claude.ref", "referenced path does not exist", key=ref))
    seen = {norm(l) for l in agents_text.split("\n")}
    for p in rules + agents:
        for n, line in enumerate(head.text(p).split("\n"), 1):
            s = norm(line)
            if len(s) >= DUP_MIN and s in seen:
                out.append(Finding(p, n, "claude.dup", "line repeats AGENTS.md; state it in one file", key=s))
    return out


OK_RULE = '---\npaths:\n  - "prod/*.yaml"\n---\n\n# prod\n\n- Start from a template.\n'
OK_AGENT = ("---\nname: rev\ndescription: Read-only review. Use proactively.\ntools: Read, Grep, Glob\n"
            "model: inherit\n---\n\nYou review.\n\n## Output\n\nVerdict.\n")
LONG = "- Every deployable is pinned and listed once in the status table.\n"
FILES = {
    AGENTS: "# AGENTS.md\n\n- `prod/`: [prod](.claude/rules/prod.md).\n- Review: [review](.claude/rules/review.md).\n"
            + LONG,
    "CLAUDE.md": "@AGENTS.md\n",
    "prod/a.yaml": "services: {}\n",
    ".claude/rules/prod.md": OK_RULE,
    UNSCOPED: "# Review\n\n- [ ] Read the rendered diff.\n",
    ".claude/agents/rev.md": OK_AGENT,
    SETTINGS: json.dumps({"permissions": {"deny": sorted(REQUIRED_DENY)}}, indent=2),
}


def _with(path, text):
    files = dict(FILES)
    files[path] = text
    return files


def _settings(**perms):
    return _with(SETTINGS, json.dumps({"permissions": perms}, indent=2))


CASES = [
    ("conforming .claude", None, FILES, None),
    ("no .claude files", None, {AGENTS: "# AGENTS.md\n", "CLAUDE.md": "@AGENTS.md\n"}, None),
    ("AGENTS.md over the cap", "claude.cap", _with(AGENTS, FILES[AGENTS] + "x\n" * AGENTS_CAP), None),
    ("always-loaded text over the cap", "claude.cap", _with(UNSCOPED, FILES[UNSCOPED] + "x\n" * LOADED_CAP), None),
    ("rule without paths", "claude.rule", _with(".claude/rules/prod.md", "# prod\n"), None),
    ("paths as a flow list", "claude.rule", _with(".claude/rules/prod.md", '---\npaths: ["prod/*.yaml"]\n---\n'), None),
    ("absolute glob", "claude.rule", _with(".claude/rules/prod.md", OK_RULE.replace('"prod/', '"/prod/')), None),
    ("import in a rule", "claude.rule", _with(".claude/rules/prod.md", OK_RULE + "@AGENTS.md\n"), None),
    ("glob matches nothing", "claude.glob", _with(".claude/rules/prod.md", OK_RULE.replace("*.yaml", "*.yml")), None),
    ("agent with Bash", "claude.agent", _with(".claude/agents/rev.md", OK_AGENT.replace("Glob\n", "Glob, Bash\n")), None),
    ("agent with Write", "claude.agent", _with(".claude/agents/rev.md", OK_AGENT.replace("Glob\n", "Glob, Write\n")),
     None),
    ("agent with Edit", "claude.agent", _with(".claude/agents/rev.md", OK_AGENT.replace("Glob\n", "Glob, Edit\n")), None),
    ("agent without tools", "claude.agent",
     _with(".claude/agents/rev.md", OK_AGENT.replace("tools: Read, Grep, Glob\n", "")), None),
    ("agent without name", "claude.agent", _with(".claude/agents/rev.md", OK_AGENT.replace("name: rev\n", "")), None),
    ("agent without description", "claude.agent",
     _with(".claude/agents/rev.md", OK_AGENT.replace("description: Read-only review. Use proactively.\n", "")), None),
    ("agent name differs from file", "claude.agent", _with(".claude/agents/rev.md", OK_AGENT.replace("rev", "other")),
     None),
    ("agent description with a colon", "claude.agent",
     _with(".claude/agents/rev.md", OK_AGENT.replace("Read-only review.", "Review: read-only.")), None),
    ("agent with an unknown model", "claude.agent", _with(".claude/agents/rev.md", OK_AGENT.replace("inherit", "gpt")),
     None),
    ("agent without an Output section", "claude.agent",
     _with(".claude/agents/rev.md", OK_AGENT.replace("## Output", "## Reply")), None),
    ("settings with an allow list", "claude.settings", _settings(deny=sorted(REQUIRED_DENY), allow=["Bash(*)"]), None),
    ("settings with a default mode", "claude.settings",
     _with(SETTINGS, json.dumps({"permissions": {"deny": sorted(REQUIRED_DENY)}, "defaultMode": "acceptEdits"})), None),
    ("settings without a required Read deny", "claude.settings",
     _settings(deny=sorted(REQUIRED_DENY - {"Read(**/.env*)"})), None),
    ("settings without a required Bash deny", "claude.settings",
     _settings(deny=sorted(REQUIRED_DENY - {"Bash(*.env*)"})), None),
    ("settings without an outward-action deny", "claude.settings",
     _settings(deny=sorted(REQUIRED_DENY - {"Bash(gh pr merge*)"})), None),
    ("settings without the WebFetch deny", "claude.settings", _settings(deny=sorted(REQUIRED_DENY - {"WebFetch"})),
     None),
    ("settings without a history deny", "claude.settings", _settings(deny=sorted(REQUIRED_DENY - {"Bash(git -C * push*)"})),
     None),
    ("settings with a scoped WebFetch deny", None, _settings(deny=sorted(REQUIRED_DENY) + ["WebFetch(domain:x.org)"]),
     None),
    ("settings deny of another tool", "claude.settings", _settings(deny=sorted(REQUIRED_DENY) + ["Task(x)"]), None),
    ("rule file not linked from AGENTS.md", "claude.ref", _with(".claude/rules/ci.md", OK_RULE), None),
    ("stale path in AGENTS.md", "claude.ref", _with(AGENTS, FILES[AGENTS] + "- `ci/common.py`\n"), None),
    ("rule repeats AGENTS.md", "claude.dup", _with(".claude/rules/prod.md", OK_RULE + LONG), None),
]
