"""status: README status table, served-name sets, candidate rules and container names."""

import re

from common import Finding, engine_kind, engines, find_line, served_names, service_tokens

RULES = [
    "status.table", "status.served-name", "status.candidate", "status.key-set", "status.container-name",
    "status.stability",
]
DIFF_RULES = ["status.stability"]
BEGIN, END = "<!-- status:begin -->", "<!-- status:end -->"


def served_set(path, rendered, services_allow, out=None):
    names = set()
    for key in engines(path, rendered, services_allow):
        svc = rendered["services"][key]
        found = served_names(engine_kind(key, svc), service_tokens(svc))
        if not found and out is not None:
            out.append(Finding(path, 0, "status.served-name", f"engine {key} has no served model name", key=key))
        names.update(found)
    return names


def parse_table(text):
    if text.count(BEGIN) != 1 or text.count(END) != 1 or text.index(BEGIN) > text.index(END):
        return None, "README.md must have exactly one status marker pair"
    start = text[: text.index(BEGIN)].count("\n") + 1
    body = text[text.index(BEGIN) + len(BEGIN): text.index(END)].strip("\n").split("\n")
    rows = []
    if len(body) < 2 or [c.strip() for c in body[0].strip().strip("|").split("|")] != ["file", "serves", "status"]:
        return None, "status table header must be | file | serves | status |"
    for i, line in enumerate(body[2:], start + 3):
        cells = [c.strip().strip("`") for c in line.strip().strip("|").split("|")]
        if len(cells) != 3:
            return None, f"status table row {i} must have 3 cells"
        rows.append((i, *cells))
    return rows, None


def check(ctx):
    out, head = [], ctx.head
    sa = ctx.services_allow
    sets, keysets = {}, {}
    for p in head.deployables() + list(ctx.templates):
        r = ctx.rendered(p)
        if r is None:
            continue
        text = head.text(p)
        for key, svc in r.get("services", {}).items():
            if svc.get("container_name") != key:
                out.append(Finding(p, find_line(text, rf"^  {re.escape(key)}:"), "status.container-name",
                                   f"service {key} must set container_name equal to its key", key=key))
        sets[p] = served_set(p, r, sa, out)
        keysets[p] = sorted(r.get("services", {}))
    if ctx.base is not None:
        for p in head.deployables():
            if ctx.base.has(p) and p in sets:
                rb = ctx.rendered_base(p)
                if rb is not None and served_set(p, rb, ctx.services_allow_base) != sets[p]:
                    out.append(Finding(p, 0, "status.stability", "served-name set changed in place; add a new file"))
    if not head.has("README.md"):
        out.append(Finding("README.md", 0, "status.table", "README.md is missing"))
        return out
    text = head.text("README.md")
    rows, err = parse_table(text)
    if err:
        out.append(Finding("README.md", 0, "status.table", err))
        return out
    seen, groups = {}, {}
    for line, f, serves, status in rows:
        if f in seen:
            out.append(Finding("README.md", line, "status.table", "file listed twice"))
            continue
        seen[f] = (line, serves, status)
        if f not in head.deployables():
            out.append(Finding("README.md", line, "status.table", "row for a file that is not a deployable"))
            continue
        if status not in ("live", "candidate"):
            out.append(Finding("README.md", line, "status.table", "status must be live or candidate"))
        want = ",".join(sorted(sets.get(f, ()))) or "-"
        if f in sets and serves != want:
            out.append(Finding("README.md", line, "status.table", f"serves must be {want}"))
        groups.setdefault(serves, []).append((f, status, line))
    for p in head.deployables():
        if p not in seen:
            out.append(Finding("README.md", 0, "status.table", f"no status row for {p}"))
    for serves, members in groups.items():
        cands = [m for m in members if m[1] == "candidate"]
        if len(cands) > 1:
            out.append(Finding("README.md", cands[1][2], "status.candidate", "at most one candidate per serving group"))
        live = [m[0] for m in members if m[1] == "live"]
        for f, _, line in cands:
            if serves != "-" and not live:
                continue
            if f in keysets and not any(keysets.get(l) == keysets[f] for l in live):
                keys = "".join(k + "\n" for k in keysets[f])
                out.append(Finding(f, 0, "status.key-set", "candidate service keys differ from the live file",
                                   key=keys))
    return out


def _compose(keys, served="glm-5.3", extra=""):
    return {"services": {k: {"container_name": k, "command": ["sglang", "serve", "--model-path", "m",
                                                               "--served-model-name", served] + (
        [extra] if extra else [])} if k.startswith("model-") else {"container_name": k} for k in keys}}


def _readme(rows):
    return "x\n" + BEGIN + "\n| file | serves | status |\n|---|---|---|\n" + "".join(
        f"| `{f}` | {s} | {st} |\n" for f, s, st in rows) + END + "\n"


A, B = "prod/glm-5.3-sglang-tp4.yaml", "prod/glm-5.3-sglang-tp8.yaml"
FILES = {A: "services:\n", "README.md": _readme([(A, "glm-5.3", "live")])}
REND = {A: _compose(["model-sg-a", "proxy-a"])}
TWO = dict(FILES, **{B: "services:\n", "README.md": _readme([(A, "glm-5.3", "live"), (B, "glm-5.3", "candidate")])})
CASES = [
    ("one live file", None, FILES, FILES, REND, REND),
    ("candidate with the same keys", None, TWO, FILES, dict(REND, **{B: _compose(["model-sg-a", "proxy-a"])}), REND),
    ("diff rule green without base README", None, FILES, {A: "services:\n"}, REND, REND),
    ("missing row", "status.table", dict(FILES, **{"README.md": _readme([])}), FILES, REND, REND),
    ("wrong serves", "status.table", dict(FILES, **{"README.md": _readme([(A, "glm-5.2", "live")])}), FILES, REND,
     REND),
    ("engine without a served name", "status.served-name", dict(FILES, **{"README.md": _readme([(A, "-", "live")])}),
     FILES,
     {A: {"services": {"model-sg-a": {"container_name": "model-sg-a", "command": ["x"]}}}}, None),
    ("two candidates", "status.candidate",
     dict(TWO, **{"prod/glm-5.3-sglang-tp2.yaml": "services:\n",
                  "README.md": _readme([("prod/glm-5.3-sglang-tp2.yaml", "glm-5.3", "candidate"), (A, "glm-5.3", "live"),
                                        (B, "glm-5.3", "candidate")])}), FILES,
     dict(REND, **{B: _compose(["model-sg-a", "proxy-a"]),
                   "prod/glm-5.3-sglang-tp2.yaml": _compose(["model-sg-a", "proxy-a"])}), REND),
    ("candidate with other keys", "status.key-set", TWO, FILES, dict(REND, **{B: _compose(["model-sg-b", "proxy-a"])}),
     REND),
    ("container_name differs", "status.container-name", FILES, FILES,
     {A: {"services": {"model-sg-a": {"container_name": "x", "command": ["--served-model-name", "glm-5.3"]}}}}, None),
    ("served name changed in place", "status.stability", dict(FILES, **{"README.md": _readme([(A, "glm-5.4", "live")])}),
     FILES, {A: _compose(["model-sg-a"], "glm-5.4")}, REND),
]
