"""status: README status table, served-name sets, candidate rules and container names.

The table is generated from the rendered files (`ci/run.py --write-status`); only `status` is written by hand.
"""

import re
from collections import OrderedDict

from common import (
    EXCEPTIONS, Finding, engine_kind, engines, find_line, flag_values, image_rows, legacy_lines, load_json, model_ref,
    served_names, service_tokens, sha256, under_legacy,
)
from pins import SNAPSHOT
from privacy import check_engine

RULES = [
    "status.table", "status.served-name", "status.candidate", "status.key-set", "status.container-name",
    "status.stability",
]
DIFF_RULES = ["status.stability"]
BEGIN, END = "<!-- status:begin -->", "<!-- status:end -->"
REPO_URL = "https://github.com/nearai/cvm-compose-files"
HEADER = ["file", "serves", "model@revision", "engine image", "topology", "privacy", "status"]
NOTE = re.compile(r"^- `([^`]+)`: \S")


def served_set(path, rendered, services_allow, out=None):
    names = set()
    for key in engines(path, rendered, services_allow):
        svc = rendered["services"][key]
        found = served_names(engine_kind(key, svc), service_tokens(svc))
        if not found and out is not None:
            out.append(Finding(path, 0, "status.served-name", f"engine {key} has no served model name", key=key))
        names.update(found)
    return names


def _int(toks, flags):
    for f in flags:
        v = flag_values(toks, f)
        if v and v[0].isdigit():
            return int(v[0])
    return 1


def _model(kind, toks):
    m = model_ref(kind, toks) or "-"
    snap = SNAPSHOT.search(m)
    if snap:
        return f"{snap.group(1)}/{snap.group(2)}@{snap.group(3)[:12]}"
    revs = [v for v in flag_values(toks, "--revision") if re.fullmatch(r"[0-9a-f]{40}", v)]
    return f"{m}@{revs[0][:12] if revs else '-'}"


def _image(svc, rows):
    image = svc.get("image", "")
    digest = image.rpartition("@")[2] if "@sha256:" in image else None
    row = (rows or {}).get(image) or next((r for k, r in (rows or {}).items() if digest and k.endswith("@" + digest)),
                                          None)
    if row is None:
        return "built on host" if svc.get("build") else "not listed"
    parts = row[1]
    if len(parts) > 3 and parts[2] != "-":
        # Link the build dir at the commit the image was built from; the dir may be gone from HEAD.
        return f"[docker/{parts[2]}@{parts[3][:7]}]({REPO_URL}/tree/{parts[3]}/docker/{parts[2]})"
    return "third-party " + parts[0].split("@")[0]


def _privacy(path, key, svc, waived):
    misses = [f for f in check_engine(path, 0, key, svc, True) if f.rule == "privacy.no-log-flag"]
    if not misses:
        return "flag set"
    return "waived" if all(sha256(f.key) in waived for f in misses) else "missing"


def file_rows(path, rendered, services_allow, rows, waived, legacy):
    """Table cells (without status) for one deployable, one row per served name and engine config."""
    name = f"`{path}`" + (" (legacy name)" if under_legacy(path, legacy) else "")
    groups = OrderedDict()
    for key in engines(path, rendered, services_allow):
        svc = rendered["services"][key]
        kind, toks = engine_kind(key, svc), service_tokens(svc)
        tp = _int(toks, ("--tp-size", "--tp", "--tensor-parallel-size"))
        dp = _int(toks, ("--dp-size", "--dp", "--data-parallel-size"))
        ep = "--enable-expert-parallel" in toks or "--enable-dp-attention" in toks
        unit = (f"dep{dp}" if ep else f"dp{dp}") if dp > 1 else f"tp{tp}"
        # One engine spans dp services when each service is a single data-parallel rank.
        span = dp if flag_values(toks, "--data-parallel-rank") else 1
        g = (tuple(sorted(served_names(kind, toks))) or ("-",), _model(kind, toks), _image(svc, rows), unit,
             _privacy(path, key, svc, waived))
        groups.setdefault(g, [0, span])[0] += 1
    if not groups:
        return [(name, "-", "-", "-", "-", "n.a.")]
    out = []
    for (served, model, image, unit, priv), (n, span) in sorted(groups.items()):
        reps = max(n // span, 1)
        topo = unit + (f"x{reps}" if reps > 1 else "")
        out += [(name, s, model, image, topo, priv) for s in served]
    return out


def table_rows(ctx):
    """Generated rows for every deployable, in path order."""
    head, sa = ctx.head, ctx.services_allow
    waived = {e.get("sha256") for e in load_json(head, EXCEPTIONS, []) if isinstance(e, dict)
              and e.get("rule") == "privacy.no-log-flag"}
    rows, legacy, out = image_rows(head), legacy_lines(head), {}
    for p in head.deployables():
        r = ctx.rendered(p)
        if r is not None:
            out[p] = file_rows(p, r, sa, rows, waived, legacy)
    return out


def render_table(generated, status):
    lines = ["| " + " | ".join(HEADER) + " |", "|" + "---|" * len(HEADER)]
    for p, rows in generated.items():
        lines += ["| " + " | ".join(r + (status.get(p, "candidate"),)) + " |" for r in rows]
    return "\n".join(lines)


def parse_table(text):
    if text.count(BEGIN) != 1 or text.count(END) != 1 or text.index(BEGIN) > text.index(END):
        return None, "README.md must have exactly one status marker pair"
    start = text[: text.index(BEGIN)].count("\n") + 1
    body = text[text.index(BEGIN) + len(BEGIN): text.index(END)].strip("\n").split("\n")
    rows = []
    if len(body) < 2 or [c.strip() for c in body[0].strip().strip("|").split("|")] != HEADER:
        return None, "status table header must be | " + " | ".join(HEADER) + " |"
    for i, line in enumerate(body[2:], start + 3):
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != len(HEADER):
            return None, f"status table row {i} must have {len(HEADER)} cells"
        rows.append((i, re.sub(r"^`([^`]+)`.*$", r"\1", cells[0]), cells))
    return rows, None


def file_status(rows):
    """{path: status} from parsed rows; the first row of a file wins."""
    out = {}
    for _, f, cells in rows:
        out.setdefault(f, cells[-1])
    return out


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
    generated, status, groups = table_rows(ctx), {}, {}
    for line, f, cells in rows:
        if f not in head.deployables():
            out.append(Finding("README.md", line, "status.table", "row for a file that is not a deployable"))
            continue
        if cells[-1] not in ("live", "candidate"):
            out.append(Finding("README.md", line, "status.table", "status must be live or candidate"))
        if status.setdefault(f, cells[-1]) != cells[-1]:
            out.append(Finding("README.md", line, "status.table", "rows of one file must have one status"))
    for p in head.deployables():
        if p not in status:
            out.append(Finding("README.md", 0, "status.table", f"no status row for {p}"))
        elif p in generated and [tuple(c[:-1]) for _, f, c in rows if f == p] != generated[p]:
            out.append(Finding("README.md", next(l for l, f, _ in rows if f == p), "status.table",
                               f"rows for {p} differ from the rendered file; run ci/run.py --write-status"))
    if [f for _, f, _ in rows] != sorted([f for _, f, _ in rows]):
        out.append(Finding("README.md", 0, "status.table", "rows must be in file path order"))
    for f, st in status.items():
        groups.setdefault(",".join(sorted(sets.get(f, ()))) or "-", []).append(
            (f, st, next(l for l, g, _ in rows if g == f)))
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
    # A served name in more than one live file needs one note below the table on how the files differ.
    shared = {}
    for f, st in status.items():
        for n in sets.get(f, ()) if st == "live" else ():
            shared.setdefault(n, []).append(f)
    notes = [m.group(1) for m in map(NOTE.match, text[text.index(END):].split("\n")) if m]
    for n, fs in sorted(shared.items()):
        if len(fs) > 1 and notes.count(n) != 1:
            out.append(Finding("README.md", 0, "status.table", f"{n} is served by {len(fs)} live files: add one "
                               f"note `- `{n}`: ...` below the table saying how they differ"))
    return out


def write_status(ctx):
    """Rewrite the README status table from the rendered files, keeping each file's status."""
    text = ctx.head.text("README.md")
    rows, err = parse_table(text)
    status = file_status(rows or [])
    table = render_table(table_rows(ctx), status)
    return text[: text.index(BEGIN) + len(BEGIN)] + "\n" + table + "\n" + text[text.index(END):]


def _compose(keys, served="glm-5.3", extra=""):
    return {"services": {k: {"container_name": k, "command": ["sglang", "serve", "--model-path", "m",
                                                               "--log-requests-level", "0",
                                                               "--served-model-name", served] + (
        [extra] if extra else [])} if k.startswith("model-") else {"container_name": k} for k in keys}}


ENGINE = ("m@-", "not listed", "tp1", "flag set")


def _readme(rows, after=""):
    """rows: (file, serves, status) or (file, serves, generated cells, status)."""
    body = "".join(f"| {r[0] if r[0].startswith('`') else '`' + r[0] + '`'} | {r[1]} | " + " | ".join(r[2] if len(r) == 4 else ENGINE) + f" | {r[-1]} |\n"
                   for r in rows)
    return "x\n" + BEGIN + "\n| " + " | ".join(HEADER) + " |\n|" + "---|" * len(HEADER) + "\n" + body + END + "\n" + after


A, B = "prod/glm-5.3-sglang-tp4.yaml", "prod/glm-5.3-sglang-tp8.yaml"
FILES = {A: "services:\n", "README.md": _readme([(A, "glm-5.3", "live")])}
REND = {A: _compose(["model-sg-a", "proxy-a"])}
TWO = dict(FILES, **{B: "services:\n", "README.md": _readme([(A, "glm-5.3", "live"), (B, "glm-5.3", "candidate")])})
LIVE2 = dict(TWO, **{"README.md": _readme([(A, "glm-5.3", "live"), (B, "glm-5.3", "live")])})
REND2 = dict(REND, **{B: _compose(["model-sg-a", "proxy-a"])})
CASES = [
    ("one live file", None, FILES, FILES, REND, REND),
    ("candidate with the same keys", None, TWO, FILES, REND2, REND),
    ("diff rule green without base README", None, FILES, {A: "services:\n"}, REND, REND),
    ("two live files with a note", None,
     dict(LIVE2, **{"README.md": LIVE2["README.md"] + "\n- `glm-5.3`: tp8 has more context.\n"}), FILES, REND2, REND),
    ("legacy name and a utility file", None,
     dict(FILES, **{"prod/util-x.yaml": "services:\n", "ci/data/legacy.txt": A + "\n",
                    "README.md": _readme([(f"`{A}` (legacy name)", "glm-5.3", "live"),
                                          ("prod/util-x.yaml", "-", ("-", "-", "-", "n.a."), "live")])}), FILES,
     dict(REND, **{"prod/util-x.yaml": {"services": {"x": {"container_name": "x"}}}}), REND),
    ("missing row", "status.table", dict(FILES, **{"README.md": _readme([])}), FILES, REND, REND),
    ("wrong serves", "status.table", dict(FILES, **{"README.md": _readme([(A, "glm-5.2", "live")])}), FILES, REND,
     REND),
    ("hand-edited topology", "status.table",
     dict(FILES, **{"README.md": _readme([(A, "glm-5.3", ("m@-", "not listed", "tp8", "flag set"), "live")])}), FILES,
     REND, REND),
    ("logging flag dropped, row not updated", "status.table", FILES, FILES,
     {A: {"services": {"model-sg-a": {"container_name": "model-sg-a", "command": [
         "sglang", "serve", "--model-path", "m", "--served-model-name", "glm-5.3"]}}}}, REND),
    ("shared model without a note", "status.table", LIVE2, FILES, REND2, REND),
    ("engine without a served name", "status.served-name",
     dict(FILES, **{"README.md": _readme([(A, "-", ("-@-", "not listed", "tp1", "flag set"), "live")])}), FILES,
     {A: {"services": {"model-sg-a": {"container_name": "model-sg-a", "command": ["--log-requests-level", "0"]}}}},
     None),
    ("two candidates", "status.candidate",
     dict(TWO, **{"prod/glm-5.3-sglang-tp2.yaml": "services:\n",
                  "README.md": _readme([("prod/glm-5.3-sglang-tp2.yaml", "glm-5.3", "candidate"), (A, "glm-5.3", "live"),
                                        (B, "glm-5.3", "candidate")])}), FILES,
     dict(REND2, **{"prod/glm-5.3-sglang-tp2.yaml": _compose(["model-sg-a", "proxy-a"])}), REND),
    ("candidate with other keys", "status.key-set", TWO, FILES, dict(REND, **{B: _compose(["model-sg-b", "proxy-a"])}),
     REND),
    ("container_name differs", "status.container-name", FILES, FILES,
     {A: {"services": {"model-sg-a": dict(REND[A]["services"]["model-sg-a"], container_name="x")}}}, None),
    ("served name changed in place", "status.stability", dict(FILES, **{"README.md": _readme([(A, "glm-5.4", "live")])}),
     FILES, {A: _compose(["model-sg-a"], "glm-5.4")}, REND),
]
