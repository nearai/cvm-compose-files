"""pins: image digests and allowlist rows, model revisions, pinned URLs, banned compose keys."""

import json
import re

from common import (
    DIGEST_REF, IMAGES, SHA40, Finding, engine_kind, engines, find_line, image_rows, key_paths, model_ref,
    service_tokens, sha256,
)

RULES = [
    "pins.digest", "pins.allow", "pins.rows", "pins.docker-dir", "pins.hf-revision", "pins.remote-code", "pins.url",
    "pins.raw-key",
]
DIFF_RULES = ["pins.rows", "pins.docker-dir"]

ROW_REF = re.compile(r"^[\w./:-]+@sha256:[0-9a-f]{64}$")
SNAPSHOT = re.compile(r"models--([^/]+)--([^/]+)/snapshots/([0-9a-f]{40})(/|$)")
HF_ID = re.compile(r"^[\w.-]+/[\w.-]+$")
URL = re.compile(r"https?://[^\s'\"<>)`\\]+")
RAW_KEY = re.compile(r"^\s*(?:-\s+)?(include|extends|env_file|build|dockerfile_inline)\s*:")


def downloads(rendered):
    """(repo, revision or None) for every `hf download` / `huggingface-cli download` in the file."""
    out = []
    for svc in rendered.get("services", {}).values():
        toks = service_tokens(svc)
        for i, t in enumerate(toks):
            if (t == "download" and i > 0 and toks[i - 1] in ("hf", "huggingface-cli")) and i + 1 < len(toks):
                repo, rev = toks[i + 1], None
                for j in range(i + 2, len(toks)):
                    if toks[j] in ("hf", "huggingface-cli", "&&", ";", "||") or toks[j].startswith("echo"):
                        break
                    if toks[j] == "--revision" and j + 1 < len(toks):
                        rev = toks[j + 1]
                    elif toks[j].startswith("--revision="):
                        rev = toks[j].split("=", 1)[1]
                out.append((repo, rev))
    return out


def check(ctx):
    out, head = [], ctx.head
    rows = image_rows(head) or {}
    base_rows = image_rows(ctx.base) if ctx.base else None
    for p in head.deployables() + list(ctx.templates):
        r = ctx.rendered(p)
        text = head.text(p)
        if r is not None:
            out += check_file(p, text, r, rows, ctx)
        out += check_raw(p, text)
    out += check_rows(ctx, rows, base_rows)
    return out


def check_file(p, text, r, rows, ctx):
    out = []
    for key, svc in sorted(r.get("services", {}).items()):
        image = svc.get("image") or ""
        line = find_line(text, rf"^  {re.escape(key)}:")
        if not DIGEST_REF.match(image):
            out.append(Finding(p, line, "pins.digest", f"service {key}: image must be repo@sha256:<64 hex> only",
                               key=image or key))
        if "@sha256:" in image and image not in rows:
            out.append(Finding(p, line, "pins.allow", f"service {key}: image has no row in {IMAGES}", key=image))
    dl = downloads(r)
    for repo, rev in dl:
        if not (rev and SHA40.match(rev)):
            out.append(Finding(p, find_line(text, re.escape(repo)), "pins.hf-revision",
                               "download must pin --revision to a 40-hex commit", key=repo))
    for key in engines(p, r, ctx.services_allow):
        svc = r["services"][key]
        toks = service_tokens(svc)
        model = model_ref(engine_kind(key, svc), toks)
        line = find_line(text, rf"^  {re.escape(key)}:")
        pinned = False
        m = SNAPSHOT.search(model or "")
        if m:
            repo = f"{m.group(1)}/{m.group(2)}"
            if (repo, m.group(3)) in dl:
                pinned = True
            else:
                out.append(Finding(p, line, "pins.hf-revision",
                                   f"engine {key}: no downloader fetches the same repo at the same revision",
                                   key=model))
        elif model and HF_ID.match(model):
            revs = [toks[i + 1] for i, t in enumerate(toks[:-1]) if t == "--revision"]
            if revs and all(SHA40.match(v) for v in revs):
                pinned = True
            else:
                out.append(Finding(p, line, "pins.hf-revision", f"engine {key}: model needs --revision <40 hex>",
                                   key=model))
        else:
            out.append(Finding(p, line, "pins.hf-revision",
                               f"engine {key}: model is not a pinned snapshot or a repo with a revision",
                               key=model or key))
        if "--trust-remote-code" in toks and not pinned:
            out.append(Finding(p, line, "pins.remote-code", f"engine {key}: --trust-remote-code needs a pinned model",
                               key=model or key))
    return out


def check_raw(p, text):
    out = []
    lines, kinds, paths = key_paths(text)
    for n, (line, kind, kp) in enumerate(zip(lines, kinds, paths), 1):
        if kind == "code":
            m = RAW_KEY.match(line)
            if m:
                svc = kp[1] if len(kp) > 1 and kp[0] == "services" else ""
                out.append(Finding(p, n, "pins.raw-key", f"`{m.group(1)}` is not allowed in deployable files",
                                   key=f"{svc}:{m.group(1)}"))
        for url in URL.findall(line):
            if not url_pinned(url):
                out.append(Finding(p, n, "pins.url", "URL must pin a 40-hex commit or revision", key=url))
    return out


def url_pinned(url):
    m = re.match(r"https?://raw\.githubusercontent\.com/[^/]+/[^/]+/([^/]+)", url)
    if m:
        return bool(SHA40.match(m.group(1)))
    m = re.match(r"https?://github\.com/[^/]+/[^/]+/(?:archive|blob|raw|tree)/([^/.]+)", url)
    if m:
        return bool(SHA40.match(m.group(1)))
    m = re.match(r"https?://(?:huggingface\.co|hf\.co)/.+?/(?:resolve|blob|raw|tree)/([^/]+)", url)
    if m:
        return bool(SHA40.match(m.group(1)))
    return True


def check_rows(ctx, rows, base_rows):
    out, head = [], ctx.head
    if not head.has(IMAGES):
        return [Finding(IMAGES, 0, "pins.rows", "image allowlist is missing")]
    lines = head.text(IMAGES).splitlines()
    if len(set(lines)) != len(lines) or len({l.split()[0] for l in lines if l.split()}) != len(lines):
        out.append(Finding(IMAGES, 0, "pins.rows", "rows must be unique per image"))
    for ref, (n, parts) in rows.items():
        if len(parts) != 4 or not ROW_REF.match(parts[0]):
            out.append(Finding(IMAGES, n, "pins.rows", "row must be: repo@sha256:<64> tag dir commit"))
            continue
        _, tag, d, commit = parts
        if not (commit == "-" or SHA40.match(commit)) or (d == "-") != (tag == "-"):
            out.append(Finding(IMAGES, n, "pins.rows", "commit must be 40 hex or -; dir and tag go together"))
        if base_rows is not None and ref not in base_rows and d != "-":
            engine = ref.split("@")[0].rsplit("/", 1)[-1].split(":")[0]
            if d != f"{engine}-{tag}" or not any(p.startswith(f"docker/{d}/") for p in head.paths):
                out.append(Finding(IMAGES, n, "pins.rows", "new row: dir must be <engine>-<tag> and exist"))
    if ctx.base is not None:
        base_dirs = {p.split("/")[1] for p in ctx.base.paths if p.startswith("docker/") and p.count("/") >= 2}
        new_dirs = sorted({p.split("/")[1] for p in head.paths if p.startswith("docker/") and p.count("/") >= 2}
                          - base_dirs)
        for d in new_dirs:
            df = f"docker/{d}/Dockerfile"
            if not head.has(df):
                out.append(Finding(df, 0, "pins.docker-dir", "new build dir needs a Dockerfile"))
                continue
            text = head.text(df)
            for n, line in enumerate(text.splitlines(), 1):
                m = re.match(r"(?i)^\s*FROM\s+(?:--\S+\s+)*(\S+)", line)
                if m and not (DIGEST_REF.match(m.group(1)) and m.group(1) in rows):
                    out.append(Finding(df, n, "pins.docker-dir", "FROM must be a digest with a row", key=m.group(1)))
    return out


D = "a" * 64
IMG = f"docker.io/nearaidev/sglang@sha256:{D}"
REV = "c" * 40
ROWS = f"{IMG} x-r1 sglang-x-r1 {'d' * 40}\n"
SNAP = f"/root/.cache/huggingface/hub/models--org--m/snapshots/{REV}"
ENGINE = {"container_name": "model-sg-a", "image": IMG,
          "command": ["sglang", "serve", "--model-path", SNAP, "--served-model-name", "m"]}
DL = {"container_name": "model-downloader", "image": IMG, "entrypoint": ["sh", "-c"],
      "command": [f"hf download org/m --revision {REV}\n"]}
P = "prod/m-sglang-tp1.yaml"
FILES = {P: "services:\n  model-sg-a:\n    image: x\n", "ci/data/images.allow": ROWS,
         "docker/sglang-x-r1/Dockerfile": f"FROM {IMG}\n"}
REND = {P: {"services": {"model-sg-a": ENGINE, "model-downloader": DL}}}


def _svc(**kw):
    return {P: {"services": {"model-sg-a": dict(ENGINE, **kw), "model-downloader": DL}}}


CASES = [
    ("pinned engine and downloader", None, FILES, FILES, REND, None),
    ("diff rules green without base allowlist", None, FILES, {P: "x"}, REND, None),
    ("tag-only image", "pins.digest", FILES, FILES, _svc(image="nearaidev/sglang:latest"), None),
    ("digest without a row", "pins.allow", FILES, FILES, _svc(image=IMG.replace("a" * 64, "b" * 64)), None),
    ("new row with a wrong dir", "pins.rows",
     dict(FILES, **{"ci/data/images.allow": ROWS + f"x/sglang@sha256:{'e' * 64} y-r1 sglang-z-r1 {'d' * 40}\n"}), FILES,
     REND, None),
    ("new build dir with an unpinned FROM", "pins.docker-dir",
     dict(FILES, **{"docker/sglang-y-r1/Dockerfile": "FROM ubuntu:24.04\n"}), FILES, REND, None),
    ("model without revision", "pins.hf-revision", FILES, FILES,
     _svc(command=["sglang", "serve", "--model-path", "org/m", "--served-model-name", "m"]), None),
    ("remote code without revision (missing revision waived)", "pins.remote-code",
     dict(FILES, **{"ci/data/exceptions.json": json.dumps([{
         "rule": "pins.hf-revision", "path": P, "sha256": sha256("org/n"), "reason": "r", "remove_when": "w"}])}),
     FILES,
     {P: {"services": {"model-sg-a": dict(ENGINE, command=["sglang", "serve", "--model-path", "org/m", "--revision",
                                                           REV, "--trust-remote-code"]), "model-downloader": DL,
                       "model-sg-b": dict(ENGINE, container_name="model-sg-b",
                                          command=["sglang", "serve", "--model-path", "org/n", "--trust-remote-code"])}}},
     None),
    ("branch URL", "pins.url",
     dict(FILES, **{P: "services:\n  a:\n    command: curl https://raw.githubusercontent." + "com/o/r/main/f\n"}), FILES,
     REND, None),
    ("dockerfile_inline", "pins.raw-key",
     dict(FILES, **{P: "services:\n  a:\n    build:\n      dockerfile_inline: |\n        FROM x\n"}), FILES,
     REND, None),
]
