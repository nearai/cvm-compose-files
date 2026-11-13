"""Shared constants and helpers for the ci/ guards. Stdlib only."""

import hashlib
import json
import os
import re
import shlex
import subprocess
import tempfile

# Renderer: Docker 29.9.3 CLI with the compose plugin 5.7.0.
COMPOSE_IMAGE = (
    "docker.io/library/docker:29-cli"
    "@sha256:2357639ad1ea7d0194134bf3292622fd4e562483966ed674e5d7c945b5f3b28d"
)
PUBLISH_ENV = "publish"
ORG = "nearai"

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXCEPTIONS = "ci/data/exceptions.json"
EXPECTED = "ci/data/expected.json"
LEGACY = "ci/data/legacy.txt"
IMAGES = "ci/data/images.allow"
SERVICES = "ci/data/services.allow.json"
PUBLIC = "ci/data/public.json"
TEMPLATES = ("ci/templates/sglang.yaml", "ci/templates/vllm.yaml")

# Public allowlists: hosts, repo owners, org repos, IPs and runners that may appear in repo text.
with open(os.path.join(ROOT, PUBLIC)) as _f:
    _public = json.load(_f)
PUBLIC_HOSTS = set(_public["hosts"])
PUBLIC_OWNERS = set(_public["owners"])
ORG_PUBLIC_REPOS = set(_public["org_public_repos"])
IP_ALLOW = set(_public["ips"])
IP_ALLOW_PREFIX = tuple(_public["ip_prefixes"])
RUNNERS = set(_public["runners"])

LIFECYCLE = re.compile(
    r"(?i)(^|[-_.\s])(canary|candidate|test|tmp|temp|old|new|backup|wip|draft|legacy|final|copy|prev|previous"
    r"|v\d+|gpu[\s_-]?\d+|cpu[\s_-]?\d+|\d{8}|\d{4}-\d{2}-\d{2})($|[-_.\s])"
)
SHA40 = re.compile(r"^[0-9a-f]{40}$")
DIGEST_REF = re.compile(r"^[\w./-]+@sha256:[0-9a-f]{64}$")
SECRET_SUFFIX = re.compile(r"_(TOKEN|KEY|SECRET|PASSWORD|CREDENTIALS)$")
ENGINE_KEY = re.compile(r"^model-(sg|vllm)-")


def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Finding:
    def __init__(self, path, line, rule, msg, key=None):
        self.path, self.line, self.rule, self.msg = path, line or 0, rule, msg
        # The text an exception entry hashes; never printed.
        self.key = key if key is not None else path

    def __str__(self):
        return f"{self.path}:{self.line}: {self.rule}: {self.msg}"


# ---------------------------------------------------------------- git trees

def git(*args, check=True):
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True)
    if check and r.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed: {r.stderr.decode(errors='replace').strip()}")
    return r.stdout


class Tree:
    """Files of one revision. rev None = the working tree (tracked files)."""

    def __init__(self, rev=None, files=None):
        self.rev, self._cache = rev, {}
        if files is not None:
            self._cache = {p: (c.encode() if isinstance(c, str) else c) for p, c in files.items()}
            self.paths = sorted(self._cache)
        elif rev is None:
            # Untracked, not-ignored files count too, so a local run sees new files before `git add`.
            listed = git("ls-files", "-z", "--cached", "--others", "--exclude-standard").decode().split("\0")[:-1]
            self.paths = sorted({p for p in listed if os.path.isfile(os.path.join(ROOT, p))})
        else:
            self.paths = sorted(git("ls-tree", "-r", "--name-only", "-z", rev).decode().split("\0")[:-1])
        self.pathset = set(self.paths)

    def has(self, path):
        return path in self.pathset

    def data(self, path):
        if path not in self._cache:
            if self.rev is None:
                with open(os.path.join(ROOT, path), "rb") as f:
                    self._cache[path] = f.read()
            else:
                self._cache[path] = git("show", f"{self.rev}:{path}")
        return self._cache[path]

    def text(self, path):
        return self.data(path).decode("utf-8", errors="replace")

    def deployables(self):
        return [p for p in self.paths if re.fullmatch(r"prod/[^/]+\.yaml", p)]


def is_text(data):
    if b"\0" in data[:8192]:
        return False
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


# ---------------------------------------------------------------- rendering

def env_lines(services_allow):
    return "".join(f"{e['name']}={e.get('dummy', 'ci-placeholder')}\n" for e in services_allow.get("env", []))


def render_dir(directory, path, env_text, interpolate=True):
    """Render one compose file with the pinned renderer; returns parsed JSON.

    interpolate=False keeps `${NAME:-default}` and `$$` as written, with anchors resolved.
    """
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "ci.env"), "w") as f:
            f.write(env_text)
        cmd = [
            "docker", "run", "--rm", "--network", "none",
            "-v", f"{directory}:/repo:ro", "-v", f"{tmp}:/ci-env:ro", "-w", "/repo",
            COMPOSE_IMAGE, "docker", "compose", "-p", "ci", "--profile", "*",
            "--env-file", "/ci-env/ci.env", "-f", path, "config", "--format", "json",
        ]
        if not interpolate:
            cmd.insert(cmd.index("config") + 1, "--no-interpolate")
        r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RenderError(path, r.stderr.decode(errors="replace").strip().splitlines()[-1:] or ["render failed"])
    return json.loads(r.stdout)


class RenderError(Exception):
    def __init__(self, path, lines):
        super().__init__(f"{path}: {' '.join(lines)}")
        self.path = path


class Renderer:
    """Renders deployables of a Tree; base trees are exported to a temp dir."""

    def __init__(self, tree, services_allow, interpolate=True):
        self.tree, self.env, self._cache, self._dir = tree, env_lines(services_allow), {}, None
        self.interpolate = interpolate

    def directory(self):
        if self._dir is None:
            if self.tree.rev is None:
                self._dir = ROOT
            else:
                self._tmp = tempfile.TemporaryDirectory()
                arch = git("archive", "--format=tar", self.tree.rev)
                subprocess.run(["tar", "-x", "-C", self._tmp.name], input=arch, check=True)
                self._dir = self._tmp.name
        return self._dir

    def __call__(self, path):
        if path not in self._cache:
            self._cache[path] = render_dir(self.directory(), path, self.env, self.interpolate)
        return self._cache[path]


# ---------------------------------------------------------------- data files

def load_json(tree, path, default):
    if not tree or not tree.has(path):
        return default
    return json.loads(tree.text(path))


def legacy_lines(tree):
    if not tree or not tree.has(LEGACY):
        return None
    return tree.text(LEGACY).splitlines()


def under_legacy(path, entries):
    for e in entries or ():
        if path == e or (e.endswith("/") and path.startswith(e)):
            return True
    return False


def image_rows(tree):
    rows = {}
    if not tree or not tree.has(IMAGES):
        return None
    for n, line in enumerate(tree.text(IMAGES).splitlines(), 1):
        parts = line.split()
        rows[parts[0] if parts else ""] = (n, parts)
    return rows


# ---------------------------------------------------------------- build dirs

SUM_LINE = re.compile(r"^[0-9a-f]{64}\s+\*?(\S+)\s*$")


def build_inputs(tree, d):
    """Files of build dir `d` that its build or a publish workflow reads, relative to the dir.

    None means every file: a COPY source the parser cannot bound (`.`, globs, variables) or no publish
    workflow that names the dir. Lists name files by relative path or by base name; both count.
    """
    prefix = f"docker/{d}/"
    files = {p[len(prefix):] for p in tree.paths if p.startswith(prefix)}
    names = {"Dockerfile", "SHA256SUMS", "SERIES"}
    for listing in ("SHA256SUMS", "SERIES"):
        if listing in files:
            names.update(m.group(1) for m in map(SUM_LINE.match, tree.text(prefix + listing).splitlines()) if m)
    dirs = []
    if "Dockerfile" in files:
        text = re.sub(r"\\\n", " ", tree.text(prefix + "Dockerfile"))
        for line in text.splitlines():
            words = line.split()
            if not words or words[0].upper() not in ("COPY", "ADD"):
                continue
            args = words[1:]
            if any(a.startswith("--from") for a in args):
                continue
            args = [a for a in args if not a.startswith("--")]
            if args and args[0].startswith("["):
                args = json.loads(" ".join(args))
            for src in args[:-1]:
                if re.match(r"^[a-z]+://", src):
                    continue
                src = src[2:] if src.startswith("./") else src
                if src in ("", ".") or re.search(r"[*?\[$]", src):
                    return None
                if src.endswith("/") or any(f.startswith(src + "/") for f in files):
                    dirs.append(src.rstrip("/") + "/")
                names.add(src)
    flows = [tree.text(p) for p in tree.paths if re.fullmatch(r"\.github/workflows/publish-[^/]+\.yaml", p)]
    flows = [t for t in flows if re.search(rf"docker/{re.escape(d)}(?![\w.-])", t)]
    if not flows:
        return None
    for f in files:
        if any(re.search(rf"(?<![\w.-]){re.escape(f)}(?![\w.-])", t) for t in flows):
            names.add(f)
    return {f for f in files if f in names or os.path.basename(f) in names or any(f.startswith(x) for x in dirs)}


def editable_doc(tree, path):
    """True for a Markdown file in a build dir that neither the build nor a publish workflow reads."""
    m = re.fullmatch(r"docker/([^/]+)/(.+\.md)", path)
    if not m:
        return False
    inputs = build_inputs(tree, m.group(1))
    return inputs is not None and m.group(2) not in inputs


# ---------------------------------------------------------------- YAML text

BLOCK_HDR = re.compile(r"""^(\s*)(?:-\s+|[^#'"\s][^#]*?:\s+|['"][^'"]*['"]:\s+|-\s+[^#'"\s][^#]*?:\s+)[|>][-+0-9]*\s*(#.*)?$""")


def yaml_line_kinds(text):
    """Per line: 'scalar' (inside a block scalar), 'comment', 'blank' or 'code'."""
    lines = text.split("\n")
    kinds, parent = [], None
    for line in lines:
        ind = len(line) - len(line.lstrip(" "))
        if parent is not None:
            if not line.strip() or ind > parent:
                kinds.append("scalar")
                continue
            parent = None
        s = line.strip()
        if not s:
            kinds.append("blank")
        elif s.startswith("#"):
            kinds.append("comment")
        else:
            kinds.append("code")
            m = BLOCK_HDR.match(line)
            if m:
                parent = ind
    return lines, kinds


def strip_trailing_comment(s):
    """Return (code, comment) for one YAML code line, honouring quotes."""
    q = None
    for i, ch in enumerate(s):
        if q:
            if ch == q:
                q = None
        elif ch in "'\"" and (i == 0 or s[i - 1] in " :[{,-"):
            q = ch
        elif ch == "#" and (i == 0 or s[i - 1] in " \t"):
            return s[:i].rstrip(), s[i:]
    return s, ""


def yaml_comments(text):
    """(line number, comment text) for comments outside block scalars."""
    lines, kinds = yaml_line_kinds(text)
    out = []
    for n, (line, kind) in enumerate(zip(lines, kinds), 1):
        if kind == "comment":
            out.append((n, line.strip(), True))
        elif kind == "code":
            _, c = strip_trailing_comment(line)
            if c:
                out.append((n, c, False))
    return out


def key_paths(text):
    """Per line, the list of mapping keys enclosing it (by indentation)."""
    lines, kinds = yaml_line_kinds(text)
    stack, out = [], []
    last_code_path = []
    for line, kind in zip(lines, kinds):
        if kind != "code":
            out.append(last_code_path if kind == "scalar" else [k for _, k in stack])
            continue
        ind = len(line) - len(line.lstrip(" "))
        body = line.strip()
        while body.startswith("- "):
            ind += 2
            body = body[2:].lstrip()
        while stack and stack[-1][0] >= ind:
            stack.pop()
        m = re.match(r"""^(['"]?)([^'":#][^:#]*?)\1:(\s|$)""", body)
        cur = [k for _, k in stack]
        out.append(cur + [m.group(2)] if m else cur)
        last_code_path = out[-1]
        if m:
            stack.append((ind, m.group(2)))
    return lines, kinds, out


# ---------------------------------------------------------------- YAML subset parser

class YamlError(Exception):
    def __init__(self, line, msg):
        super().__init__(msg)
        self.line = line


def _scalar(s, n):
    s = s.strip()
    if not s:
        return None
    if s[0] in "&*!" or s.startswith("? ") or s.startswith("<<"):
        raise YamlError(n, "anchors, aliases, tags and complex keys are not supported here")
    if s[0] == '"':
        if len(s) < 2 or not s.endswith('"'):
            raise YamlError(n, "multi-line or unterminated double-quoted scalar")
        try:
            return json.loads(s)
        except ValueError:
            return s[1:-1]
    if s[0] == "'":
        if len(s) < 2 or not s.endswith("'"):
            raise YamlError(n, "multi-line or unterminated single-quoted scalar")
        return s[1:-1].replace("''", "'")
    if s[0] == "[":
        if not s.endswith("]"):
            raise YamlError(n, "multi-line flow sequence")
        inner = s[1:-1].strip()
        return [] if not inner else [_scalar(p, n) for p in _split_flow(inner, n)]
    if s[0] == "{":
        if not s.endswith("}"):
            raise YamlError(n, "multi-line flow mapping")
        inner, res = s[1:-1].strip(), {}
        for p in _split_flow(inner, n) if inner else []:
            k, sep, v = p.partition(":")
            if not sep:
                raise YamlError(n, "flow mapping entry without key")
            k = _scalar(k, n)
            if k in res:
                raise YamlError(n, f"duplicate key {k!r}")
            res[k] = _scalar(v, n)
        return res
    return s


def _split_flow(s, n):
    parts, depth, q, cur = [], 0, None, ""
    for ch in s:
        if q:
            cur += ch
            if ch == q:
                q = None
            continue
        if ch in "'\"":
            q = ch
        elif ch in "[{":
            raise YamlError(n, "nested flow collections are not supported here")
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return [p for p in parts if p.strip()]


def parse_yaml(text):
    """Parse a block-style YAML subset; raises YamlError on anything else."""
    raw = text.split("\n")
    lines = []
    for n, line in enumerate(raw, 1):
        if "\t" in line[: len(line) - len(line.lstrip())]:
            raise YamlError(n, "tab indentation")
        lines.append([n, len(line) - len(line.lstrip(" ")), line.strip(), line])
    pos = [0]

    def skip():
        while pos[0] < len(lines):
            s = lines[pos[0]][2]
            if s and not s.startswith("#") and s not in ("---",):
                return lines[pos[0]]
            if s == "---" and pos[0] != 0 and any(l[2] and not l[2].startswith("#") for l in lines[: pos[0]]):
                raise YamlError(lines[pos[0]][0], "multiple documents")
            pos[0] += 1
        return None

    def block_scalar(header, parent_ind, n):
        style, chomp = header[0], ("-" if "-" in header else "+" if "+" in header else "")
        body, ind = [], None
        while pos[0] < len(lines):
            ln = lines[pos[0]]
            if ln[2] and ln[1] <= parent_ind:
                break
            if ln[2] and ind is None:
                ind = ln[1]
            body.append(ln[3][ind:] if ind is not None and ln[2] else "")
            pos[0] += 1
        while body and not body[-1]:
            body.pop()
        if style == "|":
            out = "\n".join(body)
        else:
            out = " ".join(b for b in body if b)
        return out if chomp == "-" else out + "\n"

    def value_after(rest, ind, n):
        rest = strip_trailing_comment(rest)[0].strip()
        if re.fullmatch(r"[|>][-+0-9]*", rest):
            return block_scalar(rest, ind, n)
        if rest:
            v = _scalar(rest, n)
            nxt = skip()
            if nxt and nxt[1] > ind and isinstance(v, str) and rest[0] not in "'\"[{":
                raise YamlError(nxt[0], "multi-line plain scalar")
            return v
        nxt = skip()
        if nxt is None:
            return None
        if nxt[1] > ind or (nxt[1] == ind and (nxt[2].startswith("- ") or nxt[2] == "-")):
            return node(nxt[1])
        return None

    def node(ind):
        first = skip()
        if first[2].startswith("- ") or first[2] == "-":
            return seq(ind)
        return mapping(ind)

    def mapping(ind):
        res = {}
        while True:
            ln = skip()
            if ln is None or ln[1] < ind:
                return res
            if ln[1] > ind:
                raise YamlError(ln[0], "bad indentation")
            if ln[2].startswith("- "):
                return res
            m = re.match(r"""^("(?:[^"\\]|\\.)*"|'[^']*'|[^'"#\s][^#]*?):(\s+|$)(.*)$""", ln[2])
            if not m:
                raise YamlError(ln[0], "expected a mapping key")
            key = _scalar(m.group(1), ln[0])
            if key in res:
                raise YamlError(ln[0], f"duplicate key {key!r}")
            pos[0] += 1
            res[key] = value_after(m.group(3), ind, ln[0])

    def seq(ind):
        res = []
        while True:
            ln = skip()
            if ln is None or ln[1] < ind or not (ln[2].startswith("- ") or ln[2] == "-"):
                return res
            if ln[1] > ind:
                raise YamlError(ln[0], "bad indentation")
            content = ln[2][1:].lstrip()
            off = len(ln[2]) - len(content)
            if not content or content.startswith("#"):
                pos[0] += 1
                res.append(value_after("", ind, ln[0]))
            elif re.match(r"""^("(?:[^"\\]|\\.)*"|'[^']*'|[^'"#\s\[{][^#]*?):(\s+|$)""", content):
                ln[1], ln[2] = ln[1] + off, content
                res.append(mapping(ln[1]))
            else:
                pos[0] += 1
                res.append(value_after(content, ind, ln[0]))

    first = skip()
    if first is None:
        return None
    if first[1] != 0:
        raise YamlError(first[0], "document must start at column 0")
    doc = node(0)
    rest = skip()
    if rest is not None:
        raise YamlError(rest[0], "unexpected content")
    return doc


def find_line(text, pattern, start=1):
    rx = re.compile(pattern)
    for n, line in enumerate(text.split("\n"), 1):
        if n >= start and rx.search(line):
            return n
    return 0


# ---------------------------------------------------------------- engine commands

def split_cmd(value):
    """Flatten a rendered command/entrypoint to tokens, splitting `sh -c` bodies."""
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    out = []
    for it in items:
        it = str(it)
        if re.search(r"\s", it):
            body = it.replace("\\\n", " ")
            try:
                out.extend(shlex.split(body, comments=True))
            except ValueError:
                out.extend(body.split())
        else:
            out.append(it)
    return out


def flag_values(tokens, flag):
    vals = []
    for i, t in enumerate(tokens):
        if t == flag:
            vals.append(tokens[i + 1] if i + 1 < len(tokens) else "")
        elif t.startswith(flag + "="):
            vals.append(t[len(flag) + 1:])
    return vals


def service_tokens(svc):
    return split_cmd(svc.get("entrypoint")) + split_cmd(svc.get("command"))


def engine_kind(key, svc):
    toks = service_tokens(svc)
    joined = " ".join(toks)
    image = svc.get("image", "")
    if re.search(r"sglang\.launch_server|\bsglang serve\b", joined) or key.startswith("model-sg-"):
        return "sglang"
    if re.search(r"\bvllm serve\b|vllm\.entrypoints", joined) or key.startswith("model-vllm-"):
        return "vllm"
    if "sglang" in image:
        return "sglang"
    if "vllm" in image:
        return "vllm"
    return "unknown"


def has_gpu(svc):
    for d in (((svc.get("deploy") or {}).get("resources") or {}).get("reservations") or {}).get("devices") or []:
        if "gpu" in (d.get("capabilities") or []) or d.get("driver") == "nvidia":
            return True
    return bool(svc.get("gpus"))


def engines(path, rendered, services_allow):
    """Engine service keys: key prefix, or role engine in services.allow.json."""
    roles = {(s["file"], s["service"]): s.get("role") for s in services_allow.get("services", [])}
    out = []
    for key in sorted(rendered.get("services", {})):
        if ENGINE_KEY.match(key) or roles.get((path, key)) == "engine":
            out.append(key)
    return out


def model_ref(kind, tokens):
    """The model an engine loads: --model-path / --model / `vllm serve X` / first vLLM positional."""
    for flag in ("--model-path", "--model"):
        v = flag_values(tokens, flag)
        if v:
            return v[0]
    for i, t in enumerate(tokens[:-2]):
        if t == "vllm" and tokens[i + 1] == "serve" and not tokens[i + 2].startswith("-"):
            return tokens[i + 2]
    if kind == "vllm" and tokens and not tokens[0].startswith("-"):
        return tokens[0]
    return None


def served_names(kind, tokens):
    names = flag_values(tokens, "--served-model-name")
    out = []
    for i, t in enumerate(tokens):
        if t == "--served-model-name":
            j = i + 2
            while j < len(tokens) and not tokens[j].startswith("-") and tokens[j] not in ("&&", ";", "|"):
                if kind == "vllm":
                    out.append(tokens[j])
                j += 1
    names += out
    if names:
        return names
    m = model_ref(kind, tokens)
    return [m] if m else []
