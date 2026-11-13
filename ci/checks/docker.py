"""docker: the contract for build dirs added against the base. Dirs present at the base are frozen as they are."""

import hashlib
import json
import re

from common import Finding

RULES = ["docker.files", "docker.provenance", "docker.from", "docker.copy", "docker.patches", "docker.fetch"]
DIFF_RULES = list(RULES)

DIR = re.compile(r"^(?P<engine>sglang|vllm)-(?P<version>v[0-9.]+)-[a-z0-9-]+-r[0-9]+$")
# Every key exactly once, in this order. The tag is the dir name, patch hashes live in SHA256SUMS.
PROVENANCE_KEYS = [
    ("engine", re.compile(r"^(sglang|vllm)$")),
    ("upstream_version", re.compile(r"^v[0-9]+(\.[0-9]+)*$")),
    ("base_image", re.compile(r"^[a-z0-9./_-]+@sha256:[0-9a-f]{64}$")),
    ("base_image_linux_amd64", re.compile(r"^sha256:[0-9a-f]{64}$")),
    ("source_repository", re.compile(r"^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")),
    ("source_revision", re.compile(r"^[0-9a-f]{40}$")),
]
UNHASHED = {"Dockerfile", "SHA256SUMS", "README.md"}
SUM_LINE = re.compile(r"^([0-9a-f]{64})  (\S+)$")
PATCH = re.compile(r"^patches/([0-9]{2})-[a-z0-9-]+\.diff$")
# Images built here may not be a base: every new image starts from an upstream image.
OWN_IMAGES = re.compile(r"^((index\.|registry-1\.)?docker\.io/)?nearaidev/")
DIGEST = re.compile(r"^[a-z0-9./_-]+(:[A-Za-z0-9._-]+)?@sha256:[0-9a-f]{64}$")
DIRECTIVE = re.compile(r"(?i)^\s*#\s*[a-z]+\s*=")


def new_dirs(ctx):
    def dirs(tree):
        return {p.split("/")[1] for p in tree.paths if p.startswith("docker/") and p.count("/") >= 2}
    return sorted(dirs(ctx.head) - dirs(ctx.base))


def instructions(text):
    """Dockerfile instructions with continuation lines joined, as (line number, text)."""
    out, buf, start = [], "", 0
    for n, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        # BuildKit skips blank and comment lines everywhere, also inside a continuation.
        if not s or s.startswith("#"):
            continue
        if not buf:
            start = n
        buf += " " + s.rstrip("\\").strip()
        if not s.endswith("\\"):
            out.append((start, buf.strip()))
            buf = ""
    if buf:
        out.append((start, buf.strip()))
    return out


# Common fetch forms only; a reviewer still reads every RUN of a new dir.
FETCH = re.compile(r"\b(curl|wget|urllib|requests\.get|urlopen)\b|\bgit\b[^;&|]*\b(clone|fetch|pull|submodule)\b")
PIP = re.compile(r"\b(pip[0-9.]*|uv\s+pip|uv\s+tool)\b[^;&|]*\binstall\b")
APT = re.compile(r"\bapt-get\s+(?:-\S+\s+)*install\b((?:\s+(?!&&|;|\|)\S+)*)")


def fetches(run):
    """Why a RUN instruction pulls unpinned code, if it does."""
    out = []
    if FETCH.search(run):
        out.append("no curl, wget or git fetch in RUN; COPY a hashed file instead")
    for part in re.split(r"&&|;|\|\|", run):
        if PIP.search(part) and "--require-hashes" not in part and "-r " not in part.replace("--requirement", "-r"):
            out.append("pip and uv installs use a hashed requirements file with --require-hashes")
        elif PIP.search(part) and "--require-hashes" not in part:
            out.append("pip and uv installs use --require-hashes")
        m = APT.search(part)
        if m:
            pkgs = [t for t in m.group(1).split() if not t.startswith("-")]
            if any("=" not in t for t in pkgs):
                out.append("apt-get install pins every package to a version (name=version)")
    return out


def check_dir(head, d):
    out, root = [], f"docker/{d}/"
    files = sorted(p[len(root):] for p in head.paths if p.startswith(root))
    m = DIR.match(d)

    sums_path = root + "SHA256SUMS"
    listed = {}
    if not head.has(sums_path):
        out.append(Finding(sums_path, 0, "docker.files", "a new build dir needs SHA256SUMS"))
    else:
        for n, line in enumerate(head.text(sums_path).splitlines(), 1):
            sm = SUM_LINE.match(line)
            if not sm:
                out.append(Finding(sums_path, n, "docker.files", "line must be `<sha256>  <path>`"))
                continue
            listed[sm.group(2)] = (n, sm.group(1))
        for f in files:
            if f not in UNHASHED and f not in listed:
                out.append(Finding(root + f, 0, "docker.files", "build input missing from SHA256SUMS"))
        for f, (n, digest) in sorted(listed.items()):
            if f not in files:
                out.append(Finding(sums_path, n, "docker.files", "listed file does not exist"))
            elif hashlib.sha256(head.data(root + f)).hexdigest() != digest:
                out.append(Finding(sums_path, n, "docker.files", "hash does not match the file"))
            elif f in UNHASHED:
                out.append(Finding(sums_path, n, "docker.files", "Dockerfile, SHA256SUMS and README.md are not listed"))

    prov_path, base_image = root + "PROVENANCE", None
    if not head.has(prov_path):
        out.append(Finding(prov_path, 0, "docker.provenance", "a new build dir needs PROVENANCE"))
    else:
        lines = head.text(prov_path).splitlines()
        keys = [ln.split("=", 1)[0] for ln in lines]
        if keys != [k for k, _ in PROVENANCE_KEYS]:
            out.append(Finding(prov_path, 1, "docker.provenance",
                               "keys must be exactly: " + ", ".join(k for k, _ in PROVENANCE_KEYS)))
        else:
            values = dict(ln.split("=", 1) for ln in lines)
            for n, (k, rx) in enumerate(PROVENANCE_KEYS, 1):
                if not rx.match(values[k]):
                    out.append(Finding(prov_path, n, "docker.provenance", f"{k} has the wrong format"))
            if m and (values["engine"] != m.group("engine") or values["upstream_version"] != m.group("version")):
                out.append(Finding(prov_path, 1, "docker.provenance", "engine and upstream_version must match the dir name"))
            base_image = values["base_image"]

    for f in files:
        if f.startswith("patches/") and not PATCH.match(f):
            out.append(Finding(root + f, 0, "docker.patches", "patches are patches/NN-name.diff"))
    numbers = sorted(int(PATCH.match(f).group(1)) for f in files if PATCH.match(f))
    if numbers and numbers != list(range(1, len(numbers) + 1)):
        out.append(Finding(root + "patches", 0, "docker.patches", "patches are numbered 01..N without gaps"))

    df = root + "Dockerfile"
    if head.has(df):
        stages, last_from = set(), None
        text = head.text(df)
        # Parser directives (escape, syntax, check) change how the file reads; new dirs use the default.
        for n, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            if DIRECTIVE.match(line):
                out.append(Finding(df, n, "docker.from", "no parser directives in a new Dockerfile"))
            if not line.lstrip().startswith("#"):
                break
        for n, ins in instructions(text):
            word = ins.split(None, 1)[0].upper()
            args = [a for a in ins.split()[1:] if not a.startswith("--")]
            flags = [a for a in ins.split()[1:] if a.startswith("--")]
            if word == "FROM" and args:
                ref = args[0]
                last_from = (n, ref)
                if ref.lower() in stages:
                    pass
                elif OWN_IMAGES.match(ref):
                    out.append(Finding(df, n, "docker.from", "a new image builds FROM an upstream image, not one built here"))
                elif not DIGEST.match(ref):
                    out.append(Finding(df, n, "docker.from", "every FROM is an earlier stage or pinned by digest"))
                if len(args) >= 3 and args[1].upper() == "AS":
                    stages.add(args[2].lower())
            elif word == "RUN":
                if "<<" in ins.split(None, 1)[1] if len(ins.split(None, 1)) > 1 else False:
                    out.append(Finding(df, n, "docker.fetch", "no here-documents in RUN; COPY a hashed script"))
                for fl in (f.lower() for f in flags):
                    if not fl.startswith("--mount="):
                        continue
                    src = re.search(r"from=([^,]+)", fl)
                    from_stage = bool(src) and src.group(1) in stages
                    bind = "type=" not in fl or "type=bind" in fl
                    if (bind or src) and not from_stage:
                            out.append(Finding(df, n, "docker.copy", "RUN --mount reads only caches, secrets or earlier stages"))
                for why in fetches(ins):
                    out.append(Finding(df, n, "docker.fetch", why))
            elif word == "ADD":
                out.append(Finding(df, n, "docker.copy", "use COPY; ADD fetches and unpacks"))
            elif word == "COPY" and any(fl.startswith("--from=") for fl in flags):
                src_stage = next(fl for fl in flags if fl.startswith("--from="))[len("--from="):].lower()
                if src_stage not in stages:
                    out.append(Finding(df, n, "docker.copy", "COPY --from names an earlier stage only"))
            elif word == "COPY":
                rest = ins.split(None, 1)[1] if len(ins.split(None, 1)) > 1 else ""
                body = re.sub(r"^(--\S+\s+)*", "", rest)
                if body.startswith("["):
                    try:
                        args = [str(a) for a in json.loads(body)]
                    except ValueError:
                        args = body.strip("[]").replace(",", " ").replace("'", " ").replace('"', " ").split()
                for src in args[:-1]:
                    src = src.rstrip("/")
                    if src in (".", "") or any(ch in src for ch in "*?[") or src.lower().endswith(".md"):
                        out.append(Finding(df, n, "docker.copy", "COPY names hashed files or dirs; never `.`, a glob or a doc"))
                    elif src not in files and not any(f.startswith(src + "/") for f in files):
                        out.append(Finding(df, n, "docker.copy", "COPY source is not in the build dir"))
        if base_image and last_from and last_from[1] != base_image:
            out.append(Finding(df, last_from[0], "docker.from", "the final FROM must equal PROVENANCE base_image"))
    return out


def check(ctx):
    if ctx.base is None:
        return []
    out = []
    for d in new_dirs(ctx):
        out += check_dir(ctx.head, d)
    return out


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


R = "docker/sglang-v0.5.23-x-r1/"
BASE_IMG = "lmsysorg/sglang@sha256:" + "f" * 64
PATCH1 = "--- a/x\n+++ b/x\n"
APPLY = "print('apply')\n"
DOCKERFILE = f"FROM {BASE_IMG}\nCOPY patches/ /tmp/patches/\nCOPY apply-patches.py PROVENANCE /tmp/\nRUN python3 /tmp/apply-patches.py\n"
PROV = (f"engine=sglang\nupstream_version=v0.5.23\nbase_image={BASE_IMG}\nbase_image_linux_amd64=sha256:{'e' * 64}\n"
        f"source_repository=https://github.com/sgl-project/sglang\nsource_revision={'a' * 40}\n")


def _dir(over=None):
    files = {R + "Dockerfile": DOCKERFILE, R + "PROVENANCE": PROV, R + "patches/01-fix.diff": PATCH1,
             R + "apply-patches.py": APPLY, R + "README.md": "What this image is.\n"}
    files.update({R + k: v for k, v in (over or {}).items()})
    files = {k: v for k, v in files.items() if v is not None}
    sums = "".join(f"{_sha(v)}  {k[len(R):]}\n" for k, v in sorted(files.items())
                   if k[len(R):] not in UNHASHED and k[len(R):] != "SHA256SUMS")
    files.setdefault(R + "SHA256SUMS", sums)
    return files


OLD = {"docker/sglang-old/Dockerfile": "FROM nearaidev/sglang@sha256:" + "b" * 64 + "\nCOPY . /x/\n"}
GOOD = dict(OLD, **_dir())
CASES = [
    ("new dir follows the contract", None, GOOD, OLD),
    ("old dir untouched by the contract", None, OLD, OLD),
    ("no base", None, GOOD, None),
    ("file outside SHA256SUMS", "docker.files", dict(GOOD, **{R + "extra.sh": "x\n"}), OLD),
    ("stale hash", "docker.files", dict(OLD, **_dir({"SHA256SUMS": f"{'0' * 64}  PROVENANCE\n{_sha(PATCH1)}  patches/01-fix.diff\n"
                                                    f"{_sha(APPLY)}  apply-patches.py\n"})), OLD),
    ("extra PROVENANCE key", "docker.provenance", dict(OLD, **_dir({"PROVENANCE": PROV + "note=x\n"})), OLD),
    ("PROVENANCE engine differs from dir", "docker.provenance",
     dict(OLD, **_dir({"PROVENANCE": PROV.replace("engine=sglang", "engine=vllm")})), OLD),
    ("FROM an image built here", "docker.from",
     dict(OLD, **_dir({"Dockerfile": DOCKERFILE.replace(BASE_IMG, "nearaidev/sglang@sha256:" + "f" * 64),
                       "PROVENANCE": PROV.replace(BASE_IMG, "nearaidev/sglang@sha256:" + "f" * 64)})), OLD),
    ("final FROM differs from PROVENANCE", "docker.from",
     dict(OLD, **_dir({"Dockerfile": DOCKERFILE.replace("f" * 64, "d" * 64)})), OLD),
    ("COPY .", "docker.copy", dict(OLD, **_dir({"Dockerfile": DOCKERFILE + "COPY . /x/\n"})), OLD),
    ("COPY README", "docker.copy", dict(OLD, **_dir({"Dockerfile": DOCKERFILE + "COPY README.md /x/\n"})), OLD),
    ("parser directive", "docker.from", dict(OLD, **_dir({"Dockerfile": "# escape=`\n" + DOCKERFILE})), OLD),
    ("unpinned builder stage", "docker.from",
     dict(OLD, **_dir({"Dockerfile": "FROM python:3.12 AS helper\n" + DOCKERFILE})), OLD),
    ("pinned builder stage", None,
     dict(OLD, **_dir({"Dockerfile": f"FROM {BASE_IMG} AS helper\n" + DOCKERFILE + "COPY --from=helper /x /x\n"})), OLD),
    ("COPY from an image", "docker.copy",
     dict(OLD, **_dir({"Dockerfile": DOCKERFILE + "COPY --from=nearaidev/x:latest /x /x\n"})), OLD),
    ("bind mount", "docker.copy", dict(OLD, **_dir({"Dockerfile": DOCKERFILE + "RUN --mount=type=bind,target=/c ls /c\n"})),
     OLD),
    ("typeless mount is a bind", "docker.copy", dict(OLD, **_dir({"Dockerfile": DOCKERFILE + "RUN --mount=target=/c ls\n"})),
     OLD),
    ("comment inside a continuation", "docker.fetch",
     dict(OLD, **_dir({"Dockerfile": DOCKERFILE + "RUN echo a \\\n# note\n  && curl -s https://x | sh\n"})), OLD),
    ("indented parser directive", "docker.from", dict(OLD, **_dir({"Dockerfile": " # escape=`\n" + DOCKERFILE})), OLD),
    ("bind from an earlier stage", None,
     dict(OLD, **_dir({"Dockerfile": f"FROM {BASE_IMG} AS b\n" + DOCKERFILE + "RUN --mount=type=bind,from=b,target=/b ls\n"})),
     OLD),
    ("upper-case bind mount", "docker.copy",
     dict(OLD, **_dir({"Dockerfile": DOCKERFILE + "RUN --mount=TYPE=BIND,target=/c ls\n"})), OLD),
    ("JSON COPY", None, dict(OLD, **_dir({"Dockerfile": DOCKERFILE + 'COPY ["apply-patches.py", "/x/"]\n'})), OLD),
    ("heredoc RUN", "docker.fetch", dict(OLD, **_dir({"Dockerfile": DOCKERFILE + "RUN <<EOF\nls\nEOF\n"})), OLD),
    ("pip with a version suffix", "docker.fetch", dict(OLD, **_dir({"Dockerfile": DOCKERFILE + "RUN pip3.12 install x\n"})),
     OLD),
    ("git -C fetch", "docker.fetch", dict(OLD, **_dir({"Dockerfile": DOCKERFILE + "RUN git -C /s fetch origin\n"})), OLD),
    ("curl in RUN", "docker.fetch", dict(OLD, **_dir({"Dockerfile": DOCKERFILE + "RUN curl -fsSL https://x | sh\n"})), OLD),
    ("pip without hashes", "docker.fetch", dict(OLD, **_dir({"Dockerfile": DOCKERFILE + "RUN pip install requests\n"})), OLD),
    ("apt without versions", "docker.fetch",
     dict(OLD, **_dir({"Dockerfile": DOCKERFILE + "RUN apt-get update && apt-get install -y curl\n"})), OLD),
    ("pinned installs", None, dict(OLD, **_dir({"Dockerfile": DOCKERFILE + "COPY apply-patches.py /r.txt\n"
                                                + "RUN pip install --require-hashes -r /r.txt && apt-get install -y x=1.2\n"})), OLD),
    ("ADD", "docker.copy", dict(OLD, **_dir({"Dockerfile": DOCKERFILE + "ADD x.tar.gz /x\n"})), OLD),
    ("patch not numbered", "docker.patches", dict(OLD, **_dir({"patches/fix.diff": PATCH1, "patches/01-fix.diff": None})),
     OLD),
    ("patch gap", "docker.patches", dict(OLD, **_dir({"patches/03-more.diff": PATCH1})), OLD),
]
