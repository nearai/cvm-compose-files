#!/usr/bin/env python3
# How to run (one treatment replica at a time; plan only without --apply):
#   python3 scripts/kvq/deploy_kvshare_ab.py --host gpu04 --replica r3 [--apply]
#   python3 scripts/kvq/deploy_kvshare_ab.py --host gpu04 --replica r3 --rollback [--apply]
"""Deploy (or roll back) one treatment replica of the in-host shared-KV A/B on a base-tier host.

Deploy: compose/up of prod/GLM-5.3-Flash-SGL-TP2x4-W4AFP8-KVShare-AB.yaml at --tag (default: this
checkout's HEAD commit), services [model-sg-glm53-w4afp8-tp2-<replica>], the host's FULL dashboard
env. Rollback: the same replica from the host's currently deployed base file and tag. Both dry-run
first and abort unless the plan recreates exactly that replica (plus volume creation). With
--apply, waits for a NEW running container and its "ready to roll" in Loki is left to the operator.
Runbook: docs/glm53-kvshare-ab.md. Token from ~/.config/nearai/deploy.env (never printed).
"""
import argparse, json, os, re, subprocess, sys, time

GM = "https://gpu-manager.infra.near.ai"
AB_FILE = "prod/GLM-5.3-Flash-SGL-TP2x4-W4AFP8-KVShare-AB.yaml"
BASE_FILE = "prod/GLM-5.3-Flash-SGL-TP2x4-W4AFP8.yaml"
VARIANT_SUFFIX = "-kvshare-l3file-v1"
KV4_FILE = "prod/GLM-5.3-Flash-SGL-TP2x4-W4AFP8-KVShare4.yaml"
KV4_SUFFIX = "-kvshare4-l3file-v1"
SMALL_FILE = "prod/small-models.yaml"          # gpu13 base file
KV2_FILE = "prod/small-models-GLM53-KVShare2.yaml"
KV2_SUFFIX = "-kvshare2-l3file-v1"
OFF_FILES = {"r3": "prod/GLM-5.3-Flash-SGL-TP2x4-W4AFP8-HiCacheOff-34.yaml", "r4": "prod/GLM-5.3-Flash-SGL-TP2x4-W4AFP8-HiCacheOff-34.yaml",
             "r1": "prod/GLM-5.3-Flash-SGL-TP2x4-W4AFP8-HiCacheOff-12.yaml", "r2": "prod/GLM-5.3-Flash-SGL-TP2x4-W4AFP8-HiCacheOff-12.yaml"}
OFF_SUFFIX = "-hicacheoff-v1"
V7_CANARY_FILE = "prod/GLM-5.3-Flash-SGL-TP2x4-W4AFP8-V7Canary.yaml"   # gpu03 base since v0.0.479
V7_OFF_FILE = "prod/GLM-5.3-Flash-SGL-TP2x4-W4AFP8-V7-HiCacheOff-r3.yaml"
V7_OFF_SUFFIX = "-v7-hicacheoff-v1"
POOL_FILE = "prod/small-models-GLM53-KVSharePool.yaml"
LONG_V7_CANARY_FILE = "prod/GLM-5.3-Flash-SGL-TP4-W4AFP8-LongContext-V7Canary.yaml"   # gpu02 base since v0.0.479
LONG_V7_OFF_FILE = "prod/GLM-5.3-Flash-SGL-TP4-W4AFP8-LongContext-V7-HiCacheOff-r2b.yaml"
POOL_SUFFIX = "-kvsharepool-v1"
PEERKV_FILE = "prod/GLM-5.3-Flash-SGL-TP2x4-W4AFP8-V7-HiCacheOff-PeerKV.yaml"  # gpu04: r1+r2 peerkv, r3+r4 control
# Appended to the v7 fleet variant (v0.0.480): r1/r2 peerkv (+ expandable_segments:False), r3/r4 only
# expandable_segments:False.
PEERKV_SUFFIX = "-peerkv-v1"
EXPFALSE_SUFFIX = "-expfalse-v1"
TREATED = (VARIANT_SUFFIX, KV4_SUFFIX, KV2_SUFFIX, OFF_SUFFIX, V7_OFF_SUFFIX, POOL_SUFFIX, PEERKV_SUFFIX, EXPFALSE_SUFFIX, "-hicacheoff-v1")


def token():
    for line in open(os.path.expanduser("~/.config/nearai/deploy.env")):
        line = line.strip()
        if line.startswith(("GPU_MANAGER_TOKEN=", "export GPU_MANAGER_TOKEN=")):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit("GPU_MANAGER_TOKEN not found")


TOKEN = token()


def api(path, body=None, timeout=1800, check=True):
    """GET (body None) or POST JSON. Returns stdout. With check=False a curl failure (for example a
    dropped compose stream) returns (False, partial_stdout) instead of exiting."""
    cmd = ["curl", "-sS", "-N", "--max-time", str(timeout), "-H", f"Authorization: Bearer {TOKEN}"]
    if body is not None:
        cmd += ["-X", "POST", "-H", "Content-Type: application/json", "--data-binary", "@-"]
    cmd.append(f"{GM}/api/{path}")
    r = subprocess.run(cmd, input=json.dumps(body) if body is not None else None, capture_output=True, text=True)
    if not check:
        return r.returncode == 0, r.stdout
    if r.returncode != 0:
        sys.exit(f"curl failed for {path}: {r.stderr.strip()[:300]}")
    return r.stdout


def label(labels, key):
    m = re.search(re.escape(key) + r"=([^,]*)", labels)
    return m.group(1) if m else ""


def service_containers(inst, svc):
    """Every container of compose service `svc` in project work, keyed by id. Matches on the compose
    service label, not the name: an interrupted recreate leaves the new container named
    `<12 hex>_<service>`, and that container can end up being the one that runs."""
    d = json.loads(api(f"instances/{inst}/docker/ps?all=true"))
    rows = {}
    for line in d.get("output", "").splitlines():
        if not line.strip():
            continue
        c = json.loads(line)
        labels = c.get("Labels", "")
        if label(labels, "com.docker.compose.service") != svc or label(labels, "com.docker.compose.project") != "work":
            continue
        rows[c["ID"]] = {"name": c["Names"], "state": c["State"], "status": c["Status"],
                         "variant": label(labels, "nearai.otel.config_variant"),
                         "file": label(labels, "com.docker.compose.project.config_files")}
    return rows


def plan_of(out):
    for line in out.splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("event") == "plan":
            return e.get("plan")
    return None


def dry_run_ok(iid, body, svc, allow_empty=False):
    """Dry-run `body`; the plan may only create/recreate `svc` and remove leftovers of `svc`.
    allow_empty: on a re-send, an empty plan means compose already created the container but
    died before starting it (gpu-manager docker/ps does not list Created containers, even with
    ?all=true); the real up then just starts it."""
    out = api(f"instances/{iid}/compose/up", dict(body, dry_run=True))
    plan = plan_of(out)
    if plan is None:
        sys.exit(f"dry-run gave no plan: {out[-600:]}")
    touched = [x.replace("Container ", "") for x in plan.get("recreate", []) + plan.get("create", []) if x.startswith("Container ")]
    removed = plan.get("remove", []) or []
    print(f"dry-run: create={plan.get('create')} recreate={plan.get('recreate')} remove={removed}")
    # An interrupted recreate can leave the service running as `<12 hex>_<service>`.
    bad_touch = [t for t in touched if t != svc and not re.fullmatch(r"[0-9a-f]{12}_" + re.escape(svc), t)]
    bad_remove = [r for r in removed if not r.replace("Container ", "").endswith(svc)]
    if bad_touch or bad_remove or (not touched and not removed and not allow_empty):
        sys.exit(f"ABORT: plan must only (re)create {svc} (and remove leftovers of it); got touched={touched} remove={removed}")
    return touched


def done_success(out):
    for line in out.splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("event") == "done":
            return bool(e.get("success"))
    return None  # stream ended without a terminal event (client side dropped)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--replica", required=True, choices=["r1", "r2", "r3", "r4", "r1a", "r1b", "r2a", "r2b"])
    ap.add_argument("--four", action="store_true", help="host-level treatment: all four replicas share (KVShare4 file)")
    ap.add_argument("--hicache-off", action="store_true", help="arm B: this replica without HiCache (HiCacheOff-12/34 file)")
    ap.add_argument("--gpu13", action="store_true", help="arm D: gpu13 r1a/r1b share one store (small-models KVShare2 file)")
    ap.add_argument("--gpu13-pool", action="store_true", help="arm D': gpu13 r1a/r1b pool host cache (KVSharePool file)")
    ap.add_argument("--v7-hicache-off-long", action="store_true", help="long arm: gpu02 r2b = v7 canary r2a without HiCache")
    ap.add_argument("--v7-hicache-off", action="store_true", help="arm B': gpu03 r3 = v7 canary without HiCache")
    ap.add_argument("--peerkv", action="store_true", help="gpu04 peerkv A/B on the v7 fleet file: r1/r2 = + GPU peer KV + expandable_segments:False, r3/r4 = + expandable_segments:False")
    ap.add_argument("--rollback-file", default=None, help="prod file to roll back to (default: the host's base file)")
    ap.add_argument("--tag", default=None, help="commit SHA to deploy (default: this checkout's HEAD)")
    ap.add_argument("--rollback", action="store_true")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    instances = json.loads(api("instances"))
    inst = next((x for x in instances if x.get("name") == a.host), None)
    if not inst or not inst.get("env_vars"):
        sys.exit(f"host {a.host} not found or has no env_vars")
    iid, env = inst["id"], inst["env_vars"]
    work = json.loads(api(f"instances/{iid}/version")).get("projects", {}).get("work", {}).get("current", {})
    print(f"{a.host}: work tag={work.get('tag')} file={work.get('file')}")
    if work.get("file") not in (BASE_FILE, AB_FILE, KV4_FILE, SMALL_FILE, KV2_FILE, V7_CANARY_FILE, V7_OFF_FILE, POOL_FILE, LONG_V7_CANARY_FILE, LONG_V7_OFF_FILE, PEERKV_FILE, *OFF_FILES.values()):
        sys.exit("host is not on the base-tier file; refusing")
    svc = f"model-sg-glm53-w4afp8-tp2-{a.replica}"
    gpu13 = a.replica in ("r1a", "r1b") and a.host == "gpu13"
    if gpu13 != (a.gpu13 or a.gpu13_pool) and not a.rollback:
        sys.exit("r1a/r1b are gpu13's replicas: use --gpu13 (and only for them)")
    base_file = a.rollback_file or (SMALL_FILE if gpu13 else BASE_FILE)
    if a.rollback:
        # Back to the host's prod file. A tag is needed when the host's current deployment is a
        # test file (its tag is then a branch SHA, not the prod release).
        tag, file = (work.get("tag"), base_file) if work.get("file") == base_file else (None, base_file)
        if a.tag:
            tag = a.tag
        if tag is None:
            sys.exit(f"current work deployment is a test file; pass --tag <prod tag> (e.g. v0.0.475 base, v0.0.476 gpu13)")
        suffix = None
    else:
        tag = a.tag or subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        if a.peerkv:
            if a.host != "gpu04" or a.replica not in ("r1", "r2", "r3", "r4"):
                sys.exit("--peerkv is built for gpu04 r1-r4")
            file = PEERKV_FILE
            suffix = PEERKV_SUFFIX if a.replica in ("r1", "r2") else EXPFALSE_SUFFIX
        elif a.v7_hicache_off_long:
            if (a.host, a.replica) != ("gpu02", "r2b"):
                sys.exit("--v7-hicache-off-long is built for gpu02 r2b only")
            # The long-tier v7 variant ends "-v7-mr16q4", so the arm suffix is "-hicacheoff-v1" there.
            file, suffix = LONG_V7_OFF_FILE, "-hicacheoff-v1"
        elif a.gpu13_pool:
            file, suffix = POOL_FILE, POOL_SUFFIX
        elif a.v7_hicache_off:
            if a.replica != "r3":
                sys.exit("--v7-hicache-off is built for r3 only")
            file, suffix = V7_OFF_FILE, V7_OFF_SUFFIX
        elif a.gpu13:
            file, suffix = KV2_FILE, KV2_SUFFIX
        elif a.hicache_off:
            file, suffix = OFF_FILES[a.replica], OFF_SUFFIX
        elif a.four:
            file, suffix = KV4_FILE, KV4_SUFFIX
        else:
            if a.replica in ("r1", "r2"):
                sys.exit("r1/r2 are the control in the 2-way file; use --four or --hicache-off")
            file, suffix = AB_FILE, VARIANT_SUFFIX
    want_treatment = not a.rollback

    def is_target(c):
        treated = any(c["variant"].endswith(x) for x in TREATED)
        ok_variant = c["variant"].endswith(suffix) if want_treatment else not treated
        return c["state"] == "running" and ok_variant and c["file"].endswith(file)

    body = {"tag": tag, "file": file, "services": [svc], "env": env, "force_recreate": False}
    print(f"{file} @ {tag[:12]}")
    dry_run_ok(iid, body, svc)
    before = service_containers(iid, svc)
    print("now:", {k[:12]: (v["name"], v["state"], v["variant"][-20:]) for k, v in before.items()})
    if not a.apply:
        print("plan only; re-run with --apply")
        return
    old_ids = set(before)

    # The real up. A dropped stream is "outcome unknown", never failure or success: compose-manager
    # (aa9de34) kills compose via SIGPIPE when its client goes away, which can leave the new
    # container Created and never started. So: poll, and re-send once the old container is gone.
    ok, out = api(f"instances/{iid}/compose/up", dict(body, dry_run=False), check=False)
    res = done_success(out)
    print(f"compose/up stream: {'complete' if ok else 'DROPPED'}; done.success={res}")
    if res is False:
        sys.exit(f"compose/up failed: {out[-800:]}")
    deadline, resends, last_send = time.time() + 1800, 0, time.time()
    while time.time() < deadline:
        cs = service_containers(iid, svc)
        tgt = [(k, c) for k, c in cs.items() if is_target(c) and k not in old_ids]
        if tgt:
            k, c = tgt[0]
            print(f"{svc} running as {c['name']} id={k[:12]} {c['status']} variant=...{c['variant'][-24:]}")
            print(f'next: Loki {{host="{a.host}", container_name=~".*{svc}"}} |~ "shared-kv-patch|ready to roll|Traceback"')
            return
        old_running = [k for k, c in cs.items() if k in old_ids and c["state"] == "running"]
        if old_running:
            print(f"  old {svc} still running (graceful drain, up to 5 min)"); time.sleep(20); continue
        # Old one is gone and no new one runs: compose died between create and start, or is still
        # starting. Give a live compose 90 s, then re-send (fast now: nothing left to drain).
        if time.time() - last_send > 90:
            if resends >= 3:
                sys.exit(f"{svc} still not running after {resends} re-sends: {cs}")
            print(f"  {svc} not running ({ {k[:12]: (c['name'], c['state']) for k, c in cs.items()} }); re-sending up")
            dry_run_ok(iid, body, svc, allow_empty=True)
            ok, out = api(f"instances/{iid}/compose/up", dict(body, dry_run=False), check=False)
            print(f"  re-send stream: {'complete' if ok else 'DROPPED'}; done.success={done_success(out)}")
            resends, last_send = resends + 1, time.time()
        time.sleep(20)
    sys.exit(f"{svc} did not come up as a new container in 30 min")


if __name__ == "__main__":
    main()
