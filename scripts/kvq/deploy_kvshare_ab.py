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


def token():
    for line in open(os.path.expanduser("~/.config/nearai/deploy.env")):
        line = line.strip()
        if line.startswith(("GPU_MANAGER_TOKEN=", "export GPU_MANAGER_TOKEN=")):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit("GPU_MANAGER_TOKEN not found")


TOKEN = token()


def api(path, body=None, timeout=1800):
    cmd = ["curl", "-sS", "-N", "--max-time", str(timeout), "-H", f"Authorization: Bearer {TOKEN}"]
    if body is not None:
        cmd += ["-X", "POST", "-H", "Content-Type: application/json", "--data-binary", "@-"]
    cmd.append(f"{GM}/api/{path}")
    r = subprocess.run(cmd, input=json.dumps(body) if body is not None else None, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"curl failed for {path}: {r.stderr.strip()[:300]}")
    return r.stdout


def containers(inst):
    d = json.loads(api(f"instances/{inst}/docker/ps"))
    rows = {}
    for line in d.get("output", "").splitlines():
        if line.strip():
            c = json.loads(line)
            rows[c["Names"]] = {"state": c["State"], "status": c["Status"], "id": c["ID"]}
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--replica", required=True, choices=["r3", "r4"])
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
    if work.get("file") not in (BASE_FILE, AB_FILE):
        sys.exit("host is not on the base-tier file; refusing")
    svc = f"model-sg-glm53-w4afp8-tp2-{a.replica}"
    if a.rollback:
        tag, file = (work.get("tag"), BASE_FILE) if work.get("file") == BASE_FILE else (None, BASE_FILE)
        if tag is None:
            sys.exit("current work deployment is the A/B file; pass --tag <base tag> for the rollback")
        if a.tag:
            tag = a.tag
    else:
        tag = a.tag or subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        file = AB_FILE
    body = {"tag": tag, "file": file, "services": [svc], "env": env, "force_recreate": False, "dry_run": True}
    out = api(f"instances/{iid}/compose/up", body)
    plan = plan_of(out)
    if plan is None:
        sys.exit(f"dry-run gave no plan: {out[-600:]}")
    rec = [x.replace("Container ", "") for x in plan.get("recreate", []) + plan.get("create", []) if x.startswith("Container ")]
    print(f"dry-run ({file} @ {tag[:12]}): create={plan.get('create')} recreate={plan.get('recreate')} remove={plan.get('remove')}")
    if rec != [svc] or plan.get("remove"):
        sys.exit(f"ABORT: plan must recreate exactly {svc}; got {rec}, remove={plan.get('remove')}")
    if not a.apply:
        print("plan only; re-run with --apply")
        return
    old = containers(iid).get(svc, {}).get("id")
    body["dry_run"] = False
    out = api(f"instances/{iid}/compose/up", body)
    if '"success":true' not in out.replace(" ", ""):
        sys.exit(f"compose/up failed: {out[-800:]}")
    print(f"applied; waiting for a new {svc} container (old stop has a 5 min grace)")
    deadline, resent = time.time() + 1200, False
    while time.time() < deadline:
        c = containers(iid).get(svc)
        if c and c["state"] == "running" and c["id"] != old:
            print(f"{svc} running: id={c['id']} {c['status']}")
            print(f'next: Loki {{host="{a.host}", container_name="{svc}"}} |~ "shared-kv-patch|ready to roll|Traceback"')
            return
        if not resent and time.time() > deadline - 600:
            api(f"instances/{iid}/compose/up", body); resent = True
        time.sleep(20)
    sys.exit(f"{svc} did not come up as a new container in 20 min")


if __name__ == "__main__":
    main()
