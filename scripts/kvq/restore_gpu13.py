#!/usr/bin/env python3
# How to run:
#   python3 scripts/kvq/restore_gpu13.py            # plan only: what would be torn down / started
#   python3 scripts/kvq/restore_gpu13.py --apply    # do it
"""Return gpu13's GPUs 4-7 to the fleet after the in-host KV-sharing experiments.

1. Tear down the lab stack: a scoped compose/down of every service in compose project `glm53kvq`
   (prod/GLM-5.3-Flash-SGL-KVShare-Qual.yaml). Nothing in project `work` is touched.
2. Read gpu13's CURRENT `work` deployment (tag + file) from compose-manager's /version, so the
   restore uses whatever tag the owners last deployed, never a stale one.
3. compose/up, scoped to the GLM replicas r1a (GPUs 4,5) and r1b (GPUs 6,7) plus the
   glm53-ghost-aggregator when that file defines it, with the host's FULL env from the
   gpu-manager dashboard. Dry-run first: the plan may only create/recreate those services (and
   the one-shot model-downloader they depend on); anything else aborts before applying.
4. Wait for fresh r1a/r1b containers to be running, then for both engines to log
   "fired up and ready to roll" in Loki (if a Grafana token is available) or report how to check.

Needs ~/.config/nearai/deploy.env with GPU_MANAGER_TOKEN (never printed). Uses curl (system CA store).
After it finishes, confirm the OpenRouter gateway logs "Backend recovered" for gpu13:8444 and that
the long lane is back at 128 slots.
"""
import argparse, json, os, re, subprocess, sys, time

GM = "https://gpu-manager.infra.near.ai"
INSTANCE = "b760c105-40c3-4f8c-8fd4-5691c3bf470c"  # gpu13
LAB_PROJECT = "glm53kvq"
LAB_FILE = "prod/GLM-5.3-Flash-SGL-KVShare-Qual.yaml"
LAB_SERVICES = ["kvq-driver", "kvq-router", "kvq-pf", "kvq-dc", "kvq-r1", "kvq-r2", "kvq-ghost-aggregator",
                "kvq-probe", "kvq-ucxinfo", "kvq-nixl-target", "kvq-nixl-initiator", "kvq-pa", "kvq-pb"]
GLM_SERVICES = ["model-sg-glm53-w4afp8-tp2-r1a", "model-sg-glm53-w4afp8-tp2-r1b"]
OPTIONAL_SERVICES = ["glm53-ghost-aggregator"]
ALLOWED_PLAN = set(GLM_SERVICES + OPTIONAL_SERVICES + ["model-downloader"])


def token():
    for line in open(os.path.expanduser("~/.config/nearai/deploy.env")):
        line = line.strip()
        if line.startswith(("GPU_MANAGER_TOKEN=", "export GPU_MANAGER_TOKEN=")):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit("GPU_MANAGER_TOKEN not found in ~/.config/nearai/deploy.env")


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


def stream_events(out):
    events = []
    for line in out.splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return events


def containers():
    d = json.loads(api(f"instances/{INSTANCE}/docker/ps"))
    rows = {}
    for line in d.get("output", "").splitlines():
        if line.strip():
            c = json.loads(line)
            proj = re.search(r"com\.docker\.compose\.project=([^,]*)", c.get("Labels", ""))
            rows[c["Names"]] = {"state": c["State"], "status": c["Status"], "id": c["ID"],
                                "project": proj.group(1) if proj else ""}
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="tear down the lab and restore (default: plan only)")
    ap.add_argument("--keep-lab", action="store_true", help="skip the lab teardown")
    ap.add_argument("--tag", help="restore this work tag instead of the current one (with --file)")
    ap.add_argument("--file", default="prod/small-models.yaml")
    a = ap.parse_args()

    ver = json.loads(api(f"instances/{INSTANCE}/version"))
    work = ver.get("projects", {}).get("work", {}).get("current") or {}
    tag, file = work.get("tag"), work.get("file")
    if a.tag:
        print(f"current work deployment: {work.get('tag')} {work.get('file')} -> restoring {a.tag} {a.file}")
        tag, file = a.tag, a.file
    if not tag or file != "prod/small-models.yaml":
        sys.exit(f"unexpected gpu13 work deployment {work!r}; restore by hand")
    lab = ver.get("projects", {}).get(LAB_PROJECT, {}).get("current") or {}
    print(f"gpu13 work: tag={tag} file={file} commit={work.get('commit', '')[:12]}")
    print(f"lab project {LAB_PROJECT}: {lab.get('tag', 'none')} {lab.get('file', '')}")

    before = containers()
    lab_running = sorted(n for n, c in before.items() if c["project"] == LAB_PROJECT)
    print("lab containers running:", lab_running or "none")
    present = {n: c for n, c in before.items() if n in GLM_SERVICES}
    print("GLM replicas present now:", {n: c["status"] for n, c in present.items()} or "none")

    instances = json.loads(api("instances"))
    inst = next((x for x in instances if x.get("id") == INSTANCE), None)
    if not inst or not inst.get("env_vars"):
        sys.exit("could not read gpu13's env_vars from the dashboard; restore by hand")
    env = inst["env_vars"]  # full host env (never printed)
    print(f"host env: {len(env)} variables")

    services = list(GLM_SERVICES)
    ghost_running = "glm53-ghost-aggregator" in before
    if not ghost_running:
        services += OPTIONAL_SERVICES  # restored only if the work file defines it (checked by the plan)

    up = {"tag": tag, "file": file, "services": services, "env": env, "force_recreate": False, "dry_run": True}
    events = stream_events(api(f"instances/{INSTANCE}/compose/up", up))
    plan = next((e.get("plan") for e in events if e.get("event") == "plan"), None)
    errs = [e.get("data", "") for e in events if "no such service" in str(e.get("data", ""))]
    if errs and not ghost_running:
        services = list(GLM_SERVICES)  # this file has no ghost aggregator; retry without it
        up["services"] = services
        events = stream_events(api(f"instances/{INSTANCE}/compose/up", up))
        plan = next((e.get("plan") for e in events if e.get("event") == "plan"), None)
    if plan is None:
        sys.exit(f"dry-run produced no plan: {events[-3:]}")
    touched = [x.replace("Container ", "") for x in plan.get("create", []) + plan.get("recreate", []) if x.startswith("Container ")]
    removed = plan.get("remove", [])
    print("dry-run plan:", {"create": plan.get("create"), "recreate": plan.get("recreate"), "remove": removed})
    bad = [t for t in touched if t not in ALLOWED_PLAN] + removed
    if bad:
        sys.exit(f"ABORT: the plan would touch more than the GLM replicas: {bad}")

    if not a.apply:
        print("\nPlan only. Re-run with --apply to tear down the lab and restore r1a/r1b.")
        return

    if not a.keep_lab and lab_running:
        down = {"tag": lab.get("commit") or lab.get("tag"), "file": LAB_FILE, "project": LAB_PROJECT, "services": LAB_SERVICES}
        out = api(f"instances/{INSTANCE}/compose/down", down)
        if '"success":true' not in out.replace(" ", ""):
            sys.exit(f"lab teardown failed: {out[-500:]}")
        left = sorted(n for n, c in containers().items() if c["project"] == LAB_PROJECT)
        if left:
            sys.exit(f"lab containers still running after teardown: {left}")
        print("lab torn down; GPUs 4-7 free")

    up["dry_run"] = False
    out = api(f"instances/{INSTANCE}/compose/up", up)
    if '"success":true' not in out.replace(" ", ""):
        sys.exit(f"compose/up failed: {out[-800:]}")
    print("compose/up applied:", services)

    # Gate on NEW running containers (a recreate can leave one stuck in Created; re-send once).
    old_ids = {n: before[n]["id"] for n in GLM_SERVICES if n in before}
    deadline, resent = time.time() + 900, False
    while time.time() < deadline:
        now = containers()
        ok = [n for n in GLM_SERVICES if n in now and now[n]["state"] == "running" and now[n]["id"] != old_ids.get(n)]
        if len(ok) == len(GLM_SERVICES):
            break
        if not resent and time.time() > deadline - 600:
            api(f"instances/{INSTANCE}/compose/up", up); resent = True
        time.sleep(20)
    else:
        sys.exit(f"GLM replicas not running after 15 min: { {n: now.get(n) for n in GLM_SERVICES} }")
    print("r1a/r1b containers running; engines take ~10-15 min to load (pinned 325 GiB host tier each).")
    print("Next: wait for 'The server is fired up and ready to roll!' from both in Loki\n"
          '  {host="gpu13", container_name=~"model-sg-glm53-w4afp8-tp2-r1[ab]"} |= "ready to roll"\n'
          "then confirm the OpenRouter gateway logs 'Backend recovered' for gpu13:8444 (long lane back to 128 slots).")


if __name__ == "__main__":
    main()
