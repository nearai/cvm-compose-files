#!/usr/bin/env python3
"""Per-adapter prefix-cache check for an engine (or proxy) serving a base model plus LoRA adapters.

  lab_kv_adapter_check.py URL OUT.json BASE_ID ADAPTER_ID [OTHER_ADAPTER_ID]   (env LAB_TOKEN: bearer, optional)

One fresh shared prefix (~6K tokens, unique per run) followed by different questions. Expected:
  1 adapter cold   2 adapter warm   3 base cold   4 other adapter cold   5 base warm   6 adapter warm
cached = usage.prompt_tokens_details.cached_tokens ("warm" >= 90% of the prefix, "cold" <= one block).
KV correctness: with temperature 0, a warm request's text equals the same request forced cold (unique
cache_salt), for the adapter and for the base; and the other adapter (a deliberately strong probe) differs
from the base. Exit 0 only if every check holds. Point it at one engine rank, not a load balancer, unless
the balancer keeps the conversation on one rank.
"""
import json, os, sys, time, urllib.error, urllib.request, uuid

url, out = sys.argv[1].rstrip("/"), sys.argv[2]
base, adapter = sys.argv[3], sys.argv[4]
other = sys.argv[5] if len(sys.argv) > 5 else None
token = os.environ.get("LAB_TOKEN")
# Optional {"model id": "url"}: send each model to its own endpoint (one strict proxy per id).
url_map = json.loads(os.environ.get("KV_URL_MAP", "{}"))
run = uuid.uuid4().hex[:12]
words = ("river stone lantern harbor copper meadow signal orchard lattice ember quarry violet summit "
         "thread canyon marble falcon willow cobalt beacon").split()
prefix = f"Run {run}. Reference notes follow.\n" + " ".join(words[(i * 7 + i // 3) % len(words)] for i in range(5200))


def ask(model, question, salt=None, max_tokens=24):
    body = {"model": model, "temperature": 0, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prefix + "\n\nQuestion: " + question + " Answer in one short sentence."}],
            "chat_template_kwargs": {"thinking": False}}
    if salt:
        body["cache_salt"] = salt
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    t0 = time.time()
    req = urllib.request.Request(url_map.get(model, url).rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(), headers)
    try:
        r = json.load(urllib.request.urlopen(req, timeout=600))
    except urllib.error.HTTPError as e:
        raise SystemExit(f"HTTP {e.code} for model={model} salt={bool(salt)}: {e.read()[:500]!r}")
    u = r.get("usage", {})
    return {"model": model, "question": question, "salt": bool(salt), "s": round(time.time() - t0, 2),
            "prompt": u.get("prompt_tokens"), "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "text": (r["choices"][0]["message"].get("content") or "")}


steps, checks = [], []
seq = [("adapter cold", adapter, "What is the first word?", "cold"),
       ("adapter warm", adapter, "What is the second word?", "warm"),
       ("base cold", base, "What is the third word?", "cold")]
if other:
    seq.append(("other adapter cold", other, "What is the fourth word?", "cold"))
seq += [("base warm", base, "What is the fifth word?", "warm"),
        ("adapter still warm", adapter, "What is the sixth word?", "warm")]
for name, model, q, want in seq:
    r = ask(model, q)
    r["step"], r["want"] = name, want
    steps.append(r)
prefix_tokens = steps[0]["prompt"] - 40
for r in steps:
    c = r["cached"] or 0
    ok = c >= 0.9 * prefix_tokens if r["want"] == "warm" else c <= 64
    checks.append((f"{r['step']}: cached {c} of {r['prompt']} ({r['want']})", ok))

# KV content: the warm answer should match the same request forced cold (informational: FP8/speculative
# numerics can flip a late greedy token), and a strong adapter's warm answer must differ from the base's.
def similarity(a, b):
    a, b = a.split(), b.split()
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return round(n / max(1, max(len(a), len(b))), 2)

same_q = "List the first three words."
info = []
for model in [m for m in (adapter, base, other) if m]:
    warm = ask(model, same_q, max_tokens=32)
    warm2 = ask(model, same_q, max_tokens=32)
    cold = ask(model, same_q, salt=f"cold-{run}-" + __import__("hashlib").sha1(model.encode()).hexdigest()[:12], max_tokens=32)
    steps += [dict(warm, step=f"{model} text (warming)"), dict(warm2, step=f"{model} text warm"),
              dict(cold, step=f"{model} text forced cold")]
    checks.append((f"{model}: second identical request warm ({warm2['cached']})", (warm2["cached"] or 0) >= 0.9 * prefix_tokens))
    checks.append((f"{model}: forced-cold request had no cache hit ({cold['cached']})", (cold["cached"] or 0) <= 64))
    info.append({"model": model, "warm_vs_cold_similarity": similarity(warm2["text"], cold["text"])})
if other:
    o = next(s for s in steps if s["step"] == f"{other} text warm")
    b = next(s for s in steps if s["step"] == f"{base} text warm")
    checks.append(("other (strong probe) adapter's warm answer differs from base's", o["text"] != b["text"]))

result = {"url": url, "run": run, "prefix_tokens_est": prefix_tokens, "steps": steps,
          "checks": [{"check": c, "ok": ok} for c, ok in checks], "info": info, "pass": all(ok for _, ok in checks)}
json.dump(result, open(out, "w"), indent=1)
for c, ok in checks:
    print(("PASS " if ok else "FAIL ") + c)
print("INFO", json.dumps(info))
print("OVERALL", "PASS" if result["pass"] else "FAIL")
sys.exit(0 if result["pass"] else 1)
