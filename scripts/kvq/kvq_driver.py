#!/usr/bin/env python3
"""In-CVM load driver for the KV-sharing qualification (stdlib only, no ports exposed).

KVQ_TESTS (comma list, run in order):
  health    wait until every KVQ_TARGETS endpoint answers /v1/models
  gsm8k     KVQ_GSM8K_N GSM8K test questions via chat completions on the first target; accuracy
  longturn  a ~KVQ_LONG_TOKENS-token conversation with a hidden code: turn 1 cold on target A,
            turn 2 (same prefix + question) on target B (A again if one target). Reports TTFT of
            both turns, cached_tokens and whether turn 2 recalls the code (KV correctness).
            With two targets this is the cross-replica restore test (shared tier / GPU fetch).
  metrics   print ghost/cache counters scraped from KVQ_METRICS_URLS (lab engines are not in Prometheus)
  hold      print one line and sleep KVQ_HOLD_S (log-path check)
  cold      the same turn-2 prompt with a fresh salt on target B: the recompute baseline.
Results are printed as `KVQ {json}` lines; `KVQ_DONE` at the end.
"""
import concurrent.futures as cf, json, os, random, re, time, urllib.request

TARGETS = [t.strip() for t in os.environ.get("KVQ_TARGETS", "http://kvq-router:8000").split(",") if t.strip()]
MODEL = os.environ.get("KVQ_MODEL", "z-ai/glm-5.3-flash")
LAST_REASONING = []  # reasoning_content of the last chat() call
TESTS = os.environ.get("KVQ_TESTS", "health,gsm8k,longturn,cold").split(",")


def out(kind, **kw):
    print("KVQ " + json.dumps({"kind": kind, **kw}), flush=True)


def post(base, path, body, timeout=3600):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def chat(base, messages, max_tokens, thinking=True, stream=True):
    """Returns (ttft_s, total_s, text, usage)."""
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0, "stream": stream,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": thinking}}
    t0 = time.perf_counter(); ttft = None; text = []; usage = None; LAST_REASONING.clear()
    with post(base, "/v1/chat/completions", body) as r:
        for raw in r:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            ev = json.loads(line[5:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                piece = (d.get("content") or "") + (d.get("reasoning_content") or "")
                if piece and ttft is None:
                    ttft = time.perf_counter() - t0
                text.append(d.get("content") or "")
                LAST_REASONING.append(d.get("reasoning_content") or "")
    return ttft, time.perf_counter() - t0, "".join(text), usage


def health():
    # Ready = a tiny chat succeeds end to end (a router can answer /v1/models before its
    # engines are up); it also keeps first-request overheads out of later timings.
    for base in TARGETS:
        t0, last = time.time(), None
        while True:
            try:
                chat(base, [{"role": "user", "content": "Say ok."}], 8, thinking=False)
                break
            except Exception as e:  # noqa: BLE001
                last = repr(e)[:200]
            if time.time() - t0 > 3600:
                out("health", target=base, ok=False, last_error=last); raise SystemExit(1)
            time.sleep(15)
        out("health", target=base, ok=True, wait_s=round(time.time() - t0))


def gsm8k():
    n = int(os.environ.get("KVQ_GSM8K_N", "150"))
    url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
    rows = [json.loads(l) for l in urllib.request.urlopen(url, timeout=60).read().decode().splitlines() if l.strip()][:n]

    def one(row):
        q = row["question"] + "\nSolve it, then give the final answer as a plain number on the last line in the form '#### <number>'."
        try:
            _, _, text, _ = chat(TARGETS[0], [{"role": "user", "content": q}], 8192, thinking=True, stream=True)
        except Exception as e:  # noqa: BLE001
            return False, f"error {e!r}"[:200]
        gold = row["answer"].split("####")[-1].strip().replace(",", "")
        m = re.findall(r"####\s*\x24?(-?[\d,]*\.?\d+)", text) or re.findall(r"(-?[\d,]*\.?\d+)", text)
        pred = m[-1].replace(",", "").rstrip(".") if m else ""
        try:
            return abs(float(pred) - float(gold)) < 1e-6, None
        except ValueError:
            return False, None

    t0 = time.time()
    with cf.ThreadPoolExecutor(int(os.environ.get("KVQ_GSM8K_CONC", "32"))) as ex:
        res = list(ex.map(one, rows))
    errs = [e for _, e in res if e]
    out("gsm8k", n=len(rows), accuracy=round(sum(ok for ok, _ in res) / len(rows), 4), errors=len(errs),
        first_error=errs[0] if errs else None, wall_s=round(time.time() - t0))


def long_doc(tokens, seed):
    rnd = random.Random(seed)
    words = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa".split()
    code = f"{rnd.randint(100000, 999999)}-{rnd.choice(words)}"
    lines, n_lines = [], tokens // 22  # ~22 tokens per line on GLM's tokenizer (calibrated by usage below)
    for i in range(n_lines):
        lines.append(f"Record {i:06d}: {' '.join(rnd.choice(words) for _ in range(8))} value={rnd.randint(0, 10**6)}.")
        if i == n_lines // 2:
            lines.append(f"IMPORTANT: the vault access code is {code}. Remember it.")
    return "\n".join(lines), code


def longturn(name="longturn", salt=None):
    tokens = int(os.environ.get("KVQ_LONG_TOKENS", "189000"))
    seed = salt if salt is not None else int(os.environ.get("KVQ_LONG_SEED", "7"))
    doc, code = long_doc(tokens, seed)
    a, b = TARGETS[0], TARGETS[-1]
    m1 = [{"role": "user", "content": f"[session {seed}]\n{doc}\n\nRead the log above. Reply with just 'ready'."}]
    res = {"seed": seed, "target_turn1": a, "target_turn2": b}
    if name == "longturn":
        ttft1, tot1, text1, u1 = chat(a, m1, 16, thinking=False)
        res.update(turn1_ttft_s=round(ttft1 or tot1, 2), turn1_prompt_tokens=(u1 or {}).get("prompt_tokens"),
                   turn1_cached=((u1 or {}).get("prompt_tokens_details") or {}).get("cached_tokens"))
        m2 = m1 + [{"role": "assistant", "content": text1 or "ready"}]
    else:
        m2 = m1 + [{"role": "assistant", "content": "ready"}]
    m2 = m2 + [{"role": "user", "content": "What is the vault access code? Reply with the code only."}]
    ttft2, tot2, text2, u2 = chat(b, m2, 32, thinking=False)
    res.update(turn2_ttft_s=round(ttft2 or tot2, 2), turn2_total_s=round(tot2, 2),
               turn2_prompt_tokens=(u2 or {}).get("prompt_tokens"),
               turn2_cached=((u2 or {}).get("prompt_tokens_details") or {}).get("cached_tokens"),
               recalled=code in ((text2 or "") + "".join(LAST_REASONING)), answer=(text2 or "")[:80],
               reasoning="".join(LAST_REASONING)[:160], code=code)
    out(name, **res)


def metrics():
    """Print ghost / prefix-cache / HiCache counters from each KVQ_METRICS_URLS endpoint (summed per name)."""
    want = re.compile(r"^(sglang[:_](ghost_[a-z_]+|cache_hit_rate|prompt_tokens_total|cached_tokens_total|"
                      r"kv_tier_[a-z_]+|hicache_[a-z_]+|realtime_tokens_total))(\{[^}]*\})?\s+([0-9.eE+-]+)\Z")
    for url in [u.strip() for u in os.environ.get("KVQ_METRICS_URLS", "").split(",") if u.strip()]:
        try:
            body = urllib.request.urlopen(url, timeout=20).read().decode(errors="replace")
        except Exception as e:  # noqa: BLE001
            out("metrics", url=url, error=repr(e)[:200]); continue
        agg = {}
        for line in body.splitlines():
            m = want.match(line.strip())
            if m:
                key = m.group(1) + (m.group(3) or "")
                agg[key] = agg.get(key, 0.0) + float(m.group(4))
        out("metrics", url=url, values=agg)


def main():
    for t in TESTS:
        t = t.strip()
        try:
            if t == "hold":  # log-path check: print, then stay up so `docker ps`/logs can see it
                out("hold", targets=TARGETS, tests=TESTS); time.sleep(int(os.environ.get("KVQ_HOLD_S", "600")))
            elif t == "health": health()
            elif t == "gsm8k": gsm8k()
            elif t == "metrics": metrics()
            elif t == "longturn":
                base = int(os.environ.get("KVQ_LONG_SEED", "7"))
                for i in range(int(os.environ.get("KVQ_LONG_REPEAT", "1"))):
                    longturn(salt=base + i)
            elif t == "cold": longturn("cold", salt=random.randint(10**6, 10**7))
        except SystemExit:
            raise
        except Exception as e:  # noqa: BLE001
            out("error", test=t, error=repr(e)[:500])


if __name__ == "__main__":
    try:
        main()
    finally:
        print("KVQ_DONE", flush=True)
