import copy
import gc
import json
import os
import sys
import threading
import time
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask
from transformers import AutoTokenizer, pipeline

MODEL_ID = "openai/privacy-filter"
MODEL_REVISION = "7ffa9a043d54d1be65afb281eddf0ffbe629385b"


def _env(name, default, cast):
    # Tolerate absent/blank/garbage env so a typo can't crash-loop the boot.
    try:
        return cast(os.environ[name])
    except (KeyError, ValueError):
        return default


def _retry_window(max_length, cap):
    # The retry must use a strictly smaller window than the primary pass, or it
    # repeats the identical forward (clamped eager mode is 4096 -> 2048).
    window = min(cap, max_length // 2)
    return window if window >= 8 else None


# Left unbounded,
# the HF pipeline's CUDA caching allocator ratchets its reserved memory up
# under traffic and never releases it, eventually filling the card.
# Defence in depth (all tunable via env, no rebuild):
#   * PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (set in compose) is the
#     root-cause fix — reserved segments can shrink again instead of ratcheting.
#   * A watchdog returns idle cached blocks to the driver on an interval (never
#     per request: empty_cache() is a synchronizing cudaFree that would stall
#     the GPU) and hard-restarts the container if this process exceeds its VRAM
#     budget. Beside the coloc Qwen3.8 the per-process cap keeps reserved memory
#     under that budget, so the OOM bullet below is the restart path.
#   * An acute CUDA-OOM retries once with a strictly smaller window, then returns
#     503 if memory is still short. After FILTER_OOM_EXIT_AFTER consecutive 503s
#     without a successful classify, exit for a clean restart rather than serving
#     errors behind a still-healthy /v1/models probe.
# The model's sliding_window is 128, so a non-eager attention implementation
# makes attention cost linear in sequence length, not quadratic. That permits a
# single 131072-token pass at batch 1. Eager attention measured ~13 GB over
# baseline at only 16k tokens and cannot safely use the full window, so explicit
# eager mode or a rejected non-eager implementation clamps MAX_LENGTH to 4096
# and retains STRIDE-token overlapping chunks without truncating the document.
BATCH_SIZE = max(1, _env("PRIVACY_BATCH_SIZE", 8, int))
MAX_LENGTH = max(8, _env("PRIVACY_MAX_LENGTH", 4096, int))
STRIDE = max(0, min(_env("PRIVACY_STRIDE", 256, int), MAX_LENGTH - 1))
ATTN_IMPL = _env("PRIVACY_ATTN_IMPL", "sdpa", str)
# MoE experts kernel. Unset keeps the library default, which resolves to the per-expert "eager" loop for
# this model; "grouped_mm" runs all 128 experts of a layer as one grouped GEMM.
EXPERTS_IMPL = _env("PRIVACY_EXPERTS_IMPL", "", str)
_EXPERTS_KW = {"experts_implementation": EXPERTS_IMPL} if EXPERTS_IMPL else {}
GPU_MEM_LIMIT_GB = max(1.0, _env("GPU_MEM_LIMIT_GB", 36.0, float))
WATCHDOG_INTERVAL_S = max(1.0, _env("WATCHDOG_INTERVAL_S", 30.0, float))
FILTER_GPU_MEM_FRACTION = min(1.0, max(0.01, _env("FILTER_GPU_MEM_FRACTION", 0.25, float)))
OOM_RETRY_MAX_LENGTH = max(8, _env("FILTER_OOM_RETRY_MAX_LENGTH", 16384, int))
OOM_EXIT_AFTER = max(1, _env("FILTER_OOM_EXIT_AFTER", 3, int))
_oom_lock = threading.Lock()
_consecutive_oom_503 = 0

USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    # Per-process ceiling includes live tensors and the caching allocator.
    torch.cuda.set_per_process_memory_fraction(FILTER_GPU_MEM_FRACTION)

if ATTN_IMPL == "eager":
    MAX_LENGTH = 4096
    clf = pipeline(
        "token-classification",
        model=MODEL_ID,
        revision=MODEL_REVISION,
        aggregation_strategy="simple",
        device=0 if USE_CUDA else -1,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        model_kwargs=_EXPERTS_KW,
    )
else:
    try:
        clf = pipeline(
            "token-classification",
            model=MODEL_ID,
            revision=MODEL_REVISION,
            aggregation_strategy="simple",
            device=0 if USE_CUDA else -1,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            model_kwargs={"attn_implementation": ATTN_IMPL, **_EXPERTS_KW},
        )
        cfg = getattr(getattr(clf, "model", None), "config", None)
        # The private field may change; try the public form too, and safely
        # clamp below if neither is available.
        effective_attn_impl = (
            getattr(cfg, "_attn_implementation", None)
            or getattr(cfg, "attn_implementation", None)
        )
        if effective_attn_impl is None:
            MAX_LENGTH = 4096
            sys.stderr.write(
                "privacy-filter: requested=%s attention implementation could not "
                "be confirmed; clamping MAX_LENGTH=%d\n"
                % (ATTN_IMPL, MAX_LENGTH)
            )
            sys.stderr.flush()
        elif effective_attn_impl != ATTN_IMPL:
            MAX_LENGTH = 4096
            sys.stderr.write(
                "privacy-filter: requested=%s effective=%s attention implementation "
                "mismatch; falling back to eager + MAX_LENGTH=%d\n"
                % (ATTN_IMPL, effective_attn_impl, MAX_LENGTH)
            )
            sys.stderr.flush()
            clf = pipeline(
                "token-classification",
                model=MODEL_ID,
                revision=MODEL_REVISION,
                aggregation_strategy="simple",
                device=0 if USE_CUDA else -1,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                model_kwargs=_EXPERTS_KW,
            )
    except Exception as exc:
        error_text = str(exc).lower()
        is_attention_rejection = "attn_implementation" in error_text or (
            "attention" in error_text
            and any(
                marker in error_text
                for marker in ("reject", "not supported", "does not support", "unsupported")
            )
        )
        if not is_attention_rejection:
            sys.stderr.write(
                "privacy-filter: non-attention initialization failure (%s); "
                "treating as transient and restarting\n" % exc
            )
            sys.stderr.flush()
            raise

        MAX_LENGTH = 4096
        sys.stderr.write(
            "privacy-filter: attn_implementation=%s rejected (%s); "
            "falling back to eager + MAX_LENGTH=%d\n"
            % (ATTN_IMPL, exc, MAX_LENGTH)
        )
        sys.stderr.flush()
        # Let failures in the eager rebuild propagate for a clean restart.
        clf = pipeline(
            "token-classification",
            model=MODEL_ID,
            revision=MODEL_REVISION,
            aggregation_strategy="simple",
            device=0 if USE_CUDA else -1,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            model_kwargs=_EXPERTS_KW,
        )
STRIDE = max(0, min(STRIDE, MAX_LENGTH - 1))
print(json.dumps({"event": "filter_config", "max_length": MAX_LENGTH, "experts_implementation": getattr(
    getattr(getattr(clf, "model", None), "config", None), "_experts_implementation", None)}), flush=True)
# Cap the pipeline's per-forward window. The pipeline chunks inputs longer than
# this into model_max_length-token windows (see the classify() stride= arg below).
clf.tokenizer.model_max_length = MAX_LENGTH
# The retry has its own tokenizer window but shares the same model weights.
retry_max_length = _retry_window(MAX_LENGTH, OOM_RETRY_MAX_LENGTH)
if retry_max_length is not None:
    retry_tokenizer = copy.deepcopy(clf.tokenizer)
    retry_tokenizer.model_max_length = retry_max_length
    clf_retry = pipeline(
        "token-classification",
        model=clf.model,
        tokenizer=retry_tokenizer,
        aggregation_strategy="simple",
        device=0 if USE_CUDA else -1,
    )
else:
    clf_retry = None
# Separate tokenizer for usage counts only — left UNBOUNDED so input_tokens stays
# accurate for the full text (it never truncates: truncation defaults to False).
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True)


def _gpu_watchdog():
    while True:
        time.sleep(WATCHDOG_INTERVAL_S)
        torch.cuda.empty_cache()
        reserved_gb = torch.cuda.memory_reserved(0) / 1024 ** 3
        max_reserved_gb = torch.cuda.max_memory_reserved(0) / 1024 ** 3
        allocated_gb = torch.cuda.memory_allocated(0) / 1024 ** 3
        print(json.dumps({"event": "filter_mem", "reserved_gb": reserved_gb,
                          "max_reserved_gb": max_reserved_gb, "allocated_gb": allocated_gb}), flush=True)
        if reserved_gb > GPU_MEM_LIMIT_GB:
            sys.stderr.write(
                "privacy-filter: reserved %.1f GB > %.1f GB limit; "
                "exiting for a clean restart\n" % (reserved_gb, GPU_MEM_LIMIT_GB)
            )
            sys.stderr.flush()
            os._exit(1)


if USE_CUDA:
    threading.Thread(target=_gpu_watchdog, daemon=True).start()

app = FastAPI()


class ClassifyRequest(BaseModel):
    input: str | list[str]
    threshold: float = Field(default=0.0, ge=0.0, le=1.0)


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]}

def _merge_adjacent_spans(spans):
    if not spans:
        return spans

    merged = [spans[0]]
    for span in spans[1:]:
        prev = merged[-1]
        if span["category"] == prev["category"] and span["start"] == prev["end"]:
            prev["end"] = span["end"]
            prev["text"] = prev["text"] + span["text"]
            prev["score"] = min(prev["score"], span["score"])
            continue

        merged.append(span)
    return merged


@app.post("/v1/privacy/classify")
def classify(req: ClassifyRequest):
    global _consecutive_oom_503
    texts = [req.input] if isinstance(req.input, str) else req.input
    if not texts or any(not isinstance(t, str) for t in texts):
        raise HTTPException(400, "input must be a non-empty string or list of strings")

    # The pipeline runs the forward under torch.no_grad() internally.
    # No lock: classify requests stay concurrent (the OOM counter's lock
    # never covers a forward).
    # Recover after leaving each except block: while an OOM is being handled, its
    # traceback still holds the failed forward's frames and their tensors, so
    # empty_cache() could not return that memory.
    oom = False
    try:
        raw = clf(texts, batch_size=BATCH_SIZE, stride=STRIDE)
    except torch.cuda.OutOfMemoryError:
        oom = True
    if oom:
        # Don't 500-storm behind a healthy /v1/models probe: retry with
        # smaller windows without mutating the shared classifier state.
        print(json.dumps({"event": "filter_oom", "retry_max_length": retry_max_length}), flush=True)
        gc.collect()
        torch.cuda.empty_cache()
        retry_oom = clf_retry is None
        if clf_retry is not None:
            try:
                raw = clf_retry(texts, batch_size=BATCH_SIZE, stride=min(STRIDE, retry_max_length - 1))
            except torch.cuda.OutOfMemoryError:
                retry_oom = True
        if retry_oom:
            if clf_retry is not None:
                gc.collect()
                torch.cuda.empty_cache()
            with _oom_lock:
                _consecutive_oom_503 += 1
                consecutive_503 = _consecutive_oom_503
            if consecutive_503 >= OOM_EXIT_AFTER:
                print(json.dumps({"event": "filter_oom_exit", "consecutive_503": consecutive_503}), flush=True)
                sys.stderr.write(
                    "privacy-filter: %d consecutive memory-pressure 503s; "
                    "exiting for a clean restart\n" % consecutive_503
                )
                sys.stderr.flush()
                return JSONResponse(status_code=503, headers={"Retry-After": "2"},
                                    content={"error": "gpu_memory_pressure"},
                                    background=BackgroundTask(os._exit, 1))
            return JSONResponse(status_code=503, headers={"Retry-After": "2"},
                                content={"error": "gpu_memory_pressure"})

    with _oom_lock:
        _consecutive_oom_503 = 0

    # Single batched tokenize for usage counts instead of N sequential calls.
    tok_lens = [len(ids) for ids in tokenizer(texts).input_ids]

    data = []
    for i, spans in enumerate(raw):
        kept = [
            {
                "category": s["entity_group"],
                "score": float(s["score"]),
                "text": s["word"],
                "start": int(s["start"]),
                "end": int(s["end"]),
            }
            for s in spans
            if float(s["score"]) >= req.threshold
        ]
        data.append(
            {
                "index": i,
                "spans": _merge_adjacent_spans(kept),
                "usage": {"input_tokens": tok_lens[i]},
            }
        )

    return {"model": MODEL_ID, "data": data}
