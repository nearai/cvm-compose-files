#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: python3 -m scripts.test_dsv41_profile_summary
"""summarize() on a synthetic two-rank trace fixture (no GPU, no network)."""

import gzip
import json
import tempfile
import io
import os
import unittest
from contextlib import redirect_stdout
from unittest import mock
from pathlib import Path

from scripts import dsv41_profile_summary as profile


def event(name: str, ts: int, dur: int, cat: str = "kernel") -> dict:
    return {"ph": "X", "cat": cat, "name": name, "ts": ts, "dur": dur}


def write_trace(directory: Path, rank: int, base_ns: int | None, events: list[dict]) -> None:
    data: dict = {"traceEvents": events + [{"ph": "X", "cat": "cpu_op", "name": "aten::mm", "ts": 0, "dur": 1}]}
    if base_ns is not None:
        data["baseTimeNanoseconds"] = base_ns
    path = directory / f"dp{rank}_pp0_tp0_dcp0_ep{rank}_rank0.1234.pt.trace.json.gz"
    with gzip.open(path, "wt") as handle:
        json.dump(data, handle)


def write_fixture(directory: Path, rank1_base_ns: int | None = 1_050_000_000) -> None:
    # Absolute microseconds = base/1000 + ts. Rank 0's prefill kernel is at 1_000_000 us: +-60 ms is dropped.
    write_trace(directory, 0, 1_000_000_000, [
        event("void sm90::prefill_attn<x>", 0, 500),
        event("moe_dropped_near_prefill", 30_000, 7777),
        event("ncclDevKernel_AllGatherV_RING", 200_000, 100),
        event("humming_moe_gemm", 300_000, 200),
        event("Memcpy HtoD (Pageable -> Device)", 300_500, 10, "gpu_memcpy"),
    ])
    # Rank 1's first kernel is at 1_055_000 us: inside rank 0's prefill margin, so dropped across ranks.
    write_trace(directory, 1, rank1_base_ns, [
        event("moe_dropped_by_other_rank_prefill", 5_000, 999),
        event("ncclDevKernel_AllGatherV_RING", 150_000, 300),
        event("nvjet_gemm_kernel", 150_400, 50),
        event("Memcpy DtoH (Device -> Pageable)", 150_500, 20, "gpu_memcpy"),
    ])


class SummarizeTest(unittest.TestCase):
    def test_decode_only_aggregates_per_rank(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            write_fixture(Path(tmp))
            rows = profile.summarize(tmp, ranks=2)
        self.assertEqual([r["rank"] for r in rows], [0, 1])
        self.assertEqual(
            set(rows[0]),
            {"rank", "window_ms", "decode_window_ms", "busy_ms", "categories", "memcpy", "top_kernels",
             "allgatherv_us", "prefill_intervals"},
        )
        r0, r1 = rows
        self.assertEqual(r0["prefill_intervals"], 1)
        self.assertEqual(r0["window_ms"], 300.5)
        self.assertEqual(r0["decode_window_ms"], 240.5)  # 60 ms of the window sits inside the prefill margin
        self.assertEqual(r0["busy_ms"], 0.3)  # 100 + 200 + 10 us; the near-prefill kernels are dropped
        self.assertEqual(r0["categories"]["comm (nccl)"], 0.1)
        self.assertEqual(r0["categories"]["moe"], 0.2)
        self.assertEqual(r0["categories"]["other"], 0.0)
        self.assertEqual(r0["memcpy"], {"h2d_n": 1, "h2d_ms": 0.01, "d2h_n": 0, "d2h_ms": 0.0, "d2d_n": 0, "d2d_ms": 0.0})
        self.assertEqual(r0["allgatherv_us"], {"p10": 100, "p50": 100, "p90": 100, "n": 1})
        self.assertEqual(r0["top_kernels"][0], ["humming_moe_gemm", 0.2])
        self.assertTrue(all(len(name) <= 64 for name, _ in r0["top_kernels"]))
        self.assertEqual(r1["memcpy"]["d2h_n"], 1)
        self.assertEqual(r1["categories"]["gemm-dense"], 0.1)
        self.assertEqual(r1["allgatherv_us"]["p50"], 300)
        self.assertNotIn("moe_dropped_by_other_rank_prefill", json.dumps(rows))

    def test_rows_are_one_line_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            write_fixture(Path(tmp))
            for row in profile.summarize(tmp, ranks=2):
                self.assertNotIn("\n", json.dumps(row, separators=(",", ":")))

    def test_missing_rank_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            write_fixture(Path(tmp))
            with self.assertRaisesRegex(RuntimeError, r"missing trace files for ranks \[2\]"):
                profile.summarize(tmp, ranks=3)

    def test_no_files_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "no dp"):
                profile.summarize(tmp, ranks=2)

    def test_missing_base_time_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            write_fixture(Path(tmp), rank1_base_ns=None)
            with self.assertRaisesRegex(RuntimeError, "baseTimeNanoseconds"):
                profile.summarize(tmp, ranks=2)


class MainFailureTest(unittest.TestCase):
    """main() cleanup on failure: HTTP, health and sleeps are mocked; no network."""

    def run_main(self, trace_dir, post, env=None, summarize=None, settle=None):
        calls = []

        def fake_post(path, timeout, ranks):
            calls.append((path, sorted(ranks)))
            return post(path, sorted(ranks))

        environ = {"PROFILE_TRACE_DIR": str(trace_dir), "PROFILE_DELAY_S": "0", "PROFILE_DUR_S": "0", **(env or {})}
        patches = [
            mock.patch.dict(os.environ, environ),
            mock.patch.object(profile, "wait_healthy"),
            mock.patch.object(profile, "wait_traces_settled", settle or mock.DEFAULT),
            mock.patch.object(profile.time, "sleep"),
            mock.patch.object(profile, "post_ranks", fake_post),
        ]
        if summarize is not None:
            patches.append(mock.patch.object(profile, "summarize", summarize))
        out = io.StringIO()
        for p in patches:
            p.start()
        try:
            with redirect_stdout(out):
                code = profile.main()
        finally:
            mock.patch.stopall()
        return code, out.getvalue().splitlines(), calls

    def assert_single_error(self, code, lines):
        self.assertEqual(code, 1)
        self.assertEqual(len(lines), 1)
        self.assertIn("profile_error", json.loads(lines[0]))

    def test_partial_start_failure_stops_started_ranks_and_removes_traces(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_fixture(Path(tmp))

            def post(path, ranks):
                if path == "/start_profile":
                    return {r: (OSError("refused") if r == 5 else None) for r in ranks}
                return {r: None for r in ranks}

            code, lines, calls = self.run_main(tmp, post)
            self.assert_single_error(code, lines)
            self.assertIn("rank 5", lines[0])
            stops = [ranks for path, ranks in calls if path == "/stop_profile"]
            self.assertEqual(len(stops), 1)
            self.assertTrue({0, 1, 2, 3, 4, 6, 7} <= set(stops[0]))
            self.assertEqual(os.listdir(tmp), [])

    def test_summarize_failure_stops_unstopped_ranks_and_removes_traces(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_fixture(Path(tmp))

            def post(path, ranks):
                if path == "/stop_profile" and ranks == list(range(8)):
                    return {r: (OSError("timeout") if r == 3 else None) for r in ranks}
                return {r: None for r in ranks}

            # rank 3's stop fails in the main path -> error; cleanup retries only rank 3
            code, lines, calls = self.run_main(tmp, post)
            self.assert_single_error(code, lines)
            self.assertEqual([c for c in calls if c[0] == "/stop_profile"][-1], ("/stop_profile", [3]))
            self.assertEqual(os.listdir(tmp), [])

    def test_summarize_raises_after_all_stopped_removes_traces(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_fixture(Path(tmp))

            def boom(_trace_dir):
                raise RuntimeError("bad trace")

            code, lines, calls = self.run_main(tmp, lambda path, ranks: {r: None for r in ranks},
                                               summarize=boom)
            self.assert_single_error(code, lines)
            self.assertIn("bad trace", lines[0])
            self.assertEqual(os.listdir(tmp), [])

    def test_settle_timeout_is_error_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            # no trace files at all: never settles
            real_settle = profile.wait_traces_settled
            clock = iter(range(0, 10_000, 3))
            with mock.patch.object(profile.time, "monotonic", lambda: next(clock)):
                code, lines, calls = self.run_main(
                    tmp, lambda path, ranks: {r: None for r in ranks}, env={"PROFILE_SETTLE_S": "6"},
                    settle=real_settle)
            self.assert_single_error(code, lines)
            self.assertIn("not complete and stable", lines[0])
            self.assertEqual(os.listdir(tmp), [])

    def test_settle_waits_for_growing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            for rank in range(8):
                write_trace(Path(tmp), rank, 1_000_000_000, [event("k", 0, 1)])
            path = next(Path(tmp).glob("dp0_*"))
            sleeps = []

            def grow(seconds):
                sleeps.append(seconds)
                if len(sleeps) == 1:
                    with open(path, "ab") as handle:
                        handle.write(b"x")

            with mock.patch.object(profile.time, "sleep", grow):
                profile.wait_traces_settled(tmp, 120)
            self.assertEqual(len(sleeps), 2)


if __name__ == "__main__":
    unittest.main()
