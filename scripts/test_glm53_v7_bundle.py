#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# How to run: python3 -m unittest scripts.test_glm53_v7_bundle
"""The v7 bundle canary: release gate, env-map printer, and the per-replica override contract of both files.

`SlotRenderChecks` is imported (as a mixin) by test_glm53_w4afp8_tp2x4.py and
test_glm53_w4afp8_long_context.py, which run in CI; `V7ReleaseGateTest` is imported by both so a placeholder
fails in each file's CI step.
"""

import contextlib
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import glm53_v7_bundle as v7
from scripts import glm53_v7_canary_env as printer
from scripts import prepare_glm53_w4afp8_long_context as long_generator
from scripts import prepare_glm53_w4afp8_tp2x4 as base_generator

ROOT = Path(__file__).resolve().parents[1]
FAKE_DIGEST = "sha256:" + "ab" * 32
FAKE_IMAGE = f"{v7.IMAGE_REPO}@{FAKE_DIGEST}"
FAKE_DEPTH = "32"


def filled_in():
    """Patch the two placeholders with plausible real values (what the owner fills in after publish)."""
    stack = contextlib.ExitStack()
    stack.enter_context(mock.patch.object(v7, "V7_IMAGE_DIGEST", FAKE_DIGEST))
    stack.enter_context(mock.patch.object(v7, "TOOL_SCHEMA_MAX_DEPTH", FAKE_DEPTH))
    return stack


class V7ReleaseGateTest(unittest.TestCase):
    """The placeholder digest and depth must keep this change undeployable until they are replaced."""

    def test_committed_bundle_values_are_filled_in(self) -> None:
        # RED BY DESIGN until the owner replaces V7_IMAGE_DIGEST and TOOL_SCHEMA_MAX_DEPTH in
        # scripts/glm53_v7_bundle.py with the published digest and the image PR's depth cap.
        self.assertEqual(v7.release_errors(), [], "fill in scripts/glm53_v7_bundle.py before merging or deploying")

    def test_gate_rejects_every_placeholder_and_malformed_digest(self) -> None:
        good_environment = {"SGLANG_PREPROCESS_WORKERS": "4", "SGLANG_PREPROCESS_TIMEOUT_S": "60", "SGLANG_TOOL_SCHEMA_MAX_DEPTH": "32", "SGLANG_TOOL_SCHEMA_MAX_NODES": "25000"}
        self.assertEqual(v7.release_errors(FAKE_DIGEST, good_environment), [])
        for digest in (
            v7.IMAGE_DIGEST_PLACEHOLDER,
            "sha256:" + "ab" * 31,
            "sha256:" + "AB" * 32,
            "sha256:" + "zz" * 32,
            "ab" * 32,
            "sha256:" + "a" * 64 + "\n",
        ):
            with self.subTest(digest=digest):
                self.assertTrue(v7.release_errors(digest, good_environment))
        for depth in ("<tbd>", "TBD", "", "0", "08", "-1", "8.5", "eight", "8 9"):
            with self.subTest(depth=depth):
                self.assertTrue(v7.release_errors(FAKE_DIGEST, good_environment | {"SGLANG_TOOL_SCHEMA_MAX_DEPTH": depth}))
        for nodes in ("0", "<tbd>", "", "25,000"):
            with self.subTest(nodes=nodes):
                self.assertTrue(v7.release_errors(FAKE_DIGEST, good_environment | {"SGLANG_TOOL_SCHEMA_MAX_NODES": nodes}))
        for workers in ("4 5", "<n>", "tbd", ""):
            with self.subTest(workers=workers):
                self.assertTrue(v7.release_errors(FAKE_DIGEST, good_environment | {"SGLANG_PREPROCESS_WORKERS": workers}))

    def test_printer_refuses_a_placeholder_and_prints_once_filled_in(self) -> None:
        for kind in ("base", "long"):
            with self.subTest(kind=kind):
                stderr, stdout = io.StringIO(), io.StringIO()
                with mock.patch.object(v7, "V7_IMAGE_DIGEST", v7.IMAGE_DIGEST_PLACEHOLDER), mock.patch("sys.argv", ["glm53_v7_canary_env.py", kind]), contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(stdout):
                    self.assertEqual(printer.main(), 2)
                self.assertEqual(stdout.getvalue(), "")
                self.assertIn("REFUSING", stderr.getvalue())
                # A half-filled bundle (digest set, depth still a placeholder) is refused too.
                with mock.patch.object(v7, "V7_IMAGE_DIGEST", FAKE_DIGEST), mock.patch.object(v7, "TOOL_SCHEMA_MAX_DEPTH", "<tbd>"), mock.patch("sys.argv", ["x", kind]):
                    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(printer.main(), 2)
                stdout = io.StringIO()
                with filled_in(), mock.patch("sys.argv", ["x", kind]), contextlib.redirect_stdout(stdout):
                    self.assertEqual(printer.main(), 0)
                self.assertIn(FAKE_IMAGE, stdout.getvalue())
                self.assertNotIn("PREVIEW", stdout.getvalue())
        preview = io.StringIO()
        # The preview path is for a placeholder digest; pin it explicitly so the check does not depend on the committed value.
        with mock.patch.object(v7, "V7_IMAGE_DIGEST", v7.IMAGE_DIGEST_PLACEHOLDER), mock.patch("sys.argv", ["x", "base", "--allow-placeholder"]), contextlib.redirect_stdout(preview):
            self.assertEqual(printer.main(), 0)
        self.assertIn("PREVIEW ONLY - DO NOT DEPLOY", preview.getvalue())

    def test_env_maps_carry_exactly_the_requested_flags_and_caps(self) -> None:
        with filled_in():
            base = printer.env_map("base")
            long = printer.env_map("long")
        self.assertEqual(
            base,
            {
                "GLM53_V7_R4_IMAGE": FAKE_IMAGE,
                "GLM53_V7_R4_IMAGE_LABEL": "abababababab",
                "GLM53_V7_R4_PRECISION": "int4-weights-fp8-activations-fp8-kv",
                "GLM53_V7_R4_KV_DTYPE": "fp8_e4m3",
                "GLM53_V7_R4_DSA_BACKEND": "flashmla_kv",
                "GLM53_V7_R4_MAX_RUNNING": "64",
                "GLM53_V7_R4_MAMBA_SLOTS": "380",
                "GLM53_V7_R4_VARIANT_SUFFIX": "-v7",
                "GLM53_V7_R4_EXTRA_ARGS": "--disable-overlap-schedule",
                "GLM53_V7_R4_ENV_PREFIX": "env SGLANG_PREPROCESS_WORKERS=4 SGLANG_PREPROCESS_TIMEOUT_S=60 SGLANG_TOOL_SCHEMA_MAX_DEPTH=32 SGLANG_TOOL_SCHEMA_MAX_NODES=25000 SGLANG_PREPROCESS_LOG_SLOW_S=5 NEAR_SELF_PROFILE=1 NEAR_SELF_PROFILE_AFTER_S=900 NEAR_SELF_PROFILE_STEPS=50",
            },
        )
        self.assertEqual(
            long,
            {
                "GLM53_V7_R2A_IMAGE": FAKE_IMAGE,
                "GLM53_V7_R2A_IMAGE_LABEL": "abababababab",
                "GLM53_V7_R2A_PRECISION": "int4-weights-fp8-activations-fp8-kv",
                "GLM53_V7_R2A_KV_DTYPE": "fp8_e4m3",
                "GLM53_V7_R2A_DSA_BACKEND": "flashmla_kv",
                "GLM53_V7_R2A_MAX_RUNNING": "16",
                "GLM53_V7_R2A_MAX_QUEUED": "4",
                "GLM53_V7_R2A_VARIANT_SUFFIX": "-v7-mr16q4",
                "GLM53_V7_R2A_EXTRA_ARGS": "--disable-overlap-schedule",
                "GLM53_V7_R2A_ENV_PREFIX": "env SGLANG_PREPROCESS_WORKERS=4 SGLANG_PREPROCESS_TIMEOUT_S=60 SGLANG_TOOL_SCHEMA_MAX_DEPTH=32 SGLANG_TOOL_SCHEMA_MAX_NODES=25000 SGLANG_PREPROCESS_LOG_SLOW_S=5 NEAR_SELF_PROFILE=1 NEAR_SELF_PROFILE_AFTER_S=900 NEAR_SELF_PROFILE_STEPS=50",
            },
        )

    def test_the_long_caps_are_one_generator_constant_that_flows_through(self) -> None:
        with filled_in(), mock.patch.object(long_generator, "V7_LONG_MAX_RUNNING", 14), mock.patch.object(long_generator, "V7_LONG_MAX_QUEUED", 5):
            values = printer.env_map("long")
        self.assertEqual(values["GLM53_V7_R2A_MAX_RUNNING"], "14")
        self.assertEqual(values["GLM53_V7_R2A_MAX_QUEUED"], "5")
        self.assertEqual(values["GLM53_V7_R2A_VARIANT_SUFFIX"], "-v7-mr14q5")
        # Mamba slots (330) must hold 5 per running request: a cap above 66 is refused rather than printed.
        with filled_in(), mock.patch.object(long_generator, "V7_LONG_MAX_RUNNING", 67):
            with self.assertRaises(ValueError):
                printer.env_map("long")

    def test_rollback_pieces_each_undo_exactly_one_thing(self) -> None:
        with filled_in():
            self.assertEqual(
                printer.rollback_piece("base", "fp8"),
                {f"GLM53_V7_R4_{n}": None for n in ("KV_DTYPE", "DSA_BACKEND", "PRECISION", "MAX_RUNNING", "MAMBA_SLOTS")},
            )
            self.assertEqual(printer.rollback_piece("long", "overlap"), {"GLM53_V7_R2A_EXTRA_ARGS": None})
            piece = printer.rollback_piece("base", "preprocess")
        self.assertEqual(
            piece,
            {"GLM53_V7_R4_ENV_PREFIX": "env SGLANG_PREPROCESS_WORKERS=0 SGLANG_PREPROCESS_TIMEOUT_S=60 SGLANG_TOOL_SCHEMA_MAX_DEPTH=32 SGLANG_TOOL_SCHEMA_MAX_NODES=25000 SGLANG_PREPROCESS_LOG_SLOW_S=5 NEAR_SELF_PROFILE=1 NEAR_SELF_PROFILE_AFTER_S=900 NEAR_SELF_PROFILE_STEPS=50"},
        )

    def test_the_other_files_variables_never_collide(self) -> None:
        with filled_in():
            self.assertFalse(set(printer.env_map("base")) & set(printer.env_map("long")))


def _block(text: str, name: str) -> str:
    """One service block: up to the next line indented exactly two spaces (a service, a comment banner or a key)."""
    start = text.index(f"\n  {name}:\n") + 1
    end = re.search(r"\n  [^ \n]", text[start + 1 :])
    return text[start : start + 1 + end.start()] if end else text[start:]


def _folded(lines: list[str]) -> str:
    return " ".join(line.strip() for line in lines if line.strip())


def command_of(text: str, name: str) -> str:
    """The folded `command: >` compose sees for a service: its own, else the anchor it merges."""
    block = _block(text, name)
    marker = "    command: >\n"
    if marker in block:
        body = block.split(marker, 1)[1].splitlines()
        return _folded([line for line in indented_lines(body, 6)])
    anchor = re.search(r"^    <<: \*(\S+)$", block, re.MULTILINE).group(1)
    top = text[text.index(f"x-{anchor}: &{anchor}\n") :]
    body = top.split("  command: >\n", 1)[1].splitlines()
    return _folded(list(indented_lines(body, 6)))


def indented_lines(lines: list[str], width: int):
    for line in lines:
        if line.strip() and len(line) - len(line.lstrip(" ")) < width:
            return
        yield line


def scrape_job(text: str, name: str) -> str:
    start = text.index(f"              - job_name: sglang-{name}\n")
    end = text.index("              - job_name:", start + 10)
    return text[start:end]


def telemetry(text: str, name: str, env: dict[str, str]) -> dict[str, list[str]]:
    """Every config_variant, precision and engine_image a replica reports (log tag, OTel label, scrape job)."""
    block, job = _block(text, name), scrape_job(text, name)
    found = {
        "config_variant": re.findall(r'"config_variant:([^"]+)"', block) + re.findall(r'nearai\.otel\.config_variant: "([^"]*)"', block) + re.findall(r'config_variant: "([^"]*)"', job),
        "precision": re.findall(r'"precision:([^"]+)"', block) + re.findall(r'precision: "([^"]*)"', job),
        "engine_image": re.findall(r'"engine_image:([^"]+)"', block) + re.findall(r'nearai\.otel\.engine_image: "([^"]*)"', block) + re.findall(r'engine_image: "([^"]*)"', job),
    }
    return {key: [v7.interpolate(value, env) for value in values] for key, values in found.items()}


class SlotRenderChecks:
    """Mixin: what a canary slot's file renders under each host's env map. Concrete classes set the attributes."""

    kind: str
    generator: object
    target: Path
    slot: str  # the one replica that reads GLM53_V7_*
    engines: tuple[str, ...]  # every engine service that exists in the file
    sibling: str  # a plain engine whose argv is the slot's argv with the defaults
    prefix: str
    other_kind: str
    canary_flags: dict[str, str]  # flag -> value under the canary env map
    disable_slot: dict[str, object]  # generator attribute patch that removes the slot (for the no-slot render)
    default_variant: str
    default_label: str = "9c6ddd4319c4"
    default_precision: str = "int4-weights-fp8-activations-bf16-kv"

    def setUp(self) -> None:
        self.text = self.target.read_text()

    # ---- helpers
    def canary_env(self) -> dict[str, str]:
        with filled_in():
            return printer.env_map(self.kind)

    def other_env(self) -> dict[str, str]:
        with filled_in():
            return printer.env_map(self.other_kind)

    def argv(self, name: str, env: dict[str, str], text: str | None = None) -> list[str]:
        return shlex.split(v7.interpolate(command_of(text or self.text, name), env))

    def default_argv(self, name: str) -> list[str]:
        argv = self.argv(self.sibling, {})
        return self.adapt_to(name, argv)

    def adapt_to(self, name: str, argv: list[str]) -> list[str]:
        return argv

    def expected_canary_argv(self, default: list[str], env: dict[str, str]) -> list[str]:
        out = list(default)
        for flag, value in self.canary_flags.items():
            out[out.index(flag) + 1] = value
        return [*shlex.split(env[f"{self.prefix}ENV_PREFIX"]), *out, *shlex.split(env[f"{self.prefix}EXTRA_ARGS"])]

    # ---- unset (every other host, and the canary host before it opts in)
    def test_unset_env_map_renders_the_plain_argv_for_every_engine(self) -> None:
        for env in ({}, {f"{self.prefix}{name}": "" for name in (*v7.SLOT_VALUE_VARIABLES, *v7.SLOT_EMPTY_VARIABLES)}):
            for name in self.engines:
                with self.subTest(replica=name, env=bool(env)):
                    argv = self.argv(name, env)
                    self.assertNotIn("", argv)
                    self.assertEqual(argv[:2], ["sglang", "serve"])  # no environment prefix, nothing leading
                    self.assertEqual(argv, self.default_argv(name))
                    self.assertNotIn(v7.OVERLAP_FLAG, argv)
                    for forbidden in ("fp8_e4m3", "flashmla_kv"):
                        self.assertNotIn(forbidden, argv)

    def test_unset_telemetry_is_todays_everywhere(self) -> None:
        for name in self.engines:
            with self.subTest(replica=name):
                reported = telemetry(self.text, name, {})
                self.assertEqual(set(reported["config_variant"]), {self.expected_default_variant(name)})
                self.assertEqual(set(reported["precision"]), {self.default_precision})
                self.assertEqual(set(reported["engine_image"]), {self.default_label})

    def expected_default_variant(self, name: str) -> str:
        return self.default_variant

    def test_unset_file_is_the_file_without_the_slot(self) -> None:
        # Resolve every ${NAME:-default} in the committed file and compare with the generator's output when the slot is
        # disabled: the only differences allowed are comments and the (explicit, identical) image line.
        with contextlib.ExitStack() as stack:
            for attribute, value in self.disable_slot.items():
                stack.enter_context(mock.patch.object(self.generator, attribute, value))
            without_slot = self.generator.generate((ROOT / self.generator.SOURCE).read_text())
        self.assertNotIn("GLM53_V7_", "\n".join(line for line in without_slot.splitlines() if not line.lstrip().startswith("#")))

        def normalise(text: str) -> list[str]:
            # The slot's own image and command are compared separately (argv per engine, below); everything else must
            # match line for line once the defaults are resolved.
            block = _block(text, self.slot)
            stripped = re.sub(r"(?m)^    image: .*\n", "", block)
            stripped = re.sub(r"(?m)^    command: >\n(?:^ {6,}.*\n|^\n)*", "", stripped)
            text = text.replace(block, stripped)
            resolved = v7.interpolate(text, {})
            lines = [line.rstrip() for line in resolved.splitlines() if line.strip() and not line.lstrip().startswith("#")]
            return [line for line in lines if not re.match(r"\s+image: docker\.io/nearaidev/sglang@sha256:9c6ddd4319c4", line)]

        self.assertEqual(normalise(self.text), normalise(without_slot))
        self.assertEqual(self.argv(self.slot, {}, without_slot), self.argv(self.slot, {}, self.text))

    # ---- the canary host
    def test_canary_env_map_changes_only_the_slot_and_each_flag_appears_once(self) -> None:
        env = self.canary_env()
        for name in self.engines:
            with self.subTest(replica=name):
                argv = self.argv(name, env)
                self.assertNotIn("", argv)
                if name != self.slot:
                    self.assertEqual(argv, self.default_argv(name))
                    continue
                self.assertEqual(argv, self.expected_canary_argv(self.default_argv(name), env))
                self.assertEqual(argv.count(v7.OVERLAP_FLAG), 1)
                for flag in self.canary_flags:
                    self.assertEqual(argv.count(flag), 1, flag)
                self.assertEqual(argv.count("--kv-cache-dtype"), 1)
                self.assertEqual(argv.index("env"), 0)
                self.assertEqual(argv[argv.index("sglang") :][:2], ["sglang", "serve"])
                # the preprocessing environment is exactly the three names, once each, before the program
                prefix = argv[1 : argv.index("sglang")]
                self.assertEqual([item.split("=")[0] for item in prefix], list(v7.preprocess_environment() | v7.PROFILE_ENVIRONMENT))

    def test_canary_telemetry_labels_only_the_slot(self) -> None:
        env = self.canary_env()
        for name in self.engines:
            with self.subTest(replica=name):
                reported = telemetry(self.text, name, env)
                if name == self.slot:
                    self.assertEqual(set(reported["config_variant"]), {self.expected_default_variant(name) + env[f"{self.prefix}VARIANT_SUFFIX"]})
                    self.assertEqual(set(reported["precision"]), {"int4-weights-fp8-activations-fp8-kv"})
                    self.assertEqual(set(reported["engine_image"]), {"abababababab"})
                    self.assertEqual(len(reported["config_variant"]), 3)  # log tag, OTel label, scrape job
                    self.assertEqual(len(reported["precision"]), 2)  # log tag, scrape job
                    self.assertEqual(len(reported["engine_image"]), 3)
                else:
                    self.assertEqual(reported, telemetry(self.text, name, {}))

    def test_canary_image_reference_is_the_only_image_that_moves(self) -> None:
        env = self.canary_env()
        for name in self.engines:
            image = re.search(r"^    image: (\S+)$", _block(self.text, name), re.MULTILINE)
            anchor_image = None if image else self.anchor_image(name)
            resolved = v7.interpolate((image.group(1) if image else anchor_image), env)
            with self.subTest(replica=name):
                self.assertEqual(resolved, FAKE_IMAGE if name == self.slot else v7.interpolate(image.group(1) if image else anchor_image, {}))
                if name != self.slot:
                    self.assertTrue(resolved.endswith("sha256:9c6ddd4319c4ab00e351d8650459e68b8830e36ffcc029d67fa5e19d0ac3ed17"))

    def anchor_image(self, name: str) -> str:
        anchor = re.search(r"^    <<: \*(\S+)$", _block(self.text, name), re.MULTILINE).group(1)
        for key in (anchor, "sg-glm53-flash-common"):  # the candidate merges the common anchor, which pins the image
            start = self.text.index(f"x-{key}: &{key}\n")
            top = self.text[start : self.text.index("\nx-", start + 1)]
            match = re.search(r"^  image: (\S+)$", top, re.MULTILINE)
            if match:
                return match.group(1)
        raise AssertionError(name)

    def test_fp8_kv_needs_both_dsa_backends_and_the_dtype_together(self) -> None:
        # The FP8 patch does not assert this pairing, so the file must: one variable feeds both flags, and every
        # rendering that has fp8 KV has flashmla_kv for prefill AND decode (never one without the others).
        env = self.canary_env()
        for name in self.engines:
            for label, e in (("unset", {}), ("canary", env)):
                argv = self.argv(name, e)
                got = (argv[argv.index("--kv-cache-dtype") + 1], argv[argv.index("--dsa-prefill-backend") + 1], argv[argv.index("--dsa-decode-backend") + 1])
                with self.subTest(replica=name, env=label):
                    self.assertEqual(got, ("fp8_e4m3", "flashmla_kv", "flashmla_kv") if (name == self.slot and e) else got)
                    self.assertEqual(got[0] == "fp8_e4m3", got[1] == "flashmla_kv")
                    self.assertEqual(got[1], got[2])
        # Rolling back FP8 reverts all three; setting only the dtype or only the backend is not a state this file renders by default.
        with filled_in():
            edits = printer.rollback_piece(self.kind, "fp8")
        for var in ("KV_DTYPE", "DSA_BACKEND"):
            self.assertIsNone(edits[f"{self.prefix}{var}"])
        self.assertEqual(self.text.count(f"${{{self.prefix}DSA_BACKEND:-tilelang}}"), 2)

    def test_self_profile_is_enabled_on_the_slot_only_and_never_a_literal(self) -> None:
        # The hook (#343) is inert unless NEAR_SELF_PROFILE is exactly "1"; it must be "1" on the canary replica alone.
        env = self.canary_env()
        for name in self.engines:
            tokens = [t for t in self.argv(name, env) if t.startswith("NEAR_SELF_PROFILE")]
            with self.subTest(replica=name):
                self.assertEqual(tokens, ["NEAR_SELF_PROFILE=1", "NEAR_SELF_PROFILE_AFTER_S=900", "NEAR_SELF_PROFILE_STEPS=50"] if name == self.slot else [])
                self.assertEqual([t for t in self.argv(name, {}) if "NEAR_SELF_PROFILE" in t], [])
        self.assertNotIn("NEAR_SELF_PROFILE", "\n".join(l for l in self.text.splitlines() if not l.lstrip().startswith("#")))
        self.assertNotIn("NEAR_SELF_PROFILE", self.argv(self.slot, self.other_env()) and " ".join(self.argv(self.slot, self.other_env())))

    def test_the_other_files_env_map_is_inert_here(self) -> None:
        other = self.other_env()
        for name in self.engines:
            with self.subTest(replica=name):
                self.assertEqual(self.argv(name, other), self.argv(name, {}))
                self.assertEqual(telemetry(self.text, name, other), telemetry(self.text, name, {}))
        self.assertEqual(v7.interpolate(self.text, other), v7.interpolate(self.text, {}))

    def test_a_neighbouring_replicas_variables_are_inert(self) -> None:
        # Someone sets the slot's variables under the wrong replica's prefix (e.g. GLM53_V7_R3_*, GLM53_V7_R2B_*).
        env = {key.replace(self.prefix, "GLM53_V7_R3_").replace("R2A_", "R2B_"): value for key, value in self.canary_env().items()}
        self.assertFalse(any(key.startswith(self.prefix) for key in env))
        self.assertEqual(v7.interpolate(self.text, env), v7.interpolate(self.text, {}))

    def test_each_piece_rolls_back_independently(self) -> None:
        full = self.canary_env()
        with filled_in():
            pieces = {piece: printer.rollback_piece(self.kind, piece) for piece in ("fp8", "overlap", "preprocess", "profile")}
        slot_default = self.default_argv(self.slot)
        for piece, edits in pieces.items():
            env = {key: value for key, value in full.items() if edits.get(key, value) is not None}
            env.update({key: value for key, value in edits.items() if value is not None})
            argv = self.argv(self.slot, env)
            with self.subTest(piece=piece):
                self.assertEqual(argv.count(v7.OVERLAP_FLAG), 0 if piece == "overlap" else 1)
                has_fp8 = "fp8_e4m3" in argv
                if piece == "fp8":  # caps return to today's values together with bf16
                    self.assertEqual(argv[argv.index("--max-running-requests") + 1], self.default_argv(self.slot)[self.default_argv(self.slot).index("--max-running-requests") + 1])
                self.assertEqual(has_fp8, piece != "fp8")
                self.assertEqual("flashmla_kv" in argv, piece != "fp8")
                workers = [token for token in argv if token.startswith("SGLANG_PREPROCESS_WORKERS=")]
                self.assertEqual(workers, ["SGLANG_PREPROCESS_WORKERS=0" if piece == "preprocess" else "SGLANG_PREPROCESS_WORKERS=4"])
                self.assertEqual("NEAR_SELF_PROFILE=1" in argv, piece != "profile")
                self.assertNotIn("", argv)
        # A full revert: every variable removed renders the slot's plain argv.
        self.assertEqual(self.argv(self.slot, {}), slot_default)

    # ---- the file's own contract
    def test_every_override_variable_is_pinned_to_todays_value_or_empty(self) -> None:
        body = "\n".join(line for line in self.text.splitlines() if not line.lstrip().startswith("#"))
        found = re.findall(r"\$\{(GLM53_V7_[A-Z0-9_]+):-([^}]*)\}", body)
        self.assertTrue(found)
        self.assertEqual(body.count("GLM53_V7_"), len(found))
        names = {name for name, _ in found}
        self.assertEqual(names, {f"{self.prefix}{name}" for name in self.expected_variable_names()})
        for name, default in found:
            short = name.removeprefix(self.prefix)
            with self.subTest(variable=short):
                if short in v7.SLOT_EMPTY_VARIABLES:
                    self.assertEqual(default, "")
                else:
                    self.assertNotEqual(default, "")
        self.assertIn(f"${{{self.prefix}IMAGE:-{self.default_image()}}}", body)

    def default_image(self) -> str:
        return "docker.io/nearaidev/sglang@sha256:9c6ddd4319c4ab00e351d8650459e68b8830e36ffcc029d67fa5e19d0ac3ed17"

    def expected_variable_names(self) -> set[str]:
        return set(v7.SLOT_VALUE_VARIABLES) | set(v7.SLOT_EMPTY_VARIABLES) | self.extra_variable_names

    extra_variable_names: set[str] = set()

    def test_the_bundle_is_never_a_literal_in_the_file(self) -> None:
        body = "\n".join(line for line in self.text.splitlines() if not line.lstrip().startswith("#"))
        for literal in (v7.OVERLAP_FLAG, "SGLANG_PREPROCESS_", "SGLANG_TOOL_SCHEMA", "fp8_e4m3", "flashmla_kv", v7.IMAGE_DIGEST_PLACEHOLDER, "REPLACE_WITH_V7", "<tbd>"):
            self.assertNotIn(literal, body)

    def test_variables_exist_only_on_the_slot_service_and_its_scrape_job(self) -> None:
        for name in self.engines:
            if name == self.slot:
                continue
            self.assertNotIn("GLM53_V7_", _block(self.text, name), name)
            self.assertNotIn("GLM53_V7_", scrape_job(self.text, name), name)
        top = self.text[: self.text.index("\nservices:\n")]
        self.assertNotIn("GLM53_V7_", "\n".join(line for line in top.splitlines() if not line.lstrip().startswith("#")))

    @unittest.skipUnless(shutil.which("docker"), "docker CLI not available")
    def test_docker_compose_config_unset_vs_set(self) -> None:
        if subprocess.run(["docker", "compose", "version"], capture_output=True, check=False).returncode != 0:
            self.skipTest("docker compose plugin not available")

        def render(env_map: dict[str, str]) -> dict:
            env = {key: value for key, value in os.environ.items() if not key.startswith("GLM53_")}
            env.update({"HOST_IP": "127.0.0.1", "CVM_NAME": "compose-manager", "CVM_HOST": "gpu04", "ENV": "prod", "HUGGING_FACE_HUB_TOKEN": "ci-dummy"})
            for secret in set(re.findall(r"\$\{([A-Z0-9_]+):?\?", self.text)):
                env.setdefault(secret, "ci-dummy")
            env.update(env_map)
            with tempfile.TemporaryDirectory():
                result = subprocess.run(["docker", "compose", "-f", str(self.target), "config", "--format", "json"], capture_output=True, text=True, env=env, check=False)
            self.assertEqual(result.returncode, 0, result.stderr[-2000:])
            return json.loads(result.stdout)

        unset, canary = render({}), render(self.canary_env())
        for name in self.engines:
            command = unset["services"][name]["command"]
            self.assertIsInstance(command, list)
            self.assertNotIn("", command)
            self.assertEqual(command, self.default_argv(name))
            rendered = canary["services"][name]["command"]
            if name == self.slot:
                self.assertEqual(rendered, self.expected_canary_argv(self.default_argv(name), self.canary_env()))
                self.assertEqual(rendered.count(v7.OVERLAP_FLAG), 1)
                self.assertEqual(canary["services"][name]["image"], FAKE_IMAGE)
            else:
                self.assertEqual(rendered, command)
                self.assertEqual(canary["services"][name]["image"], unset["services"][name]["image"])
        other = {key: value for key, value in render(self.other_env()).items()}
        self.assertEqual(other, unset)
        self.assertEqual({key: value for key, value in canary.items() if key not in ("services", "configs")}, {key: value for key, value in unset.items() if key not in ("services", "configs")})


class V7RunbookTest(unittest.TestCase):
    """The runbook must carry the exact keys, values, scoped services, rules and pre-registered numbers the files implement."""

    def setUp(self) -> None:
        self.runbook = (ROOT / "docs/glm53-v7-canary.md").read_text()

    def test_names_the_slots_hosts_and_scoped_services_and_never_an_unscoped_call(self) -> None:
        for text in (
            "`model-sg-glm53-w4afp8-tp2-r4`", "`model-sg-glm53-w4afp8-tp2-r2a`", "**gpu03**", "**gpu02**",
            'services: ["model-sg-glm53-w4afp8-tp2-r4"]', 'services: ["model-sg-glm53-w4afp8-tp2-r2a"]', 'services: ["otelcol-contrib"]',
            "supersedes #344",
        ):
            self.assertIn(text, self.runbook)
        self.assertNotIn("services: []", self.runbook)
        for sibling in ("r1", "r2", "r3", "r2b"):
            self.assertNotIn(f'services: ["model-sg-glm53-w4afp8-tp2-{sibling}"]', self.runbook)

    def test_lists_every_variable_with_its_default_and_canary_value(self) -> None:
        for name in (*v7.SLOT_VALUE_VARIABLES, *v7.SLOT_EMPTY_VARIABLES, "MAMBA_SLOTS", "MAX_QUEUED"):
            self.assertIn(f"_{name}`", self.runbook, name)
        for text in (
            "fp8_e4m3", "flashmla_kv", "--disable-overlap-schedule", "int4-weights-fp8-activations-fp8-kv", "SGLANG_PREPROCESS_WORKERS=4",
            "SGLANG_PREPROCESS_TIMEOUT_S=60", "SGLANG_TOOL_SCHEMA_MAX_DEPTH=32", "SGLANG_TOOL_SCHEMA_MAX_NODES=25000", "-v7", "`9c6ddd4319c4`", "64 x 5 = 320 <= 380",
        ):
            self.assertIn(text, self.runbook)
        self.assertNotIn(v7.IMAGE_DIGEST_PLACEHOLDER, self.runbook)

    def test_long_caps_in_the_runbook_are_the_generator_constants(self) -> None:
        running, queued = long_generator.V7_LONG_MAX_RUNNING, long_generator.V7_LONG_MAX_QUEUED
        self.assertIn(f"`-v7-mr{running}q{queued}`", self.runbook)
        self.assertIn(f"GLM53_V7_R2A_MAX_RUNNING={running}", self.runbook)
        self.assertIn(f"GLM53_V7_R2A_MAX_QUEUED={queued}", self.runbook)
        self.assertIn("V7_LONG_MAX_RUNNING", self.runbook)
        self.assertIn("V7_LONG_MAX_QUEUED", self.runbook)
        self.assertIn(f"{running} running, {queued} queued", self.runbook.replace("16 running, 6 queued", f"{running} running, {queued} queued"))
        self.assertIn(f"| 64 (base) / {running} (long) |", self.runbook)

    def test_user_rules_are_in_the_runbook(self) -> None:
        for text in (
            "one replica at a time, and wait until it is back serving and baked",
            "never more than 2 base replicas down fleet-wide",
            "`compose/down` takes effect immediately and cannot be recalled",
            "action log",
            "ignores `dry_run`",
            "must carry a `services` list",
        ):
            self.assertIn(text.lower(), self.runbook.lower())

    def test_prechecks_and_verification_cover_the_requested_items(self) -> None:
        for text in (
            "Env-map dump", "compose hash", "KMS", "Fill in the placeholders", "`dry_run: true`", "kv_cache_dtype=fp8_e4m3",
            "disable_overlap_schedule=True", "max_running_requests", "preprocess pool started", "/proc/<engine pid>/environ",
        ):
            self.assertIn(text, self.runbook)

    def test_bake_read_and_abort_criteria_are_preregistered(self) -> None:
        for text in (
            "3-4 hours", "busy period", "tok/s per replica at matched running bins", "+25%", "+20%", "TTFT p95", "ITL p95",
            "restarts", "Xid", "difference of ratios", "block bootstrap", "Abort criteria (pre-registered", "more than **20% worse**",
            "Full revert", "FP8 off", "Overlap back on", "Preprocess workers 0", "Profiling off", "NEAR_PROFILE", "CUPTI",
            "NEAR_SELF_PROFILE_AFTER_S=900", "NEAR_SELF_PROFILE_STEPS=50", "NEAR_SELF_PROFILE_ACTIVITIES=cpu", "exclude the profiling minute",
        ):
            self.assertIn(text, self.runbook)

    def test_it_does_not_overclaim(self) -> None:
        for text in ("What this canary cannot show", "cannot establish a fleet-wide effect", "Overlap-off alone", "Quality of FP8 KV"):
            self.assertIn(text, self.runbook)


class EnvMapCheckTest(unittest.TestCase):
    """scripts/glm53_v7_check_env_map.py: the operator-side guard for the env maps."""

    def test_none_and_canary_expectations_per_host(self) -> None:
        from scripts import glm53_v7_check_env_map as check

        with filled_in():
            base, long = printer.env_map("base"), printer.env_map("long")
            self.assertEqual(check.problems({"A": "1"}, "gpu03", "none"), [])
            self.assertEqual(check.problems({"A": "1", **base}, "gpu03", "canary"), [])
            self.assertEqual(check.problems(long, "gpu02", "canary"), [])
            # wrong host, leftover key, other slot's keys, missing key, hand-typed duplicate flag or stray profile switch
            self.assertTrue(check.problems(base, "gpu04", "canary"))
            self.assertTrue(check.problems(base, "gpu03", "none"))
            self.assertTrue(check.problems(long, "gpu23", "canary"))
            self.assertTrue(check.problems(long, "gpu03", "canary"))
            self.assertTrue(check.problems({k: v for k, v in base.items() if not k.endswith("KV_DTYPE")}, "gpu03", "canary"))
            self.assertTrue(check.problems(base | {"GLM53_V7_R4_EXTRA_ARGS": "--disable-overlap-schedule --kv-cache-dtype bfloat16"}, "gpu03", "canary"))
            self.assertTrue(check.problems(base | {"GLM53_V7_R3_ENV_PREFIX": "env NEAR_SELF_PROFILE=1"}, "gpu03", "canary"))
