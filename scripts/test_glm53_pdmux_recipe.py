"""Self-consistency tests for the PD-Multiplexing + EAGLE image recipe (docker/sglang-glm53-hicache-w4afp8-pdmuxe).

The recipe is the v7 recipe plus pdmux-eagle.diff. These tests need no image: they check that the pinned hashes,
the source manifest, the Dockerfile COPY list and the publish workflow's integrity pins all describe the same files.

Run: python3 -m scripts.test_glm53_pdmux_recipe
"""

import hashlib
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "docker/sglang-glm53-hicache-w4afp8-pdmuxe"
WORKFLOW = (ROOT / ".github/workflows/publish-glm53-admission-reserve.yaml").read_text()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def block(text: str, start_marker: str, end_pattern: str) -> str:
    start = text.index(start_marker)
    return text[start : start + len(start_marker) + re.search(end_pattern, text[start + len(start_marker) :]).start()]


class PdmuxRecipeTest(unittest.TestCase):
    """The recipe is self-consistent and the publish workflow pins the files it will build."""

    def test_sha256sums_cover_every_recipe_file(self):
        listed = {}
        for line in (RECIPE / "SHA256SUMS").read_text().splitlines():
            digest, name = line.split("  ", 1)
            listed[name] = digest
        for name, digest in listed.items():
            self.assertEqual(sha256(RECIPE / name), digest, name)
        files = {p.name for p in RECIPE.iterdir()}
        self.assertEqual(files - set(listed), {"Dockerfile", "SHA256SUMS"})
        dockerfile = (RECIPE / "Dockerfile").read_text()
        for name in files - {"SHA256SUMS"}:
            self.assertIn(name, dockerfile, f"{name} is not COPYed by the Dockerfile")

    def test_apply_patches_pins_the_pdmux_patch_last(self):
        text = (RECIPE / "apply-patches.py").read_text()
        pins = re.findall(r'^\s+"([a-z0-9.-]+\.diff)": "([0-9a-f]{64})",$', text, re.M)
        self.assertEqual(len(pins), 13)
        self.assertEqual(pins[-1], ("pdmux-eagle.diff", sha256(RECIPE / "pdmux-eagle.diff")))
        self.assertNotIn("kda-zero-sync.diff", dict(pins), "this recipe is v7 + pdmux, not v8")
        for name, digest in pins:
            self.assertEqual(sha256(RECIPE / name), digest, name)

    def test_source_manifest_covers_the_17_pdmux_files(self):
        manifest = json.loads((RECIPE / "source-manifest.json").read_text())
        diff_files = re.findall(r"^\+\+\+ b/(\S+)$", (RECIPE / "pdmux-eagle.diff").read_text(), re.M)
        self.assertEqual(len(diff_files), 17)
        for name in diff_files:
            self.assertIn(name, manifest, name)
            self.assertRegex(manifest[name]["after"], r"^[0-9a-f]{64}$")
        self.assertEqual(len(manifest), 40)
        self.assertNotIn("python/sglang/srt/layers/attention/linear/kda_backend.py", manifest)
        # the two files an earlier patch also edits keep their base-image before-hash
        self.assertEqual(manifest["python/sglang/srt/managers/scheduler.py"]["before"], "6ffd1584f6bcbbd3384f1ec45e3313a7b4b010fd3f900b8f19345d4b02f5d0ba")
        self.assertEqual(manifest["python/sglang/srt/layers/quantization/w4afp8.py"]["before"], "21e9c527c9b83e350cdc35ce2bc62891cda1550934b2a5d302f0f807f752f125")

    def test_workflow_pins_match_the_recipe_files(self):
        section = block(WORKFLOW, "            hicache-w4afp8-pdmuxe)\n", r"\n              ;;")
        self.assertIn("BUILD_CONTEXT=docker/sglang-glm53-hicache-w4afp8-pdmuxe", section)
        self.assertIn("DEFAULT_TAG=glm53-hicache-w4afp8-v7-pdmuxe", section)
        self.assertIn("TEST_SCRIPT=/usr/share/nearai/glm53-hicache-w4afp8-pdmuxe/test-cpu.sh", section)
        self.assertIn("BASE_DIGEST=sha256:3eccc30709f5719c81084d1264f03ca5354b3059ec4cff62f0c5ff88c309680c", section)
        for var, name in (("DOCKERFILE", "Dockerfile"), ("PROVENANCE", "PROVENANCE"), ("SHA256SUMS", "SHA256SUMS")):
            self.assertIn(f"EXPECTED_{var}_SHA256={sha256(RECIPE / name)}", section, f"update EXPECTED_{var}_SHA256 in the workflow")
        self.assertIn("- hicache-w4afp8-pdmuxe", WORKFLOW)
        # the per-variant label checks must include the new variant
        self.assertGreaterEqual(WORKFLOW.count('"$VARIANT" == "hicache-w4afp8-pdmuxe"'), 4)


if __name__ == "__main__":
    unittest.main()
