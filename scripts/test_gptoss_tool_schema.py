#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

# ─── How to run ───
# python3 scripts/test_gptoss_tool_schema.py
# ──────────────────

"""Test the gpt-oss tool schema rewrite embedded in experiments/gptoss-120b-openrouter-baremetal-gpu31.yaml.

Extracts gptoss_tool_schema_module and gptoss_tool_schema_patch from the compose file (undoing compose's `$$`
escaping), checks the schema rewrite, and runs the patch script against a copy of the vLLM v0.31.0 function it edits.
"""

from __future__ import annotations

import copy
import re
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).resolve().parents[1]
COMPOSE: Final = ROOT / "experiments/gptoss-120b-openrouter-baremetal-gpu31.yaml"

# create_tool_definition as shipped in vllm/vllm-openai v0.31.0 (vllm/entrypoints/openai/parser/harmony_utils.py).
V031_FUNCTION: Final = '''def create_tool_definition(tool: ChatCompletionToolsParam | Tool):
    if isinstance(tool, ChatCompletionToolsParam):
        return ToolDescription.new(
            name=tool.function.name,
            description=tool.function.description or "",
            parameters=tool.function.parameters,
        )
    return ToolDescription.new(
        name=tool.name,
        description=tool.description or "",
        parameters=tool.parameters,
    )
'''


def config_content(text: str, name: str) -> str:
    m = re.search(rf"(?m)^  {name}:\n    content: \|\n((?:(?:      .*)?\n)+)", text)
    assert m, f"config {name} not found"
    raw = "".join(line[6:] + "\n" if line else "\n" for line in m.group(1).splitlines())
    single = re.findall(r"(?<!\$)\$(?!\$)", raw.replace("$$", ""))
    assert not single, f"config {name} has an unescaped $ (compose would interpolate it)"
    return raw.replace("$$", "$")


def load_module(source: str) -> types.ModuleType:
    mod = types.ModuleType("nearai_tool_schema")
    exec(compile(source, "nearai_tool_schema.py", "exec"), mod.__dict__)
    return mod


def check_rewrite(f) -> None:
    plain = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "file"},
            "mode": {"type": "string", "enum": ["r", "w"]},
            "limit": {"type": "integer", "minimum": 1},
            "tags": {"type": "array", "items": {"type": "string"}},
            "opts": {"type": "object", "properties": {"deep": {"type": "boolean"}}, "required": ["deep"]},
        },
        "required": ["path"],
    }
    snapshot = copy.deepcopy(plain)
    assert f(plain) == plain, "a plain schema must come back unchanged"
    assert plain == snapshot, "input mutated"

    pydantic = {
        "$defs": {
            "Severity": {"type": "string", "enum": ["sev1", "sev2"]},
            "Action": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["page", "close"]},
                    "due": {"type": "integer", "enum": [1, 4]},
                    "who": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
                },
                "required": ["kind", "due", "who"],
            },
            "Node": {
                "type": "object",
                "properties": {"label": {"type": "string"}, "children": {"type": "array", "items": {"$ref": "#/$defs/Node"}}},
            },
        },
        "type": "object",
        "properties": {
            "severity": {"$ref": "#/$defs/Severity", "description": "how bad"},
            "actions": {"type": "array", "items": {"$ref": "#/$defs/Action"}},
            "impact": {"anyOf": [{"type": "object", "properties": {"n": {"type": "integer"}}}, {"type": "null"}]},
            "tree": {"$ref": "#/$defs/Node"},
            "kind_only": {"$ref": "#/$defs/Action/properties/kind"},
            "remote": {"$ref": "https://example.com/schema.json"},
            "legacy": {"allOf": [{"$ref": "#/$defs/Severity"}], "description": "v1 style"},
            "either": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
            "maybe": {"type": ["object", "null"], "properties": {"x": {"type": "string"}}},
            "method": {"const": "pickup"},
            "version": {"const": 2},
            "mode": {"enum": ["a", "b", None]},
        },
        "required": ["severity", "actions"],
    }
    snapshot = copy.deepcopy(pydantic)
    out = f(pydantic)
    assert pydantic == snapshot, "input mutated"
    p = out["properties"]
    assert "$defs" not in out
    assert p["severity"] == {"type": "string", "enum": ["sev1", "sev2"], "description": "how bad"}, p["severity"]
    action = p["actions"]["items"]
    assert action["properties"]["kind"]["enum"] == ["page", "close"]
    assert action["properties"]["due"]["type"] == "integer" and "One of: 1, 4." in action["properties"]["due"]["description"]
    assert action["properties"]["who"] == {"type": "string", "default": None, "nullable": True}, action["properties"]["who"]
    assert p["impact"]["type"] == "object" and p["impact"]["nullable"] is True and "n" in p["impact"]["properties"]
    child = p["tree"]["properties"]["children"]["items"]
    assert child["type"] == "object" and "Same shape as Node" in child["description"], child
    assert p["kind_only"] == {"type": "string", "enum": ["page", "close"]}
    assert p["remote"] == {}, "an unresolvable $ref renders as before (any)"
    assert p["legacy"] == {"type": "string", "enum": ["sev1", "sev2"], "description": "v1 style"}
    assert p["either"] == {"oneOf": [{"type": "string"}, {"type": "integer"}]}
    assert p["maybe"] == {"type": "object", "nullable": True, "properties": {"x": {"type": "string"}}}
    assert p["method"] == {"type": "string", "enum": ["pickup"]}
    assert p["version"] == {"description": "Must be 2."}
    assert p["mode"] == {"enum": ["a", "b"], "type": "string", "nullable": True}

    assert f(None) is None and f(True) is True
    deep = {"$defs": {"N": {"type": "object", "properties": {"n": {"$ref": "#/$defs/N"}}}}, "$ref": "#/$defs/N"}
    f(deep)  # terminates


def check_patch(source: str) -> None:
    with tempfile.TemporaryDirectory() as d:
        script = Path(d) / "patch_tool_schema.py"
        script.write_text(source)
        target = Path(d) / "harmony_utils.py"
        head, tail = "import x\n\n\n", "\n\ndef get_developer_message():\n    pass\n"

        def run() -> str:
            r = subprocess.run([sys.executable, str(script), str(target)], capture_output=True, text=True, check=True)
            return r.stdout

        target.write_text(head + V031_FUNCTION + tail)
        out = run()
        assert "WARN: nearai_tool_schema.py not mounted" in out, out
        assert target.read_text() == head + V031_FUNCTION + tail, "patched without the module"

        (Path(d) / "nearai_tool_schema.py").write_text("")
        assert "[tool-schema] applied" in run()
        patched = target.read_text()
        assert patched.startswith(head) and patched.endswith(tail)
        assert "parameters=schema_for_harmony(tool.function.parameters)," in patched
        assert "parameters=schema_for_harmony(tool.parameters)," in patched
        compile(patched, "harmony_utils.py", "exec")
        assert "[tool-schema] already applied" in run()
        assert target.read_text() == patched

        target.write_text(head + V031_FUNCTION.replace("tool.parameters", "tool.params") + tail)
        assert "WARN: anchor not found" in run()


def main() -> None:
    text = COMPOSE.read_text()
    module = load_module(config_content(text, "gptoss_tool_schema_module"))
    check_rewrite(module.schema_for_harmony)
    check_patch(config_content(text, "gptoss_tool_schema_patch"))
    engine = text[text.index("x-gptoss-engine:"):text.index("x-vllm-proxy-common:")]
    for needle in (
        "python3 /etc/vllm/patch_tool_schema.py; exec vllm serve",
        "source: gptoss_tool_schema_patch",
        "target: /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/parser/nearai_tool_schema.py",
    ):
        assert needle in engine, f"x-gptoss-engine is missing: {needle}"
    print("gpt-oss tool schema rewrite: ok")


if __name__ == "__main__":
    main()
