"""CPU-only tests for the tool-schema size guard (no GPU, no model).

Usage: python3 test_tool_schema_guard.py --module .../utils/tool_schema_guard.py \
           --serving-chat .../entrypoints/openai/serving_chat.py

The module is loaded by path. serving_chat.py is parsed, never imported: the real
OpenAIServingChat._validate_request is extracted by ast and executed against stubs, so the wiring
(guard before jsonschema's check_schema, HTTP-400 error string) is tested on the shipped bytes.
"""

import argparse
import ast
import copy
import importlib.util
import os
import sys
import time
import types

parser = argparse.ArgumentParser()
parser.add_argument("--module", required=True)
parser.add_argument("--serving-chat", required=True)
args = parser.parse_args()


def load(env):
    """Load a fresh copy of the guard module with the given environment."""
    for name in ("SGLANG_TOOL_SCHEMA_MAX_DEPTH", "SGLANG_TOOL_SCHEMA_MAX_NODES"):
        os.environ.pop(name, None)
    os.environ.update(env)
    spec = importlib.util.spec_from_file_location("tool_schema_guard", args.module)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def chain(kind, depth):
    s = {"type": "string"}
    for _ in range(depth):
        s = {"anyOf": [s, {"type": "integer"}]} if kind == "anyOf" else {"items": s}
    return s


def tree(depth):
    """Fan-out 2 anyOf tree: 2**depth leaves, the shape that makes check_schema slow."""
    s = {"type": "string"}
    for _ in range(depth):
        s = {"anyOf": [s, copy.deepcopy(s)]}
    return s


# 1. Off by default: nothing is walked, whatever the schema looks like.
g = load({})
assert g.MAX_DEPTH == 0 and g.MAX_NODES == 0
assert g.check_tool_schema_size(chain("anyOf", 5000)) is None
cyc = {}
cyc["self"] = cyc
assert g.check_tool_schema_size(cyc) is None, "off must not even look at the schema"

# 2. Depth limit: a flat schema is depth 1, every dict/list inside adds one; boundary is exact.
g = load({"SGLANG_TOOL_SCHEMA_MAX_DEPTH": "5"})
assert g.MAX_DEPTH == 5
assert g.check_tool_schema_size({"type": "object"}) is None
five = {"a": {"b": {"c": {"d": 1}}}}  # 4 containers deep
five_exact = {"a": {"b": {"c": {"d": {"e": 1}}}}}  # 5 containers deep: allowed
six = {"a": {"b": {"c": {"d": {"e": {"f": 1}}}}}}  # 6: rejected
assert g.check_tool_schema_size(five) is None
assert g.check_tool_schema_size(five_exact) is None
reason = g.check_tool_schema_size(six)
assert reason and "SGLANG_TOOL_SCHEMA_MAX_DEPTH" in reason, reason
assert g.check_tool_schema_size({"anyOf": [{"anyOf": [{"type": "string"}]}]}) is None  # 5 deep
assert g.check_tool_schema_size({"x": [[[[[1]]]]]}) is not None  # lists count too
assert g.check_tool_schema_size([[[[{}]]]]) is None and g.check_tool_schema_size([[[[[{}]]]]])
# A wide flat schema is not deep.
assert g.check_tool_schema_size({"enum": list(range(100000)), "type": "integer"}) is None

# 3. Iterative: a 200k-deep chain and a Python-level cycle neither recurse nor hang.
deep = {}
cur = deep
for _ in range(200_000):
    cur["x"] = {}
    cur = cur["x"]
t0 = time.monotonic()
assert g.check_tool_schema_size(deep) is not None
assert time.monotonic() - t0 < 1.0
assert g.check_tool_schema_size(deep, max_depth=300_000) is None, "walks 200k levels without recursing"
assert g.check_tool_schema_size(cyc) is not None, "a cyclic structure is rejected, not looped on"
assert time.monotonic() - t0 < 1.0

# 4. Node limit: counts every container and every value inside one (scalars included), stops at
# the first excess, and is a per-request budget through `used`.
g = load({"SGLANG_TOOL_SCHEMA_MAX_NODES": "27"})
assert g.MAX_NODES == 27 and g.MAX_DEPTH == 0
# root (1+1) + properties (1+8) + 8 * ({"type": "string"} = 1+1) = 27
props = {"properties": {f"p{i}": {"type": "string"} for i in range(8)}}
assert g.check_tool_schema_size(props) is None
props["properties"]["p8"] = {"type": "string"}  # 27 + 1 + 2 = 30
reason = g.check_tool_schema_size(props)
assert reason and "SGLANG_TOOL_SCHEMA_MAX_NODES" in reason, reason
assert g.check_tool_schema_size(chain("anyOf", 100)) is not None
# Scalar children are the bypass the first version had: properties mapped to `true` and a long
# `type` list cost check_schema about as much per entry as a nested object.
g = load({"SGLANG_TOOL_SCHEMA_MAX_NODES": "1000"})
assert g.check_tool_schema_size({"properties": {f"k{i}": True for i in range(100_000)}}) is not None
assert g.check_tool_schema_size({"type": ["string"] * 200_000}) is not None
assert g.check_tool_schema_size({"enum": list(range(1_000_000))}) is not None
assert g.check_tool_schema_size({"properties": {f"k{i}": True for i in range(900)}}) is None
# Per-request budget: tools that are each under the limit still add up.
g = load({"SGLANG_TOOL_SCHEMA_MAX_NODES": "100"})
small = {"properties": {f"k{i}": True for i in range(30)}}  # 2 + ... = 33 nodes
used = [0]
assert g.check_tool_schema_size(small, used=used) is None and used[0] == 33, used
assert g.check_tool_schema_size(small, used=used) is None and used[0] == 66
assert g.check_tool_schema_size(small, used=used) is None and used[0] == 99
reason = g.check_tool_schema_size(small, used=used)
assert reason and "per request" in reason, reason
# Both limits together; explicit arguments override the env-derived ones.
g = load({"SGLANG_TOOL_SCHEMA_MAX_DEPTH": "50", "SGLANG_TOOL_SCHEMA_MAX_NODES": "5000"})
assert g.check_tool_schema_size(chain("anyOf", 10)) is None
assert "DEPTH" in g.check_tool_schema_size(chain("items", 60))
assert "NODES" in g.check_tool_schema_size(tree(11))
assert g.check_tool_schema_size(tree(11), max_depth=0, max_nodes=0) is None

# 5. Bad values leave the limit off (and never raise at import time); negatives are off too.
for bad in ("abc", "", "-3", "1.5", " "):
    g = load({"SGLANG_TOOL_SCHEMA_MAX_DEPTH": bad, "SGLANG_TOOL_SCHEMA_MAX_NODES": bad})
    assert g.MAX_DEPTH == 0 and g.MAX_NODES == 0, (bad, g.MAX_DEPTH, g.MAX_NODES)

# 6. The shipped _validate_request: extract it from serving_chat.py by ast and run it on stubs.
tree_ast = ast.parse(open(args.serving_chat).read(), filename=args.serving_chat)
cls = next(n for n in tree_ast.body if isinstance(n, ast.ClassDef) and n.name == "OpenAIServingChat")
fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_validate_request")
fn.decorator_list = []
src = ast.get_source_segment(open(args.serving_chat).read(), fn)
assert src.index("check_tool_schema_size(") < src.index("Draft202012Validator.check_schema("), (
    "the size guard must run before jsonschema's check_schema"
)
assert "check_tool_schema_size" in open(args.serving_chat).read().split("class OpenAIServingChat")[0], (
    "serving_chat.py must import the guard"
)

try:
    from jsonschema import Draft202012Validator, SchemaError
except ImportError:  # the CPU-only image has jsonschema; a bare interpreter may not
    print("jsonschema missing: skipping the _validate_request functional part")
    print("TOOL SCHEMA GUARD TESTS PASSED (guard unit tests only)")
    sys.exit(0)

check_schema_calls = []


class CountingValidator:
    @staticmethod
    def check_schema(schema):
        check_schema_calls.append(schema)
        Draft202012Validator.check_schema(schema)


class GenericParam:  # stands in for ChatCompletionMessageGenericParam
    pass


def make_validate(guard):
    ns = {
        "Optional": __import__("typing").Optional,
        "ChatCompletionRequest": object,
        "ChatCompletionMessageGenericParam": GenericParam,
        "Draft202012Validator": CountingValidator,
        "SchemaError": SchemaError,
        "normalize_json_schema_types": lambda schema: None,
        "check_tool_schema_size": guard.check_tool_schema_size,
        "logger": None,
    }
    code = compile(ast.Module(body=[fn], type_ignores=[]), args.serving_chat, "exec")
    exec(code, ns)
    return ns["_validate_request"]


def request_with(schemas):
    tools = [
        types.SimpleNamespace(function=types.SimpleNamespace(name=f"t{i}", parameters=p))
        for i, p in enumerate(schemas)
    ]
    return types.SimpleNamespace(
        messages=[types.SimpleNamespace(role="user", content="hi", tools=None)],
        return_sampling_mask=False,
        return_meta_info=False,
        tool_choice=None,
        tools=tools,
        response_format=None,
        max_completion_tokens=None,
        max_tokens=None,
    )


def self_stub():
    server_args = types.SimpleNamespace(context_length=None, allow_auto_truncate=False)
    return types.SimpleNamespace(
        _validate_media_content=lambda request: None,
        _effective_tools=lambda request: request.tools,
        tokenizer_manager=types.SimpleNamespace(server_args=server_args),
    )


ok_schema = {"type": "object", "properties": {"city": {"type": "string"}}}
bomb = tree(12)

# Off: identical to the base behaviour, check_schema sees every tool.
g = load({})
validate = make_validate(g)
assert validate(self_stub(), request_with([ok_schema, ok_schema])) is None
assert len(check_schema_calls) == 2
check_schema_calls.clear()

# On: the oversized second tool is rejected by name/index, before check_schema ever sees it.
g = load({"SGLANG_TOOL_SCHEMA_MAX_DEPTH": "16", "SGLANG_TOOL_SCHEMA_MAX_NODES": "500"})
# the 3-node ok_schema is far below the budget; the fan-out tree is far above it
validate = make_validate(g)
t0 = time.monotonic()
error = validate(self_stub(), request_with([ok_schema, bomb]))
elapsed = time.monotonic() - t0
assert error and error.startswith("Tool 1 function 'parameters' schema is too large"), error
assert "SGLANG_TOOL_SCHEMA_MAX_NODES" in error or "SGLANG_TOOL_SCHEMA_MAX_DEPTH" in error, error
assert check_schema_calls == [ok_schema], "check_schema must not run on the rejected tool"
assert elapsed < 0.5, elapsed
check_schema_calls.clear()
assert validate(self_stub(), request_with([ok_schema])) is None
assert check_schema_calls == [ok_schema]
# A schema that is invalid but small still gets the original jsonschema error.
error = validate(self_stub(), request_with([{"type": "nonsense"}]))
assert error and error.startswith("Tool 0 function has invalid 'parameters' schema"), error

print("TOOL SCHEMA GUARD TESTS PASSED")
