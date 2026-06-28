"""Tests for the Ollama tool-schema sanitizer.

Ollama (llama.cpp's json_schema_to_grammar) rejects several standard
JSON-Schema constructs with a generic 400 ("Value looks like object, but
can't find closing '}' symbol"). With a 500+ tool manifest one offender
poisons the whole request, so every schema is normalized before /api/chat.
"""

from langfuse_openai_proxy.domain.services import (
    _build_ollama_native_body,
    _sanitize_json_schema,
    _sanitize_tool_for_ollama,
)


def test_object_valued_additional_properties_is_coerced_to_bool():
    # The #1 trigger for Ollama's "can't find closing '}' symbol" error.
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "additionalProperties": {"type": "string"},
    }
    out = _sanitize_json_schema(schema)
    assert out["additionalProperties"] is False


def test_bool_additional_properties_is_preserved():
    schema = {"type": "object", "properties": {}, "additionalProperties": True}
    assert _sanitize_json_schema(schema)["additionalProperties"] is True


def test_anyof_collapses_to_first_branch():
    schema = {"anyOf": [{"type": "string"}, {"type": "null"}]}
    out = _sanitize_json_schema(schema)
    assert out.get("type") == "string"
    assert "anyOf" not in out


def test_allof_merges_branches():
    schema = {"allOf": [{"type": "object", "properties": {"x": {"type": "string"}}}]}
    out = _sanitize_json_schema(schema)
    assert out["type"] == "object"
    assert "x" in out["properties"]


def test_type_array_keeps_first_option():
    schema = {"type": ["string", "null"]}
    assert _sanitize_json_schema(schema)["type"] == "string"


def test_ref_defs_and_pattern_properties_are_dropped():
    schema = {
        "$ref": "#/$defs/Foo",
        "$defs": {"Foo": {"type": "object"}},
        "patternProperties": {"^x": {"type": "string"}},
        "type": "object",
    }
    out = _sanitize_json_schema(schema)
    assert "$ref" not in out
    assert "$defs" not in out
    assert "patternProperties" not in out


def test_nested_properties_recurse():
    schema = {
        "type": "object",
        "properties": {
            "outer": {
                "type": "object",
                "properties": {"inner": {"type": ["number", "null"]}},
                "additionalProperties": {"type": "string"},
            }
        },
    }
    out = _sanitize_json_schema(schema)
    outer = out["properties"]["outer"]
    assert outer["additionalProperties"] is False
    assert outer["properties"]["inner"]["type"] == "number"


def test_description_is_kept():
    # Semantic hints are harmless to the grammar compiler and help the model.
    schema = {"type": "string", "description": "a value"}
    assert _sanitize_json_schema(schema)["description"] == "a value"


def test_sanitize_tool_normalizes_parameters():
    tool = {
        "type": "function",
        "function": {
            "name": "do_thing",
            "description": "does a thing",
            "parameters": {
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "additionalProperties": {"type": "array"},
            },
        },
    }
    out = _sanitize_tool_for_ollama(tool)
    assert out["function"]["name"] == "do_thing"
    assert out["function"]["parameters"]["additionalProperties"] is False


def test_build_ollama_native_body_sanitizes_tools():
    body = _build_ollama_native_body(
        "qwen-haiku:4b",
        [{"role": "user", "content": "hi"}],
        {
            "think": False,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "t",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "additionalProperties": {"type": "string"},
                        },
                    },
                }
            ],
        },
    )
    assert body["tools"][0]["function"]["parameters"]["additionalProperties"] is False
    assert body["think"] is False


def _make_tool(name: str, *, complex_schema: bool = False) -> dict:
    params = (
        {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "additionalProperties": {"type": "string"},
            "$defs": {"Foo": {"type": "object"}},
        }
        if complex_schema
        else {"type": "object", "properties": {"x": {"type": "string"}}}
    )
    return {
        "type": "function",
        "function": {"name": name, "description": f"tool {name}", "parameters": params},
    }


def test_build_ollama_native_body_sanitizes_large_manifest():
    """Large manifests get full sanitized schemas — no stubbing.

    Regression: a prior count-based path blanked every tool's parameters to
    {"type":"object"} above 32 tools, stripping all argument guidance. It also
    did not fix the grammar-400 (the real cause is tool_calls.arguments shape,
    not schemas). That path is gone; all manifests now go through the sanitizer,
    which neutralizes offenders (additionalProperties objects -> bool, $defs
    dropped) while keeping the real schema so the model can generate arguments.
    """
    tools = [_make_tool(f"t{i}", complex_schema=True) for i in range(64)]
    body = _build_ollama_native_body(
        "gemma", [{"role": "user", "content": "hi"}], {"think": False, "tools": tools}
    )
    for t in body["tools"]:
        params = t["function"]["parameters"]
        # Sanitized, NOT stubbed to a bare object.
        assert params != {"type": "object"}
        # Offending constructs neutralized.
        assert params["additionalProperties"] is False
        assert "$defs" not in params
        # Real schema preserved so the model can generate correct arguments.
        assert params["properties"]["x"] == {"type": "string"}


# --- tool_calls arguments: string -> object for Ollama native /api/chat ---
# Ollama's /api/chat rejects prior assistant tool_calls whose
# function.arguments is a JSON string (OpenAI shape) with the generic
# "Value looks like object, but can't find closing '}' symbol" 400 — on every
# multi-turn agentic conversation. The body builder parses them to objects.
# This is the real root cause of the failing claude-local calls, not schemas.


def test_ollama_native_messages_parses_tool_call_args_string_to_object():
    from langfuse_openai_proxy.domain.services import _ollama_native_messages

    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "Bash", "arguments": '{"command": "ls"}'},
                }
            ],
        }
    ]
    out = _ollama_native_messages(messages)
    args = out[0]["tool_calls"][0]["function"]["arguments"]
    assert args == {"command": "ls"}
    assert isinstance(args, dict)


def test_ollama_native_messages_handles_empty_and_malformed_args():
    from langfuse_openai_proxy.domain.services import _ollama_native_messages

    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "a", "type": "function", "function": {"name": "X", "arguments": "{}"}},
                {"id": "b", "type": "function", "function": {"name": "Y", "arguments": ""}},
                {"id": "c", "type": "function", "function": {"name": "Z", "arguments": "bad{"}},
            ],
        }
    ]
    out = _ollama_native_messages(messages)
    assert out[0]["tool_calls"][0]["function"]["arguments"] == {}
    assert out[0]["tool_calls"][1]["function"]["arguments"] == {}
    assert out[0]["tool_calls"][2]["function"]["arguments"] == {}  # malformed -> {}


def test_ollama_native_messages_idempotent_on_objects_and_passes_other_msgs():
    from langfuse_openai_proxy.domain.services import _ollama_native_messages

    messages = [
        {"role": "user", "content": "hi"},  # no tool_calls -> untouched
        {"role": "tool", "tool_call_id": "x", "content": "result"},  # untouched
        {
            "role": "assistant",
            "content": "ok",
            "tool_calls": [
                {
                    "id": "a",
                    "type": "function",
                    "function": {"name": "X", "arguments": {"already": "object"}},
                }
            ],
        },
    ]
    out = _ollama_native_messages(messages)
    assert out[0] == messages[0]
    assert out[1] == messages[1]
    # Object arguments pass through unchanged.
    assert out[2]["tool_calls"][0]["function"]["arguments"] == {"already": "object"}


def test_ollama_native_messages_does_not_mutate_input():
    from langfuse_openai_proxy.domain.services import _ollama_native_messages

    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "a", "type": "function", "function": {"name": "X", "arguments": '{"k": 1}'}}
            ],
        }
    ]
    _ollama_native_messages(messages)
    # Original is still a string.
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == '{"k": 1}'


def test_build_ollama_native_body_parses_request_tool_call_args():
    """End-to-end: the body builder converts prior assistant tool_calls args to
    objects so Ollama /api/chat accepts multi-turn agentic conversations."""
    body = _build_ollama_native_body(
        "qwen-haiku:4b",
        [
            {"role": "user", "content": "run it"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "Bash", "arguments": '{"command":"echo hi"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "hi"},
            {"role": "user", "content": "again"},
        ],
        {"think": False},
    )
    args = body["messages"][1]["tool_calls"][0]["function"]["arguments"]
    assert args == {"command": "echo hi"}
    assert isinstance(args, dict)


def test_tuple_items_collapsed_to_first():
    # JSON-Schema tuple validation (items as a list) isn't supported by llama.cpp.
    schema = {"type": "array", "items": [{"type": "string"}, {"type": "number"}]}
    out = _sanitize_json_schema(schema)
    assert out["items"] == {"type": "string"}


def test_conditional_and_prefix_keywords_dropped():
    schema = {
        "type": "object",
        "prefixItems": [{"type": "string"}],
        "propertyNames": {"pattern": "^[a-z]+$"},
        "if": {"type": "object"},
        "then": {"type": "object"},
        "else": {"type": "string"},
        "not": {"type": "null"},
    }
    out = _sanitize_json_schema(schema)
    for dropped in ("prefixItems", "propertyNames", "if", "then", "else", "not"):
        assert dropped not in out


# --- const / exclusiveMinimum / content* hardening ---
# These keywords are documented llama.cpp grammar-compile offenders that
# produce the generic "can't find closing '}'" 400. Each test asserts the
# construct is neutralized to a grammar-safe equivalent.


def test_string_const_rewritten_to_enum():
    # const is the textbook "can't find closing '}'" trigger; enum:[X] is the
    # semantically-identical, grammar-safe form.
    schema = {"type": "object", "properties": {"kind": {"const": "cloud"}}}
    out = _sanitize_json_schema(schema)
    prop = out["properties"]["kind"]
    assert "const" not in prop
    assert prop["enum"] == ["cloud"]


def test_number_const_rewritten_to_enum():
    schema = {"type": "object", "properties": {"n": {"const": 42}}}
    out = _sanitize_json_schema(schema)
    assert out["properties"]["n"]["enum"] == [42]


def test_boolean_const_rewritten_to_enum():
    schema = {"type": "object", "properties": {"flag": {"const": True}}}
    out = _sanitize_json_schema(schema)
    assert out["properties"]["flag"]["enum"] == [True]


def test_null_const_rewritten_to_enum():
    schema = {"type": "object", "properties": {"x": {"const": None}}}
    out = _sanitize_json_schema(schema)
    assert out["properties"]["x"]["enum"] == [None]


def test_object_const_rewritten_to_enum():
    # Object-valued const is the worst-case llama.cpp trigger; the value is an
    # arbitrary JSON instance (not a schema), so it passes through verbatim as
    # the single enum element. The const keyword is removed.
    schema = {
        "type": "object",
        "properties": {"cfg": {"const": {"a": 1, "b": [1, 2, 3]}}},
    }
    out = _sanitize_json_schema(schema)
    assert out["properties"]["cfg"]["enum"] == [{"a": 1, "b": [1, 2, 3]}]
    assert "const" not in out["properties"]["cfg"]


def test_const_collision_with_existing_enum_prefers_const_pin():
    # If a schema (unusually) declares both const and enum, const is stricter;
    # keep the const-pin rather than emitting an invalid two-keyword node.
    schema = {"enum": ["a", "b"], "const": "b"}
    out = _sanitize_json_schema(schema)
    assert out["enum"] == ["b"]
    assert "const" not in out


def test_exclusive_minimum_number_form_converted_to_inclusive():
    # Draft-07: exclusiveMinimum: 0 → minimum: 0 (grammar-safe, slightly looser).
    schema = {"type": "integer", "exclusiveMinimum": 0}
    out = _sanitize_json_schema(schema)
    assert "exclusiveMinimum" not in out
    assert out["minimum"] == 0


def test_exclusive_maximum_number_form_converted_to_inclusive():
    schema = {"type": "integer", "exclusiveMaximum": 100}
    out = _sanitize_json_schema(schema)
    assert "exclusiveMaximum" not in out
    assert out["maximum"] == 100


def test_exclusive_minimum_object_form_extracted():
    # Draft-2020-12: exclusiveMinimum: {value: 0} → minimum: 0.
    schema = {"type": "integer", "exclusiveMinimum": {"value": 5}}
    out = _sanitize_json_schema(schema)
    assert "exclusiveMinimum" not in out
    assert out["minimum"] == 5


def test_exclusive_minimum_does_not_clobber_explicit_inclusive_bound():
    # If the author already set an explicit minimum, don't overwrite it.
    schema = {"type": "integer", "minimum": 10, "exclusiveMinimum": 0}
    out = _sanitize_json_schema(schema)
    assert out["minimum"] == 10
    assert "exclusiveMinimum" not in out


def test_content_keywords_dropped():
    # contentEncoding/contentMediaType/contentSchema are validation-only hints
    # the grammar compiler ignores at best and chokes on at worst
    # (contentSchema with an object value is a known offender).
    schema = {
        "type": "string",
        "contentEncoding": "base64",
        "contentMediaType": "image/png",
        "contentSchema": {"type": "object", "properties": {"x": {"type": "string"}}},
    }
    out = _sanitize_json_schema(schema)
    for dropped in ("contentEncoding", "contentMediaType", "contentSchema"):
        assert dropped not in out
    assert out["type"] == "string"


def test_nested_const_in_items_rewritten():
    schema = {"type": "array", "items": {"const": "fixed"}}
    out = _sanitize_json_schema(schema)
    assert out["items"]["enum"] == ["fixed"]
    assert "const" not in out["items"]


def test_full_registry_tool_sanitizes_cleanly():
    # Regression: the real Dokploy registry-create tool pins registryType via
    # const:"cloud". After sanitization it must reach Ollama as enum:["cloud"].
    tool = {
        "type": "function",
        "function": {
            "name": "registry-create",
            "parameters": {
                "type": "object",
                "properties": {
                    "registryType": {"const": "cloud"},
                    "imagePrefix": {"type": "string"},
                },
            },
        },
    }
    out = _sanitize_tool_for_ollama(tool)
    params = out["function"]["parameters"]
    assert params["properties"]["registryType"]["enum"] == ["cloud"]
    assert "const" not in params["properties"]["registryType"]
