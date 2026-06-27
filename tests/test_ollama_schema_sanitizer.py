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
