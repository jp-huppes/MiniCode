import importlib.util
import json
from pathlib import Path

from pydantic import ValidationError
import pytest

from src.tools.catalog import (
    ENVELOPE_ADAPTER,
    export_envelope_schema,
    export_normalized_envelope_schema,
    normalize_json_schema,
)

GOLDEN_FILE = Path(__file__).parent / "fixtures" / "envelope_schema.normalized.json"
HAS_JSONSCHEMA = importlib.util.find_spec("jsonschema") is not None


def _assert_no_forbidden_keywords(node: object) -> None:
    if isinstance(node, list):
        for item in node:
            _assert_no_forbidden_keywords(item)
        return

    if isinstance(node, dict):
        assert "$ref" not in node, "Schema normalizado não deve conter '$ref'"
        assert "$defs" not in node, "Schema normalizado não deve conter '$defs'"
        assert "discriminator" not in node, "Schema normalizado não deve conter 'discriminator'"
        assert "title" not in node, "Schema normalizado não deve conter 'title'"
        assert "default" not in node, "Schema normalizado não deve conter 'default'"
        for value in node.values():
            _assert_no_forbidden_keywords(value)


def _iter_variants(normalized_schema: dict) -> list[dict]:
    if "oneOf" in normalized_schema:
        return normalized_schema["oneOf"]
    if "anyOf" in normalized_schema:
        return normalized_schema["anyOf"]
    pytest.fail("Schema normalizado deve expor 'oneOf' para as variantes do envelope.")


def _validate_against_schema(payload: dict, normalized_schema: dict) -> None:
    if HAS_JSONSCHEMA:
        import jsonschema

        jsonschema.validate(instance=payload, schema=normalized_schema)
    else:
        ENVELOPE_ADAPTER.validate_python(payload)


def test_raw_and_normalized_schema_structure():
    raw_schema = export_envelope_schema()
    norm_schema = normalize_json_schema(raw_schema)

    _assert_no_forbidden_keywords(norm_schema)

    variants = _iter_variants(norm_schema)
    assert len(variants) >= 4

    tools_found = set()
    for variant in variants:
        assert variant.get("additionalProperties") is False, (
            f"Envelope deve ter additionalProperties=False: {variant}"
        )

        required_top = variant.get("required", [])
        assert "thought" not in required_top, (
            f"'thought' deve estar fora de 'required': {variant}"
        )

        props = variant.get("properties", {})
        tool_prop = props.get("tool", {})
        assert "const" in tool_prop, f"Variante deve conter 'const' em properties.tool: {variant}"
        tool_name = tool_prop["const"]
        tools_found.add(tool_name)

        args_schema = props.get("args", {})
        assert args_schema.get("additionalProperties") is False, (
            f"'args' de '{tool_name}' deve ter additionalProperties=False: {args_schema}"
        )

        if tool_name == "write_file":
            req_args = set(args_schema.get("required", []))
            assert {"path", "content"}.issubset(req_args), (
                f"'write_file.args.required' deve conter 'path' e 'content': {req_args}"
            )

    assert {"list_files", "read_file", "write_file", "done"}.issubset(tools_found)


def test_golden_file_matches_normalized_schema():
    assert GOLDEN_FILE.exists(), (
        f"Arquivo golden não encontrado em {GOLDEN_FILE}. "
        "Gere-o uma vez e versione no repositório."
    )
    expected = json.loads(GOLDEN_FILE.read_text(encoding="utf-8"))
    actual = export_normalized_envelope_schema()
    assert actual == expected, (
        "O schema normalizado divergiu do golden file. "
        "Se a mudança foi intencional, atualize tests/fixtures/envelope_schema.normalized.json."
    )


def test_valid_examples_for_all_tools():
    norm_schema = export_normalized_envelope_schema()
    valid_payloads = [
        {"thought": "listar arquivos", "tool": "list_files", "args": {"path": ".", "max_entries": 50}},
        {"thought": "ler arquivo", "tool": "read_file", "args": {"path": "README.md", "start_line": 1, "end_line": 20}},
        {"thought": "escrever arquivo", "tool": "write_file", "args": {"path": "src/app.py", "content": "print('ok')\n"}},
        {"thought": "acrescentar texto", "tool": "append_file", "args": {"path": "src/app.py", "content": "print('more')\n"}},
        {"thought": "substituir trecho", "tool": "replace_text", "args": {"path": "src/app.py", "old_str": "ok", "new_str": "done"}},
        {"thought": "rodar testes", "tool": "run_tests", "args": {"path": "tests", "args": ["-q"], "command": "pytest -q"}},
        {"thought": "concluir", "tool": "done", "args": {}},
    ]

    for payload in valid_payloads:
        _validate_against_schema(payload, norm_schema)


@pytest.mark.parametrize(
    "invalid_payload",
    [
        # file_path no lugar de path
        {"thought": "errado", "tool": "write_file", "args": {"file_path": "a.py", "content": "x"}},
        # campo extra em args
        {"thought": "extra", "tool": "read_file", "args": {"path": "a.py", "extra_field": 123}},
        # campo extra no envelope
        {"thought": "extra top", "tool": "done", "args": {}, "unexpected": True},
        # tipo errado (content numérico)
        {"thought": "tipo errado", "tool": "write_file", "args": {"path": "a.py", "content": 999}},
        # ferramenta inexistente
        {"thought": "inexistente", "tool": "delete_everything", "args": {}},
    ],
)
def test_invalid_payloads_are_rejected(invalid_payload):
    norm_schema = export_normalized_envelope_schema()

    if HAS_JSONSCHEMA:
        import jsonschema

        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=invalid_payload, schema=norm_schema)

    with pytest.raises(ValidationError):
        ENVELOPE_ADAPTER.validate_python(invalid_payload)