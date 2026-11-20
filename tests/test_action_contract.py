import pytest

from src.llm.action_contract import (
    ActionErrorCategory,
    interpretar_acao,
)
from src.llm.client import interpretar_com_retry


class FakeResponse:
    def __init__(self, content: str, finish_reason: str | None = "stop"):
        self.content = content
        self.finish_reason = finish_reason


class SequenceFakeClient:
    def __init__(self, responses: list[FakeResponse]):
        self.responses = list(responses)
        self.calls: list[list[dict]] = []

    def chat(self, messages, **kwargs):
        self.calls.append([dict(m) for m in messages])
        idx = min(len(self.calls) - 1, len(self.responses) - 1)
        return self.responses[idx]


def test_valid_action_pure_and_fenced():
    r1 = interpretar_acao('{"thought": "listar", "tool": "list_files", "args": {"path": "."}}')
    assert r1.ok
    assert r1.action is not None
    assert r1.action.tool == "list_files"

    r2 = interpretar_acao(
        'Explicação breve\n```json\n{"thought": "escrever", "tool": "write_file", "args": {"path": "a.py", "content": "x = 1"}}\n```'
    )
    assert r2.ok
    assert r2.action is not None
    assert r2.action.tool == "write_file"


def test_tolerant_json_prefix_without_fence():
    raw = 'json\n{"thought": "ler", "tool": "read_file", "args": {"path": "README.md"}}'
    res = interpretar_acao(raw)
    assert res.ok
    assert res.action is not None
    assert res.action.tool == "read_file"


@pytest.mark.parametrize(
    ("raw_text", "finish_reason", "expected_category", "expected_field_snippet"),
    [
        # nao_json
        ("Apenas texto solto sem JSON", None, ActionErrorCategory.NAO_JSON, "root"),
        # json_truncado (chave sem fechar ou finish_reason == "length")
        ('{"thought": "cortado", "tool": "write_file", "args": {"path": "a.py"', None, ActionErrorCategory.JSON_TRUNCADO, "root"),
        ('{"thought": "cortado", "tool": "write_file"', "length", ActionErrorCategory.JSON_TRUNCADO, "root"),
        # tool_invalida
        ('{"thought": "x", "tool": "shell_exec", "args": {}}', None, ActionErrorCategory.TOOL_INVALIDA, "tool"),
        # campo_extra (file_path em vez de path)
        ('{"thought": "x", "tool": "write_file", "args": {"file_path": "a.py", "content": "1"}}', None, ActionErrorCategory.CAMPO_EXTRA, "args.file_path"),
        # campo_ausente (falta path em write_file)
        ('{"thought": "x", "tool": "write_file", "args": {"content": "1"}}', None, ActionErrorCategory.CAMPO_AUSENTE, "args.path"),
        # tipo_errado (content numérico)
        ('{"thought": "x", "tool": "write_file", "args": {"path": "a.py", "content": 123}}', None, ActionErrorCategory.TIPO_ERRADO, "args.content"),
        # args_de_outra_tool (old_str/new_str em read_file)
        ('{"thought": "x", "tool": "read_file", "args": {"path": "a.py", "old_str": "a", "new_str": "b"}}', None, ActionErrorCategory.ARGS_DE_OUTRA_TOOL, "args"),
    ],
)
def test_taxonomy_and_directed_retry_message(raw_text, finish_reason, expected_category, expected_field_snippet):
    res = interpretar_acao(raw_text, finish_reason=finish_reason)
    assert not res.ok
    assert res.diagnostic is not None
    assert res.diagnostic.category == expected_category

    msg = res.diagnostic.format_retry_message()
    assert expected_field_snippet in msg
    # Garante que não reenvia o schema completo gigante
    assert "$defs" not in msg
    assert "oneOf" not in msg


def test_retry_recovers_within_max_reformulacoes():
    client = SequenceFakeClient([
        FakeResponse('{"thought": "errado", "tool": "write_file", "args": {"file_path": "x.py", "content": "1"}}'),
        FakeResponse('{"thought": "corrigido", "tool": "write_file", "args": {"path": "x.py", "content": "1"}}'),
    ])
    res = interpretar_com_retry(
        client,
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "faça algo"}],
        max_reformulacoes=2,
    )
    assert res.ok
    assert res.action is not None
    assert res.action.args["path"] == "x.py"
    assert len(client.calls) == 2
    assert "args.file_path" in client.calls[1][-1]["content"]


def test_retry_exhaustion_returns_diagnostic():
    client = SequenceFakeClient([
        FakeResponse("sem json"),
        FakeResponse("ainda sem json"),
        FakeResponse("terceira falha"),
    ])
    res = interpretar_com_retry(
        client,
        [{"role": "user", "content": "faça algo"}],
        max_reformulacoes=2,
    )
    assert not res.ok
    assert res.diagnostic is not None
    assert res.diagnostic.category == ActionErrorCategory.NAO_JSON
    assert len(client.calls) == 3