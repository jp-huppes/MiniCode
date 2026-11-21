import json
import pytest

from src.agent.orchestrator import AgentOrchestrator
from src.agent.state import State
from src.llm.action_contract import (
    ActionErrorCategory,
    interpretar_acao,
)
from src.llm.client import interpretar_com_retry
from src.tools.catalog import CATALOG_BY_NAME
from src.tools.registry import ToolRegistry


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


def _exemplo_valido_para_ferramenta(tool_name: str, spec) -> dict:
    """Gera argumentos válidos mínimos com base no schema Pydantic da ferramenta."""
    schema = spec.args_model.model_json_schema()
    props = schema.get("properties", {})
    required = set(schema.get("required", []))
    args: dict = {}

    for field_name, prop in props.items():
        if field_name not in required and tool_name != "list_files":
            continue
        p_type = prop.get("type")
        if p_type == "string":
            if field_name == "path":
                if tool_name in {"read_file", "append_file", "replace_text"}:
                    args[field_name] = "sample.txt"
                elif tool_name == "write_file":
                    args[field_name] = "written_sample.txt"
                else:
                    args[field_name] = "."
            elif field_name == "content":
                args[field_name] = "hello\n"
            elif field_name == "old_str":
                args[field_name] = "hello"
            elif field_name == "new_str":
                args[field_name] = "world"
            else:
                args[field_name] = "valor"
        elif p_type == "integer":
            args[field_name] = prop.get("minimum", 1)
        elif p_type == "array":
            args[field_name] = []
        elif p_type == "object":
            args[field_name] = {}
        elif p_type == "boolean":
            args[field_name] = False

    return args

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


def test_invalid_corpus_never_reaches_dispatch_with_file_path_or_leaks_exception(monkeypatch, tmp_path):
    dispatched_calls: list[tuple[str, dict]] = []
    original_dispatch = ToolRegistry.dispatch

    def spy_dispatch(self, name, args):
        dispatched_calls.append((name, dict(args) if isinstance(args, dict) else {}))
        return original_dispatch(self, name, args)

    monkeypatch.setattr(ToolRegistry, "dispatch", spy_dispatch)

    invalid_payloads = [
        "texto sem json",
        '{"thought": "corte", "tool": "write_file", "args": {"file_path": "x.py", "content": "1"',
        '{"thought": "alias", "tool": "write_file", "args": {"file_path": "x.py", "content": "1"}}',
        '{"thought": "tipo", "tool": "write_file", "args": {"path": "x.py", "content": 999}}',
        '{"thought": "tool", "tool": "unknown_tool", "args": {}}',
    ]

    llm = SequenceFakeClient([FakeResponse(p) for p in invalid_payloads])
    orchestrator = AgentOrchestrator(tmp_path, tmp_path / ".minicode", llm)
    final_state = orchestrator.run("Tarefa teste F9", "Plano teste F9")

    assert final_state == State.FAILED
    for _, call_args in dispatched_calls:
        assert "file_path" not in call_args


@pytest.mark.parametrize("tool_name", sorted(CATALOG_BY_NAME.keys()))
def test_roundtrip_valid_tool_from_schema_reaches_registry_dispatch(tmp_path, tool_name):
    spec = CATALOG_BY_NAME[tool_name]
    if tool_name == "done":
        envelope = json.dumps({"thought": "finalizar", "tool": "done", "args": {}})
        parsed = interpretar_acao(envelope)
        assert parsed.ok
        assert parsed.action is not None
        assert parsed.action.tool == "done"
        return

    args = _exemplo_valido_para_ferramenta(tool_name, spec)
    target_file = tmp_path / "sample.txt"
    if tool_name in {"read_file", "append_file", "replace_text"}:
        target_file.write_text("hello\n", encoding="utf-8")

    envelope = json.dumps({"thought": f"executar {tool_name}", "tool": tool_name, "args": args})
    parsed = interpretar_acao(envelope)
    assert parsed.ok, f"Falhou na validação para {tool_name}: {parsed.diagnostic}"
    assert parsed.action is not None

    registry = ToolRegistry(workspace=tmp_path)
    if tool_name in registry.tools:
        result = registry.dispatch(parsed.action.tool, parsed.action.args)
        assert result.ok is True, f"Dispatch falhou para {tool_name}: {result.error}"


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


def test_k_consecutive_invalid_outputs_lead_to_failed_with_parse_failure(tmp_path):
    llm = SequenceFakeClient([
        FakeResponse("não é json válido #1"),
        FakeResponse("não é json válido #2"),
        FakeResponse("não é json válido #3"),
        FakeResponse("não é json válido #4"),
        FakeResponse("não é json válido #5"),
    ])
    state_dir = tmp_path / ".minicode"
    orchestrator = AgentOrchestrator(tmp_path, state_dir, llm)
    final_state = orchestrator.run("Tarefa com falhas consecutivas", "Plano")

    assert final_state == State.FAILED
    assert orchestrator.state_data.state == State.FAILED.value
    assert orchestrator.state_data.failure_reason == "parse_failure"

    state_file = state_dir / "state.json"
    assert state_file.is_file()
    persisted = json.loads(state_file.read_text(encoding="utf-8"))
    assert persisted.get("state") == "FAILED"
    assert persisted.get("failure_reason") == "parse_failure"