import json
from pathlib import Path
from types import SimpleNamespace

from src.agent.orchestrator import AgentOrchestrator
from src.agent.state import State
from src.llm.routing import (
    AgentRoutingConfig,
    ROLE_REPAIR,
    ROLE_TOOL_SELECTION,
    attach_routing_to_client,
    calcular_rho,
)


class _Resp:
    def __init__(self, content: str) -> None:
        self.content = content
        self.tool_calls = None
        self.finish_reason = "stop"
        self.prompt_tokens = 10
        self.completion_tokens = 5
        self.latency_seconds = 0.01


class _ScriptedClient:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []
        self.tool_mode = "none"
        self.constrained_mode = "none"

    def chat(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": dict(kwargs)})
        if not self._responses:
            return _Resp('```json\n{"thought": "fim", "tool": "done", "args": {}}\n```')
        return _Resp(self._responses.pop(0))


def _preparar_ws_com_teste_falhando(ws: Path) -> None:
    (ws / "src").mkdir(parents=True, exist_ok=True)
    (ws / "tests").mkdir(parents=True, exist_ok=True)
    (ws / "src" / "__init__.py").write_text("", encoding="utf-8")
    (ws / "tests" / "test_calc.py").write_text(
        "from src.calc import soma\n\ndef test_soma():\n    assert soma(2, 3) == 5\n",
        encoding="utf-8",
    )


def test_repair_disabled_fails_on_first_verifying_failure(tmp_path: Path) -> None:
    """4.a: com repair_enabled=False, o primeiro VERIFYING que falhar vai para FAILED com repair_disabled."""
    ws = tmp_path / "ws"
    mc = tmp_path / ".minicode"
    _preparar_ws_com_teste_falhando(ws)

    client = _ScriptedClient(
        [
            '```json\n{"thought": "escrever bug", "tool": "write_file", "args": {"path": "src/calc.py", "content": "def soma(a, b): return a - b\\n"}}\n```',
            '```json\n{"thought": "verificar", "tool": "done", "args": {}}\n```',
            # Se entrasse em REPAIRING, esta resposta corrigiria o codigo — mas nao deve ser chamada!
            '```json\n{"thought": "reparar", "tool": "write_file", "args": {"path": "src/calc.py", "content": "def soma(a, b): return a + b\\n"}}\n```',
        ]
    )

    orch = AgentOrchestrator(
        workspace_dir=ws,
        minicode_dir=mc,
        llm_client=client,
        max_repair_attempts=3,
        repair_enabled=False,
        temperature=0.0,
    )
    final_state = orch.run("Implemente soma", "1. Escrever src/calc.py")

    assert final_state == State.FAILED
    assert orch.state_data.state == State.FAILED.value
    assert orch.state_data.failure_reason == "repair_disabled"
    assert orch.state_machine.repair_attempts == 0

    persisted = json.loads((mc / "state.json").read_text(encoding="utf-8"))
    assert persisted["state"] == "FAILED"
    assert persisted["failure_reason"] == "repair_disabled"

    # Nenhuma chamada ROLE_REPAIR ocorreu; apenas as 2 chamadas de IMPLEMENTING
    assert orch.get_role_call_counts() == {
        ROLE_TOOL_SELECTION: 2,
        ROLE_REPAIR: 0,
    }
    assert orch.compute_rho() == 1.0
    assert all(c["kwargs"].get("role") == ROLE_TOOL_SELECTION for c in client.calls)
    assert all(c["kwargs"].get("temperature") == 0.0 for c in client.calls)


def test_max_repair_attempts_ge_semantics_preserved(tmp_path: Path) -> None:
    """4.a: garante que a semantica atual do >= em max_repair_attempts nao foi alterada."""
    ws = tmp_path / "ws"
    mc = tmp_path / ".minicode"
    _preparar_ws_com_teste_falhando(ws)

    client = _ScriptedClient(
        [
            '```json\n{"thought": "v1", "tool": "write_file", "args": {"path": "src/calc.py", "content": "def soma(a, b): return 0\\n"}}\n```',
            '```json\n{"thought": "done1", "tool": "done", "args": {}}\n```',
            '```json\n{"thought": "v2", "tool": "write_file", "args": {"path": "src/calc.py", "content": "def soma(a, b): return 1\\n"}}\n```',
            '```json\n{"thought": "done2", "tool": "done", "args": {}}\n```',
        ]
    )

    orch = AgentOrchestrator(
        workspace_dir=ws,
        minicode_dir=mc,
        llm_client=client,
        max_repair_attempts=2,
        repair_enabled=True,
    )
    final_state = orch.run("Implemente soma", "1. Escrever src/calc.py")

    assert final_state == State.FAILED
    assert orch.state_data.failure_reason == "max_repair_attempts"
    # 1a falha -> tentativa=1 < 2 -> REPAIRING (repair_attempts=1)
    # 2a falha -> tentativa=2 >= 2 -> FAILED (max_repair_attempts)
    assert orch.state_machine.repair_attempts == 1


def test_role_call_counter_and_rho_calculation(tmp_path: Path) -> None:
    """4.b: contador de chamadas por papel (tool_selection e repair) e calculo de rho."""
    ws = tmp_path / "ws"
    mc = tmp_path / ".minicode"
    _preparar_ws_com_teste_falhando(ws)

    client = _ScriptedClient(
        [
            # IMPLEMENTING (2 chamadas -> tool_selection)
            '```json\n{"thought": "bug", "tool": "write_file", "args": {"path": "src/calc.py", "content": "def soma(a, b): return a - b\\n"}}\n```',
            '```json\n{"thought": "testar", "tool": "done", "args": {}}\n```',
            # REPAIRING (2 chamadas -> repair)
            '```json\n{"thought": "fix", "tool": "write_file", "args": {"path": "src/calc.py", "content": "def soma(a, b): return a + b\\n"}}\n```',
            '```json\n{"thought": "retestar", "tool": "done", "args": {}}\n```',
        ]
    )

    orch = AgentOrchestrator(
        workspace_dir=ws,
        minicode_dir=mc,
        llm_client=client,
        max_repair_attempts=2,
    )
    final_state = orch.run("Implemente soma", "1. Escrever src/calc.py")

    assert final_state == State.DONE
    counts = orch.get_role_call_counts()
    assert counts == {ROLE_TOOL_SELECTION: 2, ROLE_REPAIR: 2}
    assert orch.compute_rho() == 0.5
    assert calcular_rho(counts) == 0.5

    # Valida tambem o rastreamento em attach_routing_to_client
    routed = SimpleNamespace(model_name="m14", chat=lambda msgs, **kw: _Resp("ok"))
    attach_routing_to_client(routed, AgentRoutingConfig())
    routed.chat([{"role": "user", "content": "a"}], role=ROLE_TOOL_SELECTION)
    routed.chat([{"role": "user", "content": "b"}], role=ROLE_TOOL_SELECTION)
    routed.chat([{"role": "user", "content": "c"}], role=ROLE_TOOL_SELECTION)
    routed.chat([{"role": "user", "content": "d"}], role=ROLE_REPAIR)
    assert routed.get_role_counts() == {ROLE_TOOL_SELECTION: 3, ROLE_REPAIR: 1}
    assert calcular_rho(routed.get_role_counts()) == 0.75