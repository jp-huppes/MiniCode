import pytest
import json
from pathlib import Path
from src.llm.client import LLMClient
from src.agent.state import State, AgentStateMachine
from src.agent.schemas import AgentStateData
from src.agent.validator import ActionValidator
from src.agent.gates import ExecutionGateManager
from src.agent.executor import ToolExecutor
from src.agent.context_manager import estimar_tokens, truncar_pytest_output
from src.agent.orchestrator import AgentOrchestrator

class FakeLLM(LLMClient):
    # Cliente LLM simulado para testar o loop de execucao sem chamadas reais
    def __init__(self, responses):
        self.responses = list(responses)
        self.call_count = 0

    def chat(self, messages, tools=None):
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
            self.call_count += 1
            return resp
        return self.responses[-1]

def test_state_data_serialization(tmp_path):
    # Serializacao e carga atomico de state.json
    executor = ToolExecutor(tmp_path, tmp_path / ".minicode")
    state_data = AgentStateData(iteration=2, tests_passed=True)
    executor.save_state_atomic(state_data, task="Tarefa de Teste")
    
    state_file = tmp_path / ".minicode" / "state.json"
    assert state_file.is_file()
    
    content = json.loads(state_file.read_text(encoding="utf-8"))
    assert content["iteration"] == 2
    assert content["tests_passed"] is True
    assert content["task"] == "Tarefa de Teste"

def test_action_validator_security(tmp_path):
    # validacao de path traversal, ferramenta invalido e argumentos
    validator = ActionValidator(tmp_path)
    
    # ferramentas não permitida
    err = validator.validate("rm_rf", {"path": "."})
    assert err is not None
    assert "desconhecida" in err

    # path traversal (fuga do workspace)
    err = validator.validate("read_file", {"path": "../etc/passwd"})
    assert err is not None
    assert "sandbox" in err

    # argumento obrigatorio faltando
    err = validator.validate("write_file", {"content": "hello"})
    assert err is not None
    assert "exige" in err

    # acao valida
    err = validator.validate("list_files", {})
    assert err is None

def test_gates_and_done_rule(tmp_path):
    # Matriz de estado e transição para IMPLEMENTING
    sm = AgentStateMachine()
    sm.transition(State.SPECIFIED)
    sm.transition(State.PLANNED)
    sm.transition(State.IMPLEMENTING)
    
    state_data = AgentStateData(tests_passed=False, modified_after_last_test=True)
    gates = ExecutionGateManager(sm, state_data)

    # Em IMPLEMENTING, chamar 'done' é permitido para acionar a transição para VERIFYING (gatilho de verificação)
    err = gates.check_gate("done", {})
    assert err is None

def test_executor_and_logging(tmp_path):
    # Execucao isolada e gravacao em actions.jsonl
    executor = ToolExecutor(tmp_path, tmp_path / ".minicode")
    res = executor.execute("list_files", {})
    assert res["ok"] is True

    executor.log_action(1, "list_files", {}, None, None, res, {"prompt_tokens": 15, "completion_tokens": 10})
    log_file = tmp_path / ".minicode" / "actions.jsonl"
    assert log_file.is_file()
    
    lines = log_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["iteration"] == 1
    assert entry["tool"] == "list_files"

def test_token_estimator_and_pytest_truncation():
    # Estimativa de tokens e truncamento inteligente de saidas do pytest
    txt = "a" * 400
    tokens = estimar_tokens(txt, chars_por_token=4)
    assert tokens == 100

    long_output = "PASSED test_a\nFAILED test_b\nTraceback...\n=== short test summary info ===\nFAILED test_b - assert False\n1 failed, 1 passed"
    
    truncated = truncar_pytest_output(long_output, max_tokens=10)
    assert "truncada" in truncated

def test_orchestrator_success_loop(tmp_path):
    #Loop completo - Sucesso (escreve arquivo, roda testes com sucesso e finaliza com done)
        
        # cria diretorio de testes e um teste valido para o pytest executar
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_dummy.py").write_text("def test_ok(): assert True\n", encoding="utf-8")

    responses = [
        'json\n{"thought": "escrever código", "tool": "write_file", "args": {"path": "src/calc.py", "content": "def soma(a, b): return a + b"}}\n',
        'json\n{"thought": "rodar testes", "tool": "run_tests", "args": {}}\n',
        'json\n{"thought": "concluir tarefa", "tool": "done", "args": {}}\n'
    ]
    
    fake_llm = FakeLLM(responses)
    orchestrator = AgentOrchestrator(tmp_path, tmp_path / ".minicode", fake_llm)
    
    final_state = orchestrator.run("Criar soma", "Plano soma")
    print(f"\n--- DEBUG FAILURE_REASON: {orchestrator.state_data.failure_reason} ---")
    assert final_state in {State.VERIFYING, State.DONE}
    assert orchestrator.state_data.tests_passed is True

def test_orchestrator_max_iterations_failure(tmp_path):
    # Loop completo - estouro de interações (transiciona para FAILED com max_iterations)
    response = '```json\n{"thought": "loop infinito", "tool": "list_files", "args": {}}\n```'
    fake_llm = FakeLLM([response] * 10)
    
    orchestrator = AgentOrchestrator(tmp_path, tmp_path / ".minicode", fake_llm)
    orchestrator.state_data.max_iterations = 2
    
    final_state = orchestrator.run("Spec", "Plan")
    assert final_state == State.FAILED
    assert orchestrator.state_data.failure_reason == "max_iterations"

def test_orchestrator_parse_failure(tmp_path):
    # Loop completo - falhas consecutivas de parse JSON (transiciona para FAILED)
    fake_llm = FakeLLM(["Texto livre sem JSON válido"] * 5)
    orchestrator = AgentOrchestrator(tmp_path, tmp_path / ".minicode", fake_llm)
    
    final_state = orchestrator.run("Spec", "Plan")
    assert final_state == State.FAILED
    assert orchestrator.state_data.failure_reason == "parse_failure"