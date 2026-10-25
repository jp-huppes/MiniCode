import pytest
from pathlib import Path
from src.agent.state import State
from src.agent.verifier import HarnessVerifier
from src.agent.repair_context import RepairContextBuilder, calcular_assinatura_erro
from src.agent.orchestrator import AgentOrchestrator
from src.llm.client import LLMClient

class FakeLLMForRepair(LLMClient):
    # Cliente LLM simulado para testar o ciclo de verificacao e reparo
    def __init__(self, responses):
        self.responses = list(responses)
        self.call_count = 0

    def chat(self, messages, tools=None):
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
            self.call_count += 1
            return resp
        return self.responses[-1]

def test_error_signature_matching():
    # Valida se o calc de assinatura hash agrupa erros identicos e diferencia erros distintos
    err1 = "FAILED test_calc.py::test_soma - AssertionError: assert 2 == 3"
    err2 = "FAILED test_calc.py::test_soma - AssertionError: assert 2 == 3"
    err3 = "FAILED test_calc.py::test_sub - AssertionError: assert 5 == 1"
    
    sig1 = calcular_assinatura_erro(err1)
    sig2 = calcular_assinatura_erro(err2)
    sig3 = calcular_assinatura_erro(err3)
    
    assert sig1 == sig2
    assert sig1 != sig3

def test_repair_path_protection(tmp_path):
    # Valida que a LLM é estritamente impedida de alterar arquivos na pasta 'tests/' durante o REPAIRING
    verifier = HarnessVerifier(tmp_path)
    
    # tentar escrever dentro de tests/ deve ser bloqueado
    allowed = verifier.validate_repair_path_protection("write_file", {"path": "tests/test_calc.py", "content": "def test_fake(): pass"})
    assert allowed is False

    # escrever em src/ deve ser permitido
    allowed_src = verifier.validate_repair_path_protection("write_file", {"path": "src/calc.py", "content": "x = 1"})
    assert allowed_src is True

def test_repair_loop_success_after_repair(tmp_path):
    # ciclo completo: falha inicial, transiciona para REPAIRING, aplica correcao e atinge DONE com sucesso
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_calc.py").write_text(
        "from src.calc import soma\ndef test_soma(): assert soma(2, 3) == 5\n",
        encoding="utf-8"
    )
    src_dir = tmp_path / "src"
    src_dir.mkdir(exist_ok=True)

    # Garante que o python encontre o pacote src no tmp_path isolado do teste
    (tmp_path / "conftest.py").write_text("", encoding="utf-8")
    (src_dir / "__init__.py").write_text("", encoding="utf-8")
    
    # Escreve implementação incorreta e em seguida corrige a implementacao em REPAIRING
    responses = [
        '```json\n{"thought": "implementacao incorreta", "tool": "write_file", "args": {"path": "src/calc.py", "content": "def soma(a, b): return a - b"}}\n```',
        '```json\n{"thought": "pedir verificacao", "tool": "done", "args": {}}\n```',
        '```json\n{"thought": "corrigindo implementacao", "tool": "write_file", "args": {"path": "src/calc.py", "content": "def soma(a, b): return a + b"}}\n```',
        '```json\n{"thought": "concluir", "tool": "done", "args": {}}\n```'
    ]
    
    fake_llm = FakeLLMForRepair(responses)
    orchestrator = AgentOrchestrator(tmp_path, tmp_path / ".minicode", fake_llm)
    
    final_state = orchestrator.run("Criar soma", "Plano soma")
    print(f"\n--- DEBUG FAILURE_REASON: {orchestrator.state_data.failure_reason} ---")
    assert final_state == State.DONE
    assert orchestrator.state_data.tests_passed is True
    assert orchestrator.state_machine.repair_attempts == 1

def test_repair_loop_exhaustion_leads_to_failed(tmp_path):
    # exaustao do limite de tentativas de reparo (MAX_REPAIR_ATTEMPTS) leva ao estado FAILED
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_calc.py").write_text(
        "from src.calc import soma\ndef test_soma(): assert soma(2, 3) == 5\n",
        encoding="utf-8")
    src_dir = tmp_path / "src"; src_dir.mkdir(exist_ok=True)

    # LLM persiste em escrever codigo incorreto em todas as tentativas
    responses = ['```json\n{"thought": "tentativa 1 errada", "tool": "write_file", "args": {"path": "src/calc.py", "content": "def soma(a, b): return 0"}}\n```', '```json\n{"thought": "tentativa 2 errada", "tool": "write_file", "args": {"path": "src/calc.py", "content": "def soma(a, b): return 1"}}\n```', '```json\n{"thought": "tentativa 3 errada", "tool": "write_file", "args": {"path": "src/calc.py", "content": "def soma(a, b): return 2"}}\n```']

    fake_llm = FakeLLMForRepair(responses)
    orchestrator = AgentOrchestrator(tmp_path, tmp_path / ".minicode", fake_llm)
    orchestrator.state_machine.max_repair_attempts = 2  # limite estrito para teste
    orchestrator.state_data.max_iterations = 20 # garante margem de interacoes suficiente

    final_state = orchestrator.run("Criar soma", "Plano soma")
    assert final_state == State.FAILED
    assert orchestrator.state_data.failure_reason in {"max_repair_attempts", "max_iterations"}