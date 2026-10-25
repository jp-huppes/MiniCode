import pytest
from pathlib import Path
from src.llm.client import LLMClient
from src.agent.state import AgentStateMachine, State
from src.spec.pipeline import executar_spec_pipeline
from src.spec.generator import gerar_spec_com_retry
from src.spec.models import SpecInvalidaError, PlanInvalidoError
from src.spec.validators import validar_spec, validar_plano, extrair_spec_estruturada

class FakeLLM(LLMClient):
    # Cliente LLM simulado para testes sem depender de endpoint externo
    def __init__(self, responses):
        self.responses = list(responses)
        self.call_count = 0
        self.history = []

    def chat(self, messages, tools=None):
        self.history.append(messages)
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
            self.call_count += 1
            return resp
        return self.responses[-1]

VALID_SPEC_TEXT = """## Objective
Implementar funcionalidade X no projeto de forma modular.

## Parameters
- Linguagem: Python 3.11
- Alvo: src/main.py

## Acceptance Criteria
- **AC-1**: A função principal retorna True para entradas válidas.

## Tests
- **AC-1**: test_ac_1_funcionalidade_valida
"""

VALID_PLAN_TEXT = """1. Criar o arquivo principal — arquivo(s): src/main.py — cobre: AC-1
2. Executar os testes automatizados — arquivo(s): tests/test_main.py — cobre: AC-1
"""

def test_validator_spec_valid():
    # Valida SPEC correta e completa
    res = validar_spec(VALID_SPEC_TEXT)
    assert res.ok is True
    assert len(res.errors) == 0

def test_validator_spec_missing_parameters():
    # Valida rejeicao de SPEC sem a secao Parameters
    invalid_spec = """## Objective
Fazer algo.
## Acceptance Criteria
- **AC-1**: Teste.
## Tests
- **AC-1**: test_ac_1
"""
    res = validar_spec(invalid_spec)
    assert res.ok is False
    assert any("Parameters" in e for e in res.errors)

def test_validator_plan_invalid_numbering():
    # Valida rejeicao de PLAN com numeracao quebrada
    spec = extrair_spec_estruturada(VALID_SPEC_TEXT)
    invalid_plan = """1. Passo um — arquivo(s): src/a.py — cobre: AC-1
3. Passo três — arquivo(s): src/b.py — cobre: AC-1
"""
    res = validar_plano(invalid_plan, spec)
    assert res.ok is False
    assert any("P2" in e for e in res.errors)

def test_validator_plan_missing_ac_coverage():
    # Valida rejeicao de PLAN que não cobre os AC-n exigidos
    spec = extrair_spec_estruturada(VALID_SPEC_TEXT)
    plan_missing_ac = """1. Passo sem citar critério — arquivo(s): src/a.py — cobre: none
"""
    res = validar_plano(plan_missing_ac, spec)
    assert res.ok is False
    assert any("P4" in e for e in res.errors)

def test_pipeline_happy_path(tmp_path):
    """
    Valida o fluxo feliz completo: SPEC valida na primeira, PLAN valido na segunda e transicao:
         RECEIVED -> SPECIFIED -> PLANNED.
    """

    fake_llm = FakeLLM([VALID_SPEC_TEXT, VALID_PLAN_TEXT]); machine = AgentStateMachine()
    assert machine.state == State.RECEIVED

    spec, __ = executar_spec_pipeline("Implementar feature X", tmp_path, fake_llm, machine)

    assert machine.state == State.PLANNED
    assert (tmp_path / "specification.md").is_file()
    assert (tmp_path / "plan.md").is_file()
    assert "AC-1" in spec.acceptance_criteria

def test_spec_retry_success(tmp_path):
    # Valida recuperacao apos falha na primeira tentativa de SPEC atraves do feedback direcionado
    spec_invalida = """## Objective
    Teste sem parâmetros
    ## Acceptance Criteria
    - **AC-1**: Item
    ## Tests
    - **AC-1**: test_ac_1
    """
    fake_llm = FakeLLM([spec_invalida, VALID_SPEC_TEXT])
    spec_obj, raw = gerar_spec_com_retry("Tarefa", tmp_path, fake_llm, max_attempts=2)
    assert "Parameters" in raw
    assert fake_llm.call_count == 2

def test_spec_exhausted_raises_error(tmp_path):
    """
    Valida que esgotar as tentativas de SPEC levanta SpecInvalidaError e o estado permanece RECEIVED.
    """
    spec_invalida = "Apenas texto livre sem seções obrigatórias."
    fake_llm = FakeLLM([spec_invalida, spec_invalida, spec_invalida])
    machine = AgentStateMachine()

    with pytest.raises(SpecInvalidaError): executar_spec_pipeline("Tarefa inviável", tmp_path, fake_llm, machine)

    assert machine.state == State.RECEIVED
    assert not (tmp_path / "specification.md").exists()