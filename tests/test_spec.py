import json
import pytest

from src.agent.state import AgentStateMachine, State
from src.llm.client import LLMClient
from src.spec.generator import gerar_spec_com_retry
from src.spec.models import SpecInvalidaError
from src.spec.pipeline import executar_spec_pipeline
from src.spec.validators import (
    extrair_spec_estruturada,
    plan_json_schema,
    validar_plano,
    validar_plano_json,
    validar_spec,
    validar_spec_json,
)


class FakeLLM(LLMClient):
    # Cliente LLM simulado para testes sem depender de endpoint externo
    def __init__(self, responses):
        self.responses = list(responses)
        self.call_count = 0
        self.history = []
        self.seen_schemas = []

    def chat(self, messages, tools=None, response_schema=None):
        self.history.append(messages)
        self.seen_schemas.append(response_schema)
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
            self.call_count += 1
            return resp
        return self.responses[-1]


VALID_SPEC_JSON = json.dumps({
    "objective": "Implementar funcionalidade X no projeto de forma modular.",
    "parameters": [
        "Linguagem: Python 3.11",
        "Alvo: src/main.py",
    ],
    "acceptance_criteria": [
        {"id": "AC-1", "text": "A função principal retorna True para entradas válidas."},
    ],
    "tests": [
        {"ac": "AC-1", "name": "test_ac_1_funcionalidade_valida"},
    ],
})

VALID_PLAN_JSON = json.dumps({
    "steps": [
        {
            "n": 1,
            "action": "Criar o arquivo principal",
            "files": ["src/main.py"],
            "covers": ["AC-1"],
        },
        {
            "n": 2,
            "action": "Criar os testes automatizados",
            "files": ["tests/test_main.py"],
            "covers": ["AC-1"],
        },
    ]
})

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


def test_validator_spec_json_valid():
    res = validar_spec_json(VALID_SPEC_JSON)
    assert res.ok is True
    assert len(res.errors) == 0


def test_validator_spec_json_missing_parameters():
    invalid_spec = json.dumps({
        "objective": "Fazer algo.",
        "parameters": [],
        "acceptance_criteria": [{"id": "AC-1", "text": "Teste válido."}],
        "tests": [{"ac": "AC-1", "name": "test_ac_1"}],
    })
    res = validar_spec_json(invalid_spec)
    assert res.ok is False
    assert any("V2" in e for e in res.errors)


def test_validator_plan_json_invalid_numbering_and_enum():
    spec = extrair_spec_estruturada(VALID_SPEC_TEXT)
    schema = plan_json_schema(spec)
    covers_enum = schema["properties"]["steps"]["items"]["properties"]["covers"]["items"]["enum"]
    assert covers_enum == ["AC-1"]

    invalid_plan = json.dumps({
        "steps": [
            {"n": 1, "action": "Passo um", "files": ["src/a.py"], "covers": ["AC-1"]},
            {"n": 3, "action": "Passo três", "files": ["src/b.py"], "covers": ["AC-1"]},
        ]
    })
    res = validar_plano_json(invalid_plan, spec)
    assert res.ok is False
    assert any("P2" in e for e in res.errors)


def test_validator_plan_json_missing_ac_coverage():
    spec = extrair_spec_estruturada(VALID_SPEC_TEXT)
    plan_missing_ac = json.dumps({
        "steps": [
            {"n": 1, "action": "Passo sem cobertura", "files": ["src/a.py"], "covers": []},
        ]
    })
    res = validar_plano_json(plan_missing_ac, spec)
    assert res.ok is False
    assert any("P4" in e for e in res.errors)


def test_pipeline_json_happy_path(tmp_path):
    fake_llm = FakeLLM([VALID_SPEC_JSON, VALID_PLAN_JSON])
    machine = AgentStateMachine()
    assert machine.state == State.RECEIVED

    spec, plano_md = executar_spec_pipeline(
        "Implementar feature X",
        tmp_path,
        fake_llm,
        machine,
        use_json=True,
    )

    assert machine.state == State.PLANNED
    assert (tmp_path / "specification.md").is_file()
    assert (tmp_path / "plan.md").is_file()
    assert "AC-1" in spec.acceptance_criteria
    assert spec.parameters == ["Linguagem: Python 3.11", "Alvo: src/main.py"]
    assert "cobre: AC-1" in plano_md


def test_spec_json_retry_success(tmp_path):
    spec_invalida = json.dumps({
        "objective": "Teste sem parâmetros",
        "parameters": [],
        "acceptance_criteria": [{"id": "AC-1", "text": "Item"}],
        "tests": [{"ac": "AC-1", "name": "test_ac_1"}],
    })
    fake_llm = FakeLLM([spec_invalida, VALID_SPEC_JSON])
    spec_obj, raw = gerar_spec_com_retry("Tarefa", tmp_path, fake_llm, max_attempts=2, use_json=True)
    assert "## Parameters" in raw
    assert fake_llm.call_count == 2
    assert spec_obj.parameters == ["Linguagem: Python 3.11", "Alvo: src/main.py"]


def test_spec_json_exhausted_raises_error(tmp_path):
    spec_invalida = "Apenas texto livre sem JSON válido."
    fake_llm = FakeLLM([spec_invalida, spec_invalida, spec_invalida])
    machine = AgentStateMachine()

    with pytest.raises(SpecInvalidaError):
        executar_spec_pipeline("Tarefa inviável", tmp_path, fake_llm, machine, use_json=True)

    assert machine.state == State.RECEIVED
    assert not (tmp_path / "specification.md").exists()


def test_markdown_fallback_flag_pipeline(monkeypatch, tmp_path):
    monkeypatch.setenv("MINICODE_SPEC_PLAN_FORMAT", "markdown")
    assert validar_spec(VALID_SPEC_TEXT).ok is True
    assert validar_plano(VALID_PLAN_TEXT, extrair_spec_estruturada(VALID_SPEC_TEXT)).ok is True

    fake_llm = FakeLLM([VALID_SPEC_TEXT, VALID_PLAN_TEXT])
    machine = AgentStateMachine()
    spec, _ = executar_spec_pipeline("Implementar feature X", tmp_path, fake_llm, machine)

    assert machine.state == State.PLANNED
    assert (tmp_path / "specification.md").is_file()
    assert (tmp_path / "plan.md").is_file()
    assert "AC-1" in spec.acceptance_criteria