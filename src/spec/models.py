from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class ValidationResult:
    ok: bool; errors: List[str] = field(default_factory=list)

@dataclass
class Spec:
    raw_content: str
    objective: str = ""
    parameterss : List[str] = field(default_factory=list)
    acceptance_criteria: Dict[str, str] = field(default_factory=dict)  # ID -> descricao

@dataclass
class Plan:
    raw_content: str
    steps: List[Dict[str, Any]] = field(default_factory=list)

class SpecInvalidaError(Exception):
    # excecao levantada quando a SPEC falha em todas as tentativas de validacao
    def __init__(self, errors: List[str]):
        super().__init__(f"SPEC inválida após esgotar tentativas. Erros: {errors}")
        self.errors = errors

class PlanInvalidoError(Exception):
    # excecao levantada quando o PLAN falha em todas as tentativas de val
    def __init__(self, errors: List[str]):
        super().__init__(f"PLAN inválido após esgotar tentativas. Erros: {errors}")
        self.errors = errors