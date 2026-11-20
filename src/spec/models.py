from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ValidationResult:
    ok: bool
    errors: List[str] = field(default_factory=list)


@dataclass
class AcceptanceCriterion:
    id: str
    text: str


@dataclass
class SpecTestItem:
    ac: str
    name: str


@dataclass
class SpecDocument:
    objective: str
    parameters: List[str] = field(default_factory=list)
    acceptance_criteria: List[AcceptanceCriterion] = field(default_factory=list)
    tests: List[SpecTestItem] = field(default_factory=list)


@dataclass
class PlanStep:
    n: int
    action: str
    files: List[str] = field(default_factory=list)
    covers: List[str] = field(default_factory=list)


@dataclass
class PlanDocument:
    steps: List[PlanStep] = field(default_factory=list)


@dataclass
class Spec:
    raw_content: str
    objective: str = ""
    parameters: List[str] = field(default_factory=list)
    acceptance_criteria: Dict[str, str] = field(default_factory=dict)  # ID -> descrição
    tests: List[SpecTestItem] = field(default_factory=list)
    structured: Optional[SpecDocument] = None


@dataclass
class Plan:
    raw_content: str
    steps: List[Dict[str, Any]] = field(default_factory=list)
    structured: Optional[PlanDocument] = None


class SpecInvalidaError(Exception):
    # excecao levantada quando a SPEC falha em todas as tentativas de validacao
    def __init__(self, errors: List[str]):
        super().__init__(f"SPEC inválida após esgotar tentativas. Erros: {errors}")
        self.errors = errors


class PlanInvalidoError(Exception):
    # excecao levantada quando o PLAN falha em todas as tentativas de validacao
    def __init__(self, errors: List[str]):
        super().__init__(f"PLAN inválido após esgotar tentativas. Erros: {errors}")
        self.errors = errors