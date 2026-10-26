import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.tools.registry import ToolRegistry
from src.agent.cobertura import testes_ausentes

DEFAULT_TEST_CMD = "pytest -x --tb=short -q"


@dataclass
class TestResult:
    exit_code: int
    output: str
    duration: float
    tests_passed: bool
    write_seq_at_run: int


class HarnessVerifier:
    """
    Executa a verficacao de forma isolada pelo harness.
    Aprova somente se: pytest retorna 0 E todos os test_ac_n exigidos pela SPEC existem.
    """

    def __init__(self, workspace_dir: str | Path, test_cmd: str = DEFAULT_TEST_CMD):
        self.workspace = Path(workspace_dir).resolve()
        self.test_cmd = test_cmd or DEFAULT_TEST_CMD

    def run_verification(self, current_write_seq: int, command: Optional[str] = None) -> TestResult:
        command = command or self.test_cmd
        start_time = time.time()
        try:
            res = ToolRegistry(self.workspace).dispatch("run_tests", {"command": command})
            duration = time.time() - start_time

            output = res.output or res.error or ""
            exit_code = res.meta.get("exit_code", 0 if res.ok else 1)

            zero_passed = re.search(r"(?<!\d)0 passed", output)
            tests_passed = exit_code == 0 and not zero_passed and "no tests ran" not in output.lower()

            # cobertura: a mensagem vai no FIM da saída, para sobreviver ao truncamento do prompt de reparo
            ausentes = testes_ausentes(self.workspace)
            if ausentes:
                tests_passed = False
                output = output + (
                    "\n\nCOBERTURA INCOMPLETA: a SPEC exige os testes abaixo, mas eles não existem "
                    f"em tests/ com este nome exato: {ausentes}. Crie-os com os nomes listados."
                )

            return TestResult(
                exit_code=exit_code,
                output=output,
                duration=duration,
                tests_passed=tests_passed,
                write_seq_at_run=current_write_seq,
            )
        except Exception as e:
            return TestResult(
                exit_code=-1,
                output=f"Erro interno de execução no harness: {str(e)}",
                duration=time.time() - start_time,
                tests_passed=False,
                write_seq_at_run=current_write_seq,
            )

    def validate_repair_path_protection(self, tool_name: str, args: dict) -> bool:
        # Decisão: testes podem ser reescritos no REPAIRING.
        # A proteção contra remoção de testes exigidos é feita por run_verification (cobertura).
        return True