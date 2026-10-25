import time, re
from pathlib import Path
from dataclasses import dataclass, field
from src.tools.registry import ToolRegistry
from src.tools.base import ToolResult
from src.security.sandbox import resolve_in_root

@dataclass
class TestResult:
    exit_code: int
    output: str
    duration: float
    tests_passed: bool
    write_seq_at_run: int

class HarnessVerifier:
    """
    Executa a verificacao dos testes de forma isolada pelo harness
        Nenhuma LLM pode chamar run_tests diretamente em VERIFYING!
    """
    def __init__(self, workspace_dir: str | Path):
        self.workspace = Path(workspace_dir).resolve()

    def run_verification(self, current_write_seq: int, command: str = "pytest -x --tb=short -q") -> TestResult:
        """
        Executa o pytest fixo definido pelo harness e calcula formalmente tests_passed
            trata codigos especiais e saidas vazia/0 testes coletados como falha
        """
        start_time = time.time()
        try:
            registry = ToolRegistry(self.workspace)
            res = registry.dispatch("run_tests", {"command": command})
            duration = time.time() - start_time
            
                # como res pode ser um ToolResult ou um dict, tratamos ambos de forma segura:
            if isinstance(res, ToolResult):
                output = res.output or ""
                meta = res.meta or {}
                exit_code = meta.get("exit_code")
                if exit_code is None:
                    exit_code = 0 if res.ok else 1
            else:
                output = res.get("output", "")
                meta = res.get("meta", {})
                exit_code = meta.get("exit_code")
                if exit_code is None:
                    exit_code = res.get("exit_code", 1)

            # Condição estrita: exit_code 0 e output indicando sucesso real
            zero_passed_match = re.search(r"(?<!\d)0 passed", output)
            tests_passed = (exit_code == 0) and not zero_passed_match and ("no tests ran" not in output.lower())
            
            return TestResult(
                exit_code=exit_code,
                output=output,
                duration=duration,
                tests_passed=tests_passed,
                write_seq_at_run=current_write_seq
            )
        
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                exit_code=-1,
                output=f"Erro interno de execução no harness: {str(e)}",
                duration=duration,
                tests_passed=False,
                write_seq_at_run=current_write_seq
            )

    def validate_repair_path_protection(self, tool_name: str, args: dict) -> bool:
        """
        Protecao de testes: impede que a LLM altere arquivos dentro do diretorio'tests/' durante o REPAIRING
            Retorna True se for permitido, ou False se violar a protecao
        """
        if tool_name in {"write_file", "replace_text"}:
            path_arg = args.get("path") or args.get("file_path")
            if path_arg:
                try:
                    resolved = resolve_in_root(self.workspace, path_arg)
                    rel = resolved.relative_to(self.workspace)
                    if "tests" in rel.parts:
                        return False
                except Exception: pass
        return True