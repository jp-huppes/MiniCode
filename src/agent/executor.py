import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from src.agent.schemas import AgentStateData
from src.tools.base import ToolResult
from src.tools.registry import ToolRegistry


class ToolExecutor:
    """
    Executa ferramentas de forma isolada e segura, capturando duracao, erros
    e registrando rastreabilidade em actions.jsonl e state.json.
    """

    def __init__(self, workspace_dir: str | Path, minicode_dir: str | Path):
        self.workspace = Path(workspace_dir).resolve()
        self.minicode_dir = Path(minicode_dir).resolve()
        self.minicode_dir.mkdir(parents=True, exist_ok=True)

        self.registry = ToolRegistry(self.workspace)
        self.actions_log_path = self.minicode_dir / "actions.jsonl"
        self.state_json_path = self.minicode_dir / "state.json"

    def execute(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executa a ferramenta solicitada atraves do ToolRegistry, garantindo que
        excecoes nunca propaguem para fora do harness.
        Retorna um dicionario normalizado: {ok: bool, output: str, duration: float}.
        """
        start_time = time.time()
        try:
            result = self.registry.dispatch(tool_name, args)
            duration = time.time() - start_time
            return self._normalizar(result, duration)
        except Exception as e:
            duration = time.time() - start_time
            return {
                "ok": False,
                "output": f"Erro interno ao executar a ferramenta '{tool_name}': {str(e)}",
                "error": str(e),
                "meta": {},
                "duration": duration,
            }

    @staticmethod
    def _normalizar(result: Any, duration: float) -> Dict[str, Any]:
        if isinstance(result, ToolResult):
            ok = result.ok
            output = result.output if ok else (result.error or "Erro desconhecido na ferramenta.")
            return {
                "ok": ok,
                "output": output,
                "error": result.error,
                "meta": result.meta,
                "duration": duration,
            }
        if isinstance(result, dict):
            return {
                "ok": bool(result.get("ok", True)),
                "output": str(result.get("output", "")),
                "error": result.get("error"),
                "meta": result.get("meta", {}),
                "duration": duration,
            }
        return {"ok": True, "output": str(result), "error": None, "meta": {}, "duration": duration}

    def log_action(
        self,
        iteration: int,
        tool_name: str,
        args: Dict[str, Any],
        validator_error: Optional[str],
        gate_error: Optional[str],
        result: Optional[Dict[str, Any]],
        tokens_info: Dict[str, int],
        *,
        run_id: str = "",
        prompt: Optional[str] = None,
        raw_response: Optional[str] = None,
        parsed_action: Optional[Dict[str, Any]] = None,
        fsm_state: Optional[str] = None,
    ):
        timestamp = datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "run_id": run_id,
            "iteration": iteration,
            "fsm_state": fsm_state,
            "prompt": prompt,
            "raw_response": raw_response,
            "parsed_action": parsed_action,
            "tool": tool_name,
            "args": args,
            "validator_error": validator_error,
            "gate_error": gate_error,
            "result": result,
            "tokens": tokens_info,
            "duration": result.get("duration", 0.0) if result else 0.0,
        }

        with open(self.actions_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    def _task_existente(self) -> str:
        if not self.state_json_path.exists():
            return ""
        try:
            data = json.loads(self.state_json_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return str(data.get("task", "") or "")
        except Exception:
            pass
        return ""

    def save_state_atomic(
        self,
        state_data: AgentStateData,
        task: Optional[str] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
    ):
        meta = dict(extra_meta) if extra_meta is not None else {}
        resolved_task = task if task is not None else self._task_existente()

        state_dict = {
            "task": resolved_task,
            "state": state_data.state,
            "iteration": state_data.iteration,
            "max_iterations": state_data.max_iterations,
            "tests_passed": state_data.tests_passed,
            "modified_after_last_test": state_data.modified_after_last_test,
            "parse_failures": state_data.parse_failures,
            "last_usage": state_data.last_usage,
            "failure_reason": state_data.failure_reason,
            "timestamp": datetime.now().isoformat(),
            **meta,
        }

        temp_path = self.state_json_path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(state_dict, f, indent=2, ensure_ascii=False)

        temp_path.replace(self.state_json_path)