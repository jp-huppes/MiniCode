import sys
import json
from typing import Any, Dict, Protocol

class Renderer(Protocol):
    def handle(self, event_type: str, payload: Dict[str, Any]) -> None:
        ...
    def finish(self, result: Any) -> None:
        ...

class HumanRenderer:
    def __init__(self, verbose: bool = False, no_color: bool = False):
        self.verbose = verbose
        self.no_color = no_color or not sys.stdout.isatty()

    def _color(self, text: str, code: str) -> str:
        if self.no_color: return text

        return f"\033[{code}m{text}\033[0m"

    def handle(self, event_type: str, payload: Dict[str, Any]) -> None:
        if event_type == "state_changed":
            old, new = payload.get("old"), payload.get("new")
            print(self._color(f"\n[ESTADO] {old} ➔ {new}", "36"))
        elif event_type == "spec_ready":
            print(self._color(f"📄 Especificação gerada: {payload.get('path')}", "32"))
            if self.verbose: print(payload.get("content", ""))
        elif event_type == "plan_ready":
            print(self._color(f"📋 Plano gerado: {payload.get('path')}", "32"))
            if self.verbose: print(payload.get("content", ""))
        elif event_type == "action":
            tool = payload.get("tool")
            args = str(payload.get("args", {}))
            if len(args) > 80 and not self.verbose: args = args[:77] + "..."
            print(f"  ⚡ Tool: {tool} | Args: {args}")
            if payload.get("error"): print(self._color(f"    ❌ Erro: {payload.get('error')}", "31"))
        elif event_type == "tests_run":
            passed = payload.get("passed", False)
            symbol = "✔" if passed else "✘"
            color_code = "32" if passed else "31"
            print(self._color(f"  🧪 Testes [{symbol}]: {payload.get('command')}", color_code))
            if not passed and payload.get("output"):
                output_lines = payload.get("output").strip().splitlines()
                snippet = "\n".join(output_lines[:10])
                print(self._color(f"    Trecho da falha:\n{snippet}", "31"))
        elif event_type == "repair_attempt":
            k, K = payload.get("k"), payload.get("K")
            print(self._color(f"  🔧 Tentativa de reparo {k}/{K}...", "33"))
        sys.stdout.flush()

    def finish(self, result: Any) -> None:
        print("\n" + "="*50)
        status = getattr(result, "final_state", str(result))
        success = getattr(result, "success", status == "DONE")
        color = "32" if success else "31"
        print(self._color(f"🏁 Execução Encerrada | Status: {status} (Sucesso: {success})", color))
        if hasattr(result, "run_dir") and result.run_dir: print(f"📂 Run Dir: {result.run_dir}")
        print("="*50)

class JsonRenderer:
    def __init__(self):
        self.last_result = None

    def handle(self, event_type: str, payload: Dict[str, Any]) -> None:
        # No modo json, eventos intermediario podem ir para stderr ou ser ignorados no stdout
        if event_type == "run_finished": self.last_result = payload

    def finish(self, result: Any) -> None:
        # Serializa o resultado final em JSON limpo no stdout
        data = result if isinstance(result, dict) else {
            "final_state": getattr(result, "final_state", "UNKNOWN"),
            "success": getattr(result, "success", False),
            "run_dir": str(getattr(result, "run_dir", "")),
            "error": getattr(result, "error", None)
        }
        print(json.dumps(data, ensure_ascii=False, indent=2))

class QuietRenderer:
    def handle(self, event_type: str, payload: Dict[str, Any]) -> None:
        pass

    def finish(self, result: Any) -> None:
        status = getattr(result, "final_state", "UNKNOWN")
        print(status)