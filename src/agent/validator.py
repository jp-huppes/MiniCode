from pathlib import Path
from typing import Any, Dict, Optional

from src.agent.state import State
from src.security.sandbox import resolve_in_root
from src.tools.catalog import (
    ALLOWED_TOOLS,
    CONTENT_TOOLS,
    FILE_PATH_TOOLS,
    WRITE_TOOLS,
)

MAX_CONTENT_CHARS = 500_000


class ActionValidator:
    def __init__(self, workspace_dir, state_machine: Any = None):
        self.workspace = Path(workspace_dir).resolve()
        self.state_machine = state_machine

    def _current_state(self) -> Optional[State]:
        if self.state_machine is None:
            return None
        return getattr(self.state_machine, "state", None)

    def _is_under_tests_dir(self, path_arg: str) -> bool:
        try:
            resolved = resolve_in_root(self.workspace, path_arg)
            rel_parts = resolved.relative_to(self.workspace).parts
        except Exception:
            rel_parts = Path(path_arg).parts
        return len(rel_parts) > 0 and rel_parts[0].lower() == "tests"

    def validate(
        self,
        tool_name: str,
        args: Dict[str, Any],
        state: Optional[State] = None,
    ) -> Optional[str]:
        if tool_name not in ALLOWED_TOOLS:
            return (
                f"Ferramenta desconhecida ou não permitida: '{tool_name}'. "
                f"Ferramentas válidas: {sorted(ALLOWED_TOOLS)}"
            )
        if not isinstance(args, dict):
            return "Os argumentos da ação devem ser um objeto JSON."

        effective_state = state if state is not None else self._current_state()

        # Gate de estado para ferramentas de escrita (F7 / Camada 4)
        if tool_name in WRITE_TOOLS and effective_state is not None:
            if effective_state not in {State.IMPLEMENTING, State.REPAIRING}:
                return (
                    f"Escrita proibida no estado atual '{effective_state.value}'. "
                    "A escrita só é permitida em IMPLEMENTING ou REPAIRING."
                )

        # Sandbox de caminho (apenas 'path', sem alias 'file_path') e proteção de tests/ em REPAIRING
        if tool_name in FILE_PATH_TOOLS and "path" in args or (
            tool_name in {"write_file", "read_file", "replace_text", "append_file"}
        ):
            path_arg = args.get("path")
            exige_path = tool_name in {"write_file", "read_file", "replace_text", "append_file"}
            if path_arg is None:
                if exige_path:
                    return f"A ferramenta '{tool_name}' exige o argumento 'path'."
            else:
                if not isinstance(path_arg, str) or not path_arg.strip():
                    return f"O argumento 'path' de '{tool_name}' deve ser um texto não vazio."
                try:
                    resolve_in_root(self.workspace, path_arg)
                except Exception as e:
                    return f"Violação de sandbox no caminho '{path_arg}': {e}"

                if (
                    effective_state == State.REPAIRING
                    and tool_name in WRITE_TOOLS
                    and self._is_under_tests_dir(path_arg)
                ):
                    return "Ação rejeitada: proibido alterar arquivos na pasta 'tests/' durante o reparo."

        # Limite de tamanho de conteúdo para ferramentas de escrita com content
        if tool_name in CONTENT_TOOLS:
            content = args.get("content", "")
            if not isinstance(content, str):
                return f"O argumento 'content' de '{tool_name}' deve ser texto."
            if len(content) > MAX_CONTENT_CHARS:
                return "Conteúdo excede o limite máximo permitido para escrita em arquivo."

        return None