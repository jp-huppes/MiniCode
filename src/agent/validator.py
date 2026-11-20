from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import ValidationError

from src.security.sandbox import resolve_in_root
from src.tools.catalog import (
    ALLOWED_TOOLS,
    CATALOG_BY_NAME,
    CONTENT_TOOLS,
    FILE_PATH_TOOLS,
)
class ActionValidator:
    def __init__(self, workspace_dir):
        self.workspace = Path(workspace_dir).resolve()

    def validate(self, tool_name: str, args: Dict[str, Any]) -> Optional[str]:
        if tool_name not in ALLOWED_TOOLS:
            return (
                f"Ferramenta desconhecida ou não permitida: '{tool_name}'. "
                f"Ferramentas válidas: {sorted(ALLOWED_TOOLS)}"
            )
        if not isinstance(args, dict):
            return "Os argumentos da ação devem ser um objeto JSON."

        spec = CATALOG_BY_NAME.get(tool_name)
        parsed_args = None
        if spec is not None:
            try:
                parsed_args = spec.args_model.model_validate(args)
            except ValidationError as exc:
                faltando = [
                    ".".join(str(part) for part in err["loc"])
                    for err in exc.errors()
                    if err.get("type") == "missing"
                ]
                if faltando:
                    return f"A ferramenta '{tool_name}' exige os argumentos: {faltando}."
                return f"Argumentos inválidos para '{tool_name}': {exc.errors()}"

        if tool_name in FILE_PATH_TOOLS:
            path_arg = getattr(parsed_args, "path", None) if parsed_args is not None else args.get("path")
            if not isinstance(path_arg, str) or not path_arg.strip():
                return f"O argumento 'path' de '{tool_name}' deve ser um texto não vazio."
            try:
                resolve_in_root(self.workspace, path_arg)
            except Exception as e:
                return f"Violação de sandbox no caminho '{path_arg}': {e}"

        if tool_name in CONTENT_TOOLS:
            content = args.get("content", "")
            if not isinstance(content, str):
                return f"O argumento 'content' de '{tool_name}' deve ser texto."
            if len(content) > 500_000:
                return "Conteúdo excede o limite máximo permitido para escrita em arquivo."

        return None