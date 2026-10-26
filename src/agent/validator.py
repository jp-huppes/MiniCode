from pathlib import Path
from typing import Dict, Any, Optional
from src.tools.registry import ToolRegistry
from src.security.sandbox import resolve_in_root

ALLOWED_TOOLS = {"list_files", "read_file", "write_file", "replace_text", "append_file", "run_tests", "done"}
class ActionValidator:
    def __init__(self, workspace_dir):
        self.workspace = Path(workspace_dir).resolve()
        self.registry = ToolRegistry(self.workspace)

    def validate(self, tool_name: str, args: Dict[str, Any]) -> Optional[str]:
        # 1. catalogo
        if tool_name not in ALLOWED_TOOLS:
            return f"Ferramenta desconhecida ou não permitida: '{tool_name}'. Ferramentas válidas: {sorted(ALLOWED_TOOLS)}"
        if not isinstance(args, dict): return "Os argumentos da ação devem ser um objeto JSON."

        # 2. contrato unico: só os parametros do schema sao aceitos (ex.: 'file_path' é rejeitado)
        meta = self.registry.tools.get(tool_name)
        if meta is not None:
            schema = meta["parameters"]
            props = schema.get("properties", {})
            desconhecidos = [k for k in args if k not in props]
            if desconhecidos:
                return (f"Argumentos inválidos para '{tool_name}': {desconhecidos}. "
                        f"Parâmetros aceitos: {list(props)}.")
            faltando = [k for k in schema.get("required", []) if k not in args]
            if faltando: return f"A ferramenta '{tool_name}' exige os argumentos: {faltando}."

        # 3. sandbox e tipos
        if tool_name in {"read_file", "write_file", "replace_text", "append_file"}:
            path_arg = args.get("path")
            if not isinstance(path_arg, str) or not path_arg.strip(): return f"O argumento 'path' de '{tool_name}' deve ser um texto não vazio."
            try:
                resolve_in_root(self.workspace, path_arg)
            except Exception as e:
                return f"Violação de sandbox no caminho '{path_arg}': {e}"

        if tool_name in {"write_file", "append_file"}:
            content = args.get("content", "")
            if not isinstance(content, str): return "O argumento 'content' de 'write_file' deve ser texto."
            if len(content) > 500_000: return "Conteúdo excede o limite máximo permitido para escrita em arquivo."

        return None