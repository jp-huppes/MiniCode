from pathlib import Path
from typing import Dict, Any, Optional
from src.tools.registry import ToolRegistry
from src.security.sandbox import resolve_in_root
from src.security.validator import CommandValidator

ALLOWED_TOOLS = {"list_files", "read_file", "write_file", "replace_text", "run_tests", "done"}

class ActionValidator:
    def __init__(self, workspace_dir: str | Path):
        self.workspace = Path(workspace_dir).resolve()

    def validate(self, tool_name: str, args: Dict[str, Any]) -> Optional[str]:
        """
        Valida a acao solicitada pelo LLM. 
        Retorna uma string com o erro se invalida, ou None se aprovada.
        """
        # 1. validacao de catalogo de ferramentas
        if tool_name not in ALLOWED_TOOLS:
            return f"Ferramenta desconhecida ou não permitida: '{tool_name}'. Ferramentas válidas: {list(ALLOWED_TOOLS)}"

        # 2. validacao especifica por ferramenta
        if tool_name in {"read_file", "write_file", "replace_text"}:
            path_arg = args.get("path") or args.get("file_path")
            if not path_arg:
                return f"A ferramenta '{tool_name}' exige o argumento de caminho ('path' ou 'file_path')."
            
                # validacao de sandbox (Directory Traversal / symlinks / null bytes)
            try:
                resolve_in_root(self.workspace, path_arg)
            except Exception as e:
                return f"Violação de sandbox no caminho '{path_arg}': {e}"

                # validacao de escrita excessiva ou vazia em write_file
            if tool_name == "write_file":
                content = args.get("content", "")
                if len(content) > 500_000:  # Limite de segurança de tamanho
                    return "Conteúdo excede o limite máximo permitido para escrita em arquivo."

        elif tool_name == "run_tests":
            cmd = args.get("command", "pytest")
                # se houver comando customizado, valida pelo CommandValidator de segurança
            if cmd and cmd != "pytest":
                try: CommandValidator.validate(cmd)
                except Exception as e:
                    return f"Comando de teste rejeitado pelo validador de segurança: {e}"

        return None