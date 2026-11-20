from pathlib import Path
from typing import Any, Callable, Dict, List

from src.tools.base import ToolResult, safe_tool
from src.tools.catalog import (
    CATALOG_BY_NAME,
    export_openai_tools as catalog_export_openai_tools,
    export_prompt_text as catalog_export_prompt_text,
)
from src.tools.implementations import (
    append_file,
    list_files,
    read_file,
    replace_text,
    run_tests,
    write_file,
)

TOOL_FUNCTIONS: Dict[str, Callable[..., Any]] = {"list_files": list_files, "read_file": read_file, "write_file": write_file, "append_file": append_file, "replace_text": replace_text, "run_tests": run_tests,}


class ToolRegistry:
    def __init__(self, workspace: str | Path = "."):
        self.workspace = Path(workspace).resolve()
        self.tools: Dict[str, Dict[str, Any]] = {}
        for name, spec in CATALOG_BY_NAME.items():
            if name in TOOL_FUNCTIONS:
                schema = spec.args_model.model_json_schema()
                schema.pop("title", None)
                self.tools[name] = {
                    "description": spec.descricao,
                    "parameters": schema,
                    "escreve_arquivos": spec.escreve_arquivos,
                    "so_harness": spec.so_harness,
                    "func": TOOL_FUNCTIONS[name],}

    def export_openai_tools(self) -> List[Dict[str, Any]]:
        return catalog_export_openai_tools(include_harness_only=True)

    def export_prompt_text(self) -> str:
        return catalog_export_prompt_text(include_harness_only=True)

    @safe_tool
    def dispatch(self, name: str, args: Dict[str, Any]) -> ToolResult:
        if name not in self.tools:
            validos = ", ".join(self.tools.keys())
            return ToolResult(
                ok=False,
                error=f"Ferramenta desconhecida: '{name}'. Ferramentas validas sao: {validos}.",
            )

        func = self.tools[name]["func"]
        return func(workspace=self.workspace, **args)