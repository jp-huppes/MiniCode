import json
from typing import Any, Dict, List
from pathlib import Path

from src.tools.implementations import list_files, read_file, write_file, append_file, run_tests, replace_text
from src.tools.base import ToolResult, safe_tool
class ToolRegistry:
    def __init__(self, workspace: str | Path = "."):
        """
        Inicializa o registro de ferramentas associando cada nome aos seus respectivos metadados, schemas de parametros em padrao JSON e a func python executavel.
        """
        
        self.workspace = Path(workspace).resolve()
        self.tools = {
            "list_files": {
                "description": "Lista arquivos e diretórios do projeto recursivamente.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Caminho relativo a partir da raiz (default '.')."},
                        "max_entries": {"type": "integer", "description": "Número máximo de entradas a listar (default 200)."}
                    },
                    "required": []
                },
                "func": list_files
            },
            "read_file": {
                "description": "Lê o conteúdo de um arquivo de texto com números de linha.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Caminho relativo do arquivo a ser lido."},
                        "start_line": {"type": "integer", "description": "Linha inicial opcional (1-indexed)."},
                        "end_line": {"type": "integer", "description": "Linha final opcional."}
                    },
                    "required": ["path"]
                },
                "func": read_file
            },
            "write_file": {
                "description": "Cria ou SUBSTITUI o arquivo inteiro com o conteúdo fornecido. Para acrescentar a um arquivo existente, use append_file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Caminho relativo do arquivo."},
                        "content": {"type": "string", "description": "Conteúdo completo a ser gravado no arquivo."}
                    },
                    "required": ["path", "content"]
                },
                "func": write_file
            },
            "run_tests": {
                "description": "Executa a suíte de testes utilizando pytest e retorna o resultado.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Caminho opcional do arquivo ou diretório de testes."},
                        "args": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Argumentos adicionais para o pytest."
                        }
                    },
                    "required": []
                },
                "func": run_tests
            },
            "replace_text": {
                "description": "Substitui um trecho exato de um arquivo. 'old_str' deve aparecer exatamente uma vez.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Caminho relativo do arquivo."},
                        "old_str": {"type": "string", "description": "Trecho exato a ser substituído (copie de read_file)."},
                        "new_str": {"type": "string", "description": "Texto que substituirá o trecho."}
                    },
                    "required": ["path", "old_str", "new_str"]
                },
                "func": replace_text
            },
            "append_file": {
                "description": "Acrescenta conteúdo ao final de um arquivo (cria o arquivo se não existir). Não apaga o conteúdo existente.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Caminho relativo do arquivo."},
                        "content": {"type": "string", "description": "Texto a ser acrescentado."}
                    },
                    "required": ["path", "content"]
                },
                "func": append_file
            },
        }

    def export_openai_tools(self) -> List[Dict[str, Any]]:
        """
        Converte e exporta as ferramentas cadastradas no formato oficial de schemas suportado por chamadas nativas de Tool Calling da API da OpenAI.
        """
        openai_tools = []
        for name, meta in self.tools.items():
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": meta["description"],
                    "parameters": meta["parameters"]
                }
            })
        return openai_tools

    def export_prompt_text(self) -> str:
        """
        Formata todas as ferramentas em um bloco textual descritivo para ser injetado diretamente no System Prompt quando operando no modo Fallback JSON
        """
        lines = []
        for name, meta in self.tools.items():
            lines.append(f"- Ferramenta: `{name}`")
            lines.append(f"  Descrição: {meta['description']}")
            lines.append(f"  Parâmetros JSON Schema: {json.dumps(meta['parameters'], ensure_ascii=False)}")
        return "\n".join(lines)

    @safe_tool
    def dispatch(self, name: str, args: Dict[str, Any]) -> ToolResult:
        """ 
        Valida se a ferramenta solicitada existe no registro e a executa com os argumentos fornecidos, injetando o caminho do workspace de forma transparente e segura
        """
        if name not in self.tools:
            validos = ", ".join(self.tools.keys())
            return ToolResult(ok=False, error=f"Ferramenta desconhecida: '{name}'. Ferramentas validas sao: {validos}.")

        tool_meta = self.tools[name]; func = tool_meta["func"]
        return func(workspace=self.workspace, **args)