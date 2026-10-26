import os
import sys
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, List
from src.tools.base import ToolResult, safe_tool
from src.tools.sandbox import resolve_path
from src.tools.truncate import truncate
from src.config import COMMAND_TIMEOUT

IGNORE_DIRS = {".git", "__pycache__", ".venv", ".minicode", "node_modules"}

def _apagar_bytecode(target_path: Path) -> None:
    # evita .pyc stale quando duas escritas caem no mesmo segundo com o mesmo tamanho
    for pyc in target_path.parent.glob(f"__pycache__/{target_path.stem}.*.pyc"):
        try:
            pyc.unlink()
        except OSError:
            pass

@safe_tool
def list_files(workspace: str | Path, path: str = ".", max_entries: int = 200) -> ToolResult:
    """
    Percorre recursivamente o diretorio especificado dentro do workspace e gera uma lista formatada de arquivos e pastas, ignorando diretorios ocultos/temp e respeitando o limite maximo de entradas informado
    """

    workspace_path = Path(workspace).resolve()
    target_path = resolve_path(workspace_path, path)
    
    if not target_path.exists(): return ToolResult(ok=False, error=f"Caminho não encontrado: {path}")

    entries = []
    count = 0
    
    if target_path.is_file(): return ToolResult(ok=True, output=target_path.relative_to(workspace_path).as_posix())

    for root, dirs, files in os.walk(target_path):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        
        root_p = Path(root)
        rel_root = root_p.relative_to(workspace_path)
        
        for d in dirs:
            if count >= max_entries: break
            rel_d = rel_root / d if rel_root != Path(".") else Path(d)
            entries.append(f"{rel_d.as_posix()}/")
            count += 1
            
        for f in files:
            if count >= max_entries: break
            rel_f = rel_root / f if rel_root != Path(".") else Path(f)
            entries.append(rel_f.as_posix())
            count += 1
            
        if count >= max_entries: break

    output = "\n".join(entries)
    if count >= max_entries:
        output += f"\n\n... [Aviso: Listagem limitada a {max_entries} entradas] ..."
        
    return ToolResult(ok=True, output=output)

@safe_tool
def read_file(workspace: str | Path, path: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> ToolResult:
    """
    Le o conteudo do arquivo indicado no workspace, valida se eh de texto, aplica a numeracao de linhas e permite a filtragem por intervalo de linhas (start_line ate end_line), aplicando truncamento de output caso o conteudo seja grande
    """

    workspace_path = Path(workspace).resolve()
    target_path = resolve_path(workspace_path, path)

    if not target_path.exists() or not target_path.is_file(): return ToolResult(ok=False, error=f"Arquivo não encontrado ou inválido: {path}")

    try:
        with open(target_path, "r", encoding="utf-8") as f: lines = f.readlines()
    except UnicodeDecodeError:
        return ToolResult(ok=False, error="O arquivo parece ser binário e não pode ser lido como texto.")

    total_lines = len(lines)
    
    start = max(1, start_line) if start_line is not None else 1
    end = min(total_lines, end_line) if end_line is not None else total_lines

    if start > total_lines: return ToolResult(ok=False, error=f"start_line ({start_line}) excede o total de linhas do arquivo ({total_lines}).")

    selected_lines = lines[start - 1 : end]
    formatted_lines = [f"{i + start:4d} | {line}" for i, line in enumerate(selected_lines)]
    content = "".join(formatted_lines)
    
    content_truncated = truncate(content, max_chars=4000)
    
    return ToolResult(
        ok=True, 
        output=content_truncated,
        meta={"total_lines": total_lines, "displayed_range": [start, end]}
    )

@safe_tool
def write_file(workspace: str | Path, path: str, content: str) -> ToolResult:
    workspace_path = Path(workspace).resolve()
    target_path = resolve_path(workspace_path, path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(content)
    _apagar_bytecode(target_path)
    return ToolResult(
        ok=True,
        output=f"Arquivo '{path}' gravado com sucesso ({len(content)} caracteres).",
        meta={"modifies_files": True},
    )

@safe_tool
def replace_text(workspace: str | Path, path: str, old_str: str, new_str: str) -> ToolResult:
    workspace_path = Path(workspace).resolve()
    target_path = resolve_path(workspace_path, path)
    if not target_path.is_file():
        return ToolResult(ok=False, error=f"Arquivo não encontrado: {path}. Use list_files para verificar o caminho.")
    if not old_str:
        return ToolResult(ok=False, error="'old_str' não pode ser vazio.")

    original = target_path.read_text(encoding="utf-8")
    ocorrencias = original.count(old_str)
    if ocorrencias == 0:
        return ToolResult(ok=False, error="'old_str' não foi encontrado no arquivo. Copie o trecho exato com read_file.")
    if ocorrencias > 1:
        return ToolResult(ok=False, error=f"'old_str' aparece {ocorrencias} vezes. Inclua mais contexto para torná-lo único.")

    target_path.write_text(original.replace(old_str, new_str, 1), encoding="utf-8")
    _apagar_bytecode(target_path)
    return ToolResult(ok=True, output=f"Substituição aplicada em '{path}'.", meta={"modifies_files": True})



@safe_tool
def run_tests(workspace: str | Path, path: str = ".", args: Optional[List[str]] = None, command: Optional[str] = None) -> ToolResult:
    workspace_path = Path(workspace).resolve()

    if command:
        cmd = command.split()
        if cmd and cmd[0] == "pytest":
            cmd = [sys.executable, "-m", "pytest", *cmd[1:]]
    else:
        target_path = resolve_path(workspace_path, path)
        cmd = [sys.executable, "-m", "pytest", str(target_path), "-q"]
        if args:
            cmd.extend(args)

    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}

    try:
        result = subprocess.run(
            cmd,
            cwd=str(workspace_path),
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT,
            env=env,
        )
        combined_output = f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        truncated_output = truncate(combined_output, max_chars=4000)
        passed = (result.returncode == 0)
        return ToolResult(
            ok=True,
            output=truncated_output,
            error=None if passed else "Testes falharam.",
            meta={"tests_passed": passed, "exit_code": result.returncode},
        )
    except subprocess.TimeoutExpired:
        return ToolResult(
            ok=False,
            error=f"Timeout excedido ({COMMAND_TIMEOUT}s) ao executar os testes.",
            meta={"tests_passed": False, "exit_code": -1},
        )

@safe_tool
def append_file(workspace: str | Path, path: str, content: str) -> ToolResult:
    workspace_path = Path(workspace).resolve()
    target_path = resolve_path(workspace_path, path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    existente = target_path.read_text(encoding="utf-8") if target_path.is_file() else ""
    # garante quebra de linha entre o conteúdo antigo e o novo
    separador = "" if (not existente or existente.endswith("\n") or content.startswith("\n")) else "\n"

    with open(target_path, "a", encoding="utf-8") as f:
        f.write(separador + content)
    _apagar_bytecode(target_path)

    total_linhas = len((existente + separador + content).splitlines())
    return ToolResult(
        ok=True,
        output=f"Conteúdo anexado a '{path}'. O arquivo agora tem {total_linhas} linhas.",
        meta={"modifies_files": True},
    )