import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, List
from src.tools.base import ToolResult, safe_tool
from src.tools.sandbox import resolve_path
from src.tools.truncate import truncate

IGNORE_DIRS = {".git", "__pycache__", ".venv", ".minicode", "node_modules"}

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
    """
    Cria/sobrescreve um arq no caminho especificado dentro do workspace, criando diretorio pai automaticamente caso n existam e sinalizando nos metadados que arquivos foram modificados
    """
    workspace_path = Path(workspace).resolve()
    target_path = resolve_path(workspace_path, path)

    target_path.parent.mkdir(parents=True, exist_ok=True)

    with open(target_path, "w", encoding="utf-8") as f: f.write(content)

    return ToolResult(
        ok=True,
        output=f"Arquivo '{path}' gravado com sucesso ({len(content)} caracteres).",
        meta={"modifies_files": True}
    )

@safe_tool
def run_tests(workspace: str | Path, path: str = ".", args: Optional[List[str]] = None) -> ToolResult:
    """
    Executa a suite de testes (pytest) via subprocesso no diretorio do workspace, capturando stdout/stderr, aplicando um timeout de seguranca e retornando o resultado formatado juntamente com a flag nos metadados informando se todos os testes passaram
    """

    workspace_path = Path(workspace).resolve()
    target_path = resolve_path(workspace_path, path)

    cmd = ["pytest", str(target_path), "-q"]
    if args:
        cmd.extend(args)

    try:
        result = subprocess.run(
            cmd,
            cwd=str(workspace_path),
            capture_output=True,
            text=True,
            timeout=60
        )
        
        combined_output = f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        truncated_output = truncate(combined_output, max_chars=4000)
        
        passed = (result.returncode == 0)
        
        return ToolResult(
            ok=True,
            output=truncated_output,
            error=None if passed else "Testes falharam.",
            meta={"tests_passed": passed, "exit_code": result.returncode}
        )
    except subprocess.TimeoutExpired:
        return ToolResult(
            ok=False,
            error="Timeout excedido (60s) ao executar os testes.",
            meta={"tests_passed": False}
        )