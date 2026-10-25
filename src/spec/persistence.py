from pathlib import Path
from datetime import datetime
from src.security.sandbox import resolve_in_root

def salvar_artefato_spec(workspace_dir: str | Path, filename: str, content: str, tarefa: str = "") -> Path:
    """
    Grava specification.md ou plan.md no workspace com cabecalho de auditoria e validacao de existencia.
    """
    root = Path(workspace_dir).resolve(); path = resolve_in_root(root, filename)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cabecalho = f"<!-- Gerado por MiniCode | Timestamp: {timestamp} | Tarefa: {tarefa} -->\n\n"
    
    conteudo_completo = cabecalho + content
    path.write_text(conteudo_completo, encoding="utf-8")

    # confirmacao de existencia e integridade
    if not path.is_file() or path.stat().st_size == 0:
        raise IOError(f"Falha crítica ao gravar ou confirmar integridade do arquivo: {filename}")

    return path