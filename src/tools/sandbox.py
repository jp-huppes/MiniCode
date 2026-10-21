from pathlib import Path
from src.tools.base import ToolResult

class SandboxError(Exception):
    # excecao levantada quando ha tentativa de acesso fora do workspace
    pass

def resolve_path(workspace_dir: str | Path, relative_path: str | Path) -> Path:
    """
    resolve um caminho em relacao ao workspace e valida se permanece estritamente dentro dele.
    bloqueia caminhos absolutos maliciosos, symlinks de fuga e travessia com '../'.
    """
    workspace = Path(workspace_dir).resolve()

    target = Path(relative_path) # converte para path absoluto alvo
    if not target.is_absolute(): target = (workspace / target)
        
    target_resolved = target.resolve()

    # valida se o caminho resolvido esta no workspace
    try: target_resolved.relative_to(workspace)
    except ValueError: raise SandboxError(f"Acesso negado: o caminho '{relative_path}' foge dos limites do workspace.")

    return target_resolved