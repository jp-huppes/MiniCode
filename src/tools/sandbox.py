from pathlib import Path
from src.security.sandbox import resolve_in_root
from src.security.errors import SecurityError

class SandboxError(SecurityError):
    # Exceção de sandbox herdada de SecurityError
    pass

def resolve_path(workspace_dir: str | Path, relative_path: str | Path) -> Path:
    # Wrapper de compatibilidade que delega para o resolve_in_root centralizado em security.
    try: return resolve_in_root(workspace_dir, relative_path)
    except SecurityError as e: raise SandboxError(str(e))