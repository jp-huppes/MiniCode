from pathlib import Path
from src.security.errors import SecurityError

def resolve_in_root(root: str | Path, path: str | Path) -> Path:
    """
    Resolve o caminho em relacao a raiz do projeto.
    Bloqueia tentativas de directory traversal, caminhos absolutos maliciosos e caracteres nulos
    """

    path_str = str(path)
    if "\0" in path_str: raise SecurityError("Caminho inválido: contém caractere nulo (\\0).")

    root_path = Path(root).resolve()
    
    # constroi o caminho alvo (se absoluto, path resolve de forma absoluta)
    target = Path(path_str)
    if not target.is_absolute(): target = (root_path / target)
        
    resolved_target = target.resolve()

    # validacao usando is_relative_to (evita prefix matching errado como startswith)
    try:
        resolved_target.relative_to(root_path)
    except ValueError:
        raise SecurityError(f"Acesso negado: o caminho '{path}' foge dos limites do workspace.")

    return resolved_target