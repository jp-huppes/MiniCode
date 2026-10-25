import os
from pathlib import Path
from typing import List
from src.security.sandbox import resolve_in_root

IGNORED_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".minicode", ".pytest_cache"}
IGNORED_EXTENSIONS = {".pyc", ".png", ".jpg", ".jpeg", ".pdf", ".zip", ".exe", ".lock"}

def coletar_contexto(workspace_dir: str | Path, max_chars: int = 12000) -> str:
    """
    Coleta a arvore de arquivos e le arquivos relevantes do projeto respeitando um orcamento maximo de caracteres.
    """

    root = Path(workspace_dir).resolve()
    arvore_linhas = []
    arquivos_candidatos = []

        #   Listar arquivos ignorando ruídos
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS]
        rel_dir = Path(dirpath).relative_to(root)
        for f in filenames:
            if any(f.endswith(ext) for ext in IGNORED_EXTENSIONS): continue
            rel_path = rel_dir / f if rel_dir != Path(".") else Path(f)
            arvore_linhas.append(str(rel_path))
            arquivos_candidatos.append(rel_path)

    arvore_resumida = "Árvore de arquivos do projeto:\n" + "\n".join(f"- {p}" for p in arvore_linhas[:100])

        # priorização por heurística (README, pyproject, requirements, testes, código fonte)
    def prioridade(path: Path) -> int:
        name = path.name.lower()
        if "readme" in name or name in {"pyproject.toml", "requirements.txt"}: return 0
        if "test" in str(path).lower(): return 1
        if path.suffix == ".py": return 2
        return 3
    arquivos_candidatos.sort(key=prioridade)

        # orcamento de tamanho e leitura truncada
    bloco_conteudo = "\n\nConteudo de arquivos relevantes:\n"
    chars_atuais = len(arvore_resumida) + len(bloco_conteudo)

    for rel_path in arquivos_candidatos:
        if chars_atuais >= max_chars:
            bloco_conteudo += "\n[orcamento de contexto esgotado - arquivos restantes omitidos]"
            break
        
        try:
            full_path = resolve_in_root(root, rel_path)
            if full_path.is_file():
                content = full_path.read_text(encoding="utf-8", errors="ignore")
                    # truncamento por arquivo caso seja muito grande
                if len(content) > 2000: content = content[:1000] + "\n... [truncado] ...\n" + content[-1000:]
                
                snippet = f"\n--- ARQUIVO: {rel_path} ---\n{content}\n"
                if chars_atuais + len(snippet) > max_chars: break
                bloco_conteudo += snippet
                chars_atuais += len(snippet)
        except Exception:  continue

    return f"{arvore_resumida}\n{bloco_conteudo}"