import ast, builtins
from pathlib import Path
from typing import Optional

def verificar_nomes_indefinidos_em_teste(workspace: Path, path: str, content: str, existente: str = "") -> Optional[str]:
    p = path.replace("\\", "/")
    nome = p.rsplit("/", 1)[-1]
    if not (p.startswith("tests/") and nome.startswith("test_") and nome.endswith(".py")):
        return None
    try:
        arvore = ast.parse(existente + content)
    except SyntaxError as e:
        return f"'{path}' tem erro de sintaxe (linha {e.lineno}): {e.msg}. Corrija antes de gravar."

    ligados = set(dir(builtins)) | {"__file__", "__name__"}
    for n in ast.walk(arvore):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                ligados.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            ligados.add(n.name)
        elif isinstance(n, ast.arg):
            ligados.add(n.arg)
        elif isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            ligados.add(n.id)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            ligados.add(n.name)

    indef = sorted({n.id for n in ast.walk(arvore)
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in ligados})
    if not indef:
        return None
    sug = [f"'{m}': use `from src import {m}` ou `from src.{m} import <funções>`"
           if (Path(workspace) / "src" / f"{m}.py").is_file() else f"'{m}': importe ou defina antes de usar"
           for m in indef]
    return (f"'{path}' usa nome(s) sem import/definição: {', '.join(indef)}. "
            "Adicione os imports no topo do arquivo. " + "; ".join(sug) + ".")