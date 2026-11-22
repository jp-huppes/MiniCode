import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Set

from src.config import COMMAND_TIMEOUT

REGEX_TESTE_AC = re.compile(r"test_ac_\d+_[A-Za-z0-9_]+")


def obter_testes_exigidos(spec_path: Path) -> List[str]:
    # le a secao 
        # tests' da SPEC e devolve os nomes test_ac_n_* exigidos
    if not spec_path.is_file(): return []
    
    texto = spec_path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"^## Tests\s*$(.*?)(?=^## |\Z)", texto, re.DOTALL | re.MULTILINE)
    if not m: return []
    
    return sorted(set(REGEX_TESTE_AC.findall(m.group(1))))


def coletar_testes_existentes(workspace: Path) -> Set[str]:
    # coleta via run_tests para herdar o isolamento (rootdir, ini vazio, sys.path); -k reduz a saída ao necessário
    from src.tools.registry import ToolRegistry
    res = ToolRegistry(Path(workspace)).dispatch("run_tests", {"command": "pytest --collect-only -q -k test_ac_"})
    nomes: Set[str] = set()
    for linha in (res.output or "").splitlines():
        linha = linha.strip()
        if "::" in linha: nomes.add(linha.rsplit("::", 1)[1])
    return nomes


def testes_ausentes(workspace: Path) -> List[str]:
    # exigidos pela SPEC que n foram coletados pelo pytest
    exigidos = obter_testes_exigidos(Path(workspace) / "specification.md")
    if not exigidos: return []
    existentes = coletar_testes_existentes(Path(workspace))
    return [t for t in exigidos if t not in existentes]