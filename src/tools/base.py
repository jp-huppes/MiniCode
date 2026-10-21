import functools
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from subprocess import TimeoutExpired

logger = logging.getLogger(__name__)

@dataclass
class ToolResult:
    ok: bool
    output: str = ""
    error: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

def safe_tool(func):
    """
    Decorador que protege a execucao da ferramenta. 
    Captura excecoes comuns (timeout, fileNotFoundError, permissionError, sandboxError) e as converte em um ToolResult com ok=False, evitando que o agente quebre.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            if isinstance(result, ToolResult): return result
            
            return ToolResult(ok=True, output=str(result))
        
        except TimeoutExpired: return ToolResult(ok=False, error="Tempo limite excedido (timeout) ao executar a ferramenta.")
        except FileNotFoundError as e: return ToolResult(ok=False, error=f"Arquivo ou diretório não encontrado: {e}. Verifique o caminho ou use list_files.")
        except PermissionError as e: return ToolResult(ok=False, error=f"Permissão negada ao acessar o recurso: {e}.")
        except Exception as e:
            logger.exception(f"Erro inesperado na ferramenta {func.__name__}: {e}")
            return ToolResult(ok=False, error=f"{type(e).__name__}: {e}")

    return wrapper