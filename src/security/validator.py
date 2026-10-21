import shlex
import re
from pathlib import Path
from typing import List, Tuple
from src.security.errors import SecurityError

DENY_PATTERNS = [
    r"\b(sudo|su|shutdown|reboot|halt|mkfs|dd|fdisk|mount)\b",
    r"\brm\s+-[rf].*[/~*]",
    r"(curl|wget)\s+.*\|\s*(sh|bash)",
    r">/dev/sd",
    r"chmod\s+-R\s+777",
]

ALLOWED_EXECUTABLES = {"pytest", "python", "git", "ls", "cat", "echo"}

class CommandValidator:
    @staticmethod
    def _split_compound_commands(cmd: str) -> List[str]:
        # divide comandos compostos por ;, &&, ||, |, quebras de linha e substituições $(...) ou crases
        if any(char in cmd for char in ["$(", "`", "\n"]):
            raise SecurityError("Comandos compostos complexos, subshells ou crases não são permitidos.")
        
        # divide por ;, &&, ||, |
        segments = re.split(r";|&&|\|\||\|", cmd)
        return [seg.strip() for seg in segments if seg.strip()]

    @classmethod
    def validate(cls, cmd: str) -> List[str]:
        """
        Valida o comando de forma estrita. 
        Retorna a lista de tokens segura se aprovado, ou levanta SecurityError se violar as politicas
        """
        if not cmd or not cmd.strip(): raise SecurityError("Comando vazio não é permitido.")

        # 1. checagem de denylist em nivel de string crua
        for pattern in DENY_PATTERNS:
            if re.search(pattern, cmd, re.IGNORECASE):
                raise SecurityError(f"Comando rejeitado por política de segurança (denylist): '{cmd}'")

        # 2. divisao e analise de comandos compostos
        segments = cls._split_compound_commands(cmd)
        
        all_tokens = []
        for seg in segments:
            try: tokens = shlex.split(seg)
            except Exception as e: raise SecurityError(f"Erro ao analisar a sintaxe do comando '{seg}': {e}")
            if not tokens: continue

            exe = Path(tokens[0]).name.lower()

            # 3. verificacao de allowlist
            if exe not in ALLOWED_EXECUTABLES:
                raise SecurityError(f"Executável não permitido: '{exe}'. Permitidos: {list(ALLOWED_EXECUTABLES)}")

            # 4. restricoes especificas por subcomando
            if exe == "git":
                allowed_git_sub = {"diff", "status", "log"}
                if len(tokens) > 1 and tokens[1] not in allowed_git_sub:
                    raise SecurityError(f"Subcomando git não permitido: '{tokens[1]}'. Permitidos: {allowed_git_sub}")
            
            if exe == "python":
                # Permite apenas -m pytest ou execuções controladas
                if len(tokens) > 1 and tokens[1] == "-m" and len(tokens) > 2 and tokens[2] == "pytest": pass
                
                # ou scripts simples (sem -c perigoso)
                elif any(arg in tokens for arg in ["-c", "-exec"]):
                    raise SecurityError("Execução arbitrária de código python (-c) não é permitida.")

            all_tokens.extend(tokens)

        return all_tokens