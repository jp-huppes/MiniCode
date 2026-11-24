import os

# Limites anti-loop e de execução do MiniCode
MAX_ITERATIONS: int = int(os.getenv("MINICODE_MAX_ITERATIONS", 25))
MAX_REPAIR_ATTEMPTS: int = int(os.getenv("MINICODE_MAX_REPAIR_ATTEMPTS", 3))
COMMAND_TIMEOUT: int = int(os.getenv("MINICODE_COMMAND_TIMEOUT", 60))  # segundos

# config de janela de contexto e tokens
MAX_MODEL_LEN: int = int(os.getenv("MINICODE_MAX_MODEL_LEN", 8192))
T_RESP_RESERVE: int = int(os.getenv("MINICODE_T_RESP_RESERVE", 1024))
MAX_PARSE_FAILURES: int = int(os.getenv("MINICODE_MAX_PARSE_FAILURES", 3))