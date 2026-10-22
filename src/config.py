import os

# Limites anti-loop e de execução do MiniCode
MAX_ITERATIONS: int = int(os.getenv("MINICODE_MAX_ITERATIONS", 25))
MAX_REPAIR_ATTEMPTS: int = int(os.getenv("MINICODE_MAX_REPAIR_ATTEMPTS", 3))
COMMAND_TIMEOUT: int = int(os.getenv("MINICODE_COMMAND_TIMEOUT", 60))  # segundos