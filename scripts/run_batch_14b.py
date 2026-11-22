from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def carregar_env(env_path: Path) -> None:
    if not env_path.exists():
        return
    for linha in env_path.read_text(encoding="utf-8").splitlines():
        s = linha.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        if s.startswith("export "):
            s = s[len("export ") :].strip()
        chave, valor = s.split("=", 1)
        chave = chave.strip()
        valor = valor.strip().strip('"').strip("'")
        if chave:
            os.environ[chave] = valor


def main() -> int:
    carregar_env(Path(".env"))

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "coleta_completa.log"

    cmd = [
        sys.executable,
        "-m",
        "scripts.collect_corpus",
        "--tasks-dir",
        "corpus_tasks",
        "--out-dir",
        "corpus_raw",
        "--model",
        "Qwen/Qwen2.5-14B-Instruct-AWQ",
        "--seeds",
        "1",
        "2",
        "3",
        "--temps",
        "0.0",
        "0.3",
    ]

    with log_file.open("w", encoding="utf-8") as f:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            f.write(line)
            f.flush()
        rc = proc.wait()

    print(f"\nColeta finalizada com c?digo: {rc} | Log: {log_file}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
