import os
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

def _get_utc_timestamp() -> str:
    # Formato sem ':' para total compatibilidade com Windows
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")

class RunLogger:
    def __init__(self, workspace_root: Path, task: str):
        self.workspace_root = Path(workspace_root).resolve()
        self.minicode_dir = self.workspace_root / ".minicode"
        self.runs_dir = self.minicode_dir / "runs"
        
        # criacao do ID unico e diretório da run com tratamento de colisao
        base_run_id = _get_utc_timestamp()
        self.run_id = base_run_id
        self.run_dir = self.runs_dir / self.run_id
        
        counter = 2
        while self.run_dir.exists():
            self.run_id = f"{base_run_id}-{counter}"
            self.run_dir = self.runs_dir / self.run_id
            counter += 1
            
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # atualiza atalho 'latest'
        latest_file = self.minicode_dir / "latest"
        latest_file.write_text(self.run_id, encoding="utf-8")
        
        # arquivo de actions.jsonl
        self.actions_path = self.run_dir / "actions.jsonl"
        self._actions_file = open(self.actions_path, "a", encoding="utf-8")
        self._seq_counter = 0

    def log_action(
        self,
        state: str,
        tool: str,
        args: Dict[str, Any],
        result: Any = None,
        error: Optional[str] = None,
        duration_ms: int = 0,
        schema_version: int = 1
    ):
        self._seq_counter += 1
        
        # sanitizacao de args para write_file (evita inchar logs com codigos gigantes)
        sanitized_args = dict(args)
        if tool in {"write_file", "create_file"} and "content" in sanitized_args:
            content = sanitized_args["content"]
            import hashlib
            content_bytes = content.encode("utf-8")
            sanitized_args["content"] = {
                "path": sanitized_args.get("path"),
                "size_bytes": len(content_bytes),
                "sha256": hashlib.sha256(content_bytes).hexdigest()
            }

        # truncamento de resultado se exceder 2000 caracteres
        result_str = str(result) if result is not None else None
        result_len = len(result_str) if result_str else 0
        if result_str and result_len > 2000:
            result_str = result_str[:2000] + f"…[truncado: {result_len} chars]"

        entry = {
            "schema_version": schema_version,
            "seq": self._seq_counter,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "state": state,
            "tool": tool,
            "args": sanitized_args,
            "result": result_str,
            "result_len": result_len,
            "error": error,
            "duration_ms": duration_ms
        }

        self._actions_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._actions_file.flush() # flush imediato para garantir sobrevivencia a crashes

    def save_artifact(self, filename: str, content: str):
        artifact_path = self.run_dir / filename
        artifact_path.write_text(content, encoding="utf-8")

    def close(self):
        if self._actions_file and not self._actions_file.closed:
            self._actions_file.close()