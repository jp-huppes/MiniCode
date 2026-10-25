import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

class RunState:
    def __init__(self, run_id: str, task: str, run_dir: Path):
        self.run_id = run_id
        self.task = task
        self.run_dir = Path(run_dir)
        self.state_path = self.run_dir / "state.json"
        
        self.started_at_wall = datetime.now(timezone.utc).isoformat()
        self.started_at_mono = time.monotonic()
        
        # estrutura completa exigida pelo schema
        self.data: Dict[str, Any] = {
            "run_id": run_id,
            "task": task,
            "final_state": None,
            "failure_reason": None,
            "files_changed": [],
            "tests_run": [],
            "repair_attempts": 0,
            "passed_without_repair": True,
            "llm_calls": 0,
            "llm_retries": 0,
            "tokens": {
                "prompt": 0,
                "completion": 0,
                "total": 0
            },
            "elapsed_s": 0.0,
            "started_at": self.started_at_wall,
            "ended_at": None,
            "transitions": []
        }
        self.persist()

    def record_transition(self, from_state: str, to_state: str):
        timestamp = datetime.now(timezone.utc).isoformat()
        self.data["transitions"].append((from_state, to_state, timestamp))
        self.persist()

    def record_llm_call(self, prompt_tokens: int, completion_tokens: int, retries: int = 0):
        self.data["llm_calls"] += 1
        self.data["llm_retries"] += retries
        self.data["tokens"]["prompt"] += prompt_tokens
        self.data["tokens"]["completion"] += completion_tokens
        self.data["tokens"]["total"] = (
            self.data["tokens"]["prompt"] + self.data["tokens"]["completion"]
        )
        self.persist()

    def record_file_changed(self, file_path: str):
        if file_path not in self.data["files_changed"]:
            self.data["files_changed"].append(file_path)
        self.persist()

    def record_test_run(self, test_info: Dict[str, Any]):
        self.data["tests_run"].append(test_info)
        self.persist()

    def finalize(self, final_state: str, failure_reason: Optional[str] = None):
        self.data["final_state"] = final_state
        self.data["failure_reason"] = failure_reason
        self.data["ended_at"] = datetime.now(timezone.utc).isoformat()
        
        # Cálculo do tempo decorrido com relógio monotônico
        ended_mono = time.monotonic()
        self.data["elapsed_s"] = round(ended_mono - self.started_at_mono, 3)
        self.persist()

    def persist(self):
        # Escrita atomica: escreve no .tmp e renomeia para evitar JSON truncado em crash
        tmp_path = self.state_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")
        if self.state_path.exists():
            self.state_path.unlink()
        tmp_path.rename(self.state_path)