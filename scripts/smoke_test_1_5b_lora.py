"""Run de fumaca (Passo 3.e): executa 1 tarefa da amostra no AgentOrchestrator com 1.5B + LoRA in-process."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.agent.orchestrator import AgentOrchestrator
from src.llm.action_contract import interpretar_acao
from src.llm.client import LLMClient
from src.llm.local_peft_handler import (
    DEFAULT_ADAPTER_DIR,
    DEFAULT_BASE_MODEL_ID,
    DEFAULT_LOCAL_TOOL_MAX_TOKENS,
    register_local_peft_handler,
)


def carregar_tarefa_amostra(tasks_file: Path, task_id: str | None = None) -> tuple[str, str]:
    linhas = [l.strip() for l in tasks_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    tarefas = [json.loads(l) for l in linhas]
    if task_id:
        for t in tarefas:
            if t["task_id"] == task_id:
                return t["task_id"], t["task"]
        raise ValueError(f"Tarefa '{task_id}' nao encontrada em {tasks_file}.")
    return tarefas[0]["task_id"], tarefas[0]["task"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test do 1.5B+LoRA no AgentOrchestrator")
    parser.add_argument("--tasks-file", type=Path, default=ROOT_DIR / "scripts" / "tasks_eval.jsonl")
    parser.add_argument("--task-id", type=str, default="t1_soma")
    parser.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL_ID)
    parser.add_argument("--adapter-dir", type=Path, default=ROOT_DIR / DEFAULT_ADAPTER_DIR)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_LOCAL_TOOL_MAX_TOKENS)
    parser.add_argument("--work-dir", type=Path, default=ROOT_DIR / "eval_runs" / "smoke_step3")
    args = parser.parse_args()

    task_id, prompt = carregar_tarefa_amostra(args.tasks_file, args.task_id)
    if args.work_dir.exists():
        shutil.rmtree(args.work_dir)
    workspace = args.work_dir / "workspace"
    state_dir = args.work_dir / ".minicode"
    (workspace / "src").mkdir(parents=True, exist_ok=True)
    (workspace / "tests").mkdir(parents=True, exist_ok=True)
    (workspace / "src" / "__init__.py").write_text("", encoding="utf-8")
    (workspace / "tests" / "test_base.py").write_text("def test_ok(): assert True\n", encoding="utf-8")

    # Cliente configurado com endpoint placeholder (o ROLE_TOOL_SELECTION roda 100% in-process na GPU)
    llm_client = LLMClient(
        api_key="local-in-process",
        base_url="[http://127.0.0.1:0/v1](http://127.0.0.1:0/v1)",
        model_name=args.base_model,
        constrained_mode="none",
        default_temperature=0.0,
    )
    handler = register_local_peft_handler(
        llm_client,
        base_model_id=args.base_model,
        adapter_dir=args.adapter_dir,
        max_tokens=args.max_tokens,
        use_adapter=True,
    )

    # Sanity check rapido da alternancia Base (disable_adapter) vs LoRA antes do loop
    probe_msg = [{"role": "user", "content": f"Tarefa: {prompt}\nPrimeiro passo?"}]
    handler.set_use_adapter(False)
    probe_base = handler(probe_msg)
    handler.set_use_adapter(True)
    probe_lora = handler(probe_msg)

    t0 = time.perf_counter()
    orch = AgentOrchestrator(
        workspace_dir=workspace,
        minicode_dir=state_dir,
        llm_client=llm_client,
        temperature=0.0,
        catalogo_enxuto=True,
    )
    final_state = orch.run(prompt, f"Plano para {task_id}")
    elapsed_s = time.perf_counter() - t0

    # Le o historico de acoes gravado pelo executor
    log_file = state_dir / "action_log.jsonl"
    eventos = []
    if log_file.exists():
        eventos = [json.loads(l) for l in log_file.read_text(encoding="utf-8").splitlines() if l.strip()]

    resumo = {
        "task_id": task_id,
        "variant": handler.active_variant,
        "max_tokens": handler.max_tokens,
        "final_state": final_state.value if hasattr(final_state, "value") else str(final_state),
        "failure_reason": getattr(orch.state_data, "failure_reason", "") or "",
        "parse_failures": getattr(orch.state_data, "parse_failures", 0),
        "iterations": getattr(orch.state_data, "iteration", 0),
        "elapsed_seconds": round(elapsed_s, 2),
        "probe_base_json_ok": interpretar_acao(probe_base.content or "", probe_base.finish_reason).ok,
        "probe_lora_json_ok": interpretar_acao(probe_lora.content or "", probe_lora.finish_reason).ok,
        "probe_lora_preview": (probe_lora.content or "")[:240],
        "tools_executed": [e.get("tool") for e in eventos],
        "metrics_calls": len(llm_client.metrics.calls),
    }
    print(json.dumps(resumo, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()