import argparse
import json
import shutil
import uuid
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from src.agent.orchestrator import AgentOrchestrator
from src.agent.state import AgentStateMachine
from src.llm.client import LLMClient
from src.llm.routing import AgentRoutingConfig, attach_routing_to_client
from src.spec.pipeline import executar_spec_pipeline


def preparar_workspace(task_dir: Path, run_ws: Path) -> None:
    if run_ws.exists():
        shutil.rmtree(run_ws)
    run_ws.mkdir(parents=True, exist_ok=True)
    # Copia arquivos base da tarefa (ex.: src/ inicial, tests/, etc.) se existirem
    base_ws = task_dir / "workspace"
    if base_ws.exists():
        shutil.copytree(base_ws, run_ws, dirs_exist_ok=True)
    else:
        (run_ws / "src").mkdir(parents=True, exist_ok=True)
        tests_src = task_dir / "tests"
        if tests_src.exists():
            shutil.copytree(tests_src, run_ws / "tests", dirs_exist_ok=True)
        else:
            (run_ws / "tests").mkdir(parents=True, exist_ok=True)


def carregar_tarefas(tasks_root: Path, limit: int | None = None) -> List[Dict[str, Any]]:
    tarefas: List[Dict[str, Any]] = []
    for p in sorted(tasks_root.iterdir()):
        if not p.is_dir():
            continue
        meta_path = p / "task.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        tarefas.append({
            "task_id": meta.get("task_id", p.name),
            "prompt": meta["prompt"],
            "dir": p,
        })
        if limit is not None and len(tarefas) >= limit:
            break
    return tarefas


def main() -> None:

    env_file = Path(".env")
    if env_file.exists():
        for linha in env_file.read_text(encoding="utf-8").splitlines():
            linha = linha.strip()
            if not linha or linha.startswith("#") or "=" not in linha:
                continue
            chave, valor = linha.split("=", 1)
            chave = chave.strip()
            valor = valor.strip().strip('"').strip("'")
            os.environ.setdefault(chave, valor)

    parser = argparse.ArgumentParser(description="Coleta de corpus do professor (Qwen-14B) - Etapa 3")
    parser.add_argument("--tasks-dir", default="corpus_tasks", help="Diretório com as tarefas (60-100)")
    parser.add_argument("--out-dir", default="corpus_raw", help="Diretório de saída dos logs brutos")
    parser.add_argument("--model", required=True, help="Nome/ID do modelo Qwen-14B")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--temps", nargs="+", type=float, default=[0.0, 0.3])
    parser.add_argument("--limit-tasks", type=int, default=None)
    args = parser.parse_args()

    tasks_root = Path(args.tasks_dir)
    out_root = Path(args.out_dir)
    runs_dir = out_root / "workspaces"
    runs_dir.mkdir(parents=True, exist_ok=True)
    runs_meta_path = out_root / "runs_metadata.jsonl"

    tarefas = carregar_tarefas(tasks_root, limit=args.limit_tasks)
    if not tarefas:
        raise RuntimeError(f"Nenhuma tarefa encontrada em '{tasks_root}'.")

    for tarefa in tarefas:
        task_id = tarefa["task_id"]
        prompt_tarefa = tarefa["prompt"]
        for seed in args.seeds:
            for temp in args.temps:
                run_id = uuid.uuid4().hex[:12]
                run_base = runs_dir / task_id / f"seed_{seed}_t_{temp}_{run_id}"
                ws_dir = run_base / "workspace"
                mc_dir = run_base / ".minicode"

                preparar_workspace(tarefa["dir"], ws_dir)

                os.environ["MINICODE_TEMPERATURE"] = str(temp)
                llm = LLMClient(model_name=args.model)
                attach_routing_to_client(
                    llm,
                    AgentRoutingConfig(
                        teacher_model=args.model,
                        tool_model=args.model,
                        spec_model=args.model,
                        plan_model=args.model,
                        repair_model=args.model,
                    ),
                )
                sm = AgentStateMachine()

                status = "ok"
                error_msg = None
                try:
                    spec, plano_raw = executar_spec_pipeline(prompt_tarefa, ws_dir, llm, sm)
                    orch = AgentOrchestrator(ws_dir, mc_dir, llm, state_machine=sm, run_id=run_id)
                    orch.state_machine = sm
                    orch.state_data.state = sm.state.value
                    final_state = orch.run(spec.raw_content, plano_raw)
                    repair_attempts = orch.state_machine.repair_attempts
                except Exception as exc:
                    final_state = sm.state.value
                    repair_attempts = sm.repair_attempts
                    status = "error"
                    error_msg = str(exc)

                run_record = {
                    "timestamp": datetime.now().isoformat(),
                    "run_id": run_id,
                    "task_id": task_id,
                    "seed": seed,
                    "temperature": temp,
                    "final_state": final_state if isinstance(final_state, str) else final_state.value,
                    "repair_attempts": repair_attempts,
                    "status": status,
                    "error": error_msg,
                    "actions_log": str(mc_dir / "actions.jsonl"),
                }
                with open(runs_meta_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(run_record, ensure_ascii=False) + "\n")

                print(
                    f"[{task_id}] seed={seed} temp={temp} run_id={run_id} "
                    f"-> state={run_record['final_state']} repairs={repair_attempts} status={status}"
                )


if __name__ == "__main__":
    main()