import argparse
import json
import shutil
import subprocess
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



def obter_commit_curto() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip() or "unknown"
    except Exception:
        return "unknown"


def obter_prompt_version() -> str:
    try:
        from src.dataset.build_sft_dataset import PROMPT_VERSION
        return str(PROMPT_VERSION)
    except Exception:
        return "minicode-prompt-v2"


def obter_limites_config() -> tuple[int, int]:
    import os
    import src.config as cfg
    max_it = getattr(cfg, "MAX_ITERATIONS", None)
    if max_it is None:
        max_it = int(os.getenv("MINICODE_MAX_ITERATIONS", "15"))
    max_rep = getattr(cfg, "MAX_REPAIR_ATTEMPTS", None)
    if max_rep is None:
        max_rep = int(os.getenv("MINICODE_MAX_REPAIR_ATTEMPTS", "2"))
    return int(max_it), int(max_rep)


def ordenar_tarefas_por_manifesto(
    tarefas: List[Dict[str, Any]],
    manifest_path: Path,
) -> List[Dict[str, Any]]:
    if not manifest_path.exists():
        return tarefas
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    splits = data.get("splits", {})
    ordem: Dict[str, int] = {}
    idx = 0
    for particao in ("train", "val", "test"):
        entrada = splits.get(particao, [])
        tids = entrada.get("task_ids", []) if isinstance(entrada, dict) else entrada
        for tid in tids:
            tid_s = str(tid)
            if tid_s not in ordem:
                ordem[tid_s] = idx
                idx += 1
    return sorted(tarefas, key=lambda t: (ordem.get(t["task_id"], 10**9), t["task_id"]))


def preparar_workspace(task_dir: Path, run_ws: Path) -> None:
    if run_ws.exists(): shutil.rmtree(run_ws)
    run_ws.mkdir(parents=True, exist_ok=True)
    # copia arquivos base da tarefa (ex.: src/ inicial, tests/, etc.) se existirem
    base_ws = task_dir / "workspace"
    if base_ws.exists(): shutil.copytree(base_ws, run_ws, dirs_exist_ok=True)
    
    else:
        (run_ws / "src").mkdir(parents=True, exist_ok=True)
        tests_src = task_dir / "tests"
        if tests_src.exists():
            shutil.copytree(tests_src, run_ws / "tests", dirs_exist_ok=True)
        else:
            (run_ws / "tests").mkdir(parents=True, exist_ok=True)


def carregar_runs_concluidas(runs_meta_path: Path) -> tuple[set[tuple[str, int, float]], set[str]]:
    concluidas: set[tuple[str, int, float]] = set()
    run_ids_existentes: set[str] = set()
    if not runs_meta_path.exists():
        return concluidas, run_ids_existentes

    for linha in runs_meta_path.read_text(encoding="utf-8").splitlines():
        if not linha.strip():
            continue
        try:
            item = json.loads(linha)
        except json.JSONDecodeError:
            continue
        rid = item.get("run_id")
        if isinstance(rid, str) and rid:
            run_ids_existentes.add(rid)
        if item.get("status") == "ok" and "task_id" in item and "seed" in item and "temperature" in item:
            try:
                chave = (str(item["task_id"]), int(item["seed"]), float(item["temperature"]))
                concluidas.add(chave)
            except (TypeError, ValueError):
                continue
    return concluidas, run_ids_existentes


def carregar_tarefas(tasks_root: Path, limit: int | None = None) -> List[Dict[str, Any]]:
    tarefas: List[Dict[str, Any]] = []
    for p in sorted(tasks_root.iterdir()):
        if not p.is_dir(): continue
        meta_path = p / "task.json"
        if not meta_path.exists(): continue
        
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        tarefas.append({
            "task_id": meta.get("task_id", p.name),
            "prompt": meta["prompt"],
            "dir": p,
        })

        if limit is not None and len(tarefas) >= limit: break
    return tarefas


def main() -> None:
    env_file = Path(".env")
    if env_file.exists():
        for linha in env_file.read_text(encoding="utf-8").splitlines():
            linha = linha.strip()
            if not linha or linha.startswith("#") or "=" not in linha: continue
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
    parser.add_argument("--task-ids", nargs="+", default=None, help="Lista opcional de task_id para filtrar")
    parser.add_argument("--manifest", default="corpus_filtered/split_manifest.json", help="Manifesto de splits para ordenar treino -> val -> teste")
    parser.add_argument("--limit-tasks", type=int, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Pula combinações (task_id, seed, temperature) já concluídas (status='ok') em runs_metadata.jsonl",
    )
    args = parser.parse_args()

    tasks_root = Path(args.tasks_dir)
    out_root = Path(args.out_dir)
    runs_dir = out_root / "workspaces"
    runs_dir.mkdir(parents=True, exist_ok=True)
    runs_meta_path = out_root / "runs_metadata.jsonl"

    runs_concluidas, run_ids_existentes = carregar_runs_concluidas(runs_meta_path)

    tarefas = carregar_tarefas(tasks_root, limit=None)
    if args.task_ids:
        permitidos = {str(t) for t in args.task_ids}
        tarefas = [t for t in tarefas if t["task_id"] in permitidos]
    tarefas = ordenar_tarefas_por_manifesto(tarefas, Path(args.manifest))
    if args.limit_tasks is not None:
        tarefas = tarefas[: args.limit_tasks]
    git_commit = obter_commit_curto()
    prompt_version = obter_prompt_version()
    cfg_max_iter, cfg_max_repair = obter_limites_config()
    print(f"[Coleta] {len(tarefas)} tarefas na fila (ordem train->val->test) | commit={git_commit} | prompt={prompt_version} | max_iter={cfg_max_iter} | max_repair={cfg_max_repair}")
    if not tarefas: raise RuntimeError(f"Nenhuma tarefa encontrada em '{tasks_root}'.")

    for tarefa in tarefas:
        task_id = tarefa["task_id"]
        prompt_tarefa = tarefa["prompt"]
        for seed in args.seeds:
            for temp in args.temps:
                chave_run = (task_id, int(seed), float(temp))
                if args.resume and chave_run in runs_concluidas:
                    print(f"[{task_id}] seed={seed} temp={temp} -> pulando (--resume)")
                    continue

                run_id = uuid.uuid4().hex[:12]
                while run_id in run_ids_existentes:
                    run_id = uuid.uuid4().hex[:12]
                run_ids_existentes.add(run_id)

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
                    orch = AgentOrchestrator(ws_dir, mc_dir, llm, state_machine=sm, run_id=run_id, proteger_testes_existentes=True)
                    orch.state_machine = sm
                    orch.state_data.state = sm.state.value
                    final_state = orch.run(spec.raw_content, plano_raw)
                    repair_attempts = orch.state_machine.repair_attempts
            
                except Exception as exc:
                    final_state = "FAILED"
                    repair_attempts = sm.repair_attempts
                    status = "error"
                    error_msg = str(exc)
                    print(f"[!] Erro crítico/Timeout capturado: {exc}")
                    # força gravacao de FAILED para o --resume pular essa seed problemática
                    try:
                        mc_dir.mkdir(parents=True, exist_ok=True)
                        with open(mc_dir / "state.json", "w", encoding="utf-8") as f:
                            json.dump({"state": "FAILED", "failure_reason": "api_timeout_or_crash"}, f)
                    except Exception:
                        pass

                if status == "ok":
                    runs_concluidas.add(chave_run)

                eff_max_iter = getattr(getattr(locals().get("orch"), "state_data", None), "max_iterations", cfg_max_iter)
                eff_max_repair = getattr(getattr(locals().get("orch"), "state_data", None), "max_repair_attempts", cfg_max_repair)
                run_record = {
                    "timestamp": datetime.now().isoformat(),
                    "run_id": run_id,
                    "task_id": task_id,
                    "seed": seed,
                    "temperature": temp,
                    "commit": git_commit,
                    "prompt_version": prompt_version,
                    "max_iterations": eff_max_iter,
                    "max_repair_attempts": eff_max_repair,
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