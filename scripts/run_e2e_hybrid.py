"""Runner E2E hibrido e de ablacao (Passo 4.c).

Executa as 4 celulas experimentais sobre o mesmo cache de SPEC/PLAN gerado uma vez
com o modelo professor (14B, temperature=0.0) em experiments/spec_cache/<task_id>/:
  - Celula A: 14B (tool_selection=14B, repair=14B)
  - Celula B: 1.5B base + 14B no reparo (via disable_adapter())
  - Celula C: 1.5B+LoRA + 14B no reparo
  - Celula D: 14B sem reparo (repair_enabled=False)

Grava um registro JSONL por run com estado final, motivo, iteracoes, tokens,
latencia, tentativas de reparo, chamadas por papel e rho.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import os
from pathlib import Path

def _carregar_env_raiz(root_dir: Path) -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(root_dir / ".env", override=False)
        return
    except Exception:
        pass
    env_file = root_dir / ".env"
    if not env_file.exists():
        return
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v

ROOT_DIR = Path(__file__).resolve().parent.parent
_carregar_env_raiz(ROOT_DIR)
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.agent.orchestrator import AgentOrchestrator
from src.agent.state import AgentStateMachine, State
from src.config import MAX_ITERATIONS, MAX_REPAIR_ATTEMPTS
from src.llm.client import LLMClient
from src.llm.local_peft_handler import (
    DEFAULT_ADAPTER_DIR,
    DEFAULT_BASE_MODEL_ID,
    DEFAULT_LOCAL_TOOL_MAX_TOKENS,
    get_or_create_peft_handler,
    register_local_peft_handler,
)
from src.llm.routing import (
    AgentRoutingConfig,
    DEFAULT_TEACHER_MODEL,
    ROLE_PLAN,
    ROLE_REPAIR,
    ROLE_SPEC,
    ROLE_TOOL_SELECTION,
    attach_routing_to_client,
    calcular_rho,
)
from src.spec.pipeline import executar_spec_pipeline

CELL_SPECS: Dict[str, Dict[str, Any]] = {
    "A": {
        "label": "14B",
        "use_local_1_5b": False,
        "use_adapter": False,
        "repair_enabled": True,
    },
    "B": {
        "label": "1.5B_base+14B_repair",
        "use_local_1_5b": True,
        "use_adapter": False,
        "repair_enabled": True,
    },
    "C": {
        "label": "1.5B_lora+14B_repair",
        "use_local_1_5b": True,
        "use_adapter": True,
        "repair_enabled": True,
    },
    "D": {
        "label": "14B_no_repair",
        "use_local_1_5b": False,
        "use_adapter": False,
        "repair_enabled": False,
        "constrained_mode": False,
    },
    "E": {
        "label": "1.5B_lora_constrained+14B_repair",
        "use_local_1_5b": True,
        "use_adapter": True,
        "repair_enabled": True,
        "constrained_mode": True,
    },
}


def preparar_workspace(task_dir: Path, run_ws: Path) -> None:
    if run_ws.exists():
        shutil.rmtree(run_ws)
    run_ws.mkdir(parents=True, exist_ok=True)
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


def carregar_tarefas_teste(
    tasks_dir: Path,
    manifest_path: Optional[Path] = None,
    split_name: str = "test",
    task_ids: Optional[List[str]] = None,
    limit_tasks: Optional[int] = None,
) -> List[Dict[str, Any]]:
    ordenadas: List[str] = []
    if manifest_path and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        splits = manifest.get("splits", {})
        entrada = splits.get(split_name, [])
        if isinstance(entrada, dict):
            ordenadas = [str(t) for t in entrada.get("task_ids", [])]
        elif isinstance(entrada, list):
            ordenadas = [str(t) for t in entrada]

    if not ordenadas:
        ordenadas = sorted(
            p.name for p in tasks_dir.iterdir() if p.is_dir() and (p / "task.json").exists()
        )

    if task_ids:
        permitidos = {str(t) for t in task_ids}
        ordenadas = [tid for tid in ordenadas if tid in permitidos]

    tarefas: List[Dict[str, Any]] = []
    for tid in ordenadas:
        t_dir = tasks_dir / tid
        meta_path = t_dir / "task.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        tarefas.append(
            {
                "task_id": str(meta.get("task_id") or tid),
                "prompt": str(meta.get("prompt") or meta.get("task") or ""),
                "dir": t_dir,
            }
        )

    if limit_tasks is not None:
        tarefas = tarefas[: max(0, int(limit_tasks))]
    return tarefas


def obter_ou_gerar_spec_plan_cache(
    tarefa: Dict[str, Any],
    cache_root: Path,
    teacher_model: str,
    teacher_base_url: Optional[str] = None,
    teacher_api_key: Optional[str] = None,
    temperature: float = 0.0,
) -> Tuple[str, str, bool]:
    """Gera SPEC e PLAN uma vez com o 14B (temperature=0.0) e reutiliza em todas as celulas."""
    task_id = tarefa["task_id"]
    task_cache_dir = cache_root / task_id
    spec_path = task_cache_dir / "specification.md"
    plan_path = task_cache_dir / "plan.md"
    meta_path = task_cache_dir / "meta.json"

    if spec_path.exists() and plan_path.exists():
        spec_txt = spec_path.read_text(encoding="utf-8")
        plan_txt = plan_path.read_text(encoding="utf-8")
        if spec_txt.strip() and plan_txt.strip():
            return spec_txt, plan_txt, True

    task_cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_ws = task_cache_dir / "_tmp_ws"
    preparar_workspace(tarefa["dir"], tmp_ws)

    llm_kwargs: Dict[str, Any] = {
        "model_name": teacher_model,
        "default_temperature": float(temperature),
    }
    if teacher_base_url:
        llm_kwargs["base_url"] = teacher_base_url
    if teacher_api_key:
        llm_kwargs["api_key"] = teacher_api_key

    llm_teacher = LLMClient(**llm_kwargs)
    attach_routing_to_client(
        llm_teacher,
        AgentRoutingConfig(
            teacher_model=teacher_model,
            tool_model=teacher_model,
            spec_model=teacher_model,
            plan_model=teacher_model,
            repair_model=teacher_model,
            temperature=float(temperature),
        ),
    )

    sm = AgentStateMachine()
    t0 = time.perf_counter()
    try:
        spec, plano_raw = executar_spec_pipeline(
            tarefa["prompt"],
            tmp_ws,
            llm_teacher,
            sm,
            temperature=float(temperature),
        )
    except TypeError:
        spec, plano_raw = executar_spec_pipeline(
            tarefa["prompt"],
            tmp_ws,
            llm_teacher,
            sm,
        )
    elapsed_s = time.perf_counter() - t0

    spec_content = spec.raw_content
    plan_content = plano_raw

    spec_path.write_text(spec_content, encoding="utf-8")
    plan_path.write_text(plan_content, encoding="utf-8")

    prompt_toks = sum(c.prompt_tokens for c in llm_teacher.metrics.calls)
    comp_toks = sum(c.completion_tokens for c in llm_teacher.metrics.calls)
    meta_path.write_text(
        json.dumps(
            {
                "task_id": task_id,
                "teacher_model": teacher_model,
                "temperature": float(temperature),
                "timestamp": datetime.now().isoformat(),
                "latency_seconds": round(elapsed_s, 4),
                "prompt_tokens": prompt_toks,
                "completion_tokens": comp_toks,
                "llm_calls": len(llm_teacher.metrics.calls),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    shutil.rmtree(tmp_ws, ignore_errors=True)
    return spec_content, plan_content, False


def carregar_runs_concluidas(jsonl_path: Path) -> set[Tuple[str, str, int, bool]]:
    concluidas: set[Tuple[str, str, int, bool]] = set()
    if not jsonl_path.exists():
        return concluidas
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("status") == "ok" and "cell" in row and "task_id" in row:
            concluidas.add(
                (str(row["cell"]), str(row["task_id"]), int(row.get("seed", 1)), bool(row.get("no_spec", False)))
            )
    return concluidas


def executar_celula_tarefa(
    cell_id: str,
    tarefa: Dict[str, Any],
    spec_content: str,
    plan_content: str,
    seed: int,
    temperature: float,
    out_dir: Path,
    teacher_model: str,
    teacher_base_url: Optional[str],
    teacher_api_key: Optional[str],
    base_model_1_5b: str,
    adapter_dir: Path,
    tool_max_tokens: int,
    max_iterations: int,
    max_repair_attempts: int,
    peft_handler: Any = None,
    no_spec: bool = False,
) -> Dict[str, Any]:
    spec_cfg = CELL_SPECS[cell_id]
    task_id = tarefa["task_id"]
    run_id = uuid.uuid4().hex[:12]

    tag_dir = f"cell_{cell_id}_nospec" if no_spec else f"cell_{cell_id}"
    run_base = out_dir / "workspaces" / tag_dir / task_id / f"seed_{seed}_{run_id}"
    ws_dir = run_base / "workspace"
    mc_dir = run_base / ".minicode"

    preparar_workspace(tarefa["dir"], ws_dir)
    if no_spec:
        spec_file = ws_dir / "specification.md"
        if spec_file.exists():
            spec_file.unlink()
        spec_efetivo = (
            "(Ablacao no_spec: especificacao formal omitida. "
            "Implemente a tarefa diretamente seguindo o plano e os testes existentes.)"
        )
    else:
        (ws_dir / "specification.md").write_text(spec_content, encoding="utf-8")
        spec_efetivo = spec_content
    (ws_dir / "plan.md").write_text(plan_content, encoding="utf-8")

    llm_kwargs: Dict[str, Any] = {
        "model_name": teacher_model,
        "default_temperature": float(temperature),
    }
    if teacher_base_url:
        llm_kwargs["base_url"] = teacher_base_url
    if teacher_api_key:
        llm_kwargs["api_key"] = teacher_api_key

    llm_client = LLMClient(**llm_kwargs)

    if spec_cfg["use_local_1_5b"]:
        routing_cfg = AgentRoutingConfig(
            teacher_model=teacher_model,
            tool_model=base_model_1_5b,
            spec_model=teacher_model,
            plan_model=teacher_model,
            repair_model=teacher_model,
            temperature=float(temperature),
            max_tokens=int(tool_max_tokens),
        )
        attach_routing_to_client(llm_client, routing_cfg)
        use_constrained = bool(spec_cfg.get("constrained_mode", False))
        if use_constrained:
            llm_client.constrained_mode = "v-a"
            routing_cfg.constrained_mode = True

        if peft_handler is None:
            peft_handler = get_or_create_peft_handler(
                base_model_id=base_model_1_5b,
                adapter_dir=adapter_dir,
                max_tokens=int(tool_max_tokens),
                use_adapter=bool(spec_cfg["use_adapter"]),
                constrained_mode=use_constrained,
                metrics=llm_client.metrics,
            )
        else:
            peft_handler.set_use_adapter(bool(spec_cfg["use_adapter"]))
            peft_handler.set_constrained_mode(use_constrained)
            peft_handler.max_tokens = int(tool_max_tokens)
            peft_handler.metrics = llm_client.metrics

        register_local_peft_handler(
            llm_client,
            handler=peft_handler,
            base_model_id=base_model_1_5b,
            adapter_dir=adapter_dir,
            max_tokens=int(tool_max_tokens),
            use_adapter=bool(spec_cfg["use_adapter"]),
            constrained_mode=use_constrained,
        )
        if use_constrained:
            tool_model_name = f"{base_model_1_5b}+lora+constrained"
        else:
            tool_model_name = (
                f"{base_model_1_5b}+lora" if spec_cfg["use_adapter"] else f"{base_model_1_5b}+base"
            )
    else:
        routing_cfg = AgentRoutingConfig(
            teacher_model=teacher_model,
            tool_model=teacher_model,
            spec_model=teacher_model,
            plan_model=teacher_model,
            repair_model=teacher_model,
            no_repair=not bool(spec_cfg["repair_enabled"]),
            temperature=float(temperature),
        )
        attach_routing_to_client(llm_client, routing_cfg)
        tool_model_name = teacher_model

    # Avanca FSM deterministica ate PLANNED usando SPEC/PLAN cacheados
    sm = AgentStateMachine(max_repair_attempts=max_repair_attempts)
    sm.transition(State.SPECIFIED)
    sm.transition(State.PLANNED)

    status = "ok"
    error_msg: Optional[str] = None
    t0 = time.perf_counter()

    orch = AgentOrchestrator(
        workspace_dir=ws_dir,
        minicode_dir=mc_dir,
        llm_client=llm_client,
        max_iterations=max_iterations,
        max_repair_attempts=max_repair_attempts,
        state_machine=sm,
        run_id=run_id,
        temperature=float(temperature),
        proteger_testes_existentes=True,
        repair_enabled=bool(spec_cfg["repair_enabled"]),
    )
    orch.state_data.state = sm.state.value

    try:
        final_state = orch.run(spec_efetivo, plan_content)
        final_state_str = final_state.value if hasattr(final_state, "value") else str(final_state)
    except Exception as exc:
        final_state_str = "FAILED"
        orch.state_data.failure_reason = orch.state_data.failure_reason or "runtime_exception"
        status = "error"
        error_msg = str(exc)

    elapsed_s = time.perf_counter() - t0

    calls = getattr(llm_client.metrics, "calls", [])
    prompt_tokens = sum(int(getattr(c, "prompt_tokens", 0)) for c in calls)
    completion_tokens = sum(int(getattr(c, "completion_tokens", 0)) for c in calls)
    llm_latency_s = sum(float(getattr(c, "latency_seconds", 0.0)) for c in calls)

    role_counts = orch.get_role_call_counts()
    rho = calcular_rho(role_counts, uses_light_tool_model=bool(spec_cfg["use_local_1_5b"]))

    return {
        "timestamp": datetime.now().isoformat(),
        "run_id": run_id,
        "cell": cell_id,
        "cell_label": spec_cfg["label"],
        "task_id": task_id,
        "seed": int(seed),
        "temperature": float(temperature),
        "repair_enabled": bool(spec_cfg["repair_enabled"]),
        "constrained_mode": bool(spec_cfg.get("constrained_mode", False)),
        "no_spec": bool(no_spec),
        "teacher_model": teacher_model,
        "tool_model": tool_model_name,
        "adapter_dir": str(adapter_dir) if spec_cfg["use_adapter"] else None,
        "final_state": final_state_str,
        "failure_reason": str(getattr(orch.state_data, "failure_reason", "") or ""),
        "iterations": int(getattr(orch.state_data, "iteration", 0)),
        "tokens": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "latency_seconds": round(elapsed_s, 4),
        "llm_latency_seconds": round(llm_latency_s, 4),
        "repair_attempts": int(orch.state_machine.repair_attempts),
        "max_repair_attempts": int(max_repair_attempts),
        "role_calls": {
            ROLE_TOOL_SELECTION: int(role_counts.get(ROLE_TOOL_SELECTION, 0)),
            ROLE_REPAIR: int(role_counts.get(ROLE_REPAIR, 0)),
        },
        "rho": round(rho, 4),
        "status": status,
        "error": error_msg,
        "workspace_dir": str(ws_dir),
        "actions_log": str(mc_dir / "actions.jsonl"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Runner E2E hibrido e ablacao (4 celulas A, B, C, D)")
    parser.add_argument("--tasks-dir", type=Path, default=ROOT_DIR / "corpus_tasks")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT_DIR / "corpus_filtered" / "split_manifest.json",
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--task-ids", nargs="+", default=None)
    parser.add_argument("--limit-tasks", type=int, default=None)
    parser.add_argument("--cells", nargs="+", choices=["A", "B", "C", "D", "E"], default=["A", "B", "C", "D"])
    parser.add_argument("--no-spec", action="store_true", help="Ablacao no_spec: omite specification.md e hard gates da SPEC")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--teacher-model", type=str, default=DEFAULT_TEACHER_MODEL)
    parser.add_argument("--teacher-base-url", type=str, default=None)
    parser.add_argument("--teacher-api-key", type=str, default=None)
    parser.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL_ID)
    parser.add_argument("--adapter-dir", type=Path, default=ROOT_DIR / DEFAULT_ADAPTER_DIR)
    parser.add_argument("--tool-max-tokens", type=int, default=DEFAULT_LOCAL_TOOL_MAX_TOKENS)
    parser.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS)
    parser.add_argument("--max-repair-attempts", type=int, default=MAX_REPAIR_ATTEMPTS)
    parser.add_argument("--cache-dir", type=Path, default=ROOT_DIR / "experiments" / "spec_cache")
    parser.add_argument("--out-dir", type=Path, default=ROOT_DIR / "experiments" / "e2e_hybrid")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    tarefas = carregar_tarefas_teste(
        tasks_dir=args.tasks_dir,
        manifest_path=args.manifest,
        split_name=args.split,
        task_ids=args.task_ids,
        limit_tasks=args.limit_tasks,
    )
    if not tarefas:
        raise RuntimeError(f"Nenhuma tarefa encontrada em '{args.tasks_dir}' (split='{args.split}').")

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.out_dir / "runs.jsonl"
    concluidas = carregar_runs_concluidas(jsonl_path) if args.resume else set()

    api_base = args.teacher_base_url or os.getenv("MINICODE_API_BASE")
    api_key = args.teacher_api_key or os.getenv("MINICODE_API_KEY")
    if not api_base or not api_key:
        raise RuntimeError(
            "Defina MINICODE_API_BASE e MINICODE_API_KEY (no .env ou no PowerShell) "
            "ou passe --teacher-base-url e --teacher-api-key para conectar ao Qwen-14B."
        )

    # Pre-carrega o modelo 1.5B + LoRA uma unica vez na VRAM se B ou C estiverem ativas
    peft_handler = None
    if any(c in {"B", "C", "E"} for c in args.cells):
        peft_handler = get_or_create_peft_handler(
            base_model_id=args.base_model,
            adapter_dir=args.adapter_dir,
            max_tokens=int(args.tool_max_tokens),
            use_adapter=True,
        )

    print(
        f"[E2E Hybrid] Tarefas={len(tarefas)} ({[t['task_id'] for t in tarefas]}) | "
        f"Celulas={args.cells} | Seeds={args.seeds} | Temp={args.temperature}"
    )

    for tarefa in tarefas:
        tid = tarefa["task_id"]
        spec_txt, plan_txt, from_cache = obter_ou_gerar_spec_plan_cache(
            tarefa=tarefa,
            cache_root=args.cache_dir,
            teacher_model=args.teacher_model,
            teacher_base_url=args.teacher_base_url,
            teacher_api_key=args.teacher_api_key,
            temperature=float(args.temperature),
        )
        origem = "cache" if from_cache else "gerado_14B_t0"
        print(f"\n=== Tarefa {tid} | SPEC/PLAN: {origem} ({args.cache_dir / tid}) ===")

        for seed in args.seeds:
            for cell_id in args.cells:
                if args.resume and (cell_id, tid, int(seed), bool(args.no_spec)) in concluidas:
                    print(f"  [Cell {cell_id}] {tid} seed={seed} -> pulando (--resume)")
                    continue

                rec = executar_celula_tarefa(
                    cell_id=cell_id,
                    tarefa=tarefa,
                    spec_content=spec_txt,
                    plan_content=plan_txt,
                    seed=int(seed),
                    temperature=float(args.temperature),
                    out_dir=args.out_dir,
                    teacher_model=args.teacher_model,
                    teacher_base_url=args.teacher_base_url,
                    teacher_api_key=args.teacher_api_key,
                    base_model_1_5b=args.base_model,
                    adapter_dir=args.adapter_dir,
                    tool_max_tokens=int(args.tool_max_tokens),
                    max_iterations=int(args.max_iterations),
                    max_repair_attempts=int(args.max_repair_attempts),
                    peft_handler=peft_handler,
                    no_spec=bool(args.no_spec),
                )

                with open(jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                if rec["status"] == "ok":
                    concluidas.add((cell_id, tid, int(seed), bool(args.no_spec)))

                print(
                    f"  [Cell {cell_id} | {rec['cell_label']}] state={rec['final_state']} "
                    f"reason='{rec['failure_reason']}' iter={rec['iterations']} "
                    f"tokens={rec['total_tokens']} lat={rec['latency_seconds']}s "
                    f"repairs={rec['repair_attempts']} roles={rec['role_calls']} rho={rec['rho']}"
                )

    print(f"\n[OK] Resultados gravados em: {jsonl_path.resolve()}")


if __name__ == "__main__":
    main()