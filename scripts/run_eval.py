import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path: sys.path.insert(0, str(ROOT_DIR))

from scripts.metrics_calc import comparar_pareado_por_tarefa, resumir_celula
from src.agent.orchestrator import AgentOrchestrator
from src.llm.client import LLMClient

def _carregar_dotenv_se_existir(raiz: Path) -> None:
    env_path = raiz / ".env"
    if not env_path.is_file(): return

    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(dotenv_path=env_path, override=False)
        return
    except ImportError:
        pass

    for linha in env_path.read_text(encoding="utf-8").splitlines():
        item = linha.strip()
        if not item or item.startswith("#") or "=" not in item: continue
        chave, valor = item.split("=", 1)
        chave = chave.strip()
        valor = valor.strip().strip('"').strip("'")
        if chave and chave not in os.environ: os.environ[chave] = valor

def carregar_tarefas_jsonl(caminho: Path) -> List[Dict[str, str]]:
    tarefas: List[Dict[str, str]] = []
    with caminho.open("r", encoding="utf-8") as f:
        for num_linha, linha in enumerate(f, start=1):
            conteudo = linha.strip()
            if not conteudo: continue
            item = json.loads(conteudo)
            if "task_id" not in item or "task" not in item:
                raise ValueError(f"Linha {num_linha} de {caminho} deve conter 'task_id' e 'task'.")
            tarefas.append({"task_id": str(item["task_id"]), "task": str(item["task"])})
    if not tarefas:
        raise ValueError(f"Nenhuma tarefa encontrada em {caminho}.")
    return tarefas


def _preparar_workspace_isolado(workspace: Path, task_id: str) -> None:
    (workspace / "src").mkdir(parents=True, exist_ok=True)
    (workspace / "tests").mkdir(parents=True, exist_ok=True)
    (workspace / "src" / "__init__.py").write_text("", encoding="utf-8")
    (workspace / "tests" / "test_base.py").write_text("def test_ok(): assert True\n", encoding="utf-8")

    # prepara um arquivo com bug proposital para a tarefa de reparo (t4)
    if task_id == "t4_reparo":
        (workspace / "src" / "app.py").write_text(
            "def dobro(x: int) -> int:\n    return x * 3  # bug proposital: deveria ser x * 2\n",
            encoding="utf-8",
        )
        (workspace / "tests" / "test_app.py").write_text(
            "from src.app import dobro\n\n\ndef test_dobro():\n    assert dobro(4) == 8\n",
            encoding="utf-8",
        )


def _construir_cliente(temperature: float, constrained_mode: str) -> LLMClient:
    os.environ["MINICODE_CONSTRAINED_MODE"] = constrained_mode
    os.environ["MINICODE_TEMPERATURE"] = str(temperature)
    return LLMClient()

def rodar_celula(
    cell_name: str,
    tasks: List[Dict[str, str]],
    repeats: int,
    base_dir: Path,
    temperature: float,
    constrained_mode: str,
) -> List[Dict[str, Any]]:
    registros: List[Dict[str, Any]] = []
    llm_client = _construir_cliente(temperature=temperature, constrained_mode=constrained_mode)

    for repeat_idx in range(repeats):
        for t in tasks:
            task_id = t["task_id"]
            prompt = t["task"]
            run_root = base_dir / cell_name / f"rep_{repeat_idx}" / task_id
            workspace = run_root / "workspace"
            state_dir = run_root / ".minicode"
            _preparar_workspace_isolado(workspace, task_id)

            t0 = time.perf_counter()
            orch = AgentOrchestrator(workspace, state_dir, llm_client)
            final_state = orch.run(prompt, f"Plano para {task_id}")
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            usage = getattr(orch.state_data, "last_usage", {}) or {}
            prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
            completion_tokens = int(usage.get("completion_tokens", 0) or 0)

            if prompt_tokens == 0 and completion_tokens == 0:
                metrics_obj = getattr(orch.llm_client, "metrics", None)
                if metrics_obj is not None:
                    prompt_tokens = int(getattr(metrics_obj, "total_prompt_tokens", 0) or 0)
                    completion_tokens = int(getattr(metrics_obj, "total_completion_tokens", 0) or 0)

            registros.append({
                "cell": cell_name,
                "repeat": repeat_idx,
                "task_id": task_id,
                "temperature": temperature,
                "constrained_mode": constrained_mode,
                "final_state": final_state.value if hasattr(final_state, "value") else str(final_state),
                "failure_reason": getattr(orch.state_data, "failure_reason", "") or "",
                "parse_failures": getattr(orch.state_data, "parse_failures", 0),
                "action_attempts": list(getattr(orch, "eval_action_attempts", []) or []),
                "latency_ms": round(elapsed_ms, 2),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            })

    return registros


def salvar_jsonl(caminho: Path, registros: List[Dict[str, Any]]) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("w", encoding="utf-8") as f:
        for r in registros: f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(argv: Optional[List[str]] = None) -> None:
    _carregar_dotenv_se_existir(ROOT_DIR)
    parser = argparse.ArgumentParser(description="Runner da avaliação experimental F10 do MiniCode.")
    parser.add_argument(
        "--tasks-file",
        type=Path,
        default=Path(__file__).resolve().parent / "tasks_eval.jsonl",
        help="Caminho do arquivo .jsonl com as tarefas.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Número de execuções por tarefa em cada célula.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperatura fixa usada em todas as células.",
    )
    parser.add_argument(
        "--constrained-mode",
        type=str,
        default="none",
        help="Modo de decodificação restrita (ex.: none, v-a, v-b, v-c, v-d).",
    )
    parser.add_argument(
        "--cells",
        nargs="+",
        default=["cell_a", "cell_b"],
        help="Nomes das células a executar com os mesmos parâmetros.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT_DIR / "eval_runs",
        help="Diretório de saída para traces e relatórios.",
    )
    
    args = parser.parse_args(argv)
    tasks = carregar_tarefas_jsonl(args.tasks_file)
    resumos_por_celula: Dict[str, Any] = {}
    runs_por_celula: Dict[str, List[Dict[str, Any]]] = {}

    for cell in args.cells:
        runs = rodar_celula(
            cell_name=cell,
            tasks=tasks,
            repeats=args.repeats,
            base_dir=args.out_dir / "workspaces",
            temperature=args.temperature,
            constrained_mode=args.constrained_mode,
        )
        runs_por_celula[cell] = runs
        salvar_jsonl(args.out_dir / f"{cell}_runs.jsonl", runs)
        resumos_por_celula[cell] = resumir_celula(runs)

    comparacoes_pareadas: Dict[str, Any] = {}
    if len(args.cells) >= 2:
        base_cell = args.cells[0]
        for other_cell in args.cells[1:]:
            comparacoes_pareadas[f"{base_cell}_vs_{other_cell}"] = comparar_pareado_por_tarefa(
                runs_por_celula[base_cell],
                runs_por_celula[other_cell],
            )

    relatorio = {
        "config": {
            "tasks_count": len(tasks),
            "repeats": args.repeats,
            "temperature": args.temperature,
            "constrained_mode": args.constrained_mode,
            "cells": args.cells,
        },
        "summaries": resumos_por_celula,
        "paired_by_task": comparacoes_pareadas,
    }

    out_report = args.out_dir / "summary_report.json"
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(relatorio, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(relatorio, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()