import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def wilson_ci(successes: int, total: int, z: float = 1.96) -> Dict[str, Optional[float]]:
    if total <= 0:
        return {"p_hat": None, "low": None, "high": None}
    p = successes / total
    denom = 1.0 + (z * z) / total
    center = p + (z * z) / (2.0 * total)
    margin = z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * total)) / total)
    return {
        "p_hat": round(p, 4),
        "low": round(max(0.0, (center - margin) / denom), 4),
        "high": round(min(1.0, (center + margin) / denom), 4),
    }


def resumir_celula(registros_execucao: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Cada item em `registros_execucao` representa uma execução de tarefa e deve conter:
    - task_id: str
    - final_state: str ("DONE", "FAILED", ...)
    - failure_reason: str (ex.: "", "parse_failure", "max_iterations")
    - action_attempts: list[dict] com {"valid": bool, "error_category": Optional[str], "retries": int}
    - latency_ms: float
    - prompt_tokens: int
    - completion_tokens: int
    """
    n_action_responses = 0
    first_try_valid = 0
    pre_retry_failures = 0
    total_retries = 0
    error_categories: Counter[str] = Counter()

    done_count = 0
    parse_failure_count = 0
    latencies: List[float] = []
    tokens_total: List[int] = []

    for run in registros_execucao:
        if run.get("final_state") == "DONE":
            done_count += 1
        if run.get("final_state") == "FAILED" and run.get("failure_reason") == "parse_failure":
            parse_failure_count += 1

        latencies.append(float(run.get("latency_ms", 0.0)))
        tokens_total.append(int(run.get("prompt_tokens", 0)) + int(run.get("completion_tokens", 0)))

        for act in run.get("action_attempts", []):
            n_action_responses += 1
            total_retries += int(act.get("retries", 0))
            if act.get("valid"):
                first_try_valid += 1
            else:
                pre_retry_failures += 1
                cat = act.get("error_category") or "desconhecido"
                error_categories[cat] += 1

    n_runs = len(registros_execucao)
    return {
        "runs": n_runs,
        "action_responses_N": n_action_responses,
        "p_parse_hat": wilson_ci(pre_retry_failures, n_action_responses),
        "p_first_try_hat": wilson_ci(first_try_valid, n_action_responses),
        "mean_retries_per_action": round(total_retries / n_action_responses, 4) if n_action_responses else 0.0,
        "error_category_distribution": dict(error_categories),
        "failed_by_parse_failure": wilson_ci(parse_failure_count, n_runs),
        "done_rate": wilson_ci(done_count, n_runs),
        "mean_latency_ms": round(sum(latencies) / n_runs, 2) if n_runs else 0.0,
        "mean_total_tokens": round(sum(tokens_total) / n_runs, 2) if n_runs else 0.0,
    }


def comparar_pareado_por_tarefa(
    runs_a: Iterable[Dict[str, Any]],
    runs_b: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Compara pareadamente por `task_id` entre configuração A e configuração B."""
    por_tarefa_a: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    por_tarefa_b: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for r in runs_a:
        por_tarefa_a[str(r["task_id"])].append(r)
    for r in runs_b:
        por_tarefa_b[str(r["task_id"])].append(r)

    task_ids = sorted(set(por_tarefa_a.keys()) & set(por_tarefa_b.keys()))
    comparacoes: List[Dict[str, Any]] = []

    for tid in task_ids:
        sa = resumir_celula(por_tarefa_a[tid])
        sb = resumir_celula(por_tarefa_b[tid])
        done_a = sa["done_rate"]["p_hat"] or 0.0
        done_b = sb["done_rate"]["p_hat"] or 0.0
        parse_a = sa["p_parse_hat"]["p_hat"] or 0.0
        parse_b = sb["p_parse_hat"]["p_hat"] or 0.0

        comparacoes.append({
            "task_id": tid,
            "delta_done_rate": round(done_b - done_a, 4),
            "delta_p_parse": round(parse_b - parse_a, 4),
            "delta_mean_latency_ms": round(sb["mean_latency_ms"] - sa["mean_latency_ms"], 2),
            "delta_mean_tokens": round(sb["mean_total_tokens"] - sa["mean_total_tokens"], 2),
        })

    return comparacoes


if __name__ == "__main__":
    sample_a = [{"task_id": "t1", "final_state": "FAILED", "failure_reason": "parse_failure", "action_attempts": [{"valid": False, "error_category": "campo_extra", "retries": 2}], "latency_ms": 1200, "prompt_tokens": 500, "completion_tokens": 120}]
    sample_b = [{"task_id": "t1", "final_state": "DONE", "failure_reason": "", "action_attempts": [{"valid": True, "error_category": None, "retries": 0}], "latency_ms": 1150, "prompt_tokens": 460, "completion_tokens": 105}]
    print(json.dumps({"a": resumir_celula(sample_a), "b": resumir_celula(sample_b), "paired": comparar_pareado_por_tarefa(sample_a, sample_b)}, indent=2, ensure_ascii=False))