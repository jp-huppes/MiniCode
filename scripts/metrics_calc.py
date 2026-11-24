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

    n_action_responses = 0; first_try_valid = 0; pre_retry_failures = 0; total_retries = 0
    error_categories: Counter[str] = Counter()

    done_count = 0; parse_failure_count = 0
    latencies: List[float] = []; tokens_total: List[int] = []

    for run in registros_execucao:
        if run.get("final_state") == "DONE": done_count += 1
        if run.get("final_state") == "FAILED" and run.get("failure_reason") == "parse_failure": parse_failure_count += 1

        latencies.append(float(run.get("latency_ms", 0.0)))
        tokens_total.append(int(run.get("prompt_tokens", 0)) + int(run.get("completion_tokens", 0)))

        for act in run.get("action_attempts", []):
            n_action_responses += 1
            total_retries += int(act.get("retries", 0))
            if act.get("valid"): first_try_valid += 1
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

    for r in runs_a: por_tarefa_a[str(r["task_id"])].append(r)
    for r in runs_b: por_tarefa_b[str(r["task_id"])].append(r)

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

# === NIVEL 2: E2E HYBRID METRICS & TASK BOOTSTRAP ===
import random
import statistics


def _quantiles_iqr(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "sd": 0.0, "median": 0.0, "q1": 0.0, "q3": 0.0, "iqr": 0.0, "min": 0.0, "max": 0.0}
    s = sorted(float(v) for v in values)
    n = len(s)
    mean_v = sum(s) / n
    sd_v = statistics.stdev(s) if n > 1 else 0.0
    med_v = statistics.median(s)
    if n >= 4:
        qs = statistics.quantiles(s, n=4, method="inclusive")
        q1, q3 = qs[0], qs[2]
    else:
        q1, q3 = s[0], s[-1]
    return {
        "mean": round(mean_v, 4),
        "sd": round(sd_v, 4),
        "median": round(med_v, 4),
        "q1": round(q1, 4),
        "q3": round(q3, 4),
        "iqr": round(q3 - q1, 4),
        "min": round(s[0], 4),
        "max": round(s[-1], 4),
    }


def _extract_model_calls(run: Dict[str, Any]) -> tuple[int, int]:
    """Retorna (N_1.5B, N_14B) a partir de role_calls e tool_model/cell."""
    rc = run.get("role_calls") or {}
    n_tool = int(rc.get("tool_selection", 0))
    n_rep = int(rc.get("repair", 0))
    tool_model = str(run.get("tool_model", ""))
    cell = str(run.get("cell", ""))
    uses_1_5b = ("1.5B" in tool_model) or (cell in {"B", "C", "E"})
    if uses_1_5b:
        return n_tool, n_rep
    return 0, n_tool + n_rep


def bootstrap_ci_by_task(
    runs: List[Dict[str, Any]],
    metric_fn,
    n_resamples: int = 10000,
    seed: int = 42,
) -> Dict[str, float]:
    """Bootstrap nao-parametrico por cluster (task_id) com B reamostragens."""
    by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in runs:
        by_task[str(r["task_id"])].append(r)
    task_ids = sorted(by_task.keys())
    if not task_ids:
        return {"estimate": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}

    rng = random.Random(seed)
    est = float(metric_fn(runs))
    samples: List[float] = []
    k = len(task_ids)
    for _ in range(n_resamples):
        sampled_tids = [task_ids[rng.randrange(k)] for _ in range(k)]
        boot_runs = [r for tid in sampled_tids for r in by_task[tid]]
        samples.append(float(metric_fn(boot_runs)))
    samples.sort()
    lo_idx = max(0, int(0.025 * n_resamples))
    hi_idx = min(n_resamples - 1, int(0.975 * n_resamples))
    return {
        "estimate": round(est, 4),
        "ci95_low": round(samples[lo_idx], 4),
        "ci95_high": round(samples[hi_idx], 4),
    }


def paired_bootstrap_diff_by_task(
    runs_a: List[Dict[str, Any]],
    runs_b: List[Dict[str, Any]],
    metric_fn,
    n_resamples: int = 10000,
    seed: int = 42,
) -> Dict[str, float]:
    """Bootstrap pareado por task_id para delta = metric(B) - metric(A)."""
    by_task_a: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_task_b: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in runs_a:
        by_task_a[str(r["task_id"])].append(r)
    for r in runs_b:
        by_task_b[str(r["task_id"])].append(r)

    common_tids = sorted(set(by_task_a.keys()) & set(by_task_b.keys()))
    if not common_tids:
        return {"delta": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "p_two_sided": 1.0}

    obs_a = [r for tid in common_tids for r in by_task_a[tid]]
    obs_b = [r for tid in common_tids for r in by_task_b[tid]]
    delta_obs = float(metric_fn(obs_b)) - float(metric_fn(obs_a))

    rng = random.Random(seed)
    k = len(common_tids)
    deltas: List[float] = []
    for _ in range(n_resamples):
        sampled = [common_tids[rng.randrange(k)] for _ in range(k)]
        ba = [r for tid in sampled for r in by_task_a[tid]]
        bb = [r for tid in sampled for r in by_task_b[tid]]
        deltas.append(float(metric_fn(bb)) - float(metric_fn(ba)))

    deltas.sort()
    lo_idx = max(0, int(0.025 * n_resamples))
    hi_idx = min(n_resamples - 1, int(0.975 * n_resamples))
    # p-valor bilateral aproximado via proporcao de cruzamento do zero
    le_zero = sum(1 for d in deltas if d <= 0.0) / n_resamples
    ge_zero = sum(1 for d in deltas if d >= 0.0) / n_resamples
    p_two = min(1.0, 2.0 * min(le_zero, ge_zero))
    return {
        "delta": round(delta_obs, 4),
        "ci95_low": round(deltas[lo_idx], 4),
        "ci95_high": round(deltas[hi_idx], 4),
        "p_two_sided": round(p_two, 4),
    }


def resumir_nivel2_e2e(
    runs: List[Dict[str, Any]],
    n_resamples: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Consolida metricas do Nivel 2 por celula e comparacoes pareadas por tarefa."""
    by_cell: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in runs:
        by_cell[str(r["cell"])].append(r)

    def _tsr(rs: List[Dict[str, Any]]) -> float:
        return sum(1 for x in rs if x.get("final_state") == "DONE") / len(rs) if rs else 0.0

    def _rho_micro(rs: List[Dict[str, Any]]) -> float:
        n15 = sum(_extract_model_calls(x)[0] for x in rs)
        n14 = sum(_extract_model_calls(x)[1] for x in rs)
        return n15 / (n15 + n14) if (n15 + n14) > 0 else 0.0

    def _rho_macro(rs: List[Dict[str, Any]]) -> float:
        vals = []
        for x in rs:
            n15, n14 = _extract_model_calls(x)
            vals.append(n15 / (n15 + n14) if (n15 + n14) > 0 else 0.0)
        return sum(vals) / len(vals) if vals else 0.0

    def _mean_iter(rs: List[Dict[str, Any]]) -> float:
        return sum(float(x.get("iterations", 0)) for x in rs) / len(rs) if rs else 0.0

    def _mean_tok(rs: List[Dict[str, Any]]) -> float:
        return sum(float(x.get("total_tokens", 0)) for x in rs) / len(rs) if rs else 0.0

    def _mean_lat(rs: List[Dict[str, Any]]) -> float:
        return sum(float(x.get("latency_seconds", 0.0)) for x in rs) / len(rs) if rs else 0.0

    cells_report: Dict[str, Any] = {}
    for cell in sorted(by_cell.keys()):
        rs = by_cell[cell]
        n_runs = len(rs)
        n_done = sum(1 for x in rs if x.get("final_state") == "DONE")
        n15_tot = sum(_extract_model_calls(x)[0] for x in rs)
        n14_tot = sum(_extract_model_calls(x)[1] for x in rs)
        rho_per_run = [
            (_extract_model_calls(x)[0] / sum(_extract_model_calls(x)))
            if sum(_extract_model_calls(x)) > 0 else 0.0
            for x in rs
        ]
        reasons = Counter(
            str(x.get("failure_reason") or "unknown")
            for x in rs
            if x.get("final_state") != "DONE"
        )
        cells_report[cell] = {
            "label": rs[0].get("cell_label", cell),
            "n_runs": n_runs,
            "n_done": n_done,
            "tsr": {
                "wilson_95": wilson_ci(n_done, n_runs),
                "task_bootstrap_95": bootstrap_ci_by_task(rs, _tsr, n_resamples=n_resamples, seed=seed),
            },
            "rho": {
                "N_1_5B": n15_tot,
                "N_14B": n14_tot,
                "micro_pooled": round(_rho_micro(rs), 4),
                "macro_stats": _quantiles_iqr(rho_per_run),
                "task_bootstrap_95": bootstrap_ci_by_task(rs, _rho_macro, n_resamples=n_resamples, seed=seed),
            },
            "iterations": {
                "stats": _quantiles_iqr([float(x.get("iterations", 0)) for x in rs]),
                "task_bootstrap_95": bootstrap_ci_by_task(rs, _mean_iter, n_resamples=n_resamples, seed=seed),
            },
            "tokens": {
                "stats": _quantiles_iqr([float(x.get("total_tokens", 0)) for x in rs]),
                "task_bootstrap_95": bootstrap_ci_by_task(rs, _mean_tok, n_resamples=n_resamples, seed=seed),
            },
            "latency_seconds": {
                "stats": _quantiles_iqr([float(x.get("latency_seconds", 0.0)) for x in rs]),
                "task_bootstrap_95": bootstrap_ci_by_task(rs, _mean_lat, n_resamples=n_resamples, seed=seed),
            },
            "failure_reasons": dict(reasons),
        }

    pairs = [("A", "C"), ("B", "C"), ("A", "D"), ("A", "B")]
    paired_report: Dict[str, Any] = {}
    for ca, cb in pairs:
        if ca in by_cell and cb in by_cell:
            paired_report[f"{cb}_minus_{ca}"] = {
                "delta_tsr": paired_bootstrap_diff_by_task(by_cell[ca], by_cell[cb], _tsr, n_resamples, seed),
                "delta_rho_macro": paired_bootstrap_diff_by_task(by_cell[ca], by_cell[cb], _rho_macro, n_resamples, seed),
                "delta_iterations": paired_bootstrap_diff_by_task(by_cell[ca], by_cell[cb], _mean_iter, n_resamples, seed),
                "delta_tokens": paired_bootstrap_diff_by_task(by_cell[ca], by_cell[cb], _mean_tok, n_resamples, seed),
                "delta_latency_s": paired_bootstrap_diff_by_task(by_cell[ca], by_cell[cb], _mean_lat, n_resamples, seed),
            }

    return {"cells": cells_report, "paired_bootstrap": paired_report}
