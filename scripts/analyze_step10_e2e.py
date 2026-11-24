import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * n)) / n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def paired_bootstrap_ci(
    vals_a: Dict[str, float],
    vals_b: Dict[str, float],
    resamples: int = 10000,
    seed: int = 42,
) -> Dict[str, float]:
    """Bootstrap pareado por task_id para delta = mean(A - B)."""
    common = sorted(set(vals_a.keys()) & set(vals_b.keys()))
    n = len(common)
    if n == 0:
        return {"n_tasks": 0, "mean_delta": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "p_one_sided": 1.0}
    diffs = [vals_a[t] - vals_b[t] for t in common]
    obs = sum(diffs) / n
    rng = random.Random(seed)
    boot_means: List[float] = []
    for _ in range(resamples):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    lo_idx = max(0, int(0.025 * resamples))
    hi_idx = min(resamples - 1, int(0.975 * resamples))
    p_non_pos = sum(1 for x in boot_means if x <= 0.0) / resamples
    return {
        "n_tasks": n,
        "mean_delta": round(obs, 4),
        "ci95_low": round(boot_means[lo_idx], 4),
        "ci95_high": round(boot_means[hi_idx], 4),
        "p_one_sided_gt_zero": round(p_non_pos, 4),
    }


def diagnose_cell_c(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """10.e: Analisa falhas da Celula C (repeticoes de escrita, loops de leitura e reparos)."""
    c_runs = [r for r in runs if r.get("cell") == "C"]
    if not c_runs:
        return {"total_runs": 0, "note": "Nenhuma execucao da Celula C encontrada."}

    write_tools = {"write_file", "append_file", "replace_text"}
    read_tools = {"read_file", "list_files", "search_text"}

    per_task: List[Dict[str, Any]] = []
    agg = Counter()

    for r in sorted(c_runs, key=lambda x: (x.get("task_id", ""), x.get("seed", 0))):
        actions_path = Path(r.get("actions_log", ""))
        actions: List[Dict[str, Any]] = []
        if actions_path.exists():
            actions = [
                json.loads(line)
                for line in actions_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        consecutive_same_path_writes = 0
        identical_write_repeats = 0
        total_writes = 0
        total_reads = 0
        tool_errors = 0
        prev_write_sig = None
        prev_write_path = None
        written_paths: List[str] = []

        for act in actions:
            tool = act.get("tool", "")
            args = act.get("args") or {}
            res = act.get("result") or {}
            if tool.startswith("("):
                continue
            if not res.get("ok", True):
                tool_errors += 1
            if tool in write_tools:
                total_writes += 1
                path = str(args.get("path", ""))
                written_paths.append(path)
                sig = (tool, path, json.dumps(args, sort_keys=True, ensure_ascii=False))
                if prev_write_path == path and path:
                    consecutive_same_path_writes += 1
                if prev_write_sig == sig:
                    identical_write_repeats += 1
                prev_write_path = path
                prev_write_sig = sig
            else:
                prev_write_path = None
                prev_write_sig = None
                if tool in read_tools:
                    total_reads += 1

        state = r.get("final_state", "")
        reason = r.get("failure_reason", "")
        role_calls = r.get("role_calls") or {}
        agg["total_runs"] += 1
        if state == "DONE":
            agg["done"] += 1
        else:
            agg["failed"] += 1
            agg[f"reason:{reason}"] += 1
        if consecutive_same_path_writes > 0:
            agg["runs_with_consecutive_same_path_writes"] += 1
        if identical_write_repeats > 0:
            agg["runs_with_identical_write_repeats"] += 1
        agg["sum_consecutive_same_path_writes"] += consecutive_same_path_writes
        agg["sum_identical_write_repeats"] += identical_write_repeats
        agg["sum_tool_errors"] += tool_errors

        per_task.append({
            "task_id": r.get("task_id"),
            "seed": r.get("seed"),
            "final_state": state,
            "failure_reason": reason,
            "iterations": r.get("iterations"),
            "role_calls": role_calls,
            "rho": r.get("rho"),
            "total_writes": total_writes,
            "total_reads": total_reads,
            "consecutive_same_path_writes": consecutive_same_path_writes,
            "identical_write_repeats": identical_write_repeats,
            "tool_errors": tool_errors,
            "unique_paths_written": sorted(set(p for p in written_paths if p)),
        })

    return {
        "summary": dict(agg),
        "per_run": per_task,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Consolidacao estatistica (10.d) e diagnostico Celula C (10.e)")
    ap.add_argument("--runs-file", default="experiments/e2e_hybrid/runs.jsonl")
    ap.add_argument("--temperature", type=float, default=None, help="Filtra por temperatura especifica (ex.: 0.0)")
    ap.add_argument("--bootstrap-resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="experiments/step10")
    args = ap.parse_args()

    runs_path = Path(args.runs_file)
    if not runs_path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {runs_path}")

    all_rows = [
        json.loads(line)
        for line in runs_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.temperature is not None:
        rows = [r for r in all_rows if abs(float(r.get("temperature", 0.0)) - args.temperature) < 1e-9]
    else:
        rows = all_rows

    by_cell: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_cell[str(r["cell"])].append(r)

    cells_summary: Dict[str, Any] = {}
    task_level: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(lambda: defaultdict(dict))

    print(f"=== CONSOLIDACAO E2E (10.d) | runs={len(rows)} | temp_filter={args.temperature} ===\n")
    for cell in sorted(by_cell.keys()):
        items = by_cell[cell]
        n = len(items)
        done = sum(1 for x in items if x.get("final_state") == "DONE")
        tsr = done / n if n else 0.0
        lo, hi = wilson_ci(done, n)
        mean_rho = sum(float(x.get("rho", 0.0)) for x in items) / n
        mean_iter = sum(float(x.get("iterations", 0)) for x in items) / n
        mean_tok = sum(float(x.get("total_tokens", 0)) for x in items) / n
        latencies = sorted(float(x.get("latency_seconds", 0.0)) for x in items)
        mean_lat = sum(latencies) / n
        med_lat = latencies[n // 2] if n % 2 == 1 else 0.5 * (latencies[n // 2 - 1] + latencies[n // 2])

        reasons: Dict[str, int] = {}
        by_task_items: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for x in items:
            by_task_items[str(x["task_id"])].append(x)
            if x.get("final_state") != "DONE":
                rsn = str(x.get("failure_reason") or "unknown")
                reasons[rsn] = reasons.get(rsn, 0) + 1

        for tid, t_items in by_task_items.items():
            task_level[cell]["tsr"][tid] = sum(1.0 if z.get("final_state") == "DONE" else 0.0 for z in t_items) / len(t_items)
            task_level[cell]["rho"][tid] = sum(float(z.get("rho", 0.0)) for z in t_items) / len(t_items)
            task_level[cell]["tokens"][tid] = sum(float(z.get("total_tokens", 0)) for z in t_items) / len(t_items)
            task_level[cell]["iterations"][tid] = sum(float(z.get("iterations", 0)) for z in t_items) / len(t_items)

        label = items[0].get("cell_label", cell)
        adapter = items[0].get("adapter_dir")
        cells_summary[cell] = {
            "label": label,
            "adapter_dir": adapter,
            "n_runs": n,
            "n_tasks": len(by_task_items),
            "done": done,
            "tsr": round(tsr, 4),
            "tsr_wilson_ci95": [round(lo, 4), round(hi, 4)],
            "mean_rho": round(mean_rho, 4),
            "mean_iterations": round(mean_iter, 2),
            "mean_tokens": round(mean_tok, 1),
            "mean_latency_s": round(mean_lat, 2),
            "median_latency_s": round(med_lat, 2),
            "failure_reasons": reasons,
        }
        print(
            f"Cell {cell} ({label}): TSR={done}/{n} ({tsr:.1%}, Wilson 95% [{lo:.1%}, {hi:.1%}]) | "
            f"rho={mean_rho:.4f} | iter={mean_iter:.2f} | tok={mean_tok:.1f} | "
            f"lat_med={med_lat:.1f}s (mean={mean_lat:.1f}s) | falhas={reasons}"
        )

    # Bootstrap pareado por tarefa
    comparisons = {}
    for pair_name, c1, c2 in [("C_vs_B", "C", "B"), ("C_vs_A", "C", "A"), ("A_vs_D", "A", "D")]:
        if c1 in task_level and c2 in task_level:
            comparisons[pair_name] = {
                "delta_tsr": paired_bootstrap_ci(task_level[c1]["tsr"], task_level[c2]["tsr"], args.bootstrap_resamples, args.seed),
                "delta_rho": paired_bootstrap_ci(task_level[c1]["rho"], task_level[c2]["rho"], args.bootstrap_resamples, args.seed),
                "delta_tokens": paired_bootstrap_ci(task_level[c1]["tokens"], task_level[c2]["tokens"], args.bootstrap_resamples, args.seed),
                "delta_iterations": paired_bootstrap_ci(task_level[c1]["iterations"], task_level[c2]["iterations"], args.bootstrap_resamples, args.seed),
            }

    print("\n=== BOOTSTRAP PAREADO POR TAREFA (B=10.000) ===")
    for k, v in comparisons.items():
        dtsr = v["delta_tsr"]
        drho = v["delta_rho"]
        print(
            f"  {k}: dTSR={dtsr['mean_delta']:+.4f} [{dtsr['ci95_low']:+.4f}, {dtsr['ci95_high']:+.4f}] | "
            f"dRho={drho['mean_delta']:+.4f} [{drho['ci95_low']:+.4f}, {drho['ci95_high']:+.4f}]"
        )

    diag_c = diagnose_cell_c(rows)
    print("\n=== DIAGNOSTICO CELULA C (10.e) ===")
    print("  Resumo:", diag_c.get("summary"))
    for item in diag_c.get("per_run", []):
        print(
            f"  - {item['task_id']} (seed={item['seed']}): {item['final_state']} ({item['failure_reason'] or 'ok'}) | "
            f"iter={item['iterations']} | roles={item['role_calls']} | writes={item['total_writes']} "
            f"(same_path_consec={item['consecutive_same_path_writes']}, identical_rep={item['identical_write_repeats']}) | "
            f"reads={item['total_reads']} | tool_errs={item['tool_errors']}"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "runs_file": str(runs_path),
        "temperature_filter": args.temperature,
        "bootstrap_resamples": args.bootstrap_resamples,
        "cells": cells_summary,
        "paired_bootstrap": comparisons,
        "cell_c_diagnostics": diag_c,
    }
    out_json = out_dir / "step10_report.json"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] Relatorio completo salvo em {out_json}")


if __name__ == "__main__":
    main()