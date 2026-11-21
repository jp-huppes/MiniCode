import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


def parse_action_response(raw_text: Any) -> Optional[Dict[str, Any]]:
    """Parser congelado: extrai objeto JSON com 'tool' (ou 'action') e 'args'."""
    if not isinstance(raw_text, str):
        return None
    texto = raw_text.strip()
    if not texto:
        return None

    # Remove cercas markdown ```json ... ``` se existirem
    if texto.startswith("```"):
        linhas = texto.splitlines()
        if len(linhas) >= 3:
            linhas = linhas[1:-1] if linhas[-1].strip().startswith("```") else linhas[1:]
            texto = "\n".join(linhas).strip()

    try:
        obj = json.loads(texto)
    except Exception:
        return None

    if not isinstance(obj, dict):
        return None
    tool = obj.get("tool") or obj.get("action")
    if not tool or not isinstance(tool, str):
        return None
    args = obj.get("args")
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return None
    return {"tool": tool.strip(), "args": args}


def normalizar_args(args: Dict[str, Any]) -> Dict[str, Any]:
    norm = {}
    for k, v in args.items():
        if k in {"path", "file_path", "target_file"} and isinstance(v, str):
            norm[k] = v.replace("\\", "/").lstrip("./").strip()
        else:
            norm[k] = v
    return norm


def compute_f1_macro(y_true: List[str], y_pred: List[str]) -> float:
    labels = sorted(set(y_true) | set(y_pred))
    if not labels:
        return 0.0
    f1s = []
    for lbl in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lbl and p == lbl)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lbl and p == lbl)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lbl and p != lbl)
        if tp == 0:
            f1s.append(0.0)
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1s.append(2 * prec * rec / (prec + rec))
    return sum(f1s) / len(f1s)


def evaluate_predictions(
    examples: List[Dict[str, Any]],
    raw_responses: List[Optional[str]],
) -> Dict[str, Any]:
    assert len(examples) == len(raw_responses)
    n = len(examples)
    if n == 0:
        return {"n": 0, "acc": 0.0, "json_validity": 0.0, "exact_action_acc": 0.0, "f1_macro": 0.0, "by_stratum": {}}

    y_true_tools: List[str] = []
    y_pred_tools: List[str] = []
    per_example_correct: List[int] = []
    parsed_count = 0
    exact_full_count = 0

    by_stratum_hits: Dict[str, List[int]] = defaultdict(list)

    for ex, raw in zip(examples, raw_responses):
        ref_action = ex.get("y") or {"tool": ex.get("tool", ""), "args": {}}
        ref_tool = str(ref_action.get("tool", "")).strip()
        ref_args = normalizar_args(ref_action.get("args") or {})
        stratum = str(ex.get("fsm_state") or "UNKNOWN").upper()

        parsed = parse_action_response(raw)
        if parsed is not None:
            parsed_count += 1
            pred_tool = parsed["tool"]
            pred_args = normalizar_args(parsed["args"])
        else:
            # Resposta não parseável conta como erro na métrica primária
            pred_tool = "__INVALID_JSON__"
            pred_args = {}

        hit = 1 if pred_tool == ref_tool else 0
        per_example_correct.append(hit)
        by_stratum_hits[stratum].append(hit)

        if hit == 1 and pred_args == ref_args:
            exact_full_count += 1

        y_true_tools.append(ref_tool)
        y_pred_tools.append(pred_tool)

    stratum_acc = {
        st: round(sum(vals) / len(vals), 4)
        for st, vals in sorted(by_stratum_hits.items())
        if vals
    }

    return {
        "n": n,
        "per_example_correct": per_example_correct,
        "acc": round(sum(per_example_correct) / n, 4),
        "json_validity": round(parsed_count / n, 4),
        "exact_action_acc": round(exact_full_count / n, 4),
        "f1_macro": round(compute_f1_macro(y_true_tools, y_pred_tools), 4),
        "by_stratum": stratum_acc,
    }


def paired_bootstrap_delta_ci(
    examples: List[Dict[str, Any]],
    correct_a: List[int],
    correct_b: List[int],
    n_resamples: int = 10000,
    seed: int = 42,
) -> Dict[str, float]:
    """Bootstrap pareado por run_id (10.000 reamostragens por padrão)."""
    by_run_indices: Dict[str, List[int]] = defaultdict(list)
    for idx, ex in enumerate(examples):
        rid = str(ex.get("run_id") or f"idx_{idx}")
        by_run_indices[rid].append(idx)

    runs = sorted(by_run_indices.keys())
    if not runs:
        return {"delta_mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "p_gt_zero": 0.0}

    rng = random.Random(seed)
    deltas: List[float] = []

    for _ in range(n_resamples):
        sampled_runs = [rng.choice(runs) for _ in range(len(runs))]
        hits_a: List[int] = []
        hits_b: List[int] = []
        for rid in sampled_runs:
            for idx in by_run_indices[rid]:
                hits_a.append(correct_a[idx])
                hits_b.append(correct_b[idx])
        acc_a = sum(hits_a) / len(hits_a)
        acc_b = sum(hits_b) / len(hits_b)
        deltas.append(acc_a - acc_b)

    deltas.sort()
    low_idx = int(0.025 * (n_resamples - 1))
    high_idx = int(0.975 * (n_resamples - 1))
    p_gt_zero = sum(1 for d in deltas if d > 0.0) / n_resamples

    return {
        "delta_mean": round(sum(deltas) / n_resamples, 4),
        "ci95_low": round(deltas[low_idx], 4),
        "ci95_high": round(deltas[high_idx], 4),
        "p_gt_zero": round(p_gt_zero, 4),
    }


def majority_class_baseline(
    train_examples: List[Dict[str, Any]],
    test_examples: List[Dict[str, Any]],
) -> List[str]:
    counts = Counter(
        str((ex.get("y") or {}).get("tool") or ex.get("tool") or "")
        for ex in train_examples
        if ((ex.get("y") or {}).get("tool") or ex.get("tool"))
    )
    majority_tool = counts.most_common(1)[0][0] if counts else "read_file"
    payload = json.dumps({"tool": majority_tool, "args": {}}, ensure_ascii=False)
    return [payload for _ in test_examples]


def build_few_shot_prefix(train_examples: List[Dict[str, Any]], k: int = 3) -> str:
    if k <= 0 or not train_examples:
        return ""
    escolhidos = train_examples[:k]
    blocos = ["Exemplos de referência (do split de treino):"]
    for i, ex in enumerate(escolhidos, start=1):
        y_obj = ex.get("y") or {"tool": ex.get("tool", ""), "args": {}}
        blocos.append(
            f"--- Exemplo {i} ---\n"
            f"Estado/Prompt:\n{ex.get('x', '')}\n"
            f"Resposta esperada:\n{json.dumps(y_obj, ensure_ascii=False)}"
        )
    blocos.append("--- Agora resolva o próximo caso ---\n")
    return "\n\n".join(blocos)


def run_llm_baseline(
    test_examples: List[Dict[str, Any]],
    chat_fn: Callable[[List[Dict[str, str]], float], str],
    few_shot_prefix: str = "",
) -> List[str]:
    respostas: List[str] = []
    for ex in test_examples:
        prompt_x = str(ex.get("x", ""))
        conteudo = f"{few_shot_prefix}{prompt_x}" if few_shot_prefix else prompt_x
        msgs = [{"role": "user", "content": conteudo}]
        # Temperatura congelada em 0.0 para avaliação de baselines
        raw = chat_fn(msgs, 0.0)
        respostas.append(raw)
    return respostas


def carregar_jsonl(caminho: Path) -> List[Dict[str, Any]]:
    if not caminho.exists():
        return []
    return [json.loads(l) for l in caminho.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Etapa 7: Avaliação de baselines e bootstrap pareado por run")
    parser.add_argument("--dataset", default="corpus_filtered/dataset.jsonl")
    parser.add_argument("--manifest", default="corpus_filtered/split_manifest.json")
    parser.add_argument("--few-shot-k", type=int, default=3)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-report", default="corpus_filtered/baselines_report.json")
    args = parser.parse_args()

    dataset = carregar_jsonl(Path(args.dataset))
    manifest_path = Path(args.manifest)

    if not dataset or not manifest_path.exists():
        print("Aviso: dataset.jsonl vazio ou split_manifest.json ausente.")
        print("Para testar a lógica de métricas e bootstrap agora, use o teste simulado ou aguarde os dados reais.")
        return

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    splits = manifest.get("splits", {})
    train_tasks = set(splits.get("train", []))
    test_tasks = set(splits.get("test", []))

    train_ex = [ex for ex in dataset if str(ex.get("task_id")) in train_tasks]
    test_ex = [ex for ex in dataset if str(ex.get("task_id")) in test_tasks]

    report: Dict[str, Any] = {
        "temperature": 0.0,
        "few_shot_k": args.few_shot_k,
        "bootstrap_resamples": args.bootstrap_resamples,
        "test_examples_count": len(test_ex),
        "systems": {},
        "comparisons": {},
    }

    maj_preds = majority_class_baseline(train_ex, test_ex)
    maj_metrics = evaluate_predictions(test_ex, maj_preds)
    report["systems"]["majority_class"] = {k: v for k, v in maj_metrics.items() if k != "per_example_correct"}

    # Ganchos prontos para plugar chat_fn do 1.5B base, 1.5B few-shot e Qwen-14B:
    # Ex.:
    # base_0shot_preds = run_llm_baseline(test_ex, chat_1_5b_fn, few_shot_prefix="")
    # base_kshot_preds = run_llm_baseline(test_ex, chat_1_5b_fn, few_shot_prefix=build_few_shot_prefix(train_ex, args.few_shot_k))
    # qwen14b_preds = run_llm_baseline(test_ex, chat_14b_fn, few_shot_prefix="")

    out_path = Path(args.out_report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Relatório salvo em: {out_path.resolve()}")
    print("Baseline classe majoritária:", report["systems"]["majority_class"])


if __name__ == "__main__":
    main()