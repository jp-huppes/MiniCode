"""Avaliação Offline Nível 1 (Passo 9: 9.a a 9.e) do MiniCode.

Avalia 5 sistemas no test.jsonl congelado com minicode-prompt-v2 e temperature=0.0:
  1. majority (classe majoritária do train.jsonl)
  2. base_1_5b_zero_shot (Qwen2.5-Coder-1.5B-Instruct sem adapter)
  3. finetuned_1_5b_lora (adapter vencedor do Passo 8 por val_loss)
  4. teacher_14b_zero_shot (modelo professor 14B via API)
  5. teacher_14b_few_shot (modelo professor 14B few-shot k=3 via API)

Calcula métricas gerais e por estrato, IC 95% por cluster bootstrap em task_id,
matrizes de confusão (destacando write_file, append_file e done) e G com IC 95%.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.agent.orchestrator import montar_system_prompt
from src.dataset.build_sft_dataset import DEFAULT_SYSTEM_PROMPT, PROMPT_VERSION

TOOLS_ORDER = [
    "list_files",
    "read_file",
    "write_file",
    "append_file",
    "replace_text",
    "done",
]


def load_dotenv(env_path: Path = Path(".env")) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path.resolve()}")
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def normalize_record(row: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """Unifica registros vindos de datasets/sft_jsonl/*.jsonl ou corpus_filtered/*.jsonl."""
    meta = row.get("metadata") or {}
    msgs = row.get("messages") or []

    user_prompt = row.get("x")
    if user_prompt is None:
        for m in msgs:
            if m.get("role") == "user":
                user_prompt = m.get("content", "")
                break

    ref_y = row.get("y")
    if not isinstance(ref_y, dict):
        asst_txt = ""
        for m in msgs:
            if m.get("role") == "assistant":
                asst_txt = m.get("content", "")
                break
        if asst_txt:
            try:
                ref_y = json.loads(asst_txt)
            except Exception:
                ref_y = {"tool": str(meta.get("tool") or row.get("tool") or ""), "args": {}}
        else:
            ref_y = {"tool": str(meta.get("tool") or row.get("tool") or ""), "args": {}}

    task_id = str(row.get("task_id") or meta.get("task_id") or f"task_{idx}")
    run_id = str(row.get("run_id") or meta.get("run_id") or f"run_{idx}")
    fsm_state = str(row.get("fsm_state") or meta.get("fsm_state") or "UNKNOWN").upper()

    return {
        "idx": idx,
        "task_id": task_id,
        "run_id": run_id,
        "fsm_state": fsm_state,
        "x": str(user_prompt or ""),
        "y": ref_y,
    }


def parse_action_response(raw_text: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw_text, str):
        return None
    text = raw_text.strip()
    if not text:
        return None
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            lines = lines[1:-1] if lines[-1].strip().startswith("```") else lines[1:]
            text = "\n".join(lines).strip()
    try:
        obj = json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except Exception:
            return None
    if not isinstance(obj, dict):
        return None
    tool = obj.get("tool") or obj.get("action")
    if not isinstance(tool, str) or not tool.strip():
        return None
    args = obj.get("args")
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return None
    return {"tool": tool.strip(), "args": args}


def normalize_args(args: Dict[str, Any]) -> Dict[str, Any]:
    norm: Dict[str, Any] = {}
    for k, v in args.items():
        if v is None:
            continue
        if k in {"path", "file_path", "target_file"} and isinstance(v, str):
            norm[k] = v.replace("\\", "/").removeprefix("./").strip()
        elif isinstance(v, str):
            norm[k] = v.replace("\r\n", "\n")
        else:
            norm[k] = v
    return norm


def compute_f1_macro(y_true: List[str], y_pred: List[str]) -> float:
    labels = sorted(set(y_true) | set(y_pred))
    if not labels:
        return 0.0
    f1s: List[float] = []
    for lbl in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lbl and p == lbl)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lbl and p == lbl)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lbl and p != lbl)
        if tp == 0:
            f1s.append(0.0)
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1s.append(2.0 * prec * rec / (prec + rec))
    return sum(f1s) / len(f1s)


def evaluate_subset(
    examples: List[Dict[str, Any]],
    raw_responses: List[Optional[str]],
) -> Dict[str, Any]:
    n = len(examples)
    if n == 0:
        return {
            "n": 0,
            "Acc_tool": 0.0,
            "JSON_valid": 0.0,
            "Exact_action": 0.0,
            "F1_macro": 0.0,
        }

    y_true: List[str] = []
    y_pred: List[str] = []
    acc_hits = 0
    json_hits = 0
    exact_hits = 0

    for ex, raw in zip(examples, raw_responses):
        ref_y = ex["y"]
        ref_tool = str(ref_y.get("tool", "")).strip()
        ref_args = normalize_args(ref_y.get("args") or {})

        parsed = parse_action_response(raw)
        if parsed is not None:
            json_hits += 1
            pred_tool = parsed["tool"]
            pred_args = normalize_args(parsed["args"])
        else:
            pred_tool = "__INVALID_JSON__"
            pred_args = {}

        if pred_tool == ref_tool:
            acc_hits += 1
            if pred_args == ref_args:
                exact_hits += 1

        y_true.append(ref_tool)
        y_pred.append(pred_tool)

    return {
        "n": n,
        "Acc_tool": round(acc_hits / n, 4),
        "JSON_valid": round(json_hits / n, 4),
        "Exact_action": round(exact_hits / n, 4),
        "F1_macro": round(compute_f1_macro(y_true, y_pred), 4),
    }


def cluster_bootstrap_metrics_by_task(
    examples: List[Dict[str, Any]],
    raw_responses: List[Optional[str]],
    n_resamples: int = 10000,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """IC 95% via cluster bootstrap reamostrando task_id com reposição."""
    by_task: Dict[str, List[int]] = defaultdict(list)
    for idx, ex in enumerate(examples):
        by_task[ex["task_id"]].append(idx)

    tasks = sorted(by_task.keys())
    if not tasks:
        return {}

    rng = random.Random(seed)
    dist: Dict[str, List[float]] = {
        "Acc_tool": [],
        "JSON_valid": [],
        "Exact_action": [],
        "F1_macro": [],
    }

    for _ in range(n_resamples):
        sampled_tasks = [rng.choice(tasks) for _ in range(len(tasks))]
        idxs = [i for t in sampled_tasks for i in by_task[t]]
        sub_ex = [examples[i] for i in idxs]
        sub_raw = [raw_responses[i] for i in idxs]
        m = evaluate_subset(sub_ex, sub_raw)
        for k in dist:
            dist[k].append(m[k])

    out: Dict[str, Dict[str, float]] = {}
    lo_idx = int(0.025 * n_resamples)
    hi_idx = min(n_resamples - 1, int(0.975 * n_resamples))
    for k, vals in dist.items():
        s = sorted(vals)
        out[k] = {
            "mean": round(sum(s) / len(s), 4),
            "ci95_low": round(s[lo_idx], 4),
            "ci95_high": round(s[hi_idx], 4),
        }
    return out


def paired_bootstrap_delta_by_task(
    examples: List[Dict[str, Any]],
    raw_a: List[Optional[str]],
    raw_b: List[Optional[str]],
    n_resamples: int = 10000,
    seed: int = 42,
) -> Dict[str, float]:
    """Bootstrap pareado por task_id para delta = Acc_tool(A) - Acc_tool(B)."""
    by_task: Dict[str, List[int]] = defaultdict(list)
    for idx, ex in enumerate(examples):
        by_task[ex["task_id"]].append(idx)

    tasks = sorted(by_task.keys())
    hits_a = [
        1 if (parse_action_response(r) or {}).get("tool") == ex["y"].get("tool") else 0
        for ex, r in zip(examples, raw_a)
    ]
    hits_b = [
        1 if (parse_action_response(r) or {}).get("tool") == ex["y"].get("tool") else 0
        for ex, r in zip(examples, raw_b)
    ]

    rng = random.Random(seed)
    deltas: List[float] = []
    gt_zero = 0
    for _ in range(n_resamples):
        sampled_tasks = [rng.choice(tasks) for _ in range(len(tasks))]
        idxs = [i for t in sampled_tasks for i in by_task[t]]
        acc_a = sum(hits_a[i] for i in idxs) / len(idxs)
        acc_b = sum(hits_b[i] for i in idxs) / len(idxs)
        d = acc_a - acc_b
        deltas.append(d)
        if d > 0:
            gt_zero += 1

    s = sorted(deltas)
    lo_idx = int(0.025 * n_resamples)
    hi_idx = min(n_resamples - 1, int(0.975 * n_resamples))
    return {
        "delta_mean": round(sum(s) / len(s), 4),
        "ci95_low": round(s[lo_idx], 4),
        "ci95_high": round(s[hi_idx], 4),
        "p_gt_zero": round(gt_zero / n_resamples, 4),
    }


def compute_gain_G_with_ci(
    examples: List[Dict[str, Any]],
    raw_ft: List[Optional[str]],
    raw_base: List[Optional[str]],
    raw_teacher: List[Optional[str]],
    n_resamples: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Calcula G = (A_ft - A_base) / (A_14B - A_base) e IC 95% por task_id."""
    a_ft = evaluate_subset(examples, raw_ft)["Acc_tool"]
    a_base = evaluate_subset(examples, raw_base)["Acc_tool"]
    a_14b = evaluate_subset(examples, raw_teacher)["Acc_tool"]

    denom = a_14b - a_base
    unstable = abs(denom) < 0.03
    g_point = round((a_ft - a_base) / denom, 4) if abs(denom) > 1e-9 else None

    by_task: Dict[str, List[int]] = defaultdict(list)
    for idx, ex in enumerate(examples):
        by_task[ex["task_id"]].append(idx)
    tasks = sorted(by_task.keys())

    hits_ft = [
        1 if (parse_action_response(r) or {}).get("tool") == ex["y"].get("tool") else 0
        for ex, r in zip(examples, raw_ft)
    ]
    hits_base = [
        1 if (parse_action_response(r) or {}).get("tool") == ex["y"].get("tool") else 0
        for ex, r in zip(examples, raw_base)
    ]
    hits_14b = [
        1 if (parse_action_response(r) or {}).get("tool") == ex["y"].get("tool") else 0
        for ex, r in zip(examples, raw_teacher)
    ]

    rng = random.Random(seed)
    g_samples: List[float] = []
    skipped_zero_denom = 0
    for _ in range(n_resamples):
        sampled_tasks = [rng.choice(tasks) for _ in range(len(tasks))]
        idxs = [i for t in sampled_tasks for i in by_task[t]]
        m_ft = sum(hits_ft[i] for i in idxs) / len(idxs)
        m_base = sum(hits_base[i] for i in idxs) / len(idxs)
        m_14b = sum(hits_14b[i] for i in idxs) / len(idxs)
        d = m_14b - m_base
        if abs(d) < 1e-6:
            skipped_zero_denom += 1
            continue
        g_samples.append((m_ft - m_base) / d)

    ci_low, ci_high, g_median = None, None, None
    if g_samples:
        s = sorted(g_samples)
        g_median = round(s[len(s) // 2], 4)
        ci_low = round(s[int(0.025 * len(s))], 4)
        ci_high = round(s[min(len(s) - 1, int(0.975 * len(s)))], 4)

    return {
        "A_ft": a_ft,
        "A_base": a_base,
        "A_14B": a_14b,
        "denominator_A14B_minus_Abase": round(denom, 4),
        "unstable_denominator_warning": unstable,
        "G_point": g_point,
        "G_bootstrap_median": g_median,
        "G_ci95_low": ci_low,
        "G_ci95_high": ci_high,
        "valid_bootstrap_resamples": len(g_samples),
        "skipped_zero_denom_resamples": skipped_zero_denom,
    }


def build_confusion_matrix(
    examples: List[Dict[str, Any]],
    raw_responses: List[Optional[str]],
) -> Dict[str, Any]:
    y_true: List[str] = []
    y_pred: List[str] = []
    for ex, raw in zip(examples, raw_responses):
        ref_tool = str(ex["y"].get("tool", "")).strip()
        parsed = parse_action_response(raw)
        pred_tool = parsed["tool"] if parsed is not None else "__INVALID_JSON__"
        y_true.append(ref_tool)
        y_pred.append(pred_tool)

    extra_labels = sorted((set(y_true) | set(y_pred)) - set(TOOLS_ORDER))
    labels = [t for t in TOOLS_ORDER if t in set(y_true) | set(y_pred)] + extra_labels

    matrix: Dict[str, Dict[str, int]] = {
        t: {p: 0 for p in labels} for t in labels
    }
    for t, p in zip(y_true, y_pred):
        matrix[t][p] += 1

    known_hotspots = {}
    for target_tool in ("write_file", "append_file", "done"):
        total = sum(1 for t in y_true if t == target_tool)
        if total == 0:
            known_hotspots[target_tool] = {"support": 0, "recall": None, "confusions": {}}
            continue
        correct = matrix.get(target_tool, {}).get(target_tool, 0)
        confusions = {
            pred: cnt
            for pred, cnt in matrix.get(target_tool, {}).items()
            if pred != target_tool and cnt > 0
        }
        known_hotspots[target_tool] = {
            "support": total,
            "correct": correct,
            "recall": round(correct / total, 4),
            "error_rate": round(1.0 - (correct / total), 4),
            "confusions": confusions,
        }

    return {
        "labels": labels,
        "matrix": matrix,
        "known_error_hotspots": known_hotspots,
    }


def resolve_selected_adapter(explicit_dir: str) -> Path:
    if explicit_dir:
        p = Path(explicit_dir)
        if not p.exists():
            raise FileNotFoundError(f"Adapter especificado não encontrado: {p}")
        return p

    metrics_file = Path("experiments/step8/training_metrics.json")
    if metrics_file.exists():
        data = json.loads(metrics_file.read_text(encoding="utf-8"))
        for key in ("selected_adapter_dir", "selected_adapter", "best_adapter_path"):
            val = data.get(key)
            if isinstance(val, str) and Path(val).exists():
                return Path(val)
            if isinstance(val, dict):
                for subk in ("adapter_dir", "path"):
                    if isinstance(val.get(subk), str) and Path(val[subk]).exists():
                        return Path(val[subk])

    for cand in [
        Path("checkpoints/step8/selected_adapter"),
        Path("checkpoints/qwen25_coder_1_5b_lora/final_adapter"),
    ]:
        if cand.exists():
            return cand

    raise FileNotFoundError(
        "Adapter vencedor do Passo 8 não encontrado. Extraia step8_checkpoints.zip ou passe --adapter-dir."
    )


def run_local_1_5b_systems(
    test_ex: List[Dict[str, Any]],
    system_prompt: str,
    base_model_id: str,
    adapter_dir: Path,
    cache_path: Path,
    max_new_tokens: int = 512,
) -> Tuple[List[str], List[str], Dict[str, float]]:
    if cache_path.exists():
        cached = load_jsonl(cache_path)
        if len(cached) == len(test_ex):
            print(f"[CACHE] Reaproveitando predições 1.5B de {cache_path}")
            return (
                [r["pred_base_1_5b"] for r in cached],
                [r["pred_finetuned_1_5b"] for r in cached],
                {"source": "cache"},
            )

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"[GPU] Carregando {base_model_id} + adapter {adapter_dir} (temp=0.0)...")
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_cfg,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.eval()
    model.config.use_cache = True
    device = next(model.parameters()).device

    def infer_all(tag: str) -> Tuple[List[str], float]:
        preds: List[str] = []
        t0 = time.perf_counter()
        with torch.inference_mode():
            for i, ex in enumerate(test_ex, 1):
                msgs = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": ex["x"]},
                ]
                prompt_txt = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
                inputs = tokenizer(prompt_txt, return_tensors="pt").to(device)
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                gen_ids = out_ids[0][inputs["input_ids"].shape[1] :]
                preds.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
                if i % 25 == 0 or i == len(test_ex):
                    print(f"  [{tag}] {i}/{len(test_ex)} concluídos")
        lat = (time.perf_counter() - t0) / max(1, len(test_ex))
        return preds, round(lat, 4)

    with model.disable_adapter():
        base_preds, base_lat = infer_all("1.5B base zero-shot")

    ft_preds, ft_lat = infer_all("1.5B+LoRA")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        for ex, pb, pf in zip(test_ex, base_preds, ft_preds):
            f.write(
                json.dumps(
                    {
                        "task_id": ex["task_id"],
                        "run_id": ex["run_id"],
                        "fsm_state": ex["fsm_state"],
                        "pred_base_1_5b": pb,
                        "pred_finetuned_1_5b": pf,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return base_preds, ft_preds, {"base_mean_latency_s": base_lat, "ft_mean_latency_s": ft_lat}


def build_few_shot_prefix(train_ex: List[Dict[str, Any]], k: int = 3) -> str:
    by_tool: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ex in train_ex:
        t = str(ex["y"].get("tool", "")).strip()
        if t:
            by_tool[t].append(ex)

    picked: List[Dict[str, Any]] = []
    for t in sorted(by_tool.keys()):
        if len(picked) < k:
            picked.append(by_tool[t][0])
    for ex in train_ex:
        if len(picked) >= k:
            break
        if ex not in picked:
            picked.append(ex)

    blocks = ["Exemplos de referência (responda exatamente neste formato JSON):"]
    for idx, ex in enumerate(picked[:k], 1):
        target_json = json.dumps(ex["y"], ensure_ascii=False, sort_keys=True)
        blocks.append(f"Exemplo {idx} — Entrada:\n{ex['x']}\nSaída:\n{target_json}")
    blocks.append("--- Agora resolva o próximo caso ---")
    return "\n\n".join(blocks) + "\n\n"


def run_teacher_14b_systems(
    train_ex: List[Dict[str, Any]],
    test_ex: List[Dict[str, Any]],
    system_prompt: str,
    base_url: str,
    api_key: str,
    model_14b: str,
    cache_path: Path,
    few_shot_k: int = 3,
    max_tokens: int = 512,
) -> Tuple[List[str], List[str]]:
    if cache_path.exists():
        cached = load_jsonl(cache_path)
        if len(cached) == len(test_ex):
            print(f"[CACHE] Reaproveitando predições 14B de {cache_path}")
            return (
                [r["pred_14b_zero_shot"] for r in cached],
                [r["pred_14b_few_shot"] for r in cached],
            )

    if not base_url:
        raise RuntimeError(
            "MINICODE_API_BASE / --base-url não configurado para avaliar o modelo professor 14B."
        )

    endpoint = base_url.rstrip("/") + "/chat/completions"

    def call_chat(user_content: str) -> str:
        payload = {
            "model": model_14b,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        }
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        for attempt in range(4):
            req = urllib.request.Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                    return body["choices"][0]["message"]["content"].strip()
            except Exception as exc:
                if attempt == 3:
                    return f"__API_ERROR__: {exc}"
                time.sleep(2 ** attempt)
        return ""

    print(f"[API] Avaliando professor 14B ({model_14b}) em {len(test_ex)} exemplos (temp=0.0)...")
    zero_preds: List[str] = []
    for i, ex in enumerate(test_ex, 1):
        zero_preds.append(call_chat(ex["x"]))
        if i % 25 == 0 or i == len(test_ex):
            print(f"  [14B zero-shot] {i}/{len(test_ex)} concluídos")

    prefix = build_few_shot_prefix(train_ex, k=few_shot_k)
    few_preds: List[str] = []
    for i, ex in enumerate(test_ex, 1):
        few_preds.append(call_chat(prefix + ex["x"]))
        if i % 25 == 0 or i == len(test_ex):
            print(f"  [14B few-shot]  {i}/{len(test_ex)} concluídos")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        for ex, pz, pf in zip(test_ex, zero_preds, few_preds):
            f.write(
                json.dumps(
                    {
                        "task_id": ex["task_id"],
                        "run_id": ex["run_id"],
                        "fsm_state": ex["fsm_state"],
                        "pred_14b_zero_shot": pz,
                        "pred_14b_few_shot": pf,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return zero_preds, few_preds


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Passo 9: Avaliação Offline Nível 1")
    parser.add_argument("--train-file", default="datasets/sft_jsonl/train.jsonl")
    parser.add_argument("--test-file", default="datasets/sft_jsonl/test.jsonl")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--adapter-dir", default="")
    parser.add_argument("--base-url", default=os.getenv("MINICODE_API_BASE", os.getenv("BASELINE_BASE_URL", "")))
    parser.add_argument("--api-key", default=os.getenv("MINICODE_API_KEY", os.getenv("BASELINE_API_KEY", "")))
    parser.add_argument(
        "--model-14b",
        default=os.getenv("MINICODE_TEACHER_MODEL", os.getenv("BASELINE_MODEL", "Qwen/Qwen2.5-14B-Instruct")),
    )
    parser.add_argument("--few-shot-k", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="experiments/step9")
    parser.add_argument("--only-1-5b", action="store_true", help="Roda apenas 1.5B base e 1.5B+LoRA (útil no Colab)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    canonical_prompt = montar_system_prompt(Path("."), catalogo_enxuto=True)
    assert canonical_prompt == DEFAULT_SYSTEM_PROMPT, "Divergência entre orchestrator e build_sft_dataset!"

    raw_train = load_jsonl(Path(args.train_file))
    raw_test = load_jsonl(Path(args.test_file))
    if raw_test and "messages" in raw_test[0]:
        assert raw_test[0]["messages"][0]["content"] == canonical_prompt, "test.jsonl não usa minicode-prompt-v2!"

    train_ex = [normalize_record(r, i) for i, r in enumerate(raw_train)]
    test_ex = [normalize_record(r, i) for i, r in enumerate(raw_test)]

    unique_tasks = sorted({ex["task_id"] for ex in test_ex})
    strata_counts = dict(sorted(Counter(ex["fsm_state"] for ex in test_ex).items()))
    print(
        f"[INFO] test.jsonl: {len(test_ex)} exemplos | {len(unique_tasks)} tarefas únicas | "
        f"estratos={strata_counts} | prompt={PROMPT_VERSION} | temp=0.0"
    )

    adapter_dir = resolve_selected_adapter(args.adapter_dir)
    preds_1_5b_path = out_dir / "predictions_1_5b.jsonl"
    base_preds, ft_preds, lat_info = run_local_1_5b_systems(
        test_ex=test_ex,
        system_prompt=canonical_prompt,
        base_model_id=args.base_model,
        adapter_dir=adapter_dir,
        cache_path=preds_1_5b_path,
        max_new_tokens=args.max_new_tokens,
    )

    if args.only_1_5b:
        print(f"[OK] Predições 1.5B salvas em {preds_1_5b_path}. Finalize no PC sem --only-1-5b.")
        return

    preds_14b_path = out_dir / "predictions_14b.jsonl"
    t14_zero_preds, t14_few_preds = run_teacher_14b_systems(
        train_ex=train_ex,
        test_ex=test_ex,
        system_prompt=canonical_prompt,
        base_url=args.base_url,
        api_key=args.api_key,
        model_14b=args.model_14b,
        cache_path=preds_14b_path,
        few_shot_k=args.few_shot_k,
        max_tokens=args.max_new_tokens,
    )

    # Sistema 1: Maioria (determinístico a partir de train.jsonl)
    train_tool_counts = Counter(str(ex["y"].get("tool", "")).strip() for ex in train_ex)
    maj_tool = train_tool_counts.most_common(1)[0][0]
    maj_raw = json.dumps({"tool": maj_tool, "args": {}}, ensure_ascii=False)
    maj_preds = [maj_raw for _ in test_ex]

    systems_preds: Dict[str, List[str]] = {
        "maioria": maj_preds,
        "base_1_5b_zero_shot": base_preds,
        "1_5b_lora": ft_preds,
        "14b_zero_shot": t14_zero_preds,
        "14b_few_shot": t14_few_preds,
    }

    strata = sorted(strata_counts.keys())
    systems_report: Dict[str, Any] = {}
    confusion_report: Dict[str, Any] = {}

    for sys_name, preds in systems_preds.items():
        overall = evaluate_subset(test_ex, preds)
        overall_ci = cluster_bootstrap_metrics_by_task(
            test_ex, preds, n_resamples=args.bootstrap_resamples, seed=args.seed
        )
        by_stratum: Dict[str, Any] = {}
        for st in strata:
            st_idxs = [i for i, ex in enumerate(test_ex) if ex["fsm_state"] == st]
            st_ex = [test_ex[i] for i in st_idxs]
            st_preds = [preds[i] for i in st_idxs]
            by_stratum[st] = {
                "metrics": evaluate_subset(st_ex, st_preds),
                "bootstrap_ci95_by_task": cluster_bootstrap_metrics_by_task(
                    st_ex, st_preds, n_resamples=args.bootstrap_resamples, seed=args.seed
                ),
            }

        systems_report[sys_name] = {
            "overall": overall,
            "bootstrap_ci95_by_task": overall_ci,
            "by_stratum": by_stratum,
        }
        confusion_report[sys_name] = build_confusion_matrix(test_ex, preds)

    # Comparações pareadas por task_id
    comparisons = {
        "1_5b_lora_vs_base_1_5b": paired_bootstrap_delta_by_task(
            test_ex, ft_preds, base_preds, args.bootstrap_resamples, args.seed
        ),
        "1_5b_lora_vs_maioria": paired_bootstrap_delta_by_task(
            test_ex, ft_preds, maj_preds, args.bootstrap_resamples, args.seed
        ),
        "1_5b_lora_vs_14b_zero_shot": paired_bootstrap_delta_by_task(
            test_ex, ft_preds, t14_zero_preds, args.bootstrap_resamples, args.seed
        ),
        "1_5b_lora_vs_14b_few_shot": paired_bootstrap_delta_by_task(
            test_ex, ft_preds, t14_few_preds, args.bootstrap_resamples, args.seed
        ),
    }

    # Cálculo de G (9.e) contra 14B zero-shot e 14B few-shot
    gain_vs_zero = compute_gain_G_with_ci(
        test_ex, ft_preds, base_preds, t14_zero_preds, args.bootstrap_resamples, args.seed
    )
    gain_vs_few = compute_gain_G_with_ci(
        test_ex, ft_preds, base_preds, t14_few_preds, args.bootstrap_resamples, args.seed
    )

    h1_pass = (
        comparisons["1_5b_lora_vs_base_1_5b"]["ci95_low"] > 0
        and comparisons["1_5b_lora_vs_maioria"]["ci95_low"] > 0
    )
    h2_pass = (
        gain_vs_zero["G_point"] is not None
        and gain_vs_zero["G_point"] >= 0.5
        and systems_report["1_5b_lora"]["overall"]["JSON_valid"] >= 0.95
    )

    full_report = {
        "prompt_version": PROMPT_VERSION,
        "temperature": 0.0,
        "bootstrap_unit": "task_id",
        "bootstrap_resamples": args.bootstrap_resamples,
        "seed": args.seed,
        "adapter_dir": str(adapter_dir).replace("\\", "/"),
        "teacher_14b_model": args.model_14b,
        "test_examples_count": len(test_ex),
        "unique_test_tasks_count": len(unique_tasks),
        "strata_counts": strata_counts,
        "latency_1_5b": lat_info,
        "systems": systems_report,
        "paired_bootstrap_by_task": comparisons,
        "recovery_gain_G": {
            "vs_14b_zero_shot": gain_vs_zero,
            "vs_14b_few_shot": gain_vs_few,
        },
        "hypotheses": {
            "H1_sft_beats_base_and_majority": {
                "supported": h1_pass,
                "delta_vs_base": comparisons["1_5b_lora_vs_base_1_5b"],
                "delta_vs_majority": comparisons["1_5b_lora_vs_maioria"],
            },
            "H2_recovers_teacher_gap_with_valid_json": {
                "supported": h2_pass,
                "G_vs_14b_zero_shot": gain_vs_zero,
                "G_vs_14b_few_shot": gain_vs_few,
                "json_valid_1_5b_lora": systems_report["1_5b_lora"]["overall"]["JSON_valid"],
            },
        },
        "confusion_matrices": confusion_report,
    }

    report_path = out_dir / "level1_report.json"
    report_path.write_text(json.dumps(full_report, ensure_ascii=False, indent=2), encoding="utf-8")

    # Gera tabela Markdown pronta para colar no relatório
    md_lines = [
        "# Passo 9 — Avaliação Offline Nível 1",
        "",
        f"- **Prompt:** `{PROMPT_VERSION}` | **Temperatura:** `0.0` | **Bootstrap:** `{args.bootstrap_resamples}` reamostragens por `task_id` (`N={len(test_ex)}`, `{len(unique_tasks)}` tarefas)",
        "",
        "## Resultados Gerais (com IC 95% por `task_id`)",
        "",
        "| Sistema | Acc_tool (IC 95%) | JSON_valid (IC 95%) | Exact_action (IC 95%) | F1_macro (IC 95%) |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    for sys_name in ("maioria", "base_1_5b_zero_shot", "1_5b_lora", "14b_zero_shot", "14b_few_shot"):
        ov = systems_report[sys_name]["overall"]
        ci = systems_report[sys_name]["bootstrap_ci95_by_task"]
        md_lines.append(
            f"| `{sys_name}` | "
            f"{ov['Acc_tool']:.4f} [{ci['Acc_tool']['ci95_low']:.4f}, {ci['Acc_tool']['ci95_high']:.4f}] | "
            f"{ov['JSON_valid']:.4f} [{ci['JSON_valid']['ci95_low']:.4f}, {ci['JSON_valid']['ci95_high']:.4f}] | "
            f"{ov['Exact_action']:.4f} [{ci['Exact_action']['ci95_low']:.4f}, {ci['Exact_action']['ci95_high']:.4f}] | "
            f"{ov['F1_macro']:.4f} [{ci['F1_macro']['ci95_low']:.4f}, {ci['F1_macro']['ci95_high']:.4f}] |"
        )

    md_lines.extend([
        "",
        "## Recuperação $G$ e Conclusão de H1 / H2",
        "",
        f"- **G (vs. 14B zero-shot):** `{gain_vs_zero['G_point']}` (IC 95%: `[{gain_vs_zero['G_ci95_low']}, {gain_vs_zero['G_ci95_high']}]`, denominador $A_{{14B}}-A_{{base}}={gain_vs_zero['denominator_A14B_minus_Abase']}$)",
        f"- **G (vs. 14B few-shot):** `{gain_vs_few['G_point']}` (IC 95%: `[{gain_vs_few['G_ci95_low']}, {gain_vs_few['G_ci95_high']}]`, denominador $A_{{14B}}-A_{{base}}={gain_vs_few['denominator_A14B_minus_Abase']}$)",
        f"- **H1 suportada:** `{h1_pass}` | **H2 suportada:** `{h2_pass}`",
        "",
        "## Pontos de Erro Conhecidos (`1_5b_lora` — `write_file`, `append_file`, `done`)",
        "",
    ])
    for tool_name, info in confusion_report["1_5b_lora"]["known_error_hotspots"].items():
        md_lines.append(
            f"- **`{tool_name}`** (suporte={info['support']}): recall=`{info['recall']}`, "
            f"erro=`{info['error_rate']}`, confusões={info['confusions']}"
        )

    summary_path = out_dir / "level1_summary.md"
    summary_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print("\n" + "\n".join(md_lines))
    print(f"\n[OK] Relatório completo salvo em: {report_path}")
    print(f"[OK] Resumo Markdown salvo em:   {summary_path}")


if __name__ == "__main__":
    main()