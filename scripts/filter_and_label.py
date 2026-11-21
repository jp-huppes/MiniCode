import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def carregar_runs_metadata(runs_meta_path: Path) -> Dict[str, Dict[str, Any]]:
    runs: Dict[str, Dict[str, Any]] = {}
    if not runs_meta_path.exists():
        return runs
    for linha in runs_meta_path.read_text(encoding="utf-8").splitlines():
        if not linha.strip():
            continue
        item = json.loads(linha)
        rid = item.get("run_id")
        if rid:
            runs[rid] = item
    return runs


def normalizar_acao(parsed_action: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(parsed_action, dict):
        return None
    tool = parsed_action.get("tool") or parsed_action.get("action")
    if not tool or not isinstance(tool, str):
        return None
    args = parsed_action.get("args")
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return None
    return {"tool": tool.strip(), "args": args}


def tem_erro(val: Any) -> bool:
    if val is None:
        return False
    if isinstance(val, str):
        return len(val.strip()) > 0
    return bool(val)


def encontrar_actions_logs(raw_dir: Path) -> List[Path]:
    return sorted(raw_dir.rglob("actions.jsonl"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Etapa 4: Filtragem (F1-F5), rotulagem e auditoria")
    parser.add_argument("--raw-dir", default="corpus_raw", help="Diretório da coleta bruta (Etapa 3)")
    parser.add_argument("--out-file", default="corpus_filtered/dataset.jsonl", help="Caminho do dataset.jsonl")
    parser.add_argument("--audit-file", default="corpus_filtered/audit_sample_50.jsonl", help="Amostra de 50 para auditoria")
    parser.add_argument("--audit-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_root = Path(args.raw_dir)
    out_path = Path(args.out_file)
    audit_path = Path(args.audit_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    runs_meta = carregar_runs_metadata(raw_root / "runs_metadata.jsonl")
    action_files = encontrar_actions_logs(raw_root)

    stats = Counter({
        "total_linhas": 0,
        "drop_f1_parse": 0,
        "drop_f2_validator_or_gate": 0,
        "drop_f3_not_done": 0,
        "drop_f5_premature_done": 0,
        "drop_f4_duplicate_xy": 0,
        "mantidos": 0,
    })

    seen_xy = set()
    kept_examples: List[Dict[str, Any]] = []

    for log_file in action_files:
        for linha in log_file.read_text(encoding="utf-8").splitlines():
            if not linha.strip():
                continue
            stats["total_linhas"] += 1
            row = json.loads(linha)

            run_id = row.get("run_id")
            meta = runs_meta.get(run_id, {}) if run_id else {}
            final_state = str(meta.get("final_state") or row.get("final_state") or "").upper()

            # F1: Resposta parseável em JSON válido com tool/args
            acao = normalizar_acao(row.get("parsed_action"))
            if acao is None:
                stats["drop_f1_parse"] += 1
                continue

            val_err = row.get("validator_error")
            gate_err = row.get("gate_error")

            # F5: Remover 'done' prematuro rejeitado por gate (ou com erro)
            if acao["tool"] == "done" and (tem_erro(gate_err) or tem_erro(val_err)):
                stats["drop_f5_premature_done"] += 1
                continue

            # F2: validator_error e gate_error nulos
            if tem_erro(val_err) or tem_erro(gate_err):
                stats["drop_f2_validator_or_gate"] += 1
                continue

            # F3: Run terminou em DONE
            if final_state != "DONE":
                stats["drop_f3_not_done"] += 1
                continue

            prompt_x = row.get("prompt", "")
            y_json_canon = json.dumps(acao, sort_keys=True, ensure_ascii=False)
            xy_hash = hashlib.sha256(f"{prompt_x}\n---\n{y_json_canon}".encode("utf-8")).hexdigest()

            # F4: Remover duplicatas exatas de (x, y)
            if xy_hash in seen_xy:
                stats["drop_f4_duplicate_xy"] += 1
                continue
            seen_xy.add(xy_hash)

            example = {
                "run_id": run_id,
                "task_id": meta.get("task_id") or row.get("task_id"),
                "fsm_state": row.get("fsm_state"),
                "x": prompt_x,
                "y": acao,
                "tool": acao["tool"],
                "raw_response": row.get("raw_response"),
            }
            kept_examples.append(example)
            stats["mantidos"] += 1

    with open(out_path, "w", encoding="utf-8") as f:
        for ex in kept_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Amostra para auditoria manual (até 50 exemplos)
    rng = random.Random(args.seed)
    sample_n = min(args.audit_size, len(kept_examples))
    audit_sample = rng.sample(kept_examples, sample_n) if sample_n > 0 else []
    with open(audit_path, "w", encoding="utf-8") as f:
        for i, ex in enumerate(audit_sample, start=1):
            item = {"audit_id": i, "questionable": None, "notes": "", **ex}
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Distribuição de classes (ferramenta)
    class_counts = Counter(ex["tool"] for ex in kept_examples)
    total_kept = len(kept_examples)

    print("=== RESUMO DA FILTRAGEM (F1-F5) ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\n=== DISTRIBUIÇÃO DE CLASSES (TOOL) ===")
    if total_kept == 0:
        print("  Nenhum exemplo mantido.")
    else:
        for tool_name, cnt in class_counts.most_common():
            pct = (cnt / total_kept) * 100.0
            flag = " [ATENÇÃO: > 60%, F1 macro obrigatório]" if pct > 60.0 else ""
            print(f"  {tool_name}: {cnt} ({pct:.2f}%){flag}")

    print(f"\nDataset salvo em: {out_path.resolve()}")
    print(f"Amostra de auditoria ({sample_n} itens) salva em: {audit_path.resolve()}")


if __name__ == "__main__":
    main()