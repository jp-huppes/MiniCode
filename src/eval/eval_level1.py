import json
import math
import re
from pathlib import Path
from typing import Any

# Ferramentas do catálogo do agente (sem exigir campos opcionais removidos quando None)
ALLOWED_TOOLS: dict[str, set[str]] = {
    "read_file": {"path", "start_line", "end_line"},
    "write_file": {"path", "content"},
    "append_file": {"path", "content"},
    "replace_text": {"path", "old_text", "new_text"},
    "list_files": {"path"},
    "done": {"summary"},
}

REQUIRED_ARGS: dict[str, set[str]] = {
    "read_file": {"path"},
    "write_file": {"path", "content"},
    "append_file": {"path", "content"},
    "replace_text": {"path", "old_text", "new_text"},
    "list_files": set(),
    "done": set(),
}


def strip_markdown_fence(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    cleaned = re.sub(r"^json\s*\n", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def try_parse_json_obj(raw: Any) -> tuple[bool, dict[str, Any] | None]:
    if isinstance(raw, dict):
        return True, raw
    if not isinstance(raw, str):
        return False, None

    candidate = strip_markdown_fence(raw)
    try:
        parsed = json.loads(candidate)
    except Exception:
        return False, None

    if not isinstance(parsed, dict):
        return False, None
    return True, parsed


def _strip_none_recursive(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip_none_recursive(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, list):
        return [_strip_none_recursive(item) for item in obj if item is not None]
    return obj


def is_schema_valid(raw: Any) -> tuple[bool, dict[str, Any] | None]:
    ok_json, obj = try_parse_json_obj(raw)
    if not ok_json or obj is None:
        return False, None

    # Exige exatamente as chaves 'tool' e 'args'
    if set(obj.keys()) != {"tool", "args"}:
        return False, None

    tool = obj.get("tool")
    raw_args = obj.get("args")

    if not isinstance(tool, str) or tool not in ALLOWED_TOOLS:
        return False, None
    if not isinstance(raw_args, dict):
        return False, None

    args = _strip_none_recursive(raw_args)
    if not isinstance(args, dict):
        return False, None

    allowed = ALLOWED_TOOLS[tool]
    required = REQUIRED_ARGS[tool]

    if not required.issubset(set(args.keys())):
        return False, None
    if not set(args.keys()).issubset(allowed):
        return False, None

    for k, v in args.items():
        if k in {"path", "content", "old_text", "new_text", "summary"}:
            if not isinstance(v, str):
                return False, None
        elif k in {"start_line", "end_line"}:
            if not isinstance(v, int) or isinstance(v, bool):
                return False, None

    return True, {"tool": tool, "args": args}


def normalize_args(args: dict[str, Any]) -> dict[str, Any]:
    cleaned = _strip_none_recursive(args)
    if not isinstance(cleaned, dict):
        return {}
    norm = {}
    for k, v in cleaned.items():
        if isinstance(v, str):
            norm[k] = v.replace("\r\n", "\n")
        else:
            norm[k] = v
    return norm


def wilson_or_normal_ci95(p_hat: float, n: int) -> tuple[float, float]:
    """IC 95% normal (p_hat +- 1.96 * sqrt(p*(1-p)/n))."""
    if n == 0:
        return (0.0, 0.0)
    margin = 1.96 * math.sqrt(max(0.0, p_hat * (1.0 - p_hat)) / n)
    return (max(0.0, p_hat - margin), min(1.0, p_hat + margin))


def evaluate_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        raise ValueError("Conjunto de avaliação vazio.")

    schema_valid_count = 0
    tool_correct_count = 0
    args_correct_count = 0
    errors: list[dict[str, Any]] = []

    for idx, row in enumerate(rows):
        target_raw = row.get("target", row.get("expected", row.get("y")))
        pred_raw = row.get("prediction", row.get("raw_output"))

        _, target_obj = try_parse_json_obj(target_raw)
        if target_obj is None:
            raise ValueError(f"Linha {idx} possui target/y inválido: {target_raw!r}")

        target_tool = target_obj["tool"]
        target_args = normalize_args(target_obj.get("args", {}))

        valid_schema, pred_obj = is_schema_valid(pred_raw)
        if valid_schema and pred_obj is not None:
            schema_valid_count += 1
            pred_tool = pred_obj["tool"]
            pred_args = normalize_args(pred_obj.get("args", {}))

            tool_ok = (pred_tool == target_tool)
            args_ok = tool_ok and (pred_args == target_args)
        else:
            _, loose_obj = try_parse_json_obj(pred_raw)
            pred_tool = loose_obj.get("tool") if isinstance(loose_obj, dict) else None
            tool_ok = False
            args_ok = False

        if tool_ok:
            tool_correct_count += 1
        if args_ok:
            args_correct_count += 1

        if not args_ok:
            errors.append({
                "index": idx,
                "schema_valid": valid_schema,
                "target_tool": target_tool,
                "pred_tool": pred_tool,
                "tool_correct": tool_ok,
                "args_correct": args_ok,
            })

    def pack_metric(count: int) -> dict[str, Any]:
        acc = count / n
        low, high = wilson_or_normal_ci95(acc, n)
        return {
            "count": count,
            "total": n,
            "rate": round(acc, 4),
            "ci95_low": round(low, 4),
            "ci95_high": round(high, 4),
        }

    return {
        "n_test": n,
        "SchemaValid": pack_metric(schema_valid_count),
        "Acc_tool": pack_metric(tool_correct_count),
        "Acc_args": pack_metric(args_correct_count),
        "n_errors": len(errors),
        "errors_sample": errors[:25],
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Avaliação Nível 1 - Tool Selection")
    parser.add_argument("--input", required=True, help="Caminho para predicoes_teste.jsonl")
    parser.add_argument("--output", default=None, help="Caminho opcional para salvar relatório JSON")
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    report = evaluate_predictions(rows)
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def load_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


if __name__ == "__main__":
    main()