import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List


def buscar_referencias_codigo(root: Path) -> List[Dict[str, Any]]:
    padroes = [
        r"max_tokens",
        r"MAX_TOKENS",
        r"max_iterations",
        r"MAX_ITERATIONS",
        r"max_repair_attempts",
        r"MAX_REPAIR_ATTEMPTS",
        r"context_window",
        r"max_context",
    ]
    regex = re.compile("|".join(padroes))
    achados: List[Dict[str, Any]] = []

    for pasta in ["src", "scripts", "tests"]:
        alvo = root / pasta
        if not alvo.exists(): continue
        for arq in sorted(alvo.rglob("*.py")):
            if arq.name == "inspect_max_tokens.py": continue
            for num_linha, linha in enumerate(arq.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
                if regex.search(linha):
                    achados.append({
                        "arquivo": str(arq.relative_to(root)),
                        "linha": num_linha,
                        "trecho": linha.strip(),
                    })
    return achados


def inspecionar_corpus_tasks(tasks_dir: Path) -> Dict[str, Any]:
    if not tasks_dir.exists(): return {"encontrado": False}

    tamanhos_prompt_chars: List[int] = []
    tamanhos_teste_chars: List[int] = []

    for task_json in sorted(tasks_dir.rglob("task.json")):
        try:
            data = json.loads(task_json.read_text(encoding="utf-8"))
            texto = json.dumps(data, ensure_ascii=False)
            tamanhos_prompt_chars.append(len(texto))
        except Exception:
            pass

    for test_py in sorted(tasks_dir.rglob("test_*.py")):
        tamanhos_teste_chars.append(len(test_py.read_text(encoding="utf-8", errors="ignore")))

    def resumo(valores: List[int]) -> Dict[str, Any]:
        if not valores: return {"count": 0}
        ordenados = sorted(valores)
        p95 = ordenados[min(len(ordenados) - 1, int(0.95 * (len(ordenados) - 1)))]
        return {
            "count": len(valores),
            "min_chars": min(valores),
            "avg_chars": round(sum(valores) / len(valores), 1),
            "p95_chars": p95,
            "max_chars": max(valores),
            "est_tokens_avg": round((sum(valores) / len(valores)) / 4, 1),
            "est_tokens_p95": round(p95 / 4, 1),
        }

    return {
        "encontrado": True,
        "task_json": resumo(tamanhos_prompt_chars),
        "test_files": resumo(tamanhos_teste_chars),
    }


def inspecionar_logs_existentes(root: Path) -> Dict[str, Any]:
    arquivos_jsonl = [
        p for p in sorted(root.rglob("*.jsonl"))
        if "corpus_filtered" not in p.parts
    ]
    linhas_totais = 0
    comprimentos_resposta_chars: List[int] = []
    comprimentos_prompt_chars: List[int] = []

    for arq in arquivos_jsonl:
        for linha in arq.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not linha.strip(): continue
            linhas_totais += 1
            try:
                obj = json.loads(linha)
            except Exception: continue
            if isinstance(obj, dict):
                for chave in ["raw_response", "response", "content", "output"]:
                    val = obj.get(chave)
                    if isinstance(val, str) and val.strip():
                        comprimentos_resposta_chars.append(len(val))
                        break
                for chave in ["prompt", "x", "input"]:
                    val = obj.get(chave)
                    if isinstance(val, str) and val.strip():
                        comprimentos_prompt_chars.append(len(val))
                        break

    def resumo_lista(vals: List[int]) -> Dict[str, Any]:
        if not vals: return {"count": 0}
        ord_v = sorted(vals)
        p95 = ord_v[min(len(ord_v) - 1, int(0.95 * (len(ord_v) - 1)))]
        return {
            "count": len(vals),
            "avg_chars": round(sum(vals) / len(vals), 1),
            "p95_chars": p95,
            "max_chars": max(vals),
            "est_tokens_p95": round(p95 / 4, 1),
            "est_tokens_max": round(max(vals) / 4, 1),
        }

    return {
        "arquivos_jsonl_encontrados": [str(p.relative_to(root)) for p in arquivos_jsonl],
        "linhas_totais": linhas_totais,
        "respostas": resumo_lista(comprimentos_resposta_chars),
        "prompts": resumo_lista(comprimentos_prompt_chars),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspeciona configurações de max_tokens e tamanhos no projeto")
    parser.add_argument("--root", default=".", help="Raiz do repositório MiniCode")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    refs = buscar_referencias_codigo(root)
    tasks_info = inspecionar_corpus_tasks(root / "corpus_tasks")
    logs_info = inspecionar_logs_existentes(root)

    print("=== 1. REFERÊNCIAS NO CÓDIGO (max_tokens / limites) ===")
    if not refs:
        print("  Nenhuma ocorrência explícita encontrada em src/, scripts/ ou tests/.")
    else:
        for r in refs:
            print(f"  {r['arquivo']}:{r['linha']} -> {r['trecho']}")

    print("\n=== 2. ESTIMATIVA EM CORPUS_TASKS ===")
    print(" ", json.dumps(tasks_info, ensure_ascii=False, indent=2))

    print("\n=== 3. LOGS .JSONL EXISTENTES NO REPOSITÓRIO ===")
    print(" ", json.dumps(logs_info, ensure_ascii=False, indent=2))

    print("\n=== 4. SUGESTÃO PRÁTICA PARA max_tokens ===")
    p95_resp_tok = logs_info.get("respostas", {}).get("est_tokens_p95")
    if p95_resp_tok:
        sugerido = max(256, min(1024, int(p95_resp_tok * 1.5)))
        print(f"  Com base no p95 das respostas em logs (~{p95_resp_tok} tokens), um teto seguro de saída é ~{sugerido}.")
    else:
        print("  Ainda não há respostas reais em logs para medir p95 de saída.")
        print("  Para ações JSON curtas (thought + tool + args), 256 ou 512 costuma bastar; se write_file trouxer arquivos inteiros, 512 ou 1024 dá mais margem na RTX 4050 (4 GB VRAM).")


if __name__ == "__main__":
    main()