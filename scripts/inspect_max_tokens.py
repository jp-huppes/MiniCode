import argparse
import json
from pathlib import Path
from typing import List


def estimar_tokens(texto: str) -> int:
    if not texto:
        return 0
    return max(1, (len(texto) + 3) // 4)


def percentil(valores: List[int], p: float) -> int:
    if not valores:
        return 0
    ordenados = sorted(valores)
    idx = min(len(ordenados) - 1, max(0, int(round((p / 100.0) * (len(ordenados) - 1)))))
    return ordenados[idx]


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspeciona orçamento de tokens de x e y no dataset")
    parser.add_argument("--input-file", default="corpus_filtered/probe_dataset.jsonl", help="Arquivo .jsonl de entrada")
    parser.add_argument("--top", type=int, default=5, help="Quantidade de maiores exemplos para exibir")
    args = parser.parse_args()

    caminho = Path(args.input_file)
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")

    x_tokens: List[int] = []
    y_tokens: List[int] = []
    tot_tokens: List[int] = []
    registros = []

    for i, linha in enumerate(caminho.read_text(encoding="utf-8").splitlines(), start=1):
        if not linha.strip():
            continue
        item = json.loads(linha)
        x_str = item.get("x", "")
        y_obj = item.get("y", {})
        y_str = json.dumps(y_obj, ensure_ascii=False) if not isinstance(y_obj, str) else y_obj

        tx = estimar_tokens(str(x_str))
        ty = estimar_tokens(y_str)
        tt = tx + ty

        x_tokens.append(tx)
        y_tokens.append(ty)
        tot_tokens.append(tt)
        registros.append((tt, tx, ty, i, item.get("task_id"), item.get("tool")))

    print(f"Arquivo: {caminho} | Exemplos: {len(registros)}")
    if not registros:
        return

    for rotulo, arr in [("x (prompt)", x_tokens), ("y (alvo)", y_tokens), ("x + y (total)", tot_tokens)]:
        print(
            f"  {rotulo:14s} -> min={min(arr)} | p50={percentil(arr, 50)} | "
            f"p95={percentil(arr, 95)} | max={max(arr)}"
        )

    print(f"\nTop {args.top} maiores exemplos (total estimado x+y):")
    for tt, tx, ty, idx, task_id, tool in sorted(registros, reverse=True)[: args.top]:
        print(f"  linha={idx} | task={task_id} | tool={tool} | total={tt} (x={tx}, y={ty})")


if __name__ == "__main__":
    main()
