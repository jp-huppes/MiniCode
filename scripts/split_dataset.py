import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


def chave_grupo(item: Dict[str, Any], idx: int) -> str:
    task_id = item.get("task_id")
    if task_id:
        return f"task::{task_id}"
    run_id = item.get("run_id")
    if run_id:
        return f"run::{run_id}"
    return f"row::{idx}"


def dividir_grupos(
    grupos: List[Tuple[str, List[Dict[str, Any]]]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict[str, List[Dict[str, Any]]]:
    rng = random.Random(seed)
    grupos_embaralhados = list(grupos)
    rng.shuffle(grupos_embaralhados)

    total_exemplos = sum(len(exemplos) for _, exemplos in grupos_embaralhados)
    alvo_train = int(round(total_exemplos * train_ratio))
    alvo_val = int(round(total_exemplos * val_ratio))

    splits: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    grupos_por_split: Dict[str, set] = {"train": set(), "val": set(), "test": set()}

    for gid, exemplos in grupos_embaralhados:
        n_train = len(splits["train"])
        n_val = len(splits["val"])

        if n_train < alvo_train:
            destino = "train"
        elif n_val < alvo_val:
            destino = "val"
        else:
            destino = "test"

        splits[destino].extend(exemplos)
        grupos_por_split[destino].add(gid)

    return splits, grupos_por_split


def salvar_jsonl(caminho: Path, registros: List[Dict[str, Any]]) -> None:
    with open(caminho, "w", encoding="utf-8") as f:
        for reg in registros:
            f.write(json.dumps(reg, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Divide dataset.jsonl em train/val/test sem vazamento por task_id")
    parser.add_argument("--input-file", default="corpus_filtered/probe_dataset.jsonl", help="Arquivo .jsonl de entrada")
    parser.add_argument("--out-dir", default="corpus_filtered/splits", help="Diretório de saída dos splits")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.train_ratio <= 0 or args.val_ratio < 0 or (args.train_ratio + args.val_ratio) >= 1.0:
        raise ValueError("Use --train-ratio e --val-ratio positivos com soma < 1.0 (o restante vai para test).")

    in_path = Path(args.input_file)
    if not in_path.exists():
        raise FileNotFoundError(f"Arquivo de entrada não encontrado: {in_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    total_linhas = 0

    for idx, linha in enumerate(in_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not linha.strip():
            continue
        total_linhas += 1
        item = json.loads(linha)
        gid = chave_grupo(item, idx)
        buckets[gid].append(item)

    grupos = list(buckets.items())
    splits, grupos_por_split = dividir_grupos(grupos, args.train_ratio, args.val_ratio, args.seed)

    for nome in ("train", "val", "test"):
        salvar_jsonl(out_dir / f"{nome}.jsonl", splits[nome])

    # Verificação explícita de vazamento entre conjuntos
    vaz_tv = grupos_por_split["train"] & grupos_por_split["val"]
    vaz_tt = grupos_por_split["train"] & grupos_por_split["test"]
    vaz_vt = grupos_por_split["val"] & grupos_por_split["test"]
    vazamentos = vaz_tv | vaz_tt | vaz_vt

    print(f"Entrada: {in_path} | Total exemplos: {total_linhas} | Total grupos: {len(grupos)}")
    for nome in ("train", "val", "test"):
        print(
            f"  {nome:5s}: {len(splits[nome]):4d} exemplos | "
            f"{len(grupos_por_split[nome]):3d} grupos"
        )
    print(f"Vazamento de grupos entre splits: {len(vazamentos)}")
    if vazamentos:
        raise RuntimeError(f"Vazamento detectado entre splits: {sorted(vazamentos)}")
    print(f"Splits salvos em: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
