import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


def chave_grupo(item: Dict[str, Any], idx: int) -> str:
    task_id = item.get("task_id")
    if task_id:
        return f"task::{task_id}"
    run_id = item.get("run_id")
    if run_id:
        return f"run::{run_id}"
    return f"row::{idx}"


def carregar_mapa_manifesto(manifest_path: Path) -> Tuple[Dict[str, str], Dict[str, Set[str]]]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    splits_raw = data.get("splits")
    if not isinstance(splits_raw, dict):
        raise ValueError(f"Manifesto inválido em '{manifest_path}': chave 'splits' ausente.")

    task_to_split: Dict[str, str] = {}
    grupos_manifesto: Dict[str, Set[str]] = {"train": set(), "val": set(), "test": set()}

    for nome in ("train", "val", "test"):
        entrada = splits_raw.get(nome, [])
        if isinstance(entrada, dict):
            tarefas = entrada.get("task_ids") or list(entrada.keys())
        elif isinstance(entrada, list):
            tarefas = entrada
        else:
            raise ValueError(f"Formato inesperado em splits['{nome}'] no manifesto.")

        for tid in tarefas:
            tid_str = str(tid)
            if tid_str in task_to_split and task_to_split[tid_str] != nome:
                raise RuntimeError(f"Vazamento no manifesto: '{tid_str}' aparece em múltiplos splits.")
            task_to_split[tid_str] = nome
            grupos_manifesto[nome].add(f"task::{tid_str}")

    return task_to_split, grupos_manifesto


def dividir_grupos(
    grupos: List[Tuple[str, List[Dict[str, Any]]]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Set[str]]]:
    rng = random.Random(seed)
    grupos_embaralhados = list(grupos)
    rng.shuffle(grupos_embaralhados)

    total_exemplos = sum(len(exemplos) for _, exemplos in grupos_embaralhados)
    alvo_train = int(round(total_exemplos * train_ratio))
    alvo_val = int(round(total_exemplos * val_ratio))

    splits: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    grupos_por_split: Dict[str, Set[str]] = {"train": set(), "val": set(), "test": set()}

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
    parser.add_argument("--input-file", default="corpus_filtered/dataset.jsonl", help="Arquivo .jsonl de entrada")
    parser.add_argument("--out-dir", default="corpus_filtered", help="Diretório de saída dos splits")
    parser.add_argument(
        "--manifest",
        default="corpus_filtered/split_manifest.json",
        help="Caminho para split_manifest.json congelado (G5). Passe '' para split proporcional.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

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

    manifest_path = Path(args.manifest) if args.manifest else None
    if manifest_path and manifest_path.exists():
        task_to_split, _ = carregar_mapa_manifesto(manifest_path)
        splits: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
        grupos_por_split: Dict[str, Set[str]] = {"train": set(), "val": set(), "test": set()}

        for gid, exemplos in buckets.items():
            tid = gid.split("::", 1)[1] if gid.startswith("task::") else None
            if not tid or tid not in task_to_split:
                raise ValueError(f"Tarefa '{tid}' ({gid}) não encontrada em '{manifest_path}'.")
            destino = task_to_split[tid]
            splits[destino].extend(exemplos)
            grupos_por_split[destino].add(gid)
        modo = f"manifesto ({manifest_path})"
    else:
        if args.train_ratio <= 0 or args.val_ratio < 0 or (args.train_ratio + args.val_ratio) >= 1.0:
            raise ValueError("Use --train-ratio e --val-ratio positivos com soma < 1.0 (o restante vai para test).")
        grupos = list(buckets.items())
        splits, grupos_por_split = dividir_grupos(grupos, args.train_ratio, args.val_ratio, args.seed)
        modo = "proporcional"

    for nome in ("train", "val", "test"):
        salvar_jsonl(out_dir / f"{nome}.jsonl", splits[nome])

    vaz_tv = grupos_por_split["train"] & grupos_por_split["val"]
    vaz_tt = grupos_por_split["train"] & grupos_por_split["test"]
    vaz_vt = grupos_por_split["val"] & grupos_por_split["test"]
    vazamentos = vaz_tv | vaz_tt | vaz_vt

    print(f"Entrada: {in_path} | Modo: {modo} | Total exemplos: {total_linhas} | Total grupos: {len(buckets)}")
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
