import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Set


def sha256_lista(ids: List[str]) -> str:
    payload = json.dumps(ids, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def coletar_task_ids(dataset_path: Path, tasks_dir: Path) -> List[str]:
    task_ids: Set[str] = set()

    if dataset_path.exists():
        for linha in dataset_path.read_text(encoding="utf-8").splitlines():
            if not linha.strip(): continue
            item = json.loads(linha)
            tid = item.get("task_id")
            if tid: task_ids.add(str(tid))

    # fallback para corpus_tasks caso dataset.jsonl ainda esteja vazio/inexistente
    if not task_ids and tasks_dir.exists():
        for p in sorted(tasks_dir.iterdir()):
            if not p.is_dir(): continue
            meta_file = p / "task.json"
            if meta_file.exists():
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                task_ids.add(str(meta.get("task_id", p.name)))
            else:
                task_ids.add(p.name)

    return sorted(task_ids)


def particionar_tarefas(
    task_ids: List[str],
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
) -> Dict[str, List[str]]:
    ordenadas = sorted(task_ids)
    rng = random.Random(seed)
    embaralhadas = list(ordenadas)
    rng.shuffle(embaralhadas)

    n = len(embaralhadas)
    if n == 0: return {"train": [], "val": [], "test": []}

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    if n >= 3:
        n_train = max(1, min(n_train, n - 2))
        n_val = max(1, min(n_val, n - n_train - 1))

    train_ids = sorted(embaralhadas[:n_train])
    val_ids = sorted(embaralhadas[n_train : n_train + n_val])
    test_ids = sorted(embaralhadas[n_train + n_val :])

    assert set(train_ids).isdisjoint(set(val_ids)), "Interseção inválida entre treino e validação."
    assert set(train_ids).isdisjoint(set(test_ids)), "Interseção inválida entre treino e teste."
    assert set(val_ids).isdisjoint(set(test_ids)), "Interseção inválida entre validação e teste."
    assert set(train_ids) | set(val_ids) | set(test_ids) == set(ordenadas), "Faltam tarefas nas partições."

    return {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Etapa 5: Gera split_manifest.json congelado por task_id")
    parser.add_argument("--dataset", default="corpus_filtered/dataset.jsonl", help="Caminho do dataset.jsonl")
    parser.add_argument("--tasks-dir", default="corpus_tasks", help="Diretório fallback de tarefas")
    parser.add_argument("--out-file", default="corpus_filtered/split_manifest.json", help="Arquivo de saída")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    tasks_dir = Path(args.tasks_dir)
    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    task_ids = coletar_task_ids(dataset_path, tasks_dir)
    if not task_ids:
        raise RuntimeError("Nenhuma task_id encontrada em dataset.jsonl ou corpus_tasks.")

    splits = particionar_tarefas(
        task_ids,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    manifest: Dict[str, Any] = {
        "split_unit": "task_id",
        "seed": args.seed,
        "ratios_target": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": round(1.0 - args.train_ratio - args.val_ratio, 4),
        },
        "counts": {
            "total_tasks": len(task_ids),
            "train_tasks": len(splits["train"]),
            "val_tasks": len(splits["val"]),
            "test_tasks": len(splits["test"]),
        },
        "hashes_sha256": {
            "train": sha256_lista(splits["train"]),
            "val": sha256_lista(splits["val"]),
            "test": sha256_lista(splits["test"]),
        },
        "splits": splits,
    }
    manifest["hashes_sha256"]["manifest"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()

    out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Manifesto salvo em: {out_path.resolve()}")
    print(
        f"Tarefas ({len(task_ids)}): "
        f"train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}"
    )
    print(f"SHA-256 manifest: {manifest['hashes_sha256']['manifest']}")


if __name__ == "__main__":
    main()