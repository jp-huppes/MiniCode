from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import shutil
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    TrainerCallback,
    set_seed,
)


class _EmptyCacheCallback(TrainerCallback):
    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def on_evaluate(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
from trl import SFTConfig, SFTTrainer

from src.dataset.build_sft_dataset import PROMPT_VERSION


@dataclass
class Step8Config:
    model_id: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    train_file: str = "datasets/sft_jsonl/train.jsonl"
    val_file: str = "datasets/sft_jsonl/val.jsonl"
    manifest_file: str = "datasets/sft_jsonl/sft_manifest.json"
    checkpoints_root: str = "checkpoints/step8"
    experiments_dir: str = "experiments/step8"
    prompt_version: str = PROMPT_VERSION
    max_seq_length: int = 2048

    # QLoRA / LoRA (paridade com notebooks/train.ipynb)
    use_qlora_4bit: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    # SFT (batch efetivo = 1 * 16 = 16)
    num_train_epochs: int = 3
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16

    # Protocolo Passo 8
    seeds: tuple[int, ...] = (42, 43, 44)
    data_fractions: tuple[float, ...] = (0.25, 0.50, 1.00)
    curve_seed: int = 42


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_directory(directory: Path) -> dict[str, Any]:
    files = sorted(p for p in directory.rglob("*") if p.is_file())
    bundle_hasher = hashlib.sha256()
    per_file: dict[str, str] = {}
    for p in files:
        rel = p.relative_to(directory).as_posix()
        digest = sha256_file(p)
        per_file[rel] = digest
        bundle_hasher.update(rel.encode("utf-8"))
        bundle_hasher.update(b"\0")
        bundle_hasher.update(digest.encode("ascii"))
        bundle_hasher.update(b"\n")

    weights_file = directory / "adapter_model.safetensors"
    if not weights_file.exists():
        weights_file = directory / "adapter_model.bin"

    return {
        "adapter_bundle_sha256": bundle_hasher.hexdigest(),
        "adapter_model_sha256": per_file.get(weights_file.name, ""),
        "adapter_config_sha256": per_file.get("adapter_config.json", ""),
        "files_sha256": per_file,
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path.resolve()}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_nested_stratified_subsets(
    train_rows: list[dict[str, Any]],
    fractions: tuple[float, ...],
    seed: int = 42,
) -> dict[float, list[dict[str, Any]]]:
    """
    Cria subconjuntos aninhados (25% ⊂ 50% ⊂ 100%) estratificados pela tool alvo,
    garantindo que a curva de dados adicione exemplos de forma monotônica e determinística.
    """
    fracs_sorted = sorted(fractions)
    assert fracs_sorted[-1] == 1.0, "A maior fração deve ser 1.0 (100%)."

    by_tool: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(train_rows):
        tool = (row.get("y") or {}).get("tool")
        if not tool:
            assistant_obj = json.loads(row["messages"][-1]["content"])
            tool = str(assistant_obj.get("tool", "unknown"))
        by_tool[tool].append(idx)

    rng = random.Random(seed)
    shuffled_by_tool: dict[str, list[int]] = {}
    for tool in sorted(by_tool.keys()):
        indices = list(by_tool[tool])
        rng.shuffle(indices)
        shuffled_by_tool[tool] = indices

    subsets: dict[float, list[dict[str, Any]]] = {}
    for frac in fracs_sorted:
        if math.isclose(frac, 1.0):
            subsets[frac] = list(train_rows)
            continue

        chosen_indices: list[int] = []
        for tool, indices in shuffled_by_tool.items():
            k = max(1, int(round(len(indices) * frac)))
            chosen_indices.extend(indices[:k])

        # Mantém ordenação original antes do shuffle interno do Trainer
        chosen_indices.sort()
        subsets[frac] = [train_rows[i] for i in chosen_indices]

    return subsets


def make_completion_only_formatter(tokenizer, max_seq_length: int):
    def _format(row: dict[str, Any]) -> dict[str, Any]:
        messages = row.get("messages", [])
        if len(messages) < 3:
            raise ValueError("Esperado ao menos 3 mensagens: system, user e assistant.")

        prompt_text = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        prompt_ids = tokenizer(
            prompt_text,
            truncation=True,
            max_length=max_seq_length,
            add_special_tokens=False,
        )["input_ids"]
        full_ids = tokenizer(
            full_text,
            truncation=True,
            max_length=max_seq_length,
            add_special_tokens=False,
        )["input_ids"]

        prompt_len = min(len(prompt_ids), len(full_ids))
        labels = [-100] * prompt_len + full_ids[prompt_len:]
        if not any(lbl != -100 for lbl in labels):
            raise ValueError("Exemplo totalmente mascarado após truncamento!")

        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        }

    return _format


def cleanup_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_single_training(
    run_id: str,
    seed: int,
    fraction: float,
    train_subset: list[dict[str, Any]],
    val_ds: Dataset,
    tokenizer,
    cfg: Step8Config,
    force: bool = False,
) -> dict[str, Any]:
    run_dir = Path(cfg.checkpoints_root) / run_id
    final_adapter_dir = run_dir / "final_adapter"
    result_file = run_dir / "run_result.json"

    if (
        not force
        and result_file.exists()
        and (final_adapter_dir / "adapter_config.json").exists()
    ):
        cached = json.loads(result_file.read_text(encoding="utf-8"))
        print(
            f"[SKIP/CACHE] {run_id} já concluído -> best_val_loss={cached['best_val_loss']:.4f} "
            f"(sha256={cached['hashes']['adapter_model_sha256'][:12]}...)"
        )
        return cached

    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\n======================================================================\n"
        f"[RUN] {run_id} | seed={seed} | fraction={fraction:.2f} | n_train={len(train_subset)}\n"
        f"======================================================================"
    )

    cleanup_gpu()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)

    formatter = make_completion_only_formatter(tokenizer, cfg.max_seq_length)
    train_ds = Dataset.from_list(train_subset).map(
        formatter,
        remove_columns=list(train_subset[0].keys()),
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        quantization_config=bnb_config,
        dtype=torch.float16,
        attn_implementation="sdpa",
        device_map="auto",
    )
    base_model.config.use_cache = False
    base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)

    # Garante compatibilidade estrita com T4 (sem BFloat16 residual)
    for mod in model.modules():
        if getattr(mod, "compute_dtype", None) == torch.bfloat16:
            mod.compute_dtype = torch.float16
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.to(torch.float32)
        elif p.dtype == torch.bfloat16:
            p.data = p.data.to(torch.float16)
    for b in model.buffers():
        if b.dtype == torch.bfloat16:
            b.data = b.data.to(torch.float16)
    model.config.dtype = torch.float16
    model.config.torch_dtype = torch.float16

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable_params == 18_464_768, f"Esperado 18464768, obtido {trainable_params}"

    steps_per_epoch = max(
        1,
        math.ceil(
            len(train_ds)
            / (cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps)
        ),
    )
    logging_steps = max(1, min(5, steps_per_epoch // 2))

    sft_args = SFTConfig(
        output_dir=str(run_dir / "trainer_ckpts"),
        max_length=cfg.max_seq_length,
        optim="adamw_8bit",
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=logging_steps,
        fp16=False,
        bf16=False,
        seed=seed,
        data_seed=seed,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_num_proc=1,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, pad_to_multiple_of=8),
        callbacks=[_EmptyCacheCallback()],
    )

    t0 = time.perf_counter()
    train_result = trainer.train()
    elapsed_sec = round(time.perf_counter() - t0, 2)

    final_adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(final_adapter_dir))
    tokenizer.save_pretrained(str(final_adapter_dir))

    # Extrai histórico por época
    eval_by_epoch: dict[int, float] = {}
    train_losses_by_epoch: dict[int, list[float]] = defaultdict(list)
    for entry in trainer.state.log_history:
        if "eval_loss" in entry and "epoch" in entry:
            ep = max(1, int(round(float(entry["epoch"]))))
            eval_by_epoch[ep] = float(entry["eval_loss"])
        if "loss" in entry and "epoch" in entry:
            ep = max(1, int(math.ceil(float(entry["epoch"]) - 1e-9)))
            train_losses_by_epoch[ep].append(float(entry["loss"]))

    best_epoch = min(eval_by_epoch, key=lambda e: eval_by_epoch[e])
    best_val_loss = float(eval_by_epoch[best_epoch])
    if trainer.state.best_metric is not None:
        best_val_loss = float(trainer.state.best_metric)

    # Remove checkpoints intermediários pesados mantendo apenas final_adapter/
    ckpt_tmp = run_dir / "trainer_ckpts"
    if ckpt_tmp.exists():
        shutil.rmtree(ckpt_tmp)

    hashes = sha256_directory(final_adapter_dir)

    result_payload = {
        "run_id": run_id,
        "seed": seed,
        "data_fraction": fraction,
        "n_train_examples": len(train_subset),
        "n_val_examples": len(val_ds),
        "trainable_params": trainable_params,
        "elapsed_sec": elapsed_sec,
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 6),
        "val_loss_by_epoch": {str(k): round(v, 6) for k, v in sorted(eval_by_epoch.items())},
        "train_loss_mean_by_epoch": {
            str(k): round(sum(v) / len(v), 6)
            for k, v in sorted(train_losses_by_epoch.items())
        },
        "final_train_loss": round(float(train_result.metrics.get("train_loss", 0.0)), 6),
        "adapter_dir": final_adapter_dir.as_posix(),
        "hashes": hashes,
        "log_history": trainer.state.log_history,
    }

    result_file.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")
    print(
        f"[OK] {run_id} finalizado em {elapsed_sec}s | best_epoch={best_epoch} | "
        f"best_val_loss={best_val_loss:.4f} | adapter_sha256={hashes['adapter_model_sha256'][:16]}..."
    )

    if hasattr(trainer, 'accelerator'):
        trainer.accelerator.free_memory()
    del trainer, model, base_model
    cleanup_gpu()
    return result_payload


def render_step8_plots(
    seed_runs: list[dict[str, Any]],
    curve_runs: list[dict[str, Any]],
    mean_val: float,
    std_val: float,
    selected_run_id: str,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    # Painel 1: Estabilidade entre seeds (100% treino)
    seeds_labels = [f"seed={r['seed']}" for r in seed_runs]
    val_vals = [r["best_val_loss"] for r in seed_runs]
    colors = ["#2ca02c" if r["run_id"] == selected_run_id else "#1f77b4" for r in seed_runs]

    bars = axes[0].bar(seeds_labels, val_vals, color=colors, width=0.5, edgecolor="#222222")
    axes[0].axhline(mean_val, color="#d62728", linestyle="--", linewidth=1.5, label=f"Média = {mean_val:.4f}")
    axes[0].axhspan(
        mean_val - std_val,
        mean_val + std_val,
        color="#d62728",
        alpha=0.12,
        label=f"±1 DP = {std_val:.4f}",
    )
    for bar, v in zip(bars, val_vals):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            v + max(val_vals) * 0.012,
            f"{v:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    axes[0].set_ylim(0, max(val_vals) * 1.22)
    axes[0].set_title("Passo 8.a/8.b — Val Loss por Seed (100% Treino)")
    axes[0].set_ylabel("Melhor Validation Loss (menor = melhor)")
    axes[0].grid(axis="y", linestyle=":", alpha=0.5)
    axes[0].legend(loc="lower right", fontsize=8.5)

    # Painel 2: Curva de escala de dados (25%, 50%, 100%, seed=42)
    n_examples = [r["n_train_examples"] for r in curve_runs]
    frac_labels = [f"{int(round(r['data_fraction'] * 100))}%\n({r['n_train_examples']})" for r in curve_runs]
    curve_vals = [r["best_val_loss"] for r in curve_runs]

    axes[1].plot(n_examples, curve_vals, marker="o", linewidth=2.2, markersize=8, color="#9467bd")
    for x, y, lbl in zip(n_examples, curve_vals, frac_labels):
        axes[1].annotate(
            f"{y:.4f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
    axes[1].set_xticks(n_examples)
    axes[1].set_xticklabels(frac_labels)
    axes[1].set_ylim(min(curve_vals) * 0.85, max(curve_vals) * 1.15)
    axes[1].set_title("Passo 8.c — Curva de Dados (seed=42)")
    axes[1].set_xlabel("Tamanho do Conjunto de Treino")
    axes[1].set_ylabel("Melhor Validation Loss")
    axes[1].grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Passo 8 — Treino Final QLoRA (Seeds + Curva de Dados)")
    parser.add_argument("--force", action="store_true", help="Reexecuta mesmo se já houver run_result.json")
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.92)
    cfg = Step8Config()
    train_path = Path(cfg.train_file)
    val_path = Path(cfg.val_file)
    manifest_path = Path(cfg.manifest_file)
    exp_dir = Path(cfg.experiments_dir)
    ckpt_root = Path(cfg.checkpoints_root)
    exp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_root.mkdir(parents=True, exist_ok=True)

    # Carrega APENAS train.jsonl e val.jsonl (test.jsonl NÃO é lido no Passo 8)
    train_rows = load_jsonl(train_path)
    val_rows = load_jsonl(val_path)
    assert len(train_rows) == 527, f"Esperado 527 exemplos de treino, obtido {len(train_rows)}"
    assert len(val_rows) == 67, f"Esperado 67 exemplos de validação, obtido {len(val_rows)}"

    input_hashes = {
        "train_jsonl_sha256": sha256_file(train_path),
        "val_jsonl_sha256": sha256_file(val_path),
        "sft_manifest_sha256": sha256_file(manifest_path) if manifest_path.exists() else "",
    }
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_train_sha = manifest.get("splits", {}).get("train", {}).get("sha256")
        expected_val_sha = manifest.get("splits", {}).get("val", {}).get("sha256")
        if expected_train_sha:
            assert input_hashes["train_jsonl_sha256"] == expected_train_sha, "SHA-256 de train.jsonl diverge do manifesto!"
        if expected_val_sha:
            assert input_hashes["val_jsonl_sha256"] == expected_val_sha, "SHA-256 de val.jsonl diverge do manifesto!"

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    val_formatter = make_completion_only_formatter(tokenizer, cfg.max_seq_length)
    val_ds = Dataset.from_list(val_rows).map(
        val_formatter,
        remove_columns=list(val_rows[0].keys()),
    )

    subsets_by_frac = build_nested_stratified_subsets(
        train_rows=train_rows,
        fractions=cfg.data_fractions,
        seed=cfg.curve_seed,
    )

    # Agenda os 5 treinos consolidados (sem duplicar seed=42, frac=1.00)
    schedule: list[tuple[str, int, float, list[dict[str, Any]]]] = [
        ("seed42_frac100", 42, 1.00, subsets_by_frac[1.00]),
        ("seed43_frac100", 43, 1.00, subsets_by_frac[1.00]),
        ("seed44_frac100", 44, 1.00, subsets_by_frac[1.00]),
        ("seed42_frac025", 42, 0.25, subsets_by_frac[0.25]),
        ("seed42_frac050", 42, 0.50, subsets_by_frac[0.50]),
    ]

    all_results: dict[str, dict[str, Any]] = {}
    for run_id, seed, frac, subset in schedule:
        all_results[run_id] = run_single_training(
            run_id=run_id,
            seed=seed,
            fraction=frac,
            train_subset=subset,
            val_ds=val_ds,
            tokenizer=tokenizer,
            cfg=cfg,
            force=args.force,
        )

    # 8.a & 8.b: Estatísticas das 3 seeds no dataset completo
    seed_runs = [all_results[f"seed{s}_frac100"] for s in cfg.seeds]
    seed_val_losses = [r["best_val_loss"] for r in seed_runs]
    mean_val_loss = statistics.mean(seed_val_losses)
    std_val_loss_sample = statistics.stdev(seed_val_losses)
    std_val_loss_pop = statistics.pstdev(seed_val_losses)

    # 8.c: Curva de dados (25%, 50%, 100% com seed=42)
    curve_runs = [
        all_results["seed42_frac025"],
        all_results["seed42_frac050"],
        all_results["seed42_frac100"],
    ]

    # 8.d: Seleção estrita pela menor validation loss entre os 3 adapters completos (100% treino)
    best_seed_run = min(seed_runs, key=lambda r: (r["best_val_loss"], r["seed"]))
    selected_dir = ckpt_root / "selected_adapter"
    if selected_dir.exists():
        shutil.rmtree(selected_dir)
    shutil.copytree(Path(best_seed_run["adapter_dir"]), selected_dir)
    selected_hashes = sha256_directory(selected_dir)
    assert (
        selected_hashes["adapter_bundle_sha256"]
        == best_seed_run["hashes"]["adapter_bundle_sha256"]
    )

    metrics_summary = {
        "step": "step8_final_training",
        "model_id": cfg.model_id,
        "prompt_version": cfg.prompt_version,
        "test_set_touched": False,
        "hyperparameters": {
            **asdict(cfg),
            "target_modules": list(cfg.target_modules),
            "seeds": list(cfg.seeds),
            "data_fractions": list(cfg.data_fractions),
        },
        "dataset_splits_sha256": input_hashes,
        "multi_seed_100pct": {
            "seeds": list(cfg.seeds),
            "n_runs": len(seed_runs),
            "mean_val_loss": round(mean_val_loss, 6),
            "std_val_loss": round(std_val_loss_sample, 6),
            "pstdev_val_loss": round(std_val_loss_pop, 6),
            "runs": [
                {
                    "run_id": r["run_id"],
                    "seed": r["seed"],
                    "n_train_examples": r["n_train_examples"],
                    "best_epoch": r["best_epoch"],
                    "best_val_loss": r["best_val_loss"],
                    "val_loss_by_epoch": r["val_loss_by_epoch"],
                    "final_train_loss": r["final_train_loss"],
                    "elapsed_sec": r["elapsed_sec"],
                    "adapter_dir": r["adapter_dir"],
                    "adapter_model_sha256": r["hashes"]["adapter_model_sha256"],
                    "adapter_bundle_sha256": r["hashes"]["adapter_bundle_sha256"],
                }
                for r in seed_runs
            ],
        },
        "data_scaling_curve": {
            "seed": cfg.curve_seed,
            "sampling": "nested_stratified_by_tool",
            "points": [
                {
                    "run_id": r["run_id"],
                    "data_fraction": r["data_fraction"],
                    "n_train_examples": r["n_train_examples"],
                    "best_epoch": r["best_epoch"],
                    "best_val_loss": r["best_val_loss"],
                    "val_loss_by_epoch": r["val_loss_by_epoch"],
                    "final_train_loss": r["final_train_loss"],
                    "elapsed_sec": r["elapsed_sec"],
                    "adapter_model_sha256": r["hashes"]["adapter_model_sha256"],
                    "adapter_bundle_sha256": r["hashes"]["adapter_bundle_sha256"],
                }
                for r in curve_runs
            ],
        },
        "selected_adapter": {
            "selection_rule": "min_val_loss_across_full_train_seeds",
            "selected_run_id": best_seed_run["run_id"],
            "selected_seed": best_seed_run["seed"],
            "selected_epoch": best_seed_run["best_epoch"],
            "selected_val_loss": best_seed_run["best_val_loss"],
            "selected_adapter_dir": selected_dir.as_posix(),
            "source_adapter_dir": best_seed_run["adapter_dir"],
            "adapter_model_sha256": best_seed_run["hashes"]["adapter_model_sha256"],
            "adapter_config_sha256": best_seed_run["hashes"]["adapter_config_sha256"],
            "adapter_bundle_sha256": best_seed_run["hashes"]["adapter_bundle_sha256"],
            "files_sha256": best_seed_run["hashes"]["files_sha256"],
        },
    }

    metrics_file = exp_dir / "training_metrics.json"
    metrics_file.write_text(json.dumps(metrics_summary, indent=2), encoding="utf-8")

    plot_file = exp_dir / "step8_curves.png"
    render_step8_plots(
        seed_runs=seed_runs,
        curve_runs=curve_runs,
        mean_val=mean_val_loss,
        std_val=std_val_loss_sample,
        selected_run_id=best_seed_run["run_id"],
        out_path=plot_file,
    )

    print("\n==================== RESUMO PASSO 8 (8.a - 8.e) ====================")
    for r in seed_runs:
        print(
            f"  Seed {r['seed']} (100%): best_val_loss={r['best_val_loss']:.4f} "
            f"(ep {r['best_epoch']}) | sha256={r['hashes']['adapter_model_sha256'][:16]}..."
        )
    print(
        f"  Média Val Loss (3 seeds): {mean_val_loss:.4f} ± {std_val_loss_sample:.4f} (amostral) "
        f"| ± {std_val_loss_pop:.4f} (populacional)"
    )
    print("  Curva de dados (seed=42):")
    for r in curve_runs:
        print(
            f"    {int(round(r['data_fraction']*100)):3d}% ({r['n_train_examples']:3d} ex): "
            f"best_val_loss={r['best_val_loss']:.4f} (ep {r['best_epoch']})"
        )
    print(
        f"  Adapter escolhido (8.d): {best_seed_run['run_id']} "
        f"(val_loss={best_seed_run['best_val_loss']:.4f}, "
        f"sha256={best_seed_run['hashes']['adapter_model_sha256']})"
    )
    print(f"  Métricas salvas em: {metrics_file.resolve()}")
    print(f"  Gráfico salvo em  : {plot_file.resolve()}")


if __name__ == "__main__":
    main()