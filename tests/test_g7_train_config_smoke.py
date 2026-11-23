import json
from pathlib import Path


def _load_consolidated_train_config_cls():
    nb_path = Path("notebooks/train.ipynb")
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    for cell in nb.get("cells", []):
        src = "".join(cell.get("source", []))
        if "class TrainConfig:" in src:
            ns: dict = {}
            exec(src, ns)
            return ns["TrainConfig"]
    raise AssertionError("Definição de TrainConfig não encontrada em notebooks/train.ipynb")


def test_g7_train_config_consolidated_and_compatible(tmp_path: Path):
    TrainConfig = _load_consolidated_train_config_cls()
    cfg = TrainConfig(output_dir=str(tmp_path / "ckpt"))

    assert cfg.model_id == "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    assert cfg.model_name == cfg.model_id
    assert cfg.train_file == cfg.train_path
    assert cfg.val_file == cfg.val_path
    assert cfg.test_file == cfg.test_path
    assert cfg.use_qlora_4bit is True
    assert cfg.use_4bit is True
    assert bool(cfg.system_prompt.strip())
    assert len(cfg.target_modules) == 7
