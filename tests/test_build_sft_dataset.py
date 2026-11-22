import json
import sys
from pathlib import Path
import pytest

from src.dataset import build_sft_dataset


def test_build_sft_dataset_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    manifest_path = tmp_path / "split_manifest.json"
    out_dir = tmp_path / "sft_out"

    exemplos = [
        {
            "run_id": "run-1",
            "task_id": "t-train-1",
            "fsm_state": "CODING",
            "x": "Escreva a fun??o soma em src/calc.py",
            "y": {"tool": "write_file", "args": {"path": "src/calc.py", "content": "def soma(a, b): return a + b"}},
        },
        {
            "run_id": "run-1",
            "task_id": "t-train-2",
            "fsm_state": "CODING",
            "x": "Leia src/calc.py",
            "y": {"tool": "read_file", "args": {"path": "src/calc.py"}},
        },
        {
            "run_id": "run-2",
            "task_id": "t-val-1",
            "fsm_state": "VERIFYING",
            "x": "Liste os arquivos do projeto",
            "y": {"tool": "list_files", "args": {}},
        },
        {
            "run_id": "run-3",
            "task_id": "t-test-1",
            "fsm_state": "DONE",
            "x": "Conclua a tarefa",
            "y": {"tool": "done", "args": {}},
        },
        {
            "run_id": "run-9",
            "task_id": "t-sem-split",
            "fsm_state": "CODING",
            "x": "Exemplo fora do manifesto",
            "y": {"tool": "list_files", "args": {}},
        },
    ]

    dataset_path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in exemplos) + "\n",
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(
            {
                "splits": {
                    "train": ["t-train-1", "t-train-2"],
                    "val": ["t-val-1"],
                    "test": ["t-test-1"],
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_sft_dataset.py",
            "--dataset",
            str(dataset_path),
            "--manifest",
            str(manifest_path),
            "--out-dir",
            str(out_dir),
        ],
    )

    build_sft_dataset.main()

    train_lines = (out_dir / "train.jsonl").read_text(encoding="utf-8").strip().splitlines()
    val_lines = (out_dir / "val.jsonl").read_text(encoding="utf-8").strip().splitlines()
    test_lines = (out_dir / "test.jsonl").read_text(encoding="utf-8").strip().splitlines()
    sft_manifest = json.loads((out_dir / "sft_manifest.json").read_text(encoding="utf-8"))

    assert len(train_lines) == 2
    assert len(val_lines) == 1
    assert len(test_lines) == 1
    assert sft_manifest["skipped_unassigned_task_ids"] == 1

    primeiro = json.loads(train_lines[0])
    assert [m["role"] for m in primeiro["messages"]] == ["system", "user", "assistant"]
    assert primeiro["messages"][1]["content"] == "Escreva a fun??o soma em src/calc.py"
    assert json.loads(primeiro["messages"][2]["content"]) == {
        "tool": "write_file",
        "args": {"path": "src/calc.py", "content": "def soma(a, b): return a + b"},
    }
    assert primeiro["metadata"]["split"] == "train"
    assert primeiro["metadata"]["task_id"] == "t-train-1"
    assert primeiro["metadata"]["tool"] == "write_file"


def test_montar_registro_suporta_prompt_string_y_e_unknown() -> None:
    reg_prompt_str = build_sft_dataset.montar_registro_sft(
        {
            "run_id": "r1",
            "task_id": "t1",
            "fsm_state": "CODING",
            "prompt": "Liste os arquivos",
            "y": '{"tool": "list_files", "args": {}}',
        },
        split_name="train",
        system_prompt="SYS",
    )
    assert reg_prompt_str is not None
    assert reg_prompt_str["messages"][1]["content"] == "Liste os arquivos"
    assert reg_prompt_str["messages"][2]["content"] == '{"tool": "list_files", "args": {}}'
    assert reg_prompt_str["metadata"]["tool"] == "unknown"

    reg_args_invalidos = build_sft_dataset.montar_registro_sft(
        {
            "run_id": "r2",
            "task_id": "t2",
            "fsm_state": "CODING",
            "x": "A??o sem args dict",
            "y": {"tool": "done", "args": ["nao", "e", "dict"]},
        },
        split_name="val",
        system_prompt="SYS",
    )
    assert reg_args_invalidos is not None
    assert json.loads(reg_args_invalidos["messages"][2]["content"]) == {"tool": "done", "args": {}}
    assert reg_args_invalidos["metadata"]["tool"] == "done"

    assert (
        build_sft_dataset.montar_registro_sft(
            {"task_id": "t3", "x": "   ", "y": {"tool": "done", "args": {}}},
            split_name="test",
            system_prompt="SYS",
        )
        is None
    )


def test_dataset_vazio_retorna_sem_erro(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    empty_dataset = tmp_path / "empty.jsonl"
    empty_dataset.write_text("", encoding="utf-8")
    out_dir = tmp_path / "out_empty"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_sft_dataset.py",
            "--dataset",
            str(empty_dataset),
            "--manifest",
            str(tmp_path / "nao_existe.json"),
            "--out-dir",
            str(out_dir),
        ],
    )

    build_sft_dataset.main()
    captured = capsys.readouterr()
    assert "nenhum exemplo encontrado" in captured.out
    assert not out_dir.exists()


@pytest.mark.parametrize(
    ("splits", "trecho_erro"),
    [
        ({"train": ["t1", "t2"], "val": ["t2"], "test": ["t3"]}, "Vazamento entre train e val"),
        ({"train": ["t1"], "val": ["t2"], "test": ["t1"]}, "Vazamento entre train e test"),
        ({"train": ["t1"], "val": ["t2", "t3"], "test": ["t3"]}, "Vazamento entre val e test"),
    ],
)
def test_validar_splits_detecta_vazamento(splits: dict, trecho_erro: str) -> None:
    with pytest.raises(ValueError, match=trecho_erro):
        build_sft_dataset.validar_splits_sem_vazamento(splits)
