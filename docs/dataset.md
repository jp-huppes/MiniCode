# Dataset e Metodologia de Construção — Parte 2

## 1. Origem e Coleta (`scripts/collect_corpus.py`)
- **Benchmark base (`corpus_tasks/`):** 60 tarefas autônomas de programação em Python (`task_001` a `task_060`) com enunciado (`task.md`) e suíte `pytest`.
- **Coleta via Professor:** Execuções completas com `Qwen/Qwen2.5-14B-Instruct-AWQ` (`default_temperature = 0.1`), gravando `actions.jsonl`, `state.json` e `metrics.json`.

## 2. Filtragem Determinística (`F1`–`F6`) e Alinhamento de Prompt
- **F1:** JSON válido (`parse_ok = True`).
- **F2:** Ausência de erros de validação ou de gate.
- **F3:** Apenas trajetórias encerradas em `state == "DONE"`.
- **F4:** Deduplicação exata por hash do par `(contexto_normalizado, acao_canonica)`.
- **F5:** Remoção de chamadas `done` prematuras antes da verificação.
- **F6:** Validação estrita em `src/dataset/target_validator.py` (remoção recursiva de chaves `None` e bloqueio de aliases inválidos).
- **Prompt único (`minicode-prompt-v2`):** Compartilhado entre treino SFT (`src/dataset/build_sft_dataset.py`, `notebooks/train.ipynb`) e inferência (`src/agent/orchestrator.py`), cobrindo `list_files`, `read_file`, `write_file`, `replace_text`, `append_file` e `done`.

## 3. Splits Congelados (`corpus_filtered/split_manifest.json`)
Divisão por `task_id` com interseção zero (`SHA-256: a1f938ca0ef7a145532d94eef2720d6f8c9a4243ece8a9c840e3c4eea507ae4e`):

| Split | Tarefas no Manifesto | Tarefas com Runs `DONE` | Exemplos Limpos | Composição por Estado |
| :--- | :---: | :---: | :---: | :--- |
| **`train`** | 42 | 19 | **132** | 120 `IMPLEMENTING`, 12 `REPAIRING` |
| **`val`** | 6 | 3 | **43** | 38 `IMPLEMENTING`, 5 `REPAIRING` |
| **`test`** | 12 | 7 | **36** | 36 `IMPLEMENTING`, 0 `REPAIRING` |
| **Total** | **60** | **29** | **211** | **194 `IMPLEMENTING`, 17 `REPAIRING`** |

Na avaliação E2E (**Nível 2**, `scripts/run_e2e_hybrid.py`), são utilizadas **todas as 12 tarefas do split `test`** (`task_002`, `task_007`, `task_008`, `task_009`, `task_015`, `task_016`, `task_018`, `task_041`, `task_044`, `task_048`, `task_052`, `task_057`).
