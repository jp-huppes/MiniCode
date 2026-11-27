# Documentação do Dataset SFT — Parte 2

Este documento registra a construção, critério de parada da coleta, filtragem determinística, auditoria manual, particionamento sem vazamento, orçamento de tokens e formatação SFT do conjunto de dados utilizado no ajuste supervisionado (`SFT + QLoRA`) do MiniCode.

## 1. Origem, Coleta e Critério de Parada

- **Modelo Gerador (Professor):** `Qwen/Qwen2.5-14B-Instruct-AWQ` executando o loop completo do `AgentOrchestrator` sobre o benchmark curado de 60 tarefas de programação em Python (`corpus_tasks/`).
- **Rastreabilidade:** cada execução (*run*) registra metadados em `corpus_raw_full/runs_metadata.jsonl` (`run_id`, `task_id`, `seed`, `temperature`, `commit`, `prompt_version`, `max_iterations`, `max_repair_attempts`, `final_state`, `repair_attempts`) e o log passo a passo de ações em `.minicode/actions.jsonl`.
- **Alinhamento Treino = Inferência:** todas as trajetórias válidas utilizam o prompt compacto unificado `minicode-prompt-v2` (`catalogo_enxuto=True`), cobrindo exclusivamente as 6 ferramentas ativas (`list_files`, `read_file`, `write_file`, `replace_text`, `append_file`, `done`).
- **Nota sobre a Temperatura Efetiva:** conforme registrado no Desvio 3 de `preregistration.md`, a grade de coleta variou explicitamente sementes (`seed` $\in \{1, 2, 3\}$) e temperaturas nominais (`0.0` e `0.3`), enquanto `LLMClient` operou com amostragem estocástica natural (`default_temperature = 0.1`), gerando diversidade real entre trajetórias e evitando descarte excessivo por duplicação no filtro **F4**. A geração de `specification.md` e `plan.md` (`experiments/spec_cache/`) e as avaliações oficiais operam deterministicamente com `temperature = 0.0`.
- **Aplicação da Regra de Parada:** a grade planejada foi executada integralmente, totalizando **364 *runs* brutas** (`60` tarefas únicas; `192 DONE`, `170 FAILED`, `2 RECEIVED`; distribuição por `(seed, temp)`: `(1, 0.0)=63`, `(1, 0.3)=61`, `(2, 0.0)=60`, `(2, 0.3)=60`, `(3, 0.0)=60`, `(3, 0.3)=60`). Após filtragem **F1–F6**, o dataset atingiu **762 exemplos limpos** cobrindo **45/60 tarefas**, **10/12 tarefas do split `test`** e todas as **6 ferramentas** do contrato (crescimento de $3{,}61\times$ sobre a parcial inicial de 211 exemplos), satisfazendo o critério de suficiência e encerrando a coleta.

## 2. Filtragem Determinística (Regras F1–F6) e Auditoria Manual

O pipeline `scripts/filter_and_label.py` processou **3.781 passos brutos** das 364 *runs* e aplicou seis filtros determinísticos em cascata antes de exportar `corpus_filtered/dataset.jsonl`:

| Regra | Critério Determinístico | Finalidade | Descartados | Retidos |
| :--- | :--- | :--- | :---: | :---: |
| **Entrada bruta** | Passos registrados em `.minicode/actions.jsonl` (364 *runs*) | Base total da coleta | — | `3.781` |
| **F1** | Resposta estruturada em JSON válido (`parse_ok == True`) | Descartar saídas malformadas ou truncadas | `420` | `3.361` |
| **F2** | Ausência de `validator_error` e `gate_error` | Garantir conformidade com o sandbox e políticas de caminho | `464` | `2.897` |
| **F3** | Trajetórias encerradas exclusivamente em `final_state == "DONE"` | Evitar destilação de sequências que falharam nos testes | `2.002` | `895` |
| **F5** | Remoção de chamadas `done` prematuras antes da etapa de verificação | Impedir encerramento sem aprovação do `pytest` | `0` | `895` |
| **F6** | Validação estrita de schema em `src/dataset/target_validator.py` (remoção recursiva de chaves `None` e rejeição de aliases inválidos como `file_path`) | Assegurar o contrato canônico `{"tool": <nome>, "args": <dict>}` | `0` | `895` |
| **F4** | Deduplicação exata por hash do par `(contexto_normalizado, acao_canonica)` | Eliminar exemplos idênticos entre execuções | `133` | **`762`** |

### Auditoria Manual dos 50 Exemplos (`corpus_filtered/audit_sample_50.jsonl`)

- **Resultado:** **`1 / 50` questionável (`2,0%`)**, abaixo do limiar de `10%` definido em `preregistration.md` (não exigindo revisão das regras F1–F6).
- **Ocorrência identificada (`audit_id = 49`, `task_003`):** passo intermediário `write_file` em `src/pure_ops.py` válido sob o contrato JSON/sandbox, porém contendo docstring malformada (`""` em vez de aspas triplas) com `SyntaxError` antes da correção aplicada nos passos subsequentes da mesma trajetória `DONE`. Os demais `49` exemplos (`98,0%`) foram aprovados sem ressalvas.

## 3. Orçamento de Tokens e Verificação de Duplicatas

### 3.1 Distribuição de Comprimento (`scripts/inspect_max_tokens.py`, 762 exemplos)

| Segmento | Mínimo | Mediana (`p50`) | Percentil 95 (`p95`) | Máximo (`max`) | Exemplos `> 2048` tokens |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`x` (prompt de entrada)** | `223` | `376` | `597` | `1.092` | `0 / 762` (`0,0%`) |
| **`y` (ação JSON alvo)** | `7` | `41` | `119` | `286` | `0 / 762` (`0,0%`) |
| **`x + y` (total bruto)** | `263` | `412` | **`697`** | **`1.172`** | **`0 / 762` (`0,0%`)** |
| **SFT `train` (`system+user+assistant`)** | — | — | — | `1.670` | `0 / 527` (`0,0%`) |
| **SFT `val` (`system+user+assistant`)** | — | — | — | `1.537` | `0 / 67` (`0,0%`) |
| **SFT `test` (`system+user+assistant`)** | — | — | — | `1.227` | `0 / 168` (`0,0%`) |

- **Decisão de `max_length` e `max_tokens`:** como **0 exemplos (`0,0%`)** excedem `2.048` tokens mesmo considerando o template conversacional completo (`max = 1.670` tokens no `train`, margem de `378` tokens), mantém-se `max_length = 2048` sem descarte por truncamento. O `p95` de `x + y` é **`697` tokens** (e o `p95` da resposta alvo `y` é `119` tokens, `max = 286`), validando `max_new_tokens = 512` (e orçamento total $\ge 1.024$ tokens) para geração sem truncamento.

### 3.2 Inspeção de Tarefas Quase Duplicadas entre Treino e Teste

- A comparação par-a-par de similaridade lexical (`SequenceMatcher`) entre as especificações das tarefas de `test` e `train` apresentou similaridade máxima de **`0,610`** (`task_052` `filtrar_pares` vs. `task_054` `media_segura`), atribuída exclusivamente ao cabeçalho estrutural fixo das especificações (`## Objective`, `## Parameters`, `## Acceptance Criteria`, `## Tests`). **Nenhuma tarefa duplicada ou quase duplicada** existe entre treino e teste.

## 4. Formato dos Exemplos

### 4.1 Corpus Filtrado (`corpus_filtered/dataset.jsonl`)

Cada linha representa um par estado-ação supervisionado (`minicode-prompt-v2`):

```json
{
  "example_id": "task_003__run_01__step_04",
  "task_id": "task_003",
  "run_id": "run_01",
  "step_index": 4,
  "fsm_state": "IMPLEMENTING",
  "prompt_version": "minicode-prompt-v2",
  "x": "<contexto formatado com SPEC, PLAN, estado atual e histórico recente>",
  "y": {"tool": "write_file", "args": {"path": "solution.py", "content": "..."}}
}
```

### 4.2 Formato Conversacional SFT (`datasets/sft_jsonl/*.jsonl`)

O módulo `src/dataset/build_sft_dataset.py` converte cada exemplo para mensagens (`system`, `user`, `assistant`) acompanhado do manifesto `datasets/sft_jsonl/sft_manifest.json` (`skipped_unassigned_task_ids = 0`).

## 5. Partição sem Vazamento e Tabela Final do Dataset

O particionamento é executado no nível de `task_id` por `scripts/split_dataset.py` utilizando o manifesto congelado `corpus_filtered/split_manifest.json` (sem regeneração), com **interseção zero** de `task_id` entre `train`, `val` e `test`.

### 5.1 Contagens por Split, Estrato e Ferramenta

| Split | Tarefas (Manifesto / Com `DONE`) | Exemplos (`steps`) | `IMPLEMENTING` | `REPAIRING` | `write_file` | `append_file` | `done` | `read_file` | `replace_text` | `list_files` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`train`** | `42` / `31` | **`527`** (`69,16%`) | `504` (`95,64%`) | `23` (`4,36%`) | `183` | `138` | `116` | `63` | `22` | `5` |
| **`val`** | `6` / `4` | **`67`** (`8,79%`) | `62` (`92,54%`) | `5` (`7,46%`) | `26` | `7` | `11` | `19` | `3` | `1` |
| **`test`** | `12` / `10` | **`168`** (`22,05%`) | `168` (`100,00%`) | `0` (`0,00%`) | `55` | `45` | `38` | `23` | `0` | `7` |
| **Total** | **`60` / `45`** | **`762`** (`100,00%`) | **`734`** (`96,33%`) | **`28`** (`3,67%`) | **`264`** (`34,65%`) | **`190`** (`24,93%`) | **`165`** (`21,65%`) | **`105`** (`13,78%`) | **`25`** (`3,28%`) | **`13`** (`1,71%`) |

### 5.2 Integridade Criptográfica (SHA-256 dos Artefatos)

| Artefato | Papel no Pipeline | SHA-256 |
| :--- | :--- | :--- |
| `corpus_filtered/dataset.jsonl` | Corpus filtrado completo (`762` exemplos) | `f0e93906b615708f4ffd5558ac145b9732dce4921b3ad0393e9a267615583bd7` |
| `corpus_filtered/audit_sample_50.jsonl` | Amostra de auditoria manual preenchida (`1/50` questionável) | `c55445cc79f549eaeaee8d68ac8b2022b3140e8d57e71883deeea83e4e07d2be` |
| `corpus_filtered/split_manifest.json` | Manifesto congelado de splits (`seed=42`, hash interno `a1f938ca...507ae4e`) | `8607434e02f2ef3c90bc18b3199d631e9f310d2d926bde966a9355e95865c0ae` |
| `datasets/sft_jsonl/train.jsonl` | Split SFT de treino (`527` exemplos) | `28115a2bc3d9a1f7e97a98e54a1db21bf3050f2aff3a7a0a16d7314d3a348b87` |
| `datasets/sft_jsonl/val.jsonl` | Split SFT de validação (`67` exemplos) | `bf1f911e758168259fcda9a2506362f10e0565817b25453d7aa1adfba857547e` |
| `datasets/sft_jsonl/test.jsonl` | Split SFT de teste (`168` exemplos) | `8d60d7c5ec3ed06269648db752532ba912dbc836216585bd84b6e9a2e42106ef` |
| `datasets/sft_jsonl/sft_manifest.json` | Manifesto de geração SFT (`minicode-prompt-v2`) | `8263ef86131ab93c2c9b9e56f31150976cca058826028a05ae3b8e529c744b76` |
