# Documentação do Dataset SFT — Parte 2

Este documento descreve a construção, filtragem determinística, particionamento e formatação do conjunto de dados utilizado no ajuste supervisionado (SFT + QLoRA) do MiniCode.

## 1. Origem e Coleta das Trajetórias

* **Modelo Gerador (Professor):** `Qwen/Qwen2.5-14B-Instruct-AWQ` executando o loop completo do `AgentOrchestrator` sobre o conjunto curado de tarefas de programação em Python (`tasks/`).
* **Rastreabilidade:** cada execução (*run*) registra metadados em `corpus_raw_full/runs_metadata.jsonl` (`run_id`, `task_id`, `seed`, `temperature`, `commit`, `prompt_version`, `max_iterations`, `max_repair_attempts`, `final_state`, `repair_attempts`) e o log passo a passo de ações em `.minicode/actions.jsonl`.
* **Alinhamento Treino = Inferência:** todas as trajetórias válidas utilizam o prompt compacto unificado `minicode-prompt-v2` (`catalogo_enxuto=True`), cobrindo exclusivamente as 6 ferramentas ativas (`list_files`, `read_file`, `write_file`, `replace_text`, `append_file`, `done`).

### Nota sobre a Temperatura Efetiva (Passo 1.a)

* Durante a coleta multi-seed de trajetórias brutas, verificou-se no Passo 1.a a diferença entre a temperatura nominal configurada no agendador de runs e a **temperatura efetiva** aplicada na chamada ao servidor de inferência/amostragem.
* O ajuste do Passo 1.a garantiu que a temperatura efetiva de geração das trajetórias de diversidade fosse propagada corretamente e registrada em `runs_metadata.jsonl`, enquanto a geração e o cache determinístico de `specification.md` e `plan.md` (`experiments/spec_cache/`) bem como as avaliações oficiais operam com `temperature=0.0`.
* Contagem de runs brutas coletadas (pré e pós-ajuste): `[PENDENTE]`.

## 2. Filtragem Determinística (Regras F1–F6)

O pipeline `scripts/filter_and_label.py` processa os logs brutos e aplica seis filtros determinísticos em cascata antes de exportar `corpus_filtered/dataset.jsonl`:

| Regra | Critério Determinístico | Finalidade | Retidos |
| :--- | :--- | :--- | :--- |
| **F1** | Resposta estruturada em JSON válido (`parse_ok == True`) | Descartar saídas malformadas ou truncadas | `[PENDENTE]` |
| **F2** | Ausência de `validator_error` e `gate_error` | Garantir conformidade com o sandbox e políticas de caminho | `[PENDENTE]` |
| **F3** | Trajetórias encerradas exclusivamente em `final_state == "DONE"` | Evitar destilação de sequências que falharam nos testes | `[PENDENTE]` |
| **F4** | Deduplicação exata por hash do par `(contexto_normalizado, acao_canonica)` | Eliminar exemplos idênticos entre execuções | `[PENDENTE]` |
| **F5** | Remoção de chamadas `done` prematuras antes da etapa de verificação | Impedir encerramento sem aprovação do `pytest` | `[PENDENTE]` |
| **F6** | Validação estrita de schema em `src/dataset/target_validator.py` (remoção recursiva de chaves `None` e rejeição de aliases inválidos como `file_path`) | Assegurar o contrato canônico `{"tool": <nome>, "args": <dict>}` | `[PENDENTE]` |

* **Amostra de Auditoria Manual:** `corpus_filtered/audit_sample_50.jsonl` (`50` exemplos extraídos para inspeção de qualidade).
* **Orçamento de Contexto:** auditado via `scripts/inspect_max_tokens.py` para compatibilidade com `max_length = 2048` tokens (comprimento máximo observado: `[PENDENTE]`).

## 3. Formato dos Exemplos

### 3.1 Corpus Filtrado (`corpus_filtered/dataset.jsonl`)

Cada linha representa um par estado-ação supervisionado:

```json
{
  "example_id": "task_003__run_01__step_04",
  "task_id": "task_003",
  "run_id": "run_01",
  "step_index": 4,
  "state": "IMPLEMENTING",
  "prompt_version": "minicode-prompt-v2",
  "x": "<contexto formatado com SPEC, PLAN, estado atual e histórico recente>",
  "y": {"tool": "write_file", "args": {"path": "solution.py", "content": "..."}}
}
```

### 3.2 Formato Conversacional SFT (`datasets/sft_jsonl/*.jsonl`)

O módulo `src/dataset/build_sft_dataset.py` converte cada exemplo para o formato de mensagens (`system`, `user`, `assistant`) acompanhado do manifesto `datasets/sft_jsonl/sft_manifest.json` (hashes SHA-256, versão de prompt e metadados de geração).

## 4. Partição sem Vazamento (Splits Congelados)

O particionamento é realizado no nível de `task_id` por `scripts/split_dataset.py` seguindo o manifesto congelado `corpus_filtered/split_manifest.json`, garantindo **interseção zero** de tarefas entre treino, validação e teste:

| Split | Tarefas (`task_id`) | Exemplos (`steps`) | Arquivo SFT Versionado |
| :--- | :--- | :--- | :--- |
| **`train`** | `[PENDENTE]` | `[PENDENTE]` | `datasets/sft_jsonl/train.jsonl` |
| **`val`** | `[PENDENTE]` | `[PENDENTE]` | `datasets/sft_jsonl/val.jsonl` |
| **`test`** | `[PENDENTE]` | `[PENDENTE]` | `datasets/sft_jsonl/test.jsonl` |
| **Total** | `[PENDENTE]` | `[PENDENTE]` | `datasets/sft_jsonl/sft_manifest.json` |
