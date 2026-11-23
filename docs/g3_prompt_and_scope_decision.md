# Decisão G3 — System Prompt Único e Escopo do Modelo 1.5B

- **Versão do prompt (`prompt_version`):** `minicode-prompt-v2` (incrementada a partir de `minicode-prompt-v1`)
- **Status:** Decisão registrada na branch `dev/g0-pipeline` (preservando coleta ativa e sem merge antecipado).

## 1. System Prompt Único (Treino = Inferência)
- **Decisão:** Adotar um **único system prompt** com o **catálogo enxuto** (`list_files`, `read_file`, `write_file`, `append_file`, `replace_text`, `done`) e saída JSON estrita (`tool`, `args`, e opcionalmente `thought`), idêntico no SFT (`notebooks/train.ipynb` / `build_sft_dataset.py`) e na inferência do orquestrador, compatível com decodificação restrita.
- **Motivo:** Eliminar a divergência entre o `DEFAULT_SYSTEM_PROMPT` do treino e o prompt com catálogo da inferência, garantindo consistência de distribuição e rastreabilidade via `prompt_version = minicode-prompt-v2`.

## 2. Escopo do Modelo 1.5B (`tool_selection`) e Estratos
- **Roteamento atual:** O modelo `1.5B` atende `tool_selection` (incluindo endpoint próprio configurável em `AgentRoutingConfig`), enquanto `spec`, `plan` e `repair` (`REPAIRING`) seguem roteados para o modelo professor (`14B`) por padrão.
- **Política de estratos e treino:** O dataset e a avaliação devem sempre reportar métricas separadamente por estrato (`IMPLEMENTING` e `REPAIRING`), conforme `docs/task_contract.md`. O treino SFT do `1.5B` pode incluir exemplos de ambos os estratos disponíveis no split de treino (`IMPLEMENTING` e `REPAIRING`) para robustez na seleção de ferramentas, mantendo o relatório de avaliação estratificado por estado da FSM.
