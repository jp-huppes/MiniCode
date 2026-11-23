# Decis?o G3 ? System Prompt ?nico e Escopo do Modelo 1.5B

- **Vers?o do prompt (`prompt_version`):** `minicode-prompt-v2` (incrementada a partir de `minicode-prompt-v1`)
- **Status:** Decis?o registrada na branch `dev/g0-pipeline` (preservando coleta ativa e sem merge antecipado).

## 1. System Prompt ?nico (Treino = Infer?ncia)
- **Decis?o:** Adotar um **?nico system prompt** com o **cat?logo enxuto** (`list_files`, `read_file`, `write_file`, `append_file`, `replace_text`, `done`) e sa?da JSON estrita (`tool`, `args`, e opcionalmente `thought`), id?ntico no SFT (`notebooks/train.ipynb` / `build_sft_dataset.py`) e na infer?ncia do orquestrador, compat?vel com decodifica??o restrita.
- **Motivo:** Eliminar a diverg?ncia entre o `DEFAULT_SYSTEM_PROMPT` do treino e o prompt com cat?logo da infer?ncia, garantindo consist?ncia de distribui??o e rastreabilidade via `prompt_version = minicode-prompt-v2`.

## 2. Escopo do Modelo 1.5B (`tool_selection`) e Estratos
- **Roteamento atual:** O modelo `1.5B` atende `tool_selection` (incluindo endpoint pr?prio configur?vel em `AgentRoutingConfig`), enquanto `spec`, `plan` e `repair` (`REPAIRING`) seguem roteados para o modelo professor (`14B`) por padr?o.
- **Pol?tica de estratos e treino:** O dataset e a avalia??o devem sempre reportar m?tricas separadamente por estrato (`IMPLEMENTING` e `REPAIRING`), conforme `docs/task_contract.md`. O treino SFT do `1.5B` pode incluir exemplos de ambos os estratos dispon?veis no split de treino (`IMPLEMENTING` e `REPAIRING`) para robustez na sele??o de ferramentas, mantendo o relat?rio de avalia??o estratificado por estado da FSM.
