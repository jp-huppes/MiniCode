# Hipóteses Experimentais — Parte 2 (Especialização e Agente Híbrido)

Este documento formaliza as hipóteses pré-registradas da **Parte 2** do MiniCode antes da execução das avaliações finais, cobrindo a avaliação offline por passo (**Nível 1**) e a execução ponta a ponta no `AgentOrchestrator` (**Nível 2**).

## 1. Arquitetura Experimental

Na Parte 2, o agente opera com roteamento explícito por papel (`src/llm/routing.py`):

* **`ROLE_TOOL_SELECTION` (`IMPLEMENTING`)**: executado pelo modelo leve `Qwen/Qwen2.5-Coder-1.5B-Instruct` (modelo base zero-shot vs. modelo especializado via SFT + QLoRA $r = 16$, $\alpha = 32$) com o prompt unificado `minicode-prompt-v2` (`catalogo_enxuto=True`).
* **`ROLE_REPAIR` (`REPAIRING`)**: executado pelo modelo professor `Qwen/Qwen2.5-14B-Instruct-AWQ` quando a etapa de verificação (`VERIFYING`) detecta falha nos testes `pytest` ou ausência de arquivos exigidos pela `SPEC`.

Para isolar o impacto da seleção de ferramentas e do mecanismo de reparo no **Nível 2**, os artefatos `specification.md` e `plan.md` de cada tarefa do split `test` são gerados uma única vez pelo professor (`temperature=0.0`) e congelados em `experiments/spec_cache/<task_id>/`.

## 2. Hipóteses Formais

### H1 — Ganho do Fine-Tuning sobre o Modelo Base (`Acc_tool`, Nível 1)

* **Enunciado:** O modelo `Qwen/Qwen2.5-Coder-1.5B-Instruct` adaptado via SFT + QLoRA sobre trajetórias filtradas (`datasets/sft_jsonl/train.jsonl`) obtém acurácia de seleção de ferramenta ($\text{Acc}_{\text{tool}}$) estritamente superior à do mesmo modelo base zero-shot no split de teste congelado (`datasets/sft_jsonl/test.jsonl`, sem sobreposição de `task_id`).
* **Métricas:** $\text{Acc}_{\text{tool}}$ (correspondência exata do campo `tool` previsto com o alvo canônico) e $\text{JSON}_{\text{valid}}$ (taxa de respostas válidas segundo o contrato estrito).
* **Critério de Decisão:** $\Delta_{\text{H1}} = \text{Acc}_{\text{tool}}(\text{LoRA}) - \text{Acc}_{\text{tool}}(\text{Base}) > 0$, com limite inferior do intervalo de confiança de 95% via *bootstrap* pareado (10.000 reamostragens) estritamente positivo ($\text{IC}_{95\%,\text{low}} > 0$).
* **Resultados Finais:**
  * $\text{Acc}_{\text{tool}}(\text{Base}) = \text{[PENDENTE]}$ | $\text{JSON}_{\text{valid}}(\text{Base}) = \text{[PENDENTE]}$
  * $\text{Acc}_{\text{tool}}(\text{LoRA}) = \text{[PENDENTE]}$ | $\text{JSON}_{\text{valid}}(\text{LoRA}) = \text{[PENDENTE]}$
  * $\Delta_{\text{H1}} = \text{[PENDENTE]}$ ($\text{IC}_{95\%}$: $[\text{[PENDENTE]},\ \text{[PENDENTE]}]$) — **Status:** `[PENDENTE]`

### H2 — Recuperação pelo Modelo Professor ($G \ge 0{,}70$, Nível 2)

* **Enunciado:** Quando o modelo leve especializado falha na primeira tentativa de implementação (`VERIFYING` reprovado), o acionamento do modelo professor (`Qwen/Qwen2.5-14B-Instruct-AWQ`) no papel `ROLE_REPAIR` recupera pelo menos $70\%$ das tarefas recuperáveis ($G \ge 0{,}70$).
* **Definição da Taxa de Recuperação ($G$):**
  $$G = \frac{N_{\text{recuperadas}}}{N_{\text{elegíveis ao reparo}}}$$
  onde $N_{\text{elegíveis ao reparo}}$ corresponde às tarefas do split `test` que não passaram na primeira verificação e acionaram `ROLE_REPAIR`, e $N_{\text{recuperadas}}$ corresponde ao subconjunto dessas tarefas que alcançaram o estado terminal `DONE` após o ciclo de reparo.
* **Critério de Decisão:** $G \ge 0{,}70$.
* **Resultados Finais:**
  * $N_{\text{elegíveis ao reparo}} = \text{[PENDENTE]}$ | $N_{\text{recuperadas}} = \text{[PENDENTE]}$
  * $G = \text{[PENDENTE]}$ — **Status:** `[PENDENTE]`

### H3 — Impacto do Ganho de Seleção no $TSR$ do Agente (Nível 2)

* **Enunciado:** O ganho de acurácia de ferramenta obtido pelo fine-tuning (H1) traduz-se em aumento na taxa de sucesso de tarefas ($TSR$, *Task Success Rate*) na execução ponta a ponta do `AgentOrchestrator` no split `test`, tanto no modo híbrido completo (com `ROLE_REPAIR`) quanto na ablação sem reparo (`no_repair`).
* **Métricas:** $TSR = \frac{N_{\text{DONE}}}{N_{\text{total}}}$ e taxa média de delegação ao professor ($\bar{\rho}$).
* **Critério de Decisão:** $\Delta TSR = TSR(\text{LoRA}) - TSR(\text{Base}) > 0$ sob condições idênticas de orçamento (`max_iterations`, `max_repair_attempts` e `spec_cache`).
* **Resultados Finais:**
  * Agente Base (sem reparo): $TSR = \text{[PENDENTE]}$
  * Agente LoRA (sem reparo / `no_repair`): $TSR = \text{[PENDENTE]}$
  * Agente Base Híbrido (com professor): $TSR = \text{[PENDENTE]}$ | $\bar{\rho} = \text{[PENDENTE]}$
  * Agente LoRA Híbrido (com professor): $TSR = \text{[PENDENTE]}$ | $\bar{\rho} = \text{[PENDENTE]}$
  * $\Delta TSR = \text{[PENDENTE]}$ — **Status:** `[PENDENTE]`
