# Hipóteses Experimentais: Parte 2 (Especialização e Agente Híbrido)

Este documento formaliza as hipóteses pré-registradas da **Parte 2** do MiniCode e consolida os desfechos empíricos obtidos na avaliação offline por passo (**Nível 1**) e na execução ponta a ponta no `AgentOrchestrator` (**Nível 2**, Passo 10).

## 1. Arquitetura Experimental

Na Parte 2, o agente opera com roteamento explícito por papel (`src/llm/routing.py`):

- **`ROLE_TOOL_SELECTION` (`IMPLEMENTING`)**: executado pelo modelo leve `Qwen/Qwen2.5-Coder-1.5B-Instruct` (modelo base zero-shot vs. modelo especializado via SFT + QLoRA $r = 16$, $alpha = 32$, checkpoint `seed42_frac100`) com o prompt unificado `minicode-prompt-v2` (`catalogo_enxuto=True`).
- **`ROLE_REPAIR` (`REPAIRING`)**: executado pelo modelo professor `Qwen/Qwen2.5-14B-Instruct-AWQ` quando a etapa de verificação (`VERIFYING`) detecta falha nos testes `pytest` ou ausência de arquivos exigidos pela `SPEC`.

Para isolar o impacto da seleção de ferramentas e do mecanismo de reparo no **Nível 2**, os artefatos `specification.md` e `plan.md` de cada tarefa do split `test` ($n = 12$) foram congelados em `experiments/spec_cache/<task_id>/` (12/12 hashes SHA-256 validados).

## 2. Hipóteses Formais e Desfechos Consolidados

### H1. Ganho do Fine-Tuning sobre o Modelo Base (`Acc_tool`, Nível 1)

- **Enunciado:** O modelo `Qwen/Qwen2.5-Coder-1.5B-Instruct` adaptado via SFT + QLoRA sobre trajetórias filtradas (`datasets/sft_jsonl/train.jsonl`) obtém acurácia de seleção de ferramenta ($	ext{Acc}_{	ext{tool}}$) estritamente superior à do mesmo modelo base zero-shot no split de teste congelado (`datasets/sft_jsonl/test.jsonl`, $N = 168$ turnos em $10$ tarefas únicas, sem sobreposição de `task_id`).
- **Critério de Decisão:** $\Delta_{	ext{H1}} = 	ext{Acc}_{	ext{tool}}(	ext{LoRA}) - 	ext{Acc}_{	ext{tool}}(	ext{Base}) > 0$, com limite inferior do IC 95% via *cluster bootstrap* por `task_id` ($B = 10.000$) estritamente positivo ($	ext{IC}_{95\%,	ext{low}} > 0$).
- **Resultados Finais ($T = 0{,}0$):**
  - $	ext{Acc}_{	ext{tool}}(	ext{Base}) = 0{,}1607$ ($[0{,}0730,\ 0{,}2296]$) | $	ext{JSON}_{	ext{valid}}(	ext{Base}) = 0{,}7262$
  - $	ext{Acc}_{	ext{tool}}(	ext{LoRA}) = 0{,}5536$ ($[0{,}4070,\ 0{,}7258]$) | $	ext{JSON}_{	ext{valid}}(	ext{LoRA}) = 1{,}0000$
  - $\Delta_{	ext{H1}} = +0{,}3929$ ($\Delta_{	ext{boot}} = +0{,}4009$, $	ext{IC}_{95\%}$: $[+0{,}2460,\ +0{,}5854]$, $p(\Delta > 0) = 1{,}0000$) — **Status: Confirmada**

### H2. Recuperação de Desempenho em Relação ao Professor (Nível 1)

- **Enunciado:** O modelo leve especializado (`1.5B + LoRA`) recupera pelo menos $70\%$ da lacuna de acurácia de seleção de ferramenta entre o modelo base `1.5B` e o modelo professor `Qwen/Qwen2.5-14B-Instruct-AWQ` zero-shot:
  $$G = frac{	ext{Acc}_{	ext{tool}}(	ext{LoRA}) - 	ext{Acc}_{	ext{tool}}(	ext{Base})}{	ext{Acc}_{	ext{tool}}(	ext{Teacher}_{14	ext{B}}) - 	ext{Acc}_{	ext{tool}}(	ext{Base})} \ge 0{,}70$$
- **Resultados Finais ( $T = 0{,}0$):**
  - $	ext{Acc}_{	ext{tool}}(	ext{14B zero-shot}) = 0{,}6012$ ($[0{,}5316,\ 0{,}6735]$) | $	ext{Acc}_{	ext{tool}}(	ext{14B few-shot}) = 0{,}4583$ ($[0{,}3350,\ 0{,}6067]$)
  - $G = frac{0{,}5536 - 0{,}1607}{0{,}6012 - 0{,}1607} = \mathbf{0{,}8919}$ ($	ext{IC}_{95\%}$: $[0{,}6000,\ 1{,}2698]$; $G = 1{,}3202$ vs. `14B few-shot`) — **Status: Confirmada**

### H3. Viabilidade da Arquitetura Híbrida, *Repair Loop* e SDD (Nível 2)

- **Enunciado:**
  - **H3a (Efeito do LoRA no fluxo E2E — Célula C vs. Célula B):** A especialização QLoRA eleva a taxa de conclusão ($TSR$) do agente híbrido em relação ao `1.5B_base`, desviando chamadas de implementação ao modelo leve ($
ho = N_{1.5	ext{B}} / (N_{1.5	ext{B}} + N_{14	ext{B}})$).
  - **H3b (Contribuição do Repair Loop — Célula A vs. Célula D):** A habilitação do ciclo `REPAIRING` aumenta o $TSR$ do professor `14B` em comparação à ablação `no_repair`.
  - **H3c (Papel do SDD — Célula A vs. `NO_SPEC`):** A especificação formal (`specification.md`) eleva o $TSR$ do agente completo em relação à execução sem especificação.
- **Resultados Finais ($n = 12$ tarefas de teste, $T = 0{,}0$):**
  - **H3a confirmada:** $TSR$ sobe de **0,0% (0/12)** (Célula B) para **33,3% (4/12)** (Célula C, $\Delta TSR = +33{,}3	ext{ p.p.}$, $	ext{IC}_{95\%}$: $[+8{,}3,\ +58{,}3	ext{ p.p.}]$, $p = 0{,}0138$, $
ho_{	ext{pool}} = 0{,}5563$) e **41,7% (5/12)** na Célula E (`1.5B_lora + constrained`, $
ho_{	ext{pool}} = 0{,}5725$).
  - **H3b direcionalmente suportada:** Habilitar o reparo eleva o $TSR$ do `14B` de **66,7% (8/12)** (Célula D) para **75,0% (9/12)** (Célula A, $+8{,}3	ext{ p.p.}$, $p = 0{,}7102$), ao custo de $+3.315$ tokens médios por tarefa.
  - **H3c confirmada:** Manter `specification.md` eleva o $TSR$ de **41,7% (5/12)** (`NO_SPEC`) para **75,0% (9/12)** (Célula A, $+33{,}3	ext{ p.p.}$, $p = 0{,}0152$).
