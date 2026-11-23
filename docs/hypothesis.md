# Hipóteses Experimentais — Parte 2 (Especialização e Agente Híbrido)

Este documento formaliza as hipóteses avaliadas na **Parte 2** do MiniCode, cobrindo tanto a avaliação offline por passo (**Nível 1**) quanto a execução ponta a ponta no `AgentOrchestrator` (**Nível 2**).

## 1. Contexto e Arquitetura Experimental

- **`ROLE_TOOL_SELECTION` (`IMPLEMENTING`)**: executado pelo modelo leve `Qwen/Qwen2.5-Coder-1.5B-Instruct` (base vs. adaptado via SFT + QLoRA `r=16, alpha=32`) utilizando o prompt unificado `minicode-prompt-v2` (`catalogo_enxuto=True`).
- **`ROLE_REPAIR` (`REPAIRING`)**: executado pelo modelo professor `Qwen/Qwen2.5-14B-Instruct-AWQ` quando o harness (`VERIFYING`) detecta falha nos testes `pytest` ou ausência de arquivos exigidos pela `SPEC`.
- Para isolar o efeito da seleção de ferramenta e do ciclo de reparo no **Nível 2**, `specification.md` e `plan.md` de cada tarefa do split `test` são gerados uma única vez pelo professor (`temperature=0.0`) em `experiments/spec_cache/<task_id>/`.

## 2. Formulação e Desfecho das Hipóteses

### H1 — Ganho do Fine-Tuning LoRA sobre o Modelo Base (Nível 1)
- **Critério:** $\Delta = \text{Acc}_{\text{tool}}(\text{LoRA}) - \text{Acc}_{\text{tool}}(\text{Base}) > 0$ com $\text{IC}_{95\%,\text{low}} > 0$ via *bootstrap* pareado (10.000 reamostragens) em `test.jsonl` ($n = 36$).
- **Desfecho (**Confirmada**):** $\Delta_{\text{obs}} = +0{,}3611$ ($\Delta_{\text{bootstrap}} = +0{,}3605$), IC 95% $[+0{,}2222,\ +0{,}4762]$, $P(\Delta > 0) = 1{,}0000$, elevando $\text{Acc}_{\text{tool}}$ de `0,1667` para `0,5278` e $\text{JSON}_{\text{valid}}$ de `0,6667` para `1,0000`.

### H2 — Recuperação de Desempenho em Relação ao Professor (Nível 1)
- **Critério:** $G = \frac{\text{Acc}_{\text{tool}}(\text{LoRA}) - \text{Acc}_{\text{tool}}(\text{Base})}{\text{Acc}_{\text{tool}}(\text{Teacher}_{14\text{B}}) - \text{Acc}_{\text{tool}}(\text{Base})} \ge 0{,}70$.
- **Desfecho (**Confirmada**):** $G = \frac{0{,}5278 - 0{,}1667}{0{,}4722 - 0{,}1667} = 1{,}1820 \ge 0{,}70$, superando o professor zero-shot em $+5{,}56\text{ pp}$.

### H3 — Viabilidade e Limites da Arquitetura Híbrida e do Repair Loop (Nível 2)
- **H3b (Repair Loop — Célula A vs. Célula D) — Confirmada:** Habilitar o reparo elevou $TSR$ de **66,7% (8/12)** (D) para **75,0% (9/12)** (A), recuperando a `task_015`.
- **H3a (LoRA no fluxo E2E — Célula C vs. Célula B) — Parcialmente Confirmada:** O `1.5B + LoRA` (C) eliminou 100% dos `parse_failure` (0/12 vs. 3/12 na base B) e acionou o roteamento para o professor `14B` em `REPAIRING` em **12/12 tarefas** ($\bar{\rho} = 0{,}4456$ vs. $1{,}0000$ na base). Contudo, a baixa acurácia exata de argumentos ($\text{Exact}_{\text{action}} = 0{,}1111$) deixou arquivos incompletos para o `14B` corrigir nas ~8 iterações restantes (`max_iterations=15`), resultando em $TSR = 0{,}0\%$ (0/12) em B e C.
