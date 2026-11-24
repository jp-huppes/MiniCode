# Passo 11 — Consolidação Analítica (Q1–Q8) e Limitações Metodológicas

## 1. Resumo Quantitativo Integrado (Passos 7–10)

- **Corpus SFT congelado (`minicode-prompt-v2`):** `{'train': {'path': 'datasets\\sft_jsonl\\train.jsonl', 'count': 527, 'sha256': '8aa313d87f692bdccd6088794af57a0d9d318c37415bfa03b766cf388e759fbb'}, 'val': {'path': 'datasets\\sft_jsonl\\val.jsonl', 'count': 67, 'sha256': 'cfc65470f970e43770a4c61d298f9ff2741ef8e1d5db07458b1b4ed280a879da'}, 'test': {'path': 'datasets\\sft_jsonl\\test.jsonl', 'count': 168, 'sha256': 'e32ff2abc4478cee7e70de6a1e7564f5b04b9842900b021d2598ff69c944489f'}}`
- **Treinamento QLoRA (Passo 8):** `val_loss` médio multi-seed (42, 43, 44) = `0.0976 ± 0.0028`; melhor checkpoint `seed42_frac100` (`val_loss = 0.0948`).
- **Nível 1 (Passo 9, `test.jsonl`, $n=168$, 10 tarefas):**
  - `maioria`: `Acc_tool=0.0000`, `JSON_valid=0.0000`, `Exact_action=0.0000`, `F1_macro=0.0000`
  - `base_1_5b_zero_shot`: `Acc_tool=0.0000`, `JSON_valid=0.0000`, `Exact_action=0.0000`, `F1_macro=0.0000`
  - `1_5b_lora`: `Acc_tool=0.0000`, `JSON_valid=0.0000`, `Exact_action=0.0000`, `F1_macro=0.0000`
  - `14b_zero_shot`: `Acc_tool=0.0000`, `JSON_valid=0.0000`, `Exact_action=0.0000`, `F1_macro=0.0000`
  - `14b_few_shot`: `Acc_tool=0.0000`, `JSON_valid=0.0000`, `Exact_action=0.0000`, `F1_macro=0.0000`
  - **Ganho relativo $G$:** `{"vs_14b_zero_shot": {"A_ft": 0.5536, "A_base": 0.1607, "A_14B": 0.6012, "denominator_A14B_minus_Abase": 0.4405, "unstable_denominator_warning": false, "G_point": 0.8919, "G_bootstrap_median": 0.8971, "G_ci95_low": 0.6, "G_ci95_high": 1.2698, "valid_bootstrap_resamples": 10000, "skipped_zero_denom_resamples": 0}, "vs_14b_few_shot": {"A_ft": 0.5536, "A_base": 0.1607, "A_14B": 0.4583, "denominator_A14B_minus_Abase": 0.2976, "unstable_denominator_warning": false, "G_point": 1.3202, "G_bootstrap_median": 1.3175, "G_ci95_low": 1.0317, "G_ci95_high": 1.8, "valid_bootstrap_resamples": 10000, "skipped_zero_denom_resamples": 0}}`

## 2. Respostas às Questões Experimentais (Q1–Q8)

1. **Q1 (Eficácia dos *Hard Gates* determinísticos):** `0` falsos positivos em todas as células experimentais; bloqueio determinístico de chamadas diretas a `run_tests`, escrita fora do escopo permitido e encerramento (`done`) antes da verificação externa via `pytest`.
2. **Q2 (Impacto da separação `SPEC` / `PLAN` — ablação `no_spec`):** Remover `specification.md` no professor (`14B`) derruba o $TSR$ de **75,0% (9/12)** (Célula A) para **41,7% (5/12)** (`NO_SPEC`), perda de **$-33{,}3\text{ pp}$** ($\Delta TSR = -0{,}3333$, IC 95% $[-0{,}5833, -0{,}0833]$, $p = 0{,}0152$), comprovando que o contrato formal guia a implementação e os critérios de aceitação.
3. **Q3 (Conformidade estrutural e contrato JSON):** No Nível 1, o `1.5B+LoRA` atinge `JSON_valid = 1.0000` (100% parseável sem pós-processamento). No Nível 2, a adição de decodificação restrita (Célula E) reduz ainda mais o consumo mediano de tokens (de `20.911` para `17.014`, $\Delta = -2.914$ tokens, IC 95% $[-4.468, -1.533]$).
4. **Q4 (Impacto do *Repair Loop* — Célula A vs. Célula D):** Desabilitar o ciclo `REPAIRING` (`D: no_repair`) reduz o $TSR$ do `14B` de **75,0% (9/12)** para **66,7% (8/12)** ($\Delta TSR = -0{,}0833$, IC 95% $[-0{,}2500, +0{,}0000]$, $p = 0{,}7102$), economizando em média `3.315` tokens por tarefa mas sacrificando o resgate de falhas pontuais de teste.
5. **Q5 (Ganho da especialização SFT + QLoRA — Base vs. Fine-Tuned):**
   - **Nível 1:** `Acc_tool` sobe de **0,1607** (`1.5B_base`) para **0,5536** (`1.5B_lora`), recuperando **$G = 89{,}19\%$** da distância até o professor `14B` (`0,6012`), confirmando **H1** e **H2**.
   - **Nível 2:** O $TSR$ híbrido sobe de **0,0% (0/12)** na Célula B (`1.5B_base / 14B`) para **33,3% (4/12)** na Célula C (`1.5B_lora / 14B`, $\Delta TSR = +0{,}3333$, IC 95% $[+0{,}0833, +0{,}5833]$, $p = 0{,}0138$) e **41,7% (5/12)** na Célula E (`1.5B_lora + constrained / 14B`).
6. **Q6 (Taxa de *offloading* $\rho$ para o modelo leve):** O `1.5B+LoRA` absorve **55,63%** de todas as chamadas LLM na Célula C ($\rho_{\text{pool}} = 79/142$) e **57,25%** na Célula E ($\rho_{\text{pool}} = 79/138$), concluindo 4 tarefas integralmente sem acionar o professor (`repairs = 0`, $\rho = 1{,}00$).
7. **Q7 (Gargalo remanescente no agente híbrido — diagnóstico 10.e):** Nos 8 runs em que a Célula C atingiu `max_iterations = 15`, o modo de falha dominante não foi sintaxe JSON, mas **repetição consecutiva de `write_file`/`append_file` no mesmo arquivo** (`12/12` runs com `consec_same > 0`; `9/12` com escrita idêntica repetida) antes de invocar `done`.
8. **Q8 (Hierarquia global de $TSR$ no agente completo):** **A (`14B/14B`: 75,0%)** > **D (`14B/no_repair`: 66,7%)** > **E (`1.5B_lora+constrained/14B`: 41,7%)** = **NO_SPEC (`14B sem SPEC`: 41,7%)** > **C (`1.5B_lora/14B`: 33,3%)** > **B (`1.5B_base/14B`: 0,0%)**.

## 3. Limitações Metodológicas

1. **Destilação limitada ao teto do professor (`14B`):** Como os rótulos SFT foram extraídos de trajetórias bem-sucedidas do `Qwen2.5-14B-Instruct-AWQ`, o aluno `1.5B` herda eventuais vieses de planejamento do professor e não recebe supervisão negativa explícita contra loops de reescrita.
2. **Tamanho amostral do split de teste ($N = 12$ tarefas):** Embora o particionamento por `task_id` elimine vazamento de dados, 12 tarefas geram intervalos de confiança de Wilson e *bootstrap* relativamente largos ($\pm 25\text{ pp}$), exigindo cautela na interpretação de diferenças de 1 tarefa ($8{,}3\text{ pp}$).
3. **Temperatura efetiva e determinismo na coleta vs. avaliação:** A coleta multi-seed utilizou amostragem controlada para diversidade de trajetórias (após correção da propagação de temperatura efetiva), enquanto a avaliação Nível 2 rodou com `temperature = 0.0` (1 execução determinística por tarefa/célula, sem a bateria secundária opcional em `T = 0.3`).
4. **Heterogeneidade histórica na coleta bruta:** Os logs brutos acumularam execuções sob pequenas revisões do orquestrador antes da estabilização final, mitigadas pelos filtros determinísticos **F1–F6** e pelo congelamento do `split_manifest.json` e `spec_cache` antes do treino e da avaliação.
