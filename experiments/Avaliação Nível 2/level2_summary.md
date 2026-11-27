# Avaliação Nível 2 (Agente Completo) e Ablações

- **Split:** `test` congelado (12 tarefas; `experiments/spec_cache/manifest_frozen.json`, 12/12 hashes SHA-256 validados)
- **Configuração:** `temperature = 0.0`, `seed = 1`, `max_iterations = 15`, `max_repair_attempts = 2`, prompt `minicode-prompt-v2`
- **Adapter (Células C e E):** `checkpoints/step8/selected_adapter` (`seed42_frac100`, `val_loss = 0.0948`)

## 1. Tabela Principal de TSR e Eficiência por Célula

| Célula | Configuração (`IMPLEMENTING` / `REPAIRING`) | TSR ($N_{\text{DONE}}/12$) | IC Wilson 95% | IC Bootstrap 95% | $\rho_{\text{pool}}$ ($N_{1.5\text{B}}/N_{14\text{B}}$) | Iterações Média (Med / IQR) | Tokens Média (Med / IQR) | Latência Média (Med / IQR) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A** | `14B` / `14B` (*baseline* professor) | **75.0%** (9/12) | [46.8%, 91.1%] | [50.0%, 100.0%] | 0.0000 (0/98) | 9.17 (7.5 / 10.0) | 14964 (10150 / 18013) | 1816.59s (89.85s / 86.32s) |
| **B** | `1.5B_base` / `14B` (*zero-shot*) | **0.0%** (0/12) | [0.0%, 24.2%] | [0.0%, 0.0%] | 1.0000 (145/0) | 12.08 (15.0 / 2.8) | 13698 (16419 / 5315) | 49.13s (55.22s / 33.83s) |
| **C** | `1.5B_lora` / `14B` | **33.3%** (4/12) | [13.8%, 60.9%] | [8.3%, 58.3%] | 0.5563 (79/63) | 12.83 (15.0 / 5.2) | 18715 (20911 / 14235) | 134.77s (98.81s / 49.40s) |
| **D** | `14B` / *no_repair* | **66.7%** (8/12) | [39.1%, 86.2%] | [41.7%, 91.7%] | 0.0000 (0/83) | 7.83 (7.0 / 3.8) | 11649 (9332 / 5994) | 118.11s (70.45s / 145.82s) |
| **E** | `1.5B_lora + constrained` / `14B` *(10.f)* | **41.7%** (5/12) | [19.3%, 68.0%] | [16.7%, 66.7%] | 0.5725 (79/59) | 12.58 (15.0 / 5.2) | 15801 (17014 / 10694) | 119.70s (105.97s / 14.12s) |
| **NO_SPEC** | `14B` / `14B` (sem `specification.md`) *(10.f)* | **41.7%** (5/12) | [19.3%, 68.0%] | [16.7%, 66.7%] | 0.0000 (0/124) | 11.33 (13.0 / 7.2) | 18184 (19806 / 14268) | 58.78s (56.47s / 42.38s) |

## 2. Comparações Pareadas por Tarefa (Bootstrap $B = 10.000$)

| Contraste | $\Delta TSR$ | IC Bootstrap 95% | $p$-valor (bilateral) | $\Delta\text{Tokens}$ [IC 95%] |
| :--- | :---: | :---: | :---: | :---: |
| `C_minus_B` | **+0.3333** | [+0.0833, +0.5833] | 0.0138 | +5017 [+319, +9765] |
| `C_minus_A` | **-0.4167** | [-0.7500, -0.0833] | 0.0466 | +3751 [-3831, +10922] |
| `D_minus_A` | **-0.0833** | [-0.2500, +0.0000] | 0.7102 | -3315 [-7248, -343] |
| `E_minus_C` | **+0.0833** | [+0.0000, +0.2500] | 0.6916 | -2914 [-4468, -1533] |
| `E_minus_A` | **-0.3333** | [-0.6667, +0.0000] | 0.0990 | +837 [-5795, +7273] |
| `NO_SPEC_minus_A` | **-0.3333** | [-0.5833, -0.0833] | 0.0152 | +3220 [-1138, +8044] |

## 3. Diagnóstico de Falhas: Célula C e Célula E

- **Célula C (`1.5B_lora`):** 4/12 `DONE` autônomos (`task_015`, `task_041`, `task_052`, `task_057` com `repairs = 0`, $\rho = 1{,}0$); 8/12 `FAILED` por `max_iterations = 15`, marcados por escritas consecutivas no mesmo arquivo (`12/12` runs com `consec_same > 0`, `9/12` com escrita idêntica repetida) antes de acionar `done`.
- **Célula E (`1.5B_lora + constrained`):** 5/12 `DONE` (`+8,3 pp` sobre C, resgatando `task_007` no reparo e reduzindo a mediana de tokens de `20.911` para `17.014`).
- **Ablação `no_spec` (`14B` sem `specification.md`):** queda de **75,0% (9/12)** para **41,7% (5/12)** ($-33{,}3\text{ pp}$), confirmando o papel estruturante da especificação formal e dos *hard gates*.
