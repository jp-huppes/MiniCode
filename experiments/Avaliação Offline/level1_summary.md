# Avaliação Offline Nível 1

- **Prompt:** `minicode-prompt-v2` | **Temperatura:** `0.0` | **Bootstrap:** `10000` reamostragens por `task_id` (`N=168`, `10` tarefas)

## Resultados Gerais (com IC 95% por `task_id`)

| Sistema | Acc_tool (IC 95%) | JSON_valid (IC 95%) | Exact_action (IC 95%) | F1_macro (IC 95%) |
| :--- | :---: | :---: | :---: | :---: |
| `maioria` | 0.3274 [0.2512, 0.4356] | 1.0000 [1.0000, 1.0000] | 0.0000 [0.0000, 0.0000] | 0.0987 [0.0803, 0.1275] |
| `base_1_5b_zero_shot` | 0.1607 [0.0730, 0.2296] | 0.7262 [0.6111, 0.8252] | 0.0357 [0.0093, 0.0562] | 0.1020 [0.0407, 0.1723] |
| `1_5b_lora` | 0.5536 [0.4070, 0.7258] | 1.0000 [1.0000, 1.0000] | 0.2083 [0.1203, 0.3169] | 0.3700 [0.2915, 0.4846] |
| `14b_zero_shot` | 0.6012 [0.5316, 0.6735] | 1.0000 [1.0000, 1.0000] | 0.3571 [0.2283, 0.4833] | 0.4037 [0.3533, 0.4654] |
| `14b_few_shot` | 0.4583 [0.3350, 0.6067] | 1.0000 [1.0000, 1.0000] | 0.1250 [0.0930, 0.1584] | 0.2676 [0.2077, 0.3146] |

## Recuperação $G$ e Conclusão de H1 / H2

- **G (vs. 14B zero-shot):** `0.8919` (IC 95%: `[0.6, 1.2698]`, denominador $A_{14B}-A_{base}=0.4405$)
- **G (vs. 14B few-shot):** `1.3202` (IC 95%: `[1.0317, 1.8]`, denominador $A_{14B}-A_{base}=0.2976$)
- **H1 suportada:** `True` | **H2 suportada:** `True`

## Pontos de Erro Conhecidos (`1_5b_lora` — `write_file`, `append_file`, `done`)

- **`write_file`** (suporte=55): recall=`0.7273`, erro=`0.2727`, confusões={'append_file': 15}
- **`append_file`** (suporte=45): recall=`0.6889`, erro=`0.3111`, confusões={'write_file': 8, 'done': 6}
- **`done`** (suporte=38): recall=`0.5789`, erro=`0.4211`, confusões={'write_file': 7, 'append_file': 9}
