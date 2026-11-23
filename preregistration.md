# Pré-registro — Passo 1 (Seleção de Ferramenta / Próxima Ação)

> **Regra de conduta:** Este documento congela o protocolo experimental antes de qualquer ajuste (LoRA) ou avaliação final no conjunto de teste. O conjunto de teste deve ser consultado uma única vez após o congelamento. Qualquer alteração posterior a estas definições deve ser registrada explicitamente como **Desvio do pré-registro**, com justificativa técnica.

## 1. Contrato da Tarefa e Versão do Prompt
- **Contrato:** `task_contract.md`
- **Versão do prompt (`prompt_version`):** `minicode-prompt-v2`
- **Estratos reportados separadamente:** `IMPLEMENTING` e `REPAIRING`
- **Espaço de ações / rótulos ($\mathcal{T}$):** `list_files`, `read_file`, `write_file`, `done` (e `replace_text`, se implementada e habilitada no contrato).

## 2. Métricas de Avaliação

### 2.1 Métrica Primária — Acurácia da Próxima Ação (Nível Ferramenta)

$$\text{Acc} = \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}\left[\hat{t}_i = t_i\right]$$

- $N$: número total de exemplos no conjunto de teste.
- $\hat{t}_i$: ferramenta prevista pelo modelo no exemplo $i$.
- $t_i$: ferramenta de referência (rótulo) no exemplo $i$.
- $\mathbb{1}[\cdot]$: função indicadora ($1$ se verdadeiro, $0$ caso contrário).
- Respostas não parseáveis contam como **erro** na métrica primária e **não** são descartadas.

### 2.2 Métricas Secundárias
- **Validade de JSON:** fração de respostas parseáveis pelo parser congelado.
- **Acurácia de ação completa:** ferramenta e args críticos iguais (ex.: path normalizado).
- **F1 macro:** média do F1 por classe de ferramenta (obrigatório quando uma classe ultrapassar ~60%).
- **Acc por estrato:** acurácia em `IMPLEMENTING` e em `REPAIRING`.

## 3. Hipóteses Formais e Regra de Decisão

### 3.1 Ganho do Adaptador LoRA (H1)

$$\Delta = \text{Acc}_{\text{LoRA}} - \text{Acc}_{\text{base}}$$

- $\text{Acc}_{\text{LoRA}}$: acurácia do modelo `1.5B` com adaptador LoRA no teste.
- $\text{Acc}_{\text{base}}$: acurácia do mesmo modelo `1.5B` sem fine-tuning, com o mesmo prompt e decodificação.
- $\Delta$: ganho em pontos percentuais (em fração).
- **H0:** $\Delta < 0{,}15$
- **H1:** $\Delta \ge 0{,}15$

### 3.2 Aproximação do Professor (H2)

$$G = \frac{\text{Acc}_{\text{LoRA}} - \text{Acc}_{\text{base}}}{\text{Acc}_{14B} - \text{Acc}_{\text{base}}}$$

- $\text{Acc}_{14B}$: acurácia do `Qwen-14B` no mesmo teste (teto de referência).
- $G$: fração da distância base→professor que o LoRA recuperou.
- **Critério proposto:** $G \ge 0{,}70$

### 3.3 Regra de Decisão
- **H1 confirmada** se $\Delta \ge 0{,}15$ e o limite inferior do IC 95% (bootstrap pareado por run, 10.000 reamostragens) for $> 0$. Caso contrário, reportar como não confirmada, sem ajustar o limiar depois.

## 4. Filtros, Auditoria e Splits
- **Regras F1–F5:** F1 resposta parseável; F2 `validator_error` e `gate_error` nulos; F3 run terminou em `DONE`; F4 remover duplicatas exatas de $(x, y)$; F5 remover `done` prematuro rejeitado por gate.
- **Auditoria manual:** amostra de 50 exemplos; se $> 10\%$ forem questionáveis, revisar regras antes de seguir.
- **Splits congelados:** `corpus_filtered/split_manifest.json` (70% treino / 10% validação / 20% teste, por `task_id`, `seed = 42`, hashes SHA-256).

## 5. Modelos, Decodificação e Baselines
- **Decodificação:** `temperature = 0.0`, `max_tokens = 512`, mesmo prompt e mesmo parser.
- **Modelos:**
  - `1.5B` base: `Qwen/Qwen2.5-Coder-1.5B-Instruct` (inferência local em **4 bits** na RTX 4050 4 GB VRAM para validação/baselines; sem treino local na máquina com 4 GB VRAM + 8 GB RAM)
  - `1.5B + LoRA`: adaptador LoRA/QLoRA treinado no split `train` (no Google Colab, após `passo1-freeze`) sobre `Qwen/Qwen2.5-Coder-1.5B-Instruct`
  - `Qwen-14B` (professor): `Qwen/Qwen2.5-14B-Instruct-AWQ` (`https://llm.liaufms.org/v1/qwen2-5-14b-instruct-awq`)
- **Baselines (antes de treinar):**
  1. Classe majoritária
  2. `1.5B` base zero-shot
  3. `1.5B` base few-shot ($k = 3$ fixo, extraído exclusivamente do split `train`)
  4. `Qwen-14B`
  5. `1.5B + LoRA`

## 6. Congelamento (Etapa 8)
- Tag prevista: `passo1-freeze` (registrar hash do commit e do manifesto antes de treinar).
- Hash do manifesto atual (`split_manifest.json`): `a1f938ca0ef7a145532d94eef2720d6f8c9a4243ece8a9c840e3c4eea507ae4e`

## 7. Registro de Execução e Desvios Documentados (Pós-Congelamento)
- **Commit / Manifesto:** `split_manifest.json` congelado com SHA-256 `a1f938ca0ef7a145532d94eef2720d6f8c9a4243ece8a9c840e3c4eea507ae4e` (211 exemplos limpos: 132 treino / 19 tarefas, 43 validação / 3 tarefas, 36 teste / 7 tarefas; interseção zero de `task_id`).
- **Desvio 1 — Espaço de ações ($\mathcal{T}$):** Além de `list_files`, `read_file`, `write_file`, `replace_text` e `done`, o catálogo ativo incluiu `append_file` (introduzido no contrato do agente para evitar sobrescrita de arquivos de teste existentes), mantendo `minicode-prompt-v2` idêntico entre treino e inferência.
- **Desvio 2 — Hardware de execução:** O fine-tuning QLoRA (3 épocas, `r=16`, `lora_alpha=32`, `max_length=2048`) foi executado no Google Colab (Tesla T4, `float16` compute dtype) em `notebooks/train.ipynb`, e a avaliação final (`G8`) foi executada localmente em GPU NVIDIA RTX 3060 8 GB (`temp=0.0`, `max_new_tokens=512`).
- **Desfecho das Hipóteses:**
  - **H1 confirmada:** $\Delta_{	ext{obs}} = +0{,}3611$ ($\Delta_{	ext{bootstrap}} = +0{,}3605$), IC 95% $[+0{,}2222,\ +0{,}4762]$, $P(\Delta > 0) = 1{,}0000$.
  - **H2 confirmada:** $G = rac{0{,}5278 - 0{,}1667}{0{,}4722 - 0{,}1667} = 1{,}1820 \ge 0{,}70$ (superando o professor 14B zero-shot em $+5{,}56	ext{ pp}$ na métrica primária).
- **Desvio 3 — Temperatura efetiva e seeds na coleta (`scripts/collect_corpus.py`):** Confirmado via `git grep MINICODE_TEMPERATURE src/` que `MINICODE_TEMPERATURE` e as *seeds* do loop de coleta não são repassados às chamadas da API em `LLMClient` (`src/llm/client.py`), que opera com `default_temperature = 0.1` e sem *seed*. O campo `temperature` em `runs_metadata.jsonl` é nominal, preservando variação estocástica natural entre trajetórias e evitando descarte como duplicatas no filtro F4.
