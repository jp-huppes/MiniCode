# MiniCode

**MiniCode** é um agente autônomo de programação orientado a LLM. Dada uma tarefa em linguagem natural, o agente gera uma especificação formal (SPEC), planeja os passos de implementação (PLAN), executa ações de forma controlada sobre um workspace isolado (sandbox) e só se considera concluído depois que um harness de testes — independente da LLM — confirma que a suíte `pytest` passa.

O projeto é o Trabalho Prático da disciplina de Redes Neurais (UFMS), cujo modelo-alvo é o `Qwen/Qwen2.5-14B-Instruct-AWQ` servido via vLLM com uma API compatível com OpenAI.

---

## Sumário

- [Arquitetura](#arquitetura)
- [Fluxo do agente](#fluxo-do-agente)
- [Ferramentas implementadas](#ferramentas-implementadas)
- [Spec-Driven Development (SDD)](#spec-driven-development-sdd)
- [Agent Harness](#agent-harness)
- [Mecanismos de segurança](#mecanismos-de-segurança)
- [Repair Loop](#repair-loop)
- [Rastreabilidade (`.minicode/`)](#rastreabilidade-minicode)
- [Como executar](#como-executar)
- [Testes](#testes)
- [Limitações conhecidas / próximos passos](#limitações-conhecidas--próximos-passos)

---

## Arquitetura

O código é organizado em módulos com responsabilidades isoladas, todos sob `src/`:

| Módulo | Responsabilidade |
|---|---|
| `src/llm/` | Cliente HTTP para a LLM (`client.py`), retry/backoff (`retry.py`), métricas por chamada (`metrics.py`), parser defensivo de ações em JSON (`action_parser.py`) e hierarquia de erros (`errors.py`). |
| `src/spec/` | Pipeline de Spec-Driven Development: prompts (`prompts.py`), geração com retry (`generator.py`), validação programática (`validators.py`), persistência em disco (`persistence.py`), coleta de contexto do projeto (`context.py`) e orquestração do pipeline (`pipeline.py`). |
| `src/agent/` | Núcleo do agente: máquina de estados (`state.py`), orquestrador do loop principal (`orchestrator.py`), validador de ações (`validator.py`), hard gates (`gates.py`), executor de ferramentas (`executor.py`), harness de verificação (`verifier.py`), construtor de contexto de reparo (`repair_context.py`), gerenciamento de orçamento de tokens (`context_manager.py`) e schemas de estado (`schemas.py`). |
| `src/tools/` | Catálogo de ferramentas disponíveis à LLM (`registry.py`), implementações (`implementations.py`), wrapper de segurança (`base.py`), sandbox de caminho (`sandbox.py`) e truncamento de saída (`truncate.py`). |
| `src/security/` | Camada de segurança compartilhada: resolução de caminhos sandboxed (`sandbox.py`), validador de comandos de shell (`validator.py`) e exceções (`errors.py`). |
| `src/trace/` | Logger de execução append-only (`logger.py`) e snapshot de estado por run (`state.py`), usados para auditoria e para o vídeo de demonstração. |
| `src/cli/` | Ponto de entrada `minicode` (`main.py`) e renderizadores de saída — humano colorido, JSON e silencioso (`renderer.py`). |
| `src/config.py` | Limites globais (iterações, tentativas de reparo, timeout, janela de contexto), configuráveis via variáveis de ambiente. |

```
tarefa (texto)
      │
      ▼
┌─────────────┐      ┌──────────────┐      ┌────────────────────┐
│  src/spec/  │─────▶│ src/agent/   │─────▶│   src/tools/        │
│  (SDD)      │      │ orchestrator │◀────▶│   (ações sandboxed) │
└─────────────┘      │ + state (FSM)│      └────────────────────┘
                      │ + gates      │              ▲
                      │ + verifier   │              │
                      └──────┬───────┘      ┌───────┴────────┐
                             │              │ src/security/   │
                             ▼              └─────────────────┘
                      src/trace/ (.minicode/)
```

---

## Fluxo do agente

O agente é modelado como uma **máquina de estados finita (FSM)** com transições estritas (`src/agent/state.py`):

```
RECEIVED → SPECIFIED → PLANNED → IMPLEMENTING → VERIFYING ─┬─▶ DONE
                                       ▲                    │
                                       └────── REPAIRING ◀──┘
                     (qualquer estado) ───▶ FAILED
```

`DONE` e `FAILED` são estados terminais — nenhuma transição é permitida a partir deles.

Passo a passo de uma execução:

1. **RECEIVED → SPECIFIED → PLANNED**: o pipeline de SDD (`src/spec/pipeline.py`) gera e valida `specification.md` e `plan.md` antes de liberar qualquer escrita de código.
2. **PLANNED → IMPLEMENTING**: início do loop de execução.
3. **Loop IMPLEMENTING** (`AgentOrchestrator.run`), a cada iteração:
   1. monta o prompt (SPEC + PLAN + estado atual + histórico resumido + última observação);
   2. checa o orçamento de tokens (`estimar_tokens`) contra `0.8 × (MAX_MODEL_LEN − T_RESP_RESERVE)` — estoura isso e o agente vai para `FAILED` com `context_overflow`;
   3. chama a LLM e faz o parsing da ação JSON retornada (tolerante a cercas ```` ```json ```` e a texto solto ao redor);
   4. valida a ação (`ActionValidator`) — ferramenta existe? caminho é seguro? argumentos obrigatórios presentes?;
   5. checa os **hard gates** (`ExecutionGateManager`) — ver seção [Agent Harness](#agent-harness);
   6. executa a ferramenta de forma isolada (`ToolExecutor`), nunca deixando uma exceção escapar para o loop;
   7. registra a ação em `actions.jsonl` e atualiza as flags de controle (`tests_passed`, `modified_after_last_test`);
   8. se a ferramenta foi `done`, transiciona para `VERIFYING`.
4. **VERIFYING**: só o **harness** roda os testes (a LLM nunca chama `run_tests` diretamente aqui — isso é bloqueado explicitamente no orquestrador). Se os testes passam e o hard gate de consistência é satisfeito, o agente vai para `DONE`. Caso contrário, vai para `REPAIRING` (ou `FAILED`, se as tentativas de reparo se esgotaram).
5. **REPAIRING → VERIFYING**: ciclo de correção guiado pela saída real do `pytest` (ver [Repair Loop](#repair-loop)), até passar ou até `MAX_REPAIR_ATTEMPTS`.
6. **Condições de parada**: `MAX_ITERATIONS` excedido, `MAX_PARSE_FAILURES` consecutivas, `MAX_REPAIR_ATTEMPTS` esgotado ou `context_overflow` — todas levam a `FAILED` com um `failure_reason` explícito, persistido em `state.json`.

---

## Ferramentas implementadas

Registradas em `src/tools/registry.py` e expostas à LLM em dois formatos — schema OpenAI (`export_openai_tools`, para tool calling nativo) e texto para o prompt (`export_prompt_text`, para o modo fallback JSON):

| Ferramenta | Descrição | Observações |
|---|---|---|
| `list_files` | Lista arquivos/pastas recursivamente a partir de um caminho, com limite de entradas. | Ignora `.git`, `__pycache__`, `.venv`, `.minicode`, `node_modules`. |
| `read_file` | Lê um arquivo com numeração de linhas, com recorte opcional por `start_line`/`end_line`. | Trunca saída grande preservando início e fim. |
| `write_file` | Cria/sobrescreve um arquivo, criando diretórios pai automaticamente. | Só permitida em `IMPLEMENTING`/`REPAIRING` (hard gate). Limite de 500.000 caracteres. |
| `run_tests` | Executa o `pytest` via subprocesso com timeout. | **Exclusiva do harness** — a LLM nunca a invoca diretamente; o orquestrador rejeita a ação se a LLM tentar. |
| `done` | Sinaliza pedido de verificação (não é uma ferramenta de arquivo). | Tratada no orquestrador, não passa pelo `ToolRegistry`; sempre transiciona `IMPLEMENTING`/`REPAIRING` → `VERIFYING`. |

Toda ferramenta é decorada com `@safe_tool` (`src/tools/base.py`), que captura `TimeoutExpired`, `FileNotFoundError`, `PermissionError` e qualquer `Exception` genérica, convertendo-as em um `ToolResult(ok=False, ...)` — nenhuma exceção de ferramenta propaga para o loop do agente.

---

## Spec-Driven Development (SDD)

Antes de qualquer escrita de código, o agente formaliza **o que** deve ser feito (`src/spec/`):

1. **Coleta de contexto** (`context.py`): monta uma árvore de arquivos do projeto e lê, com orçamento de caracteres, os arquivos mais relevantes (README, `pyproject.toml`, testes, código-fonte, nessa ordem de prioridade).
2. **Geração da SPEC** (`generator.gerar_spec_com_retry`): prompt fixo (`prompts.SPEC_SYSTEM_PROMPT`) exige markdown estrito com `## Objective`, `## Parameters`, `## Acceptance Criteria` e `## Tests`. Critérios de aceitação são binários, identificados por `AC-n`.
3. **Validação programática da SPEC** (`validators.validar_spec`), regras V1–V7: seções obrigatórias presentes, conteúdo não vazio/placeholder (`TBD`/`N/A`), critérios com ID `AC-n` único.
4. **Retry com feedback direcionado**: se a validação falha, os erros específicos são devolvidos à LLM na próxima mensagem — até `MAX_SPEC_ATTEMPTS` tentativas, depois `SpecInvalidaError`.
5. **Persistência** (`persistence.salvar_artefato_spec`): grava `specification.md` com cabeçalho de auditoria (timestamp + tarefa) e confirma integridade (arquivo existe e não está vazio) antes de liberar a transição `RECEIVED → SPECIFIED`.
6. **Geração e validação do PLAN** (`gerar_plano_com_retry` + `validar_plano`, regras P1–P5): passos numerados sequencialmente sem buracos, cada passo associado a um arquivo e a um `AC-n`, cobertura de **100% dos critérios** da SPEC, nenhum `AC-n` inexistente citado. Mesmo mecanismo de retry com feedback, grava `plan.md` e libera `SPECIFIED → PLANNED`.

```python
spec, plan_raw = executar_spec_pipeline(tarefa, workspace_dir, llm_client, state_machine)
```
---

## Agent Harness

O "harness" é o conjunto de mecanismos que garantem que o agente só avança quando há evidência objetiva, e nunca por autoatestado da LLM:

- **Parser de ação em 3 camadas** (`action_parser.parse_acao_json`): (1) tenta o texto inteiro como JSON; (2) extrai o conteúdo de uma cerca ```` ``` ```` (com ou sem `json`); (3) varre o texto procurando um objeto `{...}` balanceado (contando chaves e respeitando strings/escapes). A primeira camada que produzir um JSON válido com `action`/`args` e uma ferramenta reconhecida vence; senão, `ParseError`.
- **Máquina de estados com tabela de transições explícita** (`TRANSITIONS`): qualquer transição fora da tabela levanta `IllegalTransition` — testado exaustivamente via produto cartesiano 8×8 em `test_state_machine.py`.
- **Hard Gates** (`src/agent/gates.py` + `src/agent/state.py`):
  - *Sem SPEC/PLAN, sem escrita*: `write_file`/`replace_text` só são permitidas em `IMPLEMENTING`/`REPAIRING`.
  - *Sem teste, sem DONE*: `validate_done_gate()` só retorna `True` se `last_test_seq > last_write_seq` **e** o último teste passou — ou seja, toda escrita invalida o atestado de testes anterior, exigindo um novo `run_tests` do harness após a última modificação. Em `IMPLEMENTING`/`REPAIRING`, o gate aceita `done` imediatamente (ele apenas *pede* verificação); o hard gate real é avaliado pelo harness ao processar `VERIFYING`.
  - *`run_tests` é exclusivo do harness*: o orquestrador rejeita explicitamente qualquer tentativa da LLM de chamar `run_tests` fora do ciclo de `VERIFYING`.
- **Orçamento de contexto** (`context_manager.py`): estimativa simples de tokens por caracteres; truncamento inteligente da saída do `pytest` preservando o `short test summary info` e as últimas linhas do relatório; histórico de ações comprimido às últimas N entradas.
- **Limites anti-loop** (`src/config.py`, todos configuráveis via variável de ambiente): `MAX_ITERATIONS` (25), `MAX_REPAIR_ATTEMPTS` (3), `MAX_PARSE_FAILURES` (3), `COMMAND_TIMEOUT` (60s).
- **Execução isolada** (`ToolExecutor.execute`): qualquer exceção na execução da ferramenta é capturada e normalizada em `{ok: False, output: "...", duration: ...}`, nunca derruba o loop do agente.

---

## Mecanismos de segurança

Camada centralizada em `src/security/`, reutilizada por `src/tools/` e `src/spec/`:

- **Sandbox de caminhos** (`sandbox.resolve_in_root`): resolve qualquer caminho relativo à raiz do workspace e bloqueia:
  - directory traversal (`../`, mesmo combinado como `a/../../fora.txt`) via `Path.resolve()` + `relative_to` (evita o erro clássico de comparar prefixos com `startswith`);
  - caminhos absolutos fora da raiz (ex.: `/etc/passwd`);
  - caracteres nulos (`\0`) no caminho;
  - manipulação direta da pasta de auditoria `.minicode/`.
- **Validador de comandos de shell** (`security.validator.CommandValidator`), usado quando `run_tests` recebe um `command` customizado:
  - denylist por regex para `sudo`, `rm -rf`, `shutdown`, `reboot`, `dd`, `mkfs`, pipes para `sh`/`bash`, `chmod -R 777`, redirecionamento para `/dev/sd*`;
  - bloqueio de comandos compostos complexos (`$(...)`, crases, múltiplas linhas) antes mesmo da divisão por `;`/`&&`/`||`/`|`;
  - **allowlist** de executáveis (`pytest`, `python`, `git`, `ls`, `cat`, `echo`), com restrições adicionais por subcomando (`git` só aceita `diff`/`status`/`log`; `python` bloqueia `-c`).
- **Proteção de testes durante o reparo** (`verifier.validate_repair_path_protection`): em `REPAIRING`, qualquer `write_file`/`replace_text` que resolva para dentro de `tests/` é bloqueada — a LLM não pode "corrigir" o teste em vez do código.
- **Limite de escrita**: `write_file` rejeita conteúdo acima de 500.000 caracteres.
- **Falha segura por padrão**: o decorador `@safe_tool` garante que qualquer erro inesperado de uma ferramenta vira um `ToolResult(ok=False)` em vez de uma exceção não tratada.

---

## Repair Loop

Quando a verificação falha, o agente não recebe apenas "falhou" — ele recebe um contexto estruturado de depuração (`src/agent/repair_context.py`):

1. **Harness roda os testes de forma independente** em `VERIFYING` (comando fixo `pytest -x --tb=short -q`), nunca a LLM.
2. Se falhar e ainda houver tentativas disponíveis, o agente transiciona para `REPAIRING` (`repair_attempts += 1`; ao atingir `MAX_REPAIR_ATTEMPTS`, a própria `transition()` já redireciona para `FAILED`).
3. **Assinatura de erro** (`calcular_assinatura_erro`): hash MD5 truncado das últimas linhas relevantes (`FAILED`, `ERROR`, `AssertionError`, linhas `E `) — usado para detectar quando o mesmo erro se repete entre tentativas e avisar explicitamente a LLM para mudar de abordagem.
4. **Prompt de reparo** (`montar_prompt_reparo`) inclui: a SPEC completa, o conteúdo **atual** (lido do disco, não em cache) de cada arquivo tocado, a saída truncada do `pytest`, e um histórico das últimas 3 tentativas com resumo do erro.
5. A LLM corrige o código (bloqueada de tocar `tests/`), chama `done` novamente, o agente volta a `VERIFYING`, e o ciclo se repete até passar ou esgotar `MAX_REPAIR_ATTEMPTS`.
---

## Rastreabilidade (`.minicode/`)

Cada execução é auditável sem depender de logs externos:

```
.minicode/
├── latest                  # atalho com o id da última run
├── actions.jsonl           # uma linha por tool call (RunLogger) — timestamp, estado, args sanitizados (hash+tamanho p/ write_file), resultado truncado
├── state.json              # snapshot atômico do estado do orquestrador (escrita via .tmp + rename)
└── runs/
    └── <timestamp-UTC>/
        ├── actions.jsonl   # log da run específica
        └── state.json      # estado final da run (RunState): transições, tokens, tentativas de reparo, arquivos alterados, tempo decorrido
```

A escrita de `state.json` é sempre atômica (grava em `.tmp` e faz `rename`), evitando JSON truncado em caso de crash no meio da execução.

---

## Como executar

### Pré-requisitos

- Python ≥ 3.10
- Acesso a um endpoint compatível com a API da OpenAI (ex.: vLLM servindo `Qwen2.5-14B-Instruct-AWQ`)

### Instalação

```bash
git clone <url-do-repositorio>
cd MiniCode
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

### Configuração

O cliente LLM lê as credenciais de variáveis de ambiente (ou de um `.env` na raiz, via `python-dotenv`):

```bash
export MINICODE_API_KEY="sua-chave"
export MINICODE_API_BASE="https://seu-endpoint/v1"
export MINICODE_MODEL_NAME="Qwen/Qwen2.5-14B-Instruct-AWQ"   # opcional, este é o default
```

Limites de execução também são configuráveis por ambiente (`src/config.py`): `MINICODE_MAX_ITERATIONS`, `MINICODE_MAX_REPAIR_ATTEMPTS`, `MINICODE_COMMAND_TIMEOUT`, `MINICODE_MAX_MODEL_LEN`, `MINICODE_T_RESP_RESERVE`, `MINICODE_MAX_PARSE_FAILURES`.

### Uso

Modo direto (one-shot):

```bash
minicode "Implementar uma função de soma em src/calc.py" --project ./meu_projeto
```

Modo REPL interativo:

```bash
minicode --project ./meu_projeto
minicode> Implementar uma função de soma em src/calc.py
minicode> /exit
```

Flags disponíveis:

| Flag | Default | Descrição |
|---|---|---|
| `--project` | `.` | Diretório de trabalho (workspace sandboxed). |
| `--max-iterations` | `25` | Limite de iterações do loop. |
| `--max-repair-attempts` | `3` | Limite de tentativas de reparo. |
| `--test-cmd` | `pytest` | Comando de teste customizado. |
| `--json` | `False` | Saída estrita em JSON (requer uma tarefa, não funciona no REPL). |
| `--verbose` | `False` | Exibe SPEC/PLAN completos e argumentos sem truncar. |
| `--quiet` | `False` | Só imprime o estado final. |
| `--no-color` | `False` | Desativa cores ANSI. |
---

## Testes

```bash
python -m pytest -q
```

Cobertura atual: **72 testes** em 10 arquivos, cobrindo parser de ação, máquina de estados (produto cartesiano completo de transições), hard gates, ferramentas + sandbox, segurança (denylist/allowlist de comandos), pipeline de SDD, orquestrador (loop de sucesso, estouro de iterações, falha de parse), repair loop (assinatura de erro, proteção de `tests/`, sucesso após reparo, exaustão de tentativas), rastreabilidade (logger + state) e CLI (parsing de argumentos, renderers, REPL).

Na revisão desta documentação, **72 de 72 passaram de forma consistente**; É interessante documentar que o teste que mais houve problemas foi o de `test_repair_loop_success_after_repair` que falhava por conta do bug de cache presente .pyc.

---

## Problema(s) conhecido(s)
Durante a fase de testes, identificou-se um comportamento limite sensível ao contexto de execução:

**Ambiguidade de Workspace ao Executar no Diretório Raiz (`--project .`)**:
  * **Descrição**: Ao executar o MiniCode apontando o argumento `--project` para o próprio diretório de desenvolvimento do framework (`--project .`), a LLM pode interpretar erroneamente a árvore de arquivos corrente e tentar gravar, modificar ou reescrever arquivos do próprio core do MiniCode (`src/` ou `tests/`) em vez de criar uma aplicação independente.
  * **Fase de Ocorrência**: Ocorre tipicamente na transição entre o planejamento (`PLANNED`) e a implementação (`IMPLEMENTING`) quando o escopo dos caminhos alvo não está estritamente confinado pelo modelo.
  * **Mitigação Recomendada**: Para evitar qualquer sobrescrita acidental nos fontes do framework, **execute sempre as tarefas em diretórios de projetos separados e isolados** fora da árvore do MiniCode (por exemplo: `--project ../demo_soma`), garantindo que o agente confine suas operações de escrita exclusivamente ao workspace da tarefa alvo!

---

## Parte 2 — Especialização via SFT + QLoRA

Na **Parte 2**, o MiniCode foi estendido com um pipeline completo de coleta de trajetórias, filtragem determinística, destilação via SFT + QLoRA para o modelo leve **`Qwen/Qwen2.5-Coder-1.5B-Instruct`** e roteamento híbrido no orquestrador (`ROLE_TOOL_SELECTION` no modelo especializado 1.5B e `ROLE_REPAIR` no professor `Qwen/Qwen2.5-14B-Instruct-AWQ`).

### Pipeline de Dados e Treinamento (`G0`–`G7`)

1. **Coleta de Trajetórias (`scripts/collect_corpus.py`):** Execução instrumentada do agente sobre `corpus_tasks/` com suporte a retomada (`--resume`) e trava anti-loop.
2. **Filtragem e Rotulagem (`scripts/filter_and_label.py`):** Aplicação das regras `F1`–`F6` (JSON válido, ausência de `validator_error`/`gate_error`, runs concluídas em `DONE`, deduplicação exata e remoção de `done` prematuro) e auditoria manual em amostra de 50 exemplos (`corpus_filtered/audit_sample_50.jsonl`).
3. **Divisão sem Vazamento (`scripts/split_dataset.py`):** Partição congelada por `task_id` (`corpus_filtered/split_manifest.json`, SHA-256 `a1f938ca0ef7a145532d94eef2720d6f8c9a4243ece8a9c840e3c4eea507ae4e`), totalizando **211 exemplos limpos** em 29 tarefas concluídas:
   - **Treino (`train.jsonl`):** 132 exemplos (19 tarefas — 120 `IMPLEMENTING`, 12 `REPAIRING`).
   - **Validação (`val.jsonl`):** 43 exemplos (3 tarefas — 38 `IMPLEMENTING`, 5 `REPAIRING`).
   - **Teste (`test.jsonl`):** 36 exemplos (7 tarefas — 36 `IMPLEMENTING`).
4. **Prompt Unificado (`minicode-prompt-v2`):** Mesmo system prompt compacto (`catalogo_enxuto=True`) compartilhado entre geração SFT (`src/dataset/build_sft_dataset.py`), treino (`notebooks/train.ipynb`) e inferência (`src/agent/orchestrator.py`).
5. **Treinamento QLoRA (`notebooks/train.ipynb`):** Fine-tuning em 4-bit NF4 (`float16` compute, LoRA `r=16`, `lora_alpha=32`, `lora_dropout=0.05`, `18.464.768` parâmetros treináveis = `1,18%` do total, `max_length=2048`, 3 épocas), atingindo melhor `eval_loss = 0.1231` e acurácia por token de **94,53%** na Época 2.

### Resultados Experimentais no Conjunto de Teste (`test.jsonl`, $n = 36$, `temp = 0.0`)

| Sistema | `Acc_tool` (Primária) | `JSON_valid` | `Exact_action` (`tool+args`) | `F1_macro` | Latência média |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `majority_class` | 0.2778 | 1.0000 | 0.0000 | 0.1087 | — |
| `base_1_5b_zero_shot` (`Qwen2.5-Coder-1.5B`) | 0.1667 | 0.6667 | 0.0833 | 0.1295 | 6.50 s |
| **`finetuned_1_5b_lora` (`Qwen2.5-Coder-1.5B` + LoRA)** | **0.5278** | **1.0000** | **0.1111** | **0.3783** | 7.51 s |
| `teacher_14b_zero_shot` (`Qwen2.5-14B-AWQ`) | 0.4722 | 0.8333 | 0.2222 | 0.2627 | — |
| `teacher_14b_few_shot` ($k = 3$) | 0.3333 | 0.6944 | 0.1667 | 0.2076 | — |

- **Bootstrap Pareado por `run_id` (10.000 reamostragens, IC 95%):**
  - `finetuned_1_5b_lora` vs. `base_1_5b_zero_shot`: **$\Delta = +0.3605$** ($+36.05	ext{ pp}$), IC 95% **$[+0.2222,\ +0.4762]$**, $P(\Delta > 0) = 1.0000$ (**H1 confirmada**).
  - `finetuned_1_5b_lora` vs. `majority_class`: **$\Delta = +0.2440$** ($+24.40	ext{ pp}$), IC 95% **$[+0.0882,\ +0.4062]$**, $P(\Delta > 0) = 0.9993$.
  - **Recuperação do Professor (H2):** $G = 1.1820$ ($118.2\%$ da distância base $ightarrow$ professor 14B, superando o critério $G \ge 0.70$).
- **Análise de Erros e Validação E2E no `AgentOrchestrator`:**
  - O adaptador LoRA atingiu **100% de validade JSON** (`36/36`) e **100% de acerto em `write_file`** (`10/10`), concentrando os erros residuais na fronteira entre `append_file` e `write_file` (`8/15`) e no critério de parada `done` (`6/9` continuando com `append_file`).
  - Na simulação end-to-end dentro do `AgentOrchestrator` (`task_002`), o modelo 1.5B + LoRA emitiu 100% de chamadas JSON estruturalmente válidas, mas entrou em repetição de escrita de teste até ser contido pelos guardrails anti-loop e pelo limite `max_iterations` — validando empiricamente a arquitetura híbrida (`ROLE_TOOL_SELECTION` no 1.5B e `ROLE_REPAIR` no 14B).

### Reprodução dos Experimentos da Parte 2

```bash
# Avaliação offline dos baselines e bootstrap pareado
python -m scripts.eval_baselines --dataset corpus_filtered/dataset.jsonl --manifest corpus_filtered/split_manifest.json --out-report corpus_filtered/baselines_report.json

# Relatórios salvos do G6 e G8
# - corpus_filtered/baselines_report.json
# - corpus_filtered/eval_g8_report.json
# - corpus_filtered/error_analysis_g8.json
