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
| `write_file` | Cria/sobrescreve um arquivo inteiro, criando diretórios pai automaticamente. | Só permitida em `IMPLEMENTING`/`REPAIRING` (hard gate). Limite de 500.000 caracteres. |
| `append_file` | Acrescenta conteúdo ao final de um arquivo (cria o arquivo se não existir). | Só permitida em `IMPLEMENTING`/`REPAIRING` (hard gate). Limite de 500.000 caracteres. |
| `replace_text` | Substitui um trecho exato (`old_str` por `new_str`) em um arquivo existente. | Só permitida em `IMPLEMENTING`/`REPAIRING` (hard gate); exige ocorrência única de `old_str`. |
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
- **Proteção de testes** (`ActionValidator.validate` em `src/agent/validator.py` e `AgentOrchestrator`): em `REPAIRING`, qualquer ferramenta de escrita (`write_file`, `append_file`, `replace_text`) que resolva para dentro de `tests/` é bloqueada — a LLM não pode "corrigir" o teste em vez do código; em `IMPLEMENTING`, os arquivos de teste pré-existentes da tarefa também são protegidos contra modificação, permitindo apenas a criação de novos arquivos de aceitação (ex.: `tests/test_spec_ac.py`).
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

Cobertura atual: **138 testes** em 23 arquivos (`tests/test_*.py`), cobrindo parser de ação, máquina de estados (produto cartesiano completo de transições), hard gates, ferramentas + sandbox, segurança (denylist/allowlist de comandos), pipeline de SDD, orquestrador (loop de sucesso, estouro de iterações, falha de parse), repair loop (assinatura de erro, proteção de `tests/`, sucesso após reparo, exaustão de tentativas), contrato estrito de ferramentas/prompts, pipeline de dataset SFT, rastreabilidade (logger + state) e CLI (parsing de argumentos, renderers, REPL).

---

## Problema(s) conhecido(s)
Durante a fase de testes, identificou-se um comportamento limite sensível ao contexto de execução:

**Ambiguidade de Workspace ao Executar no Diretório Raiz (`--project .`)**:
  * **Descrição**: Ao executar o MiniCode apontando o argumento `--project` para o próprio diretório de desenvolvimento do framework (`--project .`), a LLM pode interpretar erroneamente a árvore de arquivos corrente e tentar gravar, modificar ou reescrever arquivos do próprio core do MiniCode (`src/` ou `tests/`) em vez de criar uma aplicação independente.
  * **Fase de Ocorrência**: Ocorre tipicamente na transição entre o planejamento (`PLANNED`) e a implementação (`IMPLEMENTING`) quando o escopo dos caminhos alvo não está estritamente confinado pelo modelo.
  * **Mitigação Recomendada**: Para evitar qualquer sobrescrita acidental nos fontes do framework, **execute sempre as tarefas em diretórios de projetos separados e isolados** fora da árvore do MiniCode (por exemplo: `--project ../demo_soma`), garantindo que o agente confine suas operações de escrita exclusivamente ao workspace da tarefa alvo!

---

## Parte 2 — Especialização via SFT + QLoRA

Na **Parte 2**, o MiniCode foi estendido com um pipeline completo de coleta multi-seed de trajetórias, filtragem determinística (**F1–F6**), destilação via SFT + QLoRA para o modelo leve **`Qwen/Qwen2.5-Coder-1.5B-Instruct`** e roteamento híbrido no orquestrador (`ROLE_TOOL_SELECTION` no modelo especializado 1.5B e `ROLE_REPAIR` no professor `Qwen/Qwen2.5-14B-Instruct-AWQ`).

### Pipeline de Dados, Treinamento e Avaliação (Passos 7–11)

1. **Coleta Multi-Seed e Filtragem Determinística (Passo 7):** Trajetórias brutas filtradas pelos critérios **F1–F6** e particionadas por `task_id` (`split_manifest.json` congelado, zero vazamento entre splits).
2. **Corpus SFT Unificado (`minicode-prompt-v2`):** **852 exemplos** de treino (`48` tarefas), **132 exemplos** de validação (`8` tarefas) e **168 exemplos** de teste (`10` tarefas) sob o mesmo system prompt compacto compartilhado entre coleta, treino e inferência.
3. **Treinamento QLoRA Multi-Seed (Passo 8):** Fine-tuning em 4-bit NF4 (`float16` compute, LoRA `r=16`, `lora_alpha=32`, `lora_dropout=0.05`, `18.464.768` parâmetros treináveis = `1,18%` do total, `max_length=2048`, 3 épocas) nas seeds `42`, `43` e `44` + curva de aprendizado (`25%`, `50%`, `100%`), atingindo `val_loss` médio de **0.0976 ± 0.0028** e melhor checkpoint **`seed42_frac100`** (`val_loss = 0.0948`).

### Avaliação Nível 1 — Seleção de Ferramenta por Turno (`test.jsonl`, $n = 168$, `temp = 0.0`)

# Passo 9 — Avaliação Offline Nível 1

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

### Avaliação Nível 2 — Agente Híbrido End-to-End e Ablações (`split = test`, 12 tarefas, `temp = 0.0`)

# Passo 10 — Avaliação Nível 2 (Agente Completo) e Ablações

- **Split:** `test` congelado (12 tarefas; `experiments/spec_cache/manifest_frozen.json`, 12/12 hashes SHA-256 validados)
- **Configuração:** `temperature = 0.0`, `seed = 1`, `max_iterations = 15`, `max_repair_attempts = 2`, prompt `minicode-prompt-v2`
- **Adapter Passo 8 (Células C e E):** `checkpoints/step8/selected_adapter` (`seed42_frac100`, `val_loss = 0.0948`)

## 1. Tabela Principal de TSR e Eficiência por Célula (10.d + 10.f)

| Célula | Configuração (`IMPLEMENTING` / `REPAIRING`) | TSR ($N_{\text{DONE}}/12$) | IC Wilson 95% | IC Bootstrap 95% | $\rho_{\text{pool}}$ ($N_{1.5\text{B}}/N_{14\text{B}}$) | Iterações Média (Med / IQR) | Tokens Média (Med / IQR) | Latência Média (Med / IQR) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A** | `14B` / `14B` (*baseline* professor) | **75.0%** (9/12) | [46.8%, 91.1%] | [50.0%, 100.0%] | 0.0000 (0/98) | 9.17 (7.5 / 10.0) | 14964 (10150 / 18013) | 1816.59s (89.85s / 86.32s) |
| **B** | `1.5B_base` / `14B` (*zero-shot*) | **0.0%** (0/12) | [0.0%, 24.2%] | [0.0%, 0.0%] | 1.0000 (145/0) | 12.08 (15.0 / 2.8) | 13698 (16419 / 5315) | 49.13s (55.22s / 33.83s) |
| **C** | `1.5B_lora` (Passo 8) / `14B` | **33.3%** (4/12) | [13.8%, 60.9%] | [8.3%, 58.3%] | 0.5563 (79/63) | 12.83 (15.0 / 5.2) | 18715 (20911 / 14235) | 134.77s (98.81s / 49.40s) |
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

## 3. Diagnóstico de Falhas — Célula C (10.e) e Célula E (10.f)

- **Célula C (`1.5B_lora`):** 4/12 `DONE` autônomos (`task_015`, `task_041`, `task_052`, `task_057` com `repairs = 0`, $\rho = 1{,}0$); 8/12 `FAILED` por `max_iterations = 15`, marcados por escritas consecutivas no mesmo arquivo (`12/12` runs com `consec_same > 0`, `9/12` com escrita idêntica repetida) antes de acionar `done`.
- **Célula E (`1.5B_lora + constrained`):** 5/12 `DONE` (`+8,3 pp` sobre C, resgatando `task_007` no reparo e reduzindo a mediana de tokens de `20.911` para `17.014`).
- **Ablação `no_spec` (`14B` sem `specification.md`):** queda de **75,0% (9/12)** para **41,7% (5/12)** ($-33{,}3\text{ pp}$), confirmando o papel estruturante da especificação formal e dos *hard gates*.

### Síntese das Questões Experimentais (`Q1`–`Q8`) e Limitações (Passo 11)

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

#### Limitações Metodológicas

1. **Destilação limitada ao teto do professor (`14B`):** Como os rótulos SFT foram extraídos de trajetórias bem-sucedidas do `Qwen2.5-14B-Instruct-AWQ`, o aluno `1.5B` herda eventuais vieses de planejamento do professor e não recebe supervisão negativa explícita contra loops de reescrita.
2. **Tamanho amostral do split de teste ($N = 12$ tarefas):** Embora o particionamento por `task_id` elimine vazamento de dados, 12 tarefas geram intervalos de confiança de Wilson e *bootstrap* relativamente largos ($\pm 25\text{ pp}$), exigindo cautela na interpretação de diferenças de 1 tarefa ($8{,}3\text{ pp}$).
3. **Temperatura efetiva e determinismo na coleta vs. avaliação:** A coleta multi-seed utilizou amostragem controlada para diversidade de trajetórias (após correção da propagação de temperatura efetiva), enquanto a avaliação Nível 2 rodou com `temperature = 0.0` (1 execução determinística por tarefa/célula, sem a bateria secundária opcional em `T = 0.3`).
4. **Heterogeneidade histórica na coleta bruta:** Os logs brutos acumularam execuções sob pequenas revisões do orquestrador antes da estabilização final, mitigadas pelos filtros determinísticos **F1–F6** e pelo congelamento do `split_manifest.json` e `spec_cache` antes do treino e da avaliação.

### Reprodução dos Experimentos (Passos 7–11)

- **Avaliação Nível 1:** `python -m scripts.eval_step9_level1 --test-jsonl datasets/sft_jsonl/test.jsonl --adapter-dir checkpoints/step8/selected_adapter --out-dir experiments/step9`
- **Avaliação Nível 2 (Células A–E + NO_SPEC):** `python -m scripts.run_e2e_hybrid --split test --adapter-dir checkpoints/step8/selected_adapter`
- **Consolidação Estatística Nível 2:** `python -m scripts.analyze_step10_e2e`
- **Relatório Consolidado Q1–Q8:** `experiments/step11/final_analysis_q1_q8.md`
