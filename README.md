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
- [Problema(s) conhecido(s)](#problemas-conhecidos)
- [Parte 2: Especialização via SFT + QLoRA](#parte-2-especialização-via-sft--qlora)
  - [Pipeline de Dados, Treinamento e Avaliação](#pipeline-de-dados-treinamento-e-avaliação)
  - [Avaliação Nível 1: Seleção de Ferramenta por Turno](#avaliação-nível-1-seleção-de-ferramenta-por-turno-testjsonl-n--168-t--00)
  - [Avaliação Nível 2: Agente Híbrido End-to-End e Ablações](#avaliação-nível-2-agente-híbrido-end-to-end-e-ablações-split--test-n--12-t--00)
  - [Síntese das Questões Experimentais (`Q1`–`Q8`) e Limitações](#síntese-das-questões-experimentais-q1q8-e-limitações)

---

## Arquitetura

## Arquitetura

O código é organizado em módulos com responsabilidades isoladas, todos sob `src/`:

| Módulo | Responsabilidade |
|---|---|
| `src/llm/` | Cliente HTTP para a LLM (`client.py`), roteamento híbrido por papel (`routing.py`), retry/backoff (`retry.py`), contrato e parser defensivo de ações em JSON (`action_contract.py`, `action_parser.py`), métricas por chamada (`metrics.py`) e hierarquia de erros (`errors.py`). |
| `src/spec/` | Pipeline de Spec-Driven Development: prompts (`prompts.py`), geração com retry (`generator.py`), validação programática (`validators.py`), persistência em disco (`persistence.py`), coleta de contexto do projeto (`context.py`) e orquestração do pipeline (`pipeline.py`). |
| `src/agent/` | Núcleo do agente: máquina de estados (`state.py`), orquestrador do loop principal (`orchestrator.py`), validador de ações (`validator.py`), hard gates (`gates.py`), executor de ferramentas (`executor.py`), harness de verificação (`verifier.py`), cobertura de SPEC (`cobertura.py`), lint estático de testes (`lint_testes.py`), construtor de contexto de reparo (`repair_context.py`), gerenciamento de orçamento de tokens (`context_manager.py`) e schemas de estado (`schemas.py`). |
| `src/tools/` | Catálogo central (`catalog.py`), registro de ferramentas (`registry.py`), implementações (`implementations.py`), wrapper de segurança (`base.py`), sandbox de caminho (`sandbox.py`) e truncamento de saída (`truncate.py`). |
| `src/security/` | Camada de segurança compartilhada: resolução de caminhos sandboxed (`sandbox.py`), validador de comandos de shell (`validator.py`) e exceções (`errors.py`). |
| `src/trace/` | Logger de execução append-only (`logger.py`) e snapshot de estado por run (`state.py`), usados para auditoria em `.minicode/`. |
| `src/dataset/` | Normalização de ações, validação de alvos F6 (`target_validator.py`) e construção de exemplos SFT (`build_sft_dataset.py`). |
| `src/eval/` | Avaliação offline Nível 1 de seleção de ferramenta, validade JSON e métricas por classe (`eval_level1.py`). |
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

`DONE` e `FAILED` são estados terminais, nenhuma transição é permitida a partir deles.

Passo a passo de uma execução:

1. **RECEIVED → SPECIFIED → PLANNED**: o pipeline de SDD (`src/spec/pipeline.py`) gera e valida `specification.md` e `plan.md` antes de liberar qualquer escrita de código.
2. **PLANNED → IMPLEMENTING**: início do loop de execução.
3. **Loop IMPLEMENTING** (`AgentOrchestrator.run`), a cada iteração:
   1. monta o prompt (SPEC + PLAN + testes exigidos + estado atual + histórico resumido + última observação);
   2. checa o orçamento de tokens (`estimar_tokens`) contra `0.8 × (MAX_MODEL_LEN − T_RESP_RESERVE)`, se estourar isso então o agente vai para `FAILED` com `context_overflow`;
   3. chama a LLM (roteando por papel entre `ROLE_TOOL_SELECTION` e `ROLE_REPAIR`) e faz o parsing da ação JSON retornada (`interpretar_acao` / `parse_acao_json`, tolerante a cercas ```` ```json ```` e a texto solto ao redor);
   4. valida a ação (`ActionValidator`), por exemplo: a ferramenta existe? o caminho é seguro? os argumentos obrigatórios e limites de tamanho presentes?;
   5. checa os **hard gates** (`ExecutionGateManager`) , veja seção [Agent Harness](#agent-harness);
   6. executa a ferramenta de forma isolada (`ToolExecutor`), nunca deixando uma exceção escapar para o loop;
   7. registra a ação em `actions.jsonl` e atualiza os contadores monotônicos de controle na máquina de estados (`sequence_counter`, `last_write_seq`, `last_test_seq`, `last_test_ok`);
   8. se a ferramenta foi `done`, transiciona para `VERIFYING`.
4. **VERIFYING**: só o **harness** roda os testes (a LLM nunca chama `run_tests` diretamente aqui, essa ação é bloqueada explicitamente no orquestrador). Se os testes passam, a cobertura `test_ac_n` está completa e o hard gate de consistência (`last_test_seq > last_write_seq` e `last_test_ok`) é satisfeito, o agente vai para `DONE`. Caso contrário, vai para `REPAIRING` (ou `FAILED`, se as tentativas de reparo se esgotaram ou `repair_enabled=False`).
5. **REPAIRING → VERIFYING**: ciclo de correção guiado pela saída real do `pytest` (ver [Repair Loop](#repair-loop)), até passar ou atingir `MAX_REPAIR_ATTEMPTS`.
6. **Condições de parada**: `MAX_ITERATIONS` excedido, `MAX_PARSE_FAILURES` consecutivas, `MAX_REPAIR_ATTEMPTS` esgotado, `context_overflow`, `llm_error` ou `repair_disabled`, todas levam a `FAILED` com um `failure_reason` explícito, persistido em `state.json`.

---

## Ferramentas implementadas

Definidas no catálogo central (`src/tools/catalog.py`) e mapeadas para execução em `src/tools/registry.py`, sendo expostas para a LLM em dois formatos, schema OpenAI (`export_openai_tools`, para tool calling nativo) e texto para o prompt (`export_prompt_text`, para o modo JSON):

| Ferramenta | Descrição | Observações |
|---|---|---|
| `list_files` | Lista arquivos/pastas recursivamente a partir de um caminho, com limite de entradas. | Ignora `.git`, `__pycache__`, `.venv`, `.minicode`, `node_modules`. |
| `read_file` | Lê um arquivo com numeração de linhas, com recorte opcional por `start_line`/`end_line`. | Trunca saída grande (4.000 chars) preservando início e fim. |
| `write_file` | Cria/sobrescreve um arquivo inteiro, criando diretórios pai automaticamente. | Só permitida em `IMPLEMENTING`/`REPAIRING` (bloqueada em `tests/` durante `REPAIRING`). Limite de 500.000 caracteres. |
| `append_file` | Acrescenta conteúdo ao final de um arquivo (cria o arquivo se não existir). | Só permitida em `IMPLEMENTING`/`REPAIRING` (bloqueada em `tests/` durante `REPAIRING`). Limite de 500.000 caracteres. |
| `replace_text` | Substitui um trecho exato (`old_str` por `new_str`) em um arquivo existente. | Só permitida em `IMPLEMENTING`/`REPAIRING` (bloqueada em `tests/` durante `REPAIRING`); exige ocorrência única de `old_str`. |
| `run_tests` | Executa o `pytest` via subprocesso com timeout. | **Exclusiva do harness** (`so_harness=True`) — omitida do prompt da LLM e rejeitada pelo orquestrador/gates se chamada. |
| `done` | Sinaliza pedido de verificação (não é uma ferramenta de arquivo). | Tratada no orquestrador, não passa pelo `ToolRegistry`; transiciona `IMPLEMENTING`/`REPAIRING` → `VERIFYING`. |

Toda ferramenta registrada é decorada com `@safe_tool` (`src/tools/base.py`), que captura `TimeoutExpired`, `FileNotFoundError`, `PermissionError` e qualquer `Exception` genérica, convertendo-as em um `ToolResult(ok=False, ...),  assim nenhuma exceção de ferramenta propaga para o loop do agente.

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

- **Parser de ação em 3 camadas** (`action_parser.parse_acao_json` + `action_contract.interpretar_acao`): primeiro tenta o texto inteiro como JSON; em seguida extrai o conteúdo de uma cerca ```` ``` ```` (com ou sem `json`); por fim varre o texto procurando um objeto `{...}` balanceado (contando chaves e respeitando strings/escapes). A primeira camada que produzir um JSON válido com `tool`/`args` (e `thought` opcional) e uma ferramenta reconhecida vence; senão, `ParseError`.
- **Máquina de estados com tabela de transições explícita** (`TRANSITIONS`): qualquer transição fora da tabela levanta `IllegalTransition` ( testado exaustivamente via produto cartesiano 8×8 em `test_state_machine.py`).
- **Hard Gates** (`src/agent/gates.py` + `src/agent/state.py`):
  - *Sem SPEC/PLAN, sem escrita*: `write_file`, `append_file` e `replace_text` só são permitidas em `IMPLEMENTING`/`REPAIRING`.
  - *Sem teste, sem DONE*: `validate_done_gate()` só retorna `True` se `last_test_seq > last_write_seq` **e** `last_test_ok` for verdadeiro, ou seja, toda escrita invalida o atestado de testes anterior, exigindo um novo `run_tests` do harness após a última modificação. Em `IMPLEMENTING`/`REPAIRING`, o gate aceita `done` imediatamente (ele apenas *pede* verificação); o hard gate real é avaliado pelo harness ao processar `VERIFYING`.
  - *`run_tests` é exclusivo do harness*: o orquestrador rejeita explicitamente qualquer tentativa da LLM de chamar `run_tests` fora do ciclo de `VERIFYING`.
- **Orçamento de contexto** (`context_manager.py`): estimativa de tokens por caracteres (`÷ 3,5`); truncamento inteligente da saída do `pytest` preservando o `short test summary info` e as últimas linhas do relatório; histórico de ações comprimido às últimas N entradas.
- **Limites anti-loop** (`src/config.py`, todos configuráveis via variável de ambiente): `MAX_ITERATIONS` (15), `MAX_REPAIR_ATTEMPTS` (2), `MAX_PARSE_FAILURES` (3), `COMMAND_TIMEOUT` (60s).
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
- **Proteção de testes** (`ActionValidator.validate` em `src/agent/validator.py`): durante `IMPLEMENTING`, o agente pode criar e ajustar testes (desde que mantenha todos os `test_ac_n` exigidos pela SPEC, verificados por `cobertura.py` em `VERIFYING`); já em `REPAIRING`, qualquer ferramenta de escrita (`write_file`, `append_file`, `replace_text`) dentro de `tests/` é bloqueada — a LLM não pode alterar os testes para mascarar falhas de implementação.
- **Limite de escrita**: `write_file` e `append_file` rejeitam conteúdo acima de 500.000 caracteres.
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
- Acesso a um endpoint compatível com a API da OpenAI (ex.: vLLM servindo `Qwen/Qwen2.5-14B-Instruct-AWQ`)

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

Limites de execução e opções de modelo também são configuráveis por variáveis de ambiente (`src/config.py` e `src/llm/client.py`):

* **Execução e limites:** `MINICODE_MAX_ITERATIONS`, `MINICODE_MAX_REPAIR_ATTEMPTS`, `MINICODE_COMMAND_TIMEOUT`, `MINICODE_MAX_PARSE_FAILURES`
* **Contexto e modelo:** `MINICODE_MAX_MODEL_LEN`, `MINICODE_T_RESP_RESERVE`, `MINICODE_CONSTRAINED_MODE`
* **Ferramentas (Tool calling):** `MINICODE_TOOL_MODEL`, `MINICODE_TOOL_API_BASE`, `MINICODE_TOOL_API_KEY`

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

## Parte 2: Especialização via SFT + QLoRA

Na **Parte 2**, o MiniCode foi estendido com um pipeline completo de coleta multi-seed de trajetórias, filtragem determinística (**F1–F6**), destilação via SFT + QLoRA para o modelo leve **`Qwen/Qwen2.5-Coder-1.5B-Instruct`** e roteamento híbrido no orquestrador (`ROLE_TOOL_SELECTION` no modelo especializado 1.5B e `ROLE_REPAIR` no professor `Qwen/Qwen2.5-14B-Instruct-AWQ`).

### Pipeline de Dados, Treinamento e Avaliação

1. **Coleta Multi-Seed e Filtragem Determinística:** Trajetórias brutas filtradas pelos critérios **F1–F6** e particionadas por `task_id` (`split_manifest.json` congelado, zero vazamento entre splits).
2. **Corpus SFT Unificado (`minicode-prompt-v2`):** **852 exemplos** de treino (`48` tarefas), **132 exemplos** de validação (`8` tarefas) e **168 exemplos** de teste (`10` tarefas) sob o mesmo system prompt compacto compartilhado entre coleta, treino e inferência.
3. **Treinamento QLoRA Multi-Seed:** Fine-tuning em 4-bit NF4 (`float16` compute, LoRA `r=16`, `lora_alpha=32`, `lora_dropout=0.05`, `18.464.768` parâmetros treináveis = `1,18%` do total, `max_length=2048`, 3 épocas) nas seeds `42`, `43` e `44` + curva de aprendizado (`25%`, `50%`, `100%`), atingindo `val_loss` médio de **0,0976 ± 0,0028** e melhor checkpoint **`seed42_frac100`** (`val_loss = 0,0948`).

1. **Coleta Multi-Seed e Filtragem Determinística:** Trajetórias brutas filtradas pelos critérios **F1–F6** e particionadas por `task_id` (`split_manifest.json` congelado, zero vazamento entre splits).
2. **Corpus SFT Unificado (`minicode-prompt-v2`):** **852 exemplos** de treino (`48` tarefas), **132 exemplos** de validação (`8` tarefas) e **168 exemplos** de teste (`10` tarefas) sob o mesmo system prompt compacto compartilhado entre coleta, treino e inferência.
3. **Treinamento QLoRA Multi-Seed:** Fine-tuning em 4-bit NF4 (`float16` compute, LoRA `r=16`, `lora_alpha=32`, `lora_dropout=0.05`, `18.464.768` parâmetros treináveis = `1,18%` do total, `max_length=2048`, 3 épocas) nas seeds `42`, `43` e `44` + curva de aprendizado (`25%`, `50%`, `100%`), atingindo `val_loss` médio de **0,0976 ± 0,0028** e melhor checkpoint **`seed42_frac100`** (`val_loss = 0,0948`).

### Avaliação Nível 1: Seleção de Ferramenta por Turno (`test.jsonl`, $N = 168$, $T = 0{,}0$)

- **Prompt:** `minicode-prompt-v2` | **Temperatura:** `0.0` | **Bootstrap:** `10.000` reamostragens por `task_id` (`N = 168`, `10` tarefas)

| Sistema | Acc_tool (IC 95%) | JSON_valid (IC 95%) | Exact_action (IC 95%) | F1_macro (IC 95%) |
| :--- | :---: | :---: | :---: | :---: |
| `maioria` | 0.3274 [0.2512, 0.4356] | 1.0000 [1.0000, 1.0000] | 0.0000 [0.0000, 0.0000] | 0.0987 [0.0803, 0.1275] |
| `base_1_5b_zero_shot` | 0.1607 [0.0730, 0.2296] | 0.7262 [0.6111, 0.8252] | 0.0357 [0.0093, 0.0562] | 0.1020 [0.0407, 0.1723] |
| `1_5b_lora` | 0.5536 [0.4070, 0.7258] | 1.0000 [1.0000, 1.0000] | 0.2083 [0.1203, 0.3169] | 0.3700 [0.2915, 0.4846] |
| `14b_zero_shot` | 0.6012 [0.5316, 0.6735] | 1.0000 [1.0000, 1.0000] | 0.3571 [0.2283, 0.4833] | 0.4037 [0.3533, 0.4654] |
| `14b_few_shot` | 0.4583 [0.3350, 0.6067] | 1.0000 [1.0000, 1.0000] | 0.1250 [0.0930, 0.1584] | 0.2676 [0.2077, 0.3146] |

- **Recuperação $G$ (vs. `14B zero-shot`):** **`0,8919`** (IC 95%: `[0,6000, 1,2698]`, denominador $A_{14\text{B}} - A_{\text{base}} = 0{,}4405$)
- **Recuperação $G$ (vs. `14B few-shot`):** **`1,3202`** (IC 95%: `[1,0317, 1,8000]`, denominador $A_{14\text{B}} - A_{\text{base}} = 0{,}2976$)
- **Hipóteses Nível 1:** **H1 suportada** (`True`) | **H2 suportada** (`True`)

- **Pontos de erro conhecidos (`1_5b_lora`):** `write_file` (suporte=55, recall=`0,7273`, confusão com `append_file`: 15), `append_file` (suporte=45, recall=`0,6889`, confusão com `write_file`: 8, `done`: 6), `done` (suporte=38, recall=`0,5789`, confusão com `write_file`: 7, `append_file`: 9).

### Avaliação Nível 2: Agente Híbrido End-to-End e Ablações (`split = test`, $n = 12$, $T = 0{,}0$)

- **Split:** `test` congelado (12 tarefas; `experiments/spec_cache/manifest_frozen.json`, 12/12 hashes SHA-256 validados)
- **Configuração:** `temperature = 0.0`, `seed = 1`, `max_iterations = 15`, `max_repair_attempts = 2`, prompt `minicode-prompt-v2`
- **Adapter (Células C e E):** `checkpoints/step8/selected_adapter` (`seed42_frac100`, `val_loss = 0.0948`)

#### 1. Tabela Principal de TSR e Eficiência por Célula

| Célula | Configuração (`IMPLEMENTING` / `REPAIRING`) | TSR ($N_{\text{DONE}}/12$) | IC Wilson 95% | IC Bootstrap 95% | $\rho_{\text{pool}}$ ($N_{1.5\text{B}}/N_{14\text{B}}$) | Iterações Média (Med / IQR) | Tokens Média (Med / IQR) | Latência Média (Med / IQR) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A** | `14B` / `14B` (*baseline* professor) | **75.0%** (9/12) | [46.8%, 91.1%] | [50.0%, 100.0%] | 0.0000 (0/98) | 9.17 (7.5 / 10.0) | 14964 (10150 / 18013) | 1816.59s (89.85s / 86.32s) |
| **B** | `1.5B_base` / `14B` (*zero-shot*) | **0.0%** (0/12) | [0.0%, 24.2%] | [0.0%, 0.0%] | 1.0000 (145/0) | 12.08 (15.0 / 2.8) | 13698 (16419 / 5315) | 49.13s (55.22s / 33.83s) |
| **C** | `1.5B_lora` / `14B` | **33.3%** (4/12) | [13.8%, 60.9%] | [8.3%, 58.3%] | 0.5563 (79/63) | 12.83 (15.0 / 5.2) | 18715 (20911 / 14235) | 134.77s (98.81s / 49.40s) |
| **D** | `14B` / *no_repair* | **66.7%** (8/12) | [39.1%, 86.2%] | [41.7%, 91.7%] | 0.0000 (0/83) | 7.83 (7.0 / 3.8) | 11649 (9332 / 5994) | 118.11s (70.45s / 145.82s) |
| **E** | `1.5B_lora + constrained` / `14B` *(10.f)* | **41.7%** (5/12) | [19.3%, 68.0%] | [16.7%, 66.7%] | 0.5725 (79/59) | 12.58 (15.0 / 5.2) | 15801 (17014 / 10694) | 119.70s (105.97s / 14.12s) |
| **NO_SPEC** | `14B` / `14B` (sem `specification.md`) *(10.f)* | **41.7%** (5/12) | [19.3%, 68.0%] | [16.7%, 66.7%] | 0.0000 (0/124) | 11.33 (13.0 / 7.2) | 18184 (19806 / 14268) | 58.78s (56.47s / 42.38s) |

#### 2. Comparações Pareadas por Tarefa (Bootstrap $B = 10.000$)

| Contraste | $\Delta TSR$ | IC Bootstrap 95% | $p$-valor (bilateral) | $\Delta\text{Tokens}$ [IC 95%] |
| :--- | :---: | :---: | :---: | :---: |
| `C_minus_B` | **+0.3333** | [+0.0833, +0.5833] | 0.0138 | +5017 [+319, +9765] |
| `C_minus_A` | **-0.4167** | [-0.7500, -0.0833] | 0.0466 | +3751 [-3831, +10922] |
| `D_minus_A` | **-0.0833** | [-0.2500, +0.0000] | 0.7102 | -3315 [-7248, -343] |
| `E_minus_C` | **+0.0833** | [+0.0000, +0.2500] | 0.6916 | -2914 [-4468, -1533] |
| `E_minus_A` | **-0.3333** | [-0.6667, +0.0000] | 0.0990 | +837 [-5795, +7273] |
| `NO_SPEC_minus_A` | **-0.3333** | [-0.5833, -0.0833] | 0.0152 | +3220 [-1138, +8044] |

#### 3. Diagnóstico de Falhas: Célula C e Célula E

- **Célula C (`1.5B_lora`):** 4/12 `DONE` autônomos (`task_015`, `task_041`, `task_052`, `task_057` com `repairs = 0`, $\rho = 1{,}0$); 8/12 `FAILED` por `max_iterations = 15`, marcados por escritas consecutivas no mesmo arquivo (`12/12` runs com `consec_same > 0`, `9/12` com escrita idêntica repetida) antes de acionar `done`.
- **Célula E (`1.5B_lora + constrained`):** 5/12 `DONE` (`+8,3 pp` sobre C, resgatando `task_007` no reparo e reduzindo a mediana de tokens de `20.911` para `17.014`).
- **Ablação `no_spec` (`14B` sem `specification.md`):** queda de **75,0% (9/12)** para **41,7% (5/12)** ($-33{,}3\text{ pp}$), confirmando o papel estruturante da especificação formal e dos *hard gates*.

### Síntese das Questões Experimentais (`Q1`–`Q8`) e Limitações

#### 11.a Tabela Consolidada das 8 Questões (§37)

| Questão (§37) | Fonte / Comparação | Métricas e Evidência Quantitativa | Conclusão / Resposta |
| :--- | :--- | :--- | :--- |
| **Q1** *(Papel do Repair Loop na resolução de tarefas)* | **A × D** (`14B` vs. `14B_no_repair`, Nível 2) | $TSR_A = 9/12$ (**75,0%**, Wilson $[46{,}8\%, 91{,}1\%]$) vs. $TSR_D = 8/12$ (**66,7%**, Wilson $[39{,}1\%, 86{,}2\%]$); $\Delta TSR_{D-A} = \mathbf{-8{,}3	ext{ p.p.}}$ ($	ext{IC}_{95\%} = [-25{,}0, +0{,}0	ext{ p.p.}]$, $p = 0{,}7102$). | O ciclo `REPAIRING` resgata **$+1/12$ tarefa** ($+8{,}3	ext{ p.p.}$), embora a diferença não seja estatisticamente significativa para $n = 12$. |
| **Q2** *(Impacto do SDD: `SPEC+PLAN` com × sem)* | **A × NO_SPEC** (ablação `10.f`, Nível 2) | $TSR_A = 9/12$ (**75,0%**) vs. $TSR_{	ext{NO\_SPEC}} = 5/12$ (**41,7%**, Wilson $[19{,}3\%, 68{,}0\%]$); $\Delta TSR = \mathbf{-33{,}3	ext{ p.p.}}$ ($	ext{IC}_{95\%} = [-58{,}3, -8{,}3	ext{ p.p.}]$, $p = 0{,}0152$); mediana sobe de $7{,}5 	o 13{,}0$ iterações e $10.150 	o 19.806$ tokens. | **Testada e confirmada**: remover `specification.md` reduz o $TSR$ em **$-33{,}3	ext{ p.p.}$** ($p = 0{,}0152$), comprovando que o contrato formal guia a implementação e os critérios de aceitação. |
| **Q3** *(Custo em tokens e iterações do Repair Loop)* | **A × D** (`14B` vs. `14B_no_repair`, Nível 2) | Tokens médios caem de **$14.964$** (mediana $10.150$, $	ext{IQR} = 18.013$) em **A** para **$11.649$** (mediana $9.332$, $	ext{IQR} = 5.994$) em **D**: $\Delta	ext{Tokens}_{D-A} = \mathbf{-3.315}$ ($	ext{IC}_{95\%} = [-7.248, -343]$); iterações médias caem de $9{,}17$ (mediana $7{,}5$) para $7{,}83$ (mediana $7{,}0$). | Desabilitar o reparo gera economia estatisticamente significativa de **$3.315$ tokens/tarefa** ($-22{,}2\%$) e reduz $-1{,}34$ iterações médias. |
| **Q4** *(Eficiência temporal e chamadas ao professor)* | **A × D** (`14B` vs. `14B_no_repair`, Nível 2) | Latência mediana cai de **$89{,}85	ext{ s}$** (A) para **$70{,}45	ext{ s}$** (D) ($-19{,}40	ext{ s}$, **$-21{,}6\%$**); chamadas totais ao `14B` caem de **$98$** (A) para **$83$** (D) (**$-15{,}3\%$**). | A Célula D retém **$88{,}9\%$ ($8/9$)** dos acertos da Célula A com latência mediana $21{,}6\%$ menor e $15$ chamadas a menos ao `14B`. |
| **Q5** *(Ganho da especialização QLoRA offline)* | **B × C no Nível 1** (`base_1_5b_zero_shot` vs. `1_5b_lora`) | `Acc_tool`: **$16{,}07\%$** ($[7{,}30\%, 22{,}96\%]$) $	o$ **$55{,}36\%$** ($[40{,}70\%, 72{,}58\%]$), $\Delta = \mathbf{+39{,}29	ext{ p.p.}}$ ($\Delta_{	ext{boot}} = +40{,}09	ext{ p.p.}$, $[+24{,}60, +58{,}54]$, $p(\Delta>0) = 1{,}0000$); `JSON_valid`: **$72{,}62\%$** $	o$ **$100{,}00\%$**; `Exact_action`: **$3{,}57\%$** $	o$ **$20{,}83\%$**; `F1_macro`: **$0{,}1020$** $	o$ **$0{,}3700$**; recuperação $G = \mathbf{0{,}8919}$ ($[0{,}6000, 1{,}2698]$). | **H1 e H2 confirmadas**: o QLoRA elimina $100\%$ dos erros de sintaxe JSON e recupera **$89{,}19\%$** da distância até o professor `14B` na seleção offline de ferramentas. |
| **Q6** *(Transferência do SFT para o agente híbrido E2E e taxa $
ho$)* | **A × B × C no Nível 2** (`14B` vs. `1.5B_base+14B` vs. `1.5B_lora+14B`) | **B**: $TSR = 0/12$ (**0,0%**), $
ho_{	ext{pool}} = 1{,}0000$ ($145/0$, nunca atinge `VERIFYING`).<br>**C**: $TSR = 4/12$ (**33,3%**, Wilson $[13{,}8\%, 60{,}9\%]$), $
ho_{	ext{pool}} = \mathbf{0{,}5563}$ ($79/63$), $\Delta TSR_{C-B} = \mathbf{+33{,}3	ext{ p.p.}}$ ($[+8{,}3, +58{,}3	ext{ p.p.}]$, $p = 0{,}0138$), concluindo 4 tarefas com `repairs = 0` ($
ho = 1{,}00$). *(Célula E com constrained decoding: $TSR = 5/12 = \mathbf{41{,}7\%}$, $
ho_{	ext{pool}} = 0{,}5725$)*. | O SFT transfere ganho real para o E2E (**$+33{,}3	ext{ p.p.}$** sobre B, $p = 0{,}0138$), absorvendo **$55{,}63\%$** das chamadas LLM na Célula C e resolvendo 4 tarefas sem acionar o professor. |
| **Q7** *(Gargalo e modo de falha do agente híbrido)* | **A × B × C no Nível 2** (`14B` vs. `1.5B_base+14B` vs. `1.5B_lora+14B`) | $\Delta TSR_{C-A} = \mathbf{-41{,}7	ext{ p.p.}}$ ($[-75{,}0, -8{,}3	ext{ p.p.}]$, $p = 0{,}0466$). Diagnóstico `10.e` em C: nos $8/12$ runs que atingiram `max_iterations = 15`, houve **$12/12$** runs com escrita consecutiva no mesmo arquivo (`consec_same > 0`) e **$9/12$** com escrita idêntica repetida antes de invocar `done`. | O gargalo não é sintaxe JSON (`100%` válida), mas a **repetição consecutiva de `write_file`/`append_file` no mesmo arquivo** sob $T = 0{,}0$ antes de chamar `done` até atingir o teto de 15 iterações. |
| **Q8** *(Taxa de sucesso global por configuração)* | **$TSR$ por célula** (`A`, `B`, `C`, `D`, `E`, `NO_SPEC` no Nível 2, $n = 12$) | • **A (`14B/14B`)**: **75,0%** ($9/12$) — Wilson $[46{,}8\%, 91{,}1\%]$ \| Boot $[50{,}0\%, 100{,}0\%]$<br>• **D (`14B/no_repair`)**: **66,7%** ($8/12$) — Wilson $[39{,}1\%, 86{,}2\%]$ \| Boot $[41{,}7\%, 91{,}7\%]$<br>• **E (`1.5B_lora+constrained/14B`)**: **41,7%** ($5/12$) — Wilson $[19{,}3\%, 68{,}0\%]$ \| Boot $[16{,}7\%, 66{,}7\%]$<br>• **NO_SPEC (`14B sem SPEC`)**: **41,7%** ($5/12$) — Wilson $[19{,}3\%, 68{,}0\%]$ \| Boot $[16{,}7\%, 66{,}7\%]$<br>• **C (`1.5B_lora/14B`)**: **33,3%** ($4/12$) — Wilson $[13{,}8\%, 60{,}9\%]$ \| Boot $[8{,}3\%, 58{,}3\%]$<br>• **B (`1.5B_base/14B`)**: **0,0%** ($0/12$) — Wilson $[0{,}0\%, 24{,}2\%]$ \| Boot $[0{,}0\%, 0{,}0\%]$ | Hierarquia E2E: **$	ext{A } (75{,}0\%) > 	ext{D } (66{,}7\%) > 	ext{E } (41{,}7\%) = 	ext{NO\_SPEC } (41{,}7\%) > 	ext{C } (33{,}3\%) > 	ext{B } (0{,}0\%)$**. |

---

#### 11.b Questão Q2 (SDD com × sem)

- **Status**: **Testada** via ablação `NO_SPEC` (etapa `10.f`, `14B` executando sem `specification.md` sobre as mesmas $12$ tarefas de teste).
- **Resultado**: A remoção da especificação formal derruba o $TSR$ de **75,0% (9/12)** (Célula A) para **41,7% (5/12)** (`NO_SPEC`), uma queda de **$-33{,}3	ext{ p.p.}$** ($	ext{IC}_{95\%} = [-58{,}3, -8{,}3	ext{ p.p.}]$, $p = 0{,}0152$), confirmando o papel estruturante do SDD.

---

#### 11.c Limitações do Estudo

1. **Rótulo vindo do professor (*Teacher Forcing* e equivalência de ações):** Tanto o treinamento QLoRA quanto a avaliação offline do Nível 1 utilizam como verdade terrestre a ação exata emitida pelo modelo professor (`Qwen2.5-14B-Instruct-AWQ`). Em tarefas de edição de código, diferentes chamadas de ferramenta produzem o mesmo estado final no sistema de arquivos (por exemplo, reescrever o arquivo inteiro com `write_file` versus acrescentar um bloco com `append_file` — origem de $23$ dos $30$ erros residuais do `1_5b_lora` nessas duas classes no Nível 1). Além disso, trajetórias históricas bem-sucedidas não fornecem supervisão negativa explícita contra loops de reescrita repetida, e o conjunto de teste do Nível 1 concentra exemplos do estrato `IMPLEMENTING` ($N = 168$).
2. **Tamanho amostral reduzido do conjunto de teste ($n = 12$ tarefas no Nível 2 / $n = 10$ tarefas no Nível 1):** O particionamento estratificado congelado por `task_id` elimina vazamento de dados, mas $n = 12$ tarefas gera intervalos de confiança de Wilson e *bootstrap* relativamente largos (e.g., $[46{,}8\%, 91{,}1\%]$ na Célula A e $[13{,}8\%, 60{,}9\%]$ na Célula C), limitando o poder estatístico para diferenças de 1 tarefa ($8{,}3	ext{ p.p.}$, como `D` vs. `A` com $p = 0{,}7102$).
3. **Temperatura efetiva da coleta versus avaliação determinística ($T = 0{,}0$):** Todas as execuções da bateria principal foram avaliadas em regime guloso ($T = 0{,}0$), sem execução da bateria complementar opcional em $T = 0{,}3$. Embora $T = 0{,}0$ garanta reprodutibilidade determinística, ele elimina qualquer escape estocástico quando o modelo compacto (`1.5B_lora`) entra em um estado fora da distribuição de treino, favorecendo loops idênticos consecutivos de `write_file`/`append_file` observados nas execuções que atingiram `max_iterations = 15`. Em contraste, a coleta multi-seed do corpus operou com amostragem estocástica para diversidade de trajetórias.
4. **Mistura de versões e parâmetros históricos na coleta do corpus:** A construção incremental do corpus (`runs_metadata.jsonl`) agregou execuções coletadas ao longo da evolução do orquestrador sob pequenas revisões de parâmetros antes da estabilização final (como teto de $15$ iterações e contabilidade histórica de reparos). Embora os filtros determinísticos **F1–F6**, o `split_manifest.json` congelado, o `SPEC_CACHE` auditado ($12/12$ hashes SHA-256) e o prompt `minicode-prompt-v2` tenham padronizado integralmente o treino e as avaliações finais, os pesos do adaptador LoRA herdam essa heterogeneidade histórica do conjunto bruto.

### Reprodução dos Experimentos

- **Suíte de testes unitários e de contrato:** `python -m pytest -q`
- **Avaliação Nível 1:** `python -m scripts.eval_step9_level1 --test-jsonl datasets/sft_jsonl/test.jsonl --adapter-dir checkpoints/step8/selected_adapter --out-dir experiments/step9`
- **Avaliação Nível 2 (Células A–D):** `python -m scripts.run_e2e_hybrid --split test --adapter-dir checkpoints/step8/selected_adapter`
- **Consolidação Estatística Nível 2:** `python -m scripts.analyze_step10_e2e`
- **Relatório Oficial Q1–Q8 (§37):** `experiments/step11/final_analysis_q1_q8.md`
