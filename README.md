# MiniCode

**MiniCode** é um agente autônomo de programação orientado a LLM. Dada uma tarefa em linguagem natural, o agente gera uma especificação formal (SPEC), planeja os passos de implementação (PLAN), executa ações de forma controlada sobre um workspace isolado (sandbox) e só se considera concluído depois que um harness de testes — independente da LLM — confirma que a suíte `pytest` passa.

O projeto é o Trabalho Prático 1 da disciplina de Redes Neurais (UFMS), cujo modelo-alvo é o `Qwen/Qwen2.5-14B-Instruct-AWQ` servido via vLLM com uma API compatível com OpenAI.

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

> **Nota de implementação:** `replace_text` é citada em seis módulos (`validator.py`, `gates.py`, `verifier.py`, `state.py`, `orchestrator.py`) como ferramenta de escrita permitida, mas **não existe implementação registrada** em `ToolRegistry`. Se a LLM a chamar, o `dispatch` retorna `ok=False, error="Ferramenta desconhecida"` — o agente não trava, mas perde a opção de edição cirúrgica e cai sempre para reescrever o arquivo inteiro via `write_file`. Ver [Limitações conhecidas](#limitações-conhecidas--próximos-passos).

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

> **Importante:** este pipeline está implementado e com cobertura de testes sólida (`tests/test_spec.py`), mas atualmente **não é chamado pela CLI** (`src/cli/main.py::run_task` usa um placeholder fixo em vez de `executar_spec_pipeline`). Ver [Limitações conhecidas](#limitações-conhecidas--próximos-passos) — este é o achado mais importante da revisão da Parte 1.

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

> **Bug real encontrado durante a revisão** (reproduzido de forma determinística, independente de LLM): o harness invoca o `pytest` via `subprocess` sem desativar o cache de bytecode do Python. Os arquivos `.pyc` padrão gravam o `mtime` da fonte truncado para **resolução de segundo**; quando duas escritas (`write_file`) acontecem dentro do mesmo segundo de relógio — algo comum entre uma tentativa de reparo e a próxima verificação —, o `pytest` reaproveita o `.pyc` **desatualizado** e reporta falha mesmo com o arquivo já corrigido no disco. Isso derruba `test_repair_loop_success_after_repair` de forma intermitente-determinística e pode fazer o agente esgotar `MAX_REPAIR_ATTEMPTS` por um motivo que não existe mais no código. Ver detalhes e correção sugerida em [Limitações conhecidas](#limitações-conhecidas--próximos-passos).

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

> **Atenção:** `--max-iterations`, `--max-repair-attempts` e `--test-cmd` são parseados pela CLI mas **ainda não são propagados** ao `AgentOrchestrator`/`HarnessVerifier` — hoje eles sempre usam os defaults de `src/config.py` e o comando fixo `pytest -x --tb=short -q`. Ver [Limitações conhecidas](#limitações-conhecidas--próximos-passos).

---

## Testes

```bash
python -m pytest -q
```

> Use `python -m pytest` (não o executável `pytest` isolado). O `pyproject.toml` não declara `[build-system]`/mapeamento de pacote para o layout `src/`, então `pip install -e .` registra o pacote como `UNKNOWN` e coloca `src/` (não a raiz do projeto) no `sys.path`. Rodar `python -m pytest` a partir da raiz contorna isso, pois o Python insere o diretório atual no `sys.path` automaticamente; o executável `pytest` sozinho, não.

Cobertura atual: **72 testes** em 10 arquivos, cobrindo parser de ação, máquina de estados (produto cartesiano completo de transições), hard gates, ferramentas + sandbox, segurança (denylist/allowlist de comandos), pipeline de SDD, orquestrador (loop de sucesso, estouro de iterações, falha de parse), repair loop (assinatura de erro, proteção de `tests/`, sucesso após reparo, exaustão de tentativas), rastreabilidade (logger + state) e CLI (parsing de argumentos, renderers, REPL).

Na revisão desta documentação, **71 de 72 passaram de forma consistente**; o teste restante (`test_repair_loop_success_after_repair`) falha de forma determinística pelo bug de cache de `.pyc` descrito acima — não é flakiness de ambiente.

---