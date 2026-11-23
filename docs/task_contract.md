# Contrato da Tarefa — Tool Selection (`Passo 1`)

- **Versão do contrato (`contract_version`):** `1.0.0`
- **Versão do prompt (`prompt_version`):** `minicode-prompt-v2`

## 1. Entrada ($x$)
A entrada $x$ de cada exemplo é o prompt de usuário construído pelo `AgentOrchestrator` na iteração:
- **Em `IMPLEMENTING`:** `SPECIFICATION`, `PLAN`, `TESTES EXIGIDOS PELA SPEC`, estado atual do agente (`iteration`, `tests_passed`, `modified_after_last_test`), histórico recente e última observação/resultado de ferramenta.
- **Em `REPAIRING`:** prompt gerado por `RepairContextBuilder.montar_prompt_reparo` com a `SPEC`, arquivos tocados (`touched_files`) e a saída da falha nos testes.

## 2. Saída ($y$)
A saída alvo $y$ é um objeto JSON válido contendo:
- `tool` (string obrigatória): nome da ferramenta escolhida em $\mathcal{T}$.
- `args` (objeto JSON obrigatório): argumentos da ferramenta validados pelo contrato/catálogo (usando sempre `path`, nunca `file_path`).
- `thought` (string opcional): raciocínio breve, não avaliado na métrica primária de seleção de ferramenta.

## 3. Espaço de Rótulos ($\mathcal{T}$)
Ferramentas disponíveis para a LLM no espaço de ações:
- `list_files`
- `read_file`
- `write_file`
- `append_file`
- `replace_text`
- `done`

> **Nota:** `run_tests` não faz parte do espaço de ações da LLM no loop; a verificação de testes é executada automaticamente pelo harness após `done`.

## 4. Estratos de Avaliação e Análise
Os resultados e distribuições devem ser reportados separadamente por estado da FSM:
- `IMPLEMENTING`
- `REPAIRING`

## 5. Imutabilidade e Controle de Versão
- Qualquer mudança no template do prompt, no catálogo de ferramentas ou nas regras de validação exige incrementar `prompt_version` / `contract_version` antes de novas coletas.
- O dataset coletado sob uma `prompt_version` não deve ser misturado sem rastreabilidade com execuções de outra versão.