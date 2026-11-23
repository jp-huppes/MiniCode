# Relatório de Uso de IA (AI_USAGE.md)

Este documento descreve de forma transparente o papel e a contribuição das ferramentas de IA utilizadas como apoio ao longo do desenvolvimento da Parte 1 do projeto **MiniCode** como permitido pelo professor.

## Divisão de Apoio dos Assistentes de IA

Durante o ciclo de desenvolvimento, duas IAs principais foram consultadas em diferentes frentes estratégicas e técnicas, ambas foram utilizadas em sua versão gratuita:

### 1. Claude
* **Planejamento Passo a Passo**: Auxílio na estruturação inicial das etapas e divisão modular do projeto em passo a passos para continualidade linear das implementações.
* **Pesquisa de Arquitetura e Opções**: "Discussão" sobre padrões de design mais "complexos" do projeto como alguns exemplos: a máquinas de estados determinísticas (FSM), a estratégias usada de hard gates e de isolamento de sandbox.
* **Resolução de Bugs Complexos**: Apoio na depuração de falhas de concorrência, exceções inesperadas em chamadas de ferramentas e desalinhamento de estados no loop de execução. Ou seja, tudo aquilo que o aluno ou a IA mais simples ficaram batendo cabeça para resolver.

### 2. Gemini
* **Implementação de Módulos Core**: Geração, refinamento e otimização de código para os subsistemas críticos.
* **Suíte de Testes & Cobertura**: Condução e ajuste iterativo de testes unitários com `pytest, como foi a minha primeira vez trabalhando com o pytest foi de grande ajuda.
* **Documentação Técnica**: Apoio na escrita e estruturação dos guias de arquitetura, README e relatórios de uso (principalmente no quesito de transformar o texto em algo mais fluido e entendivel ao leitor).

---

## Supervisão e Validação Humana

Embora os assistentes tenham atuado como copilotos (rip copilot) de engenharia para acelerar a escrita e o design, **nenhum código foi incorporado de forma cega**, a todo momento houve a presença do aluno corrigindo erros claros da maquina. 

Todas as implementações propostas pelo Claude e pelo Gemini passaram por rigorosa supervisão humana:
* Revisão linha por linha dos contratos de dados, tratamento de exceções e segurança do sandbox.
* Execução contínua de suítes de testes locais (`pytest`) para garantir a integridade dos hard gates.
* Validação manual de cenários de uso real na CLI e REPL do MiniCode.
* Tomada de decisão em possíveis caminhos a serem seguidos durante o projeto.


---

## Parte 2 ? Engenharia de Dados, QLoRA e Avalia??o Estat?stica

Durante a implementa??o da **Parte 2**, os assistentes de IA foram utilizados de forma iterativa e auditada nas seguintes frentes:

1. **Estabiliza??o da Coleta e Filtragem (`G0`?`G5`):**
   - Diagn?stico de gargalos de timeout e loops improdutivos em `scripts/collect_corpus.py`, adicionando suporte seguro a `--resume` e controle de repeti??o.
   - Garantia de integridade dos splits por `task_id` em `scripts/split_dataset.py` via carregamento obrigat?rio de `corpus_filtered/split_manifest.json` (SHA-256 congelado), impedindo vazamento entre treino, valida??o e teste.
2. **Alinhamento de Prompt e Pipeline SFT/QLoRA (`G3`, `G7`):**
   - Unifica??o de `minicode-prompt-v2` entre `src/dataset/build_sft_dataset.py`, `scripts/eval_baselines.py`, `notebooks/train.ipynb` e `src/agent/orchestrator.py`.
   - Depura??o da incompatibilidade de hardware `bfloat16` na GPU Tesla T4 do Google Colab (for?ando `float16` no `BitsAndBytesConfig` e precis?o nativa `float32` nos tensores trein?veis LoRA) e corre??o do diret?rio de trabalho (`cwd`) na execu??o automatizada do notebook.
3. **Avalia??o Offline, Bootstrap Pareado e An?lise de Erros (`G6`, `G8`):**
   - Execu??o dos baselines (`majority_class`, `zero_shot`, `few_shot`) e avalia??o comparativa local na RTX 3060 8 GB do `Qwen/Qwen2.5-Coder-1.5B-Instruct` base vs. fine-tuned LoRA, com *bootstrap* pareado por `run_id` (10.000 reamostragens) e taxonomia de erros (`eval_g8_report.json` e `error_analysis_g8.json`).
   - Todos os scripts, m?tricas e commits foram verificados localmente via `pytest` e revis?o manual antes da consolida??o no reposit?rio.
