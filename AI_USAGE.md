# Relatório de Uso de IA (AI_USAGE.md)

Este documento descreve de forma transparente o papel e a contribuição das ferramentas de Inteligência Artificial utilizadas como apoio ao longo do desenvolvimento da Parte 1 do projeto **MiniCode**.

## 👥 Divisão de Apoio dos Assistentes de IA

Durante o ciclo de desenvolvimento, duas IAs principais foram consultadas em diferentes frentes estratégicas e técnicas:

### 1. Claude
* **Planejamento Passo a Passo**: Auxílio na estruturação inicial das etapas e divisão modular do projeto (do Passo 1 ao Passo 11).
* **Pesquisa de Arquitetura e Opções**: Discussão sobre padrões de design para máquinas de estados determinísticas (FSM), estratégias de hard gates e isolamento de sandbox.
* **Resolução de Bugs Complexos**: Apoio na depuração de falhas de concorrência, exceções inesperadas em chamadas de ferramentas e desalinhamento de estados no loop de execução.

### 2. Gemini
* **Implementação de Módulos Core**: Geração, refinamento e otimização de código para os subsistemas críticos (pipeline de *Spec-Driven Development* em `src/spec/`, validadores programáticos, prompts estruturados, orquestrador de execução e renderizadores da CLI).
* **Suíte de Testes & Cobertura**: Condução e ajuste iterativo de testes unitários com `pytest` (alcançando 100% de aprovação na suíte de testes).
* **Documentação Técnica**: Apoio na escrita e estruturação dos guias de arquitetura, README e relatórios de uso.

---

## 👨‍💻 Supervisão e Validação Humana

Embora os assistentes tenham atuado como copilotos de engenharia para acelerar a escrita e o design, **nenhum código foi incorporado de forma cega**. 

Todas as implementações propostas pelo Claude e pelo Gemini passaram por rigorosa supervisão humana:
* Revisão linha por linha dos contratos de dados, tratamento de exceções e segurança do sandbox.
* Execução contínua de suítes de testes locais (`pytest`) para garantir a integridade dos hard gates.
* Validação manual de cenários de uso real na CLI e REPL do MiniCode.