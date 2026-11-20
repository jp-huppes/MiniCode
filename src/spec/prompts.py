from typing import Dict, Iterable, List, Optional

SPEC_SYSTEM_PROMPT = """Você é um Engenheiro de Requisitos Sênior especializado em Spec-Driven Development (Desenvolvimento Orientado por Especificação).
Sua missão é analisar a tarefa solicitada e o contexto técnico do projeto para criar um documento de especificação rigoroso, testável e sem ambiguidades.

REGRAS CRÍTICAS:
1. NÃO escreva código de implementação. Apenas especifique o que deve ser feito.
2. A saída OBRIGATÓRIA deve ser estritamente em Markdown com exatamente estas quatro seções de cabeçalho:
  - ## Objective
  - ## Parameters
  - ## Acceptance Criteria
  - ## Tests
3. Cada Critério de Aceitação deve ser binário (passa/falha), sem termos vagos ("rápido", "limpo"), e possuir um ID estável (AC-1, AC-2...).
4. Na seção ## Tests, associe cada AC-n a um teste específico no formato exato `test_ac_<n>_<descricao>`.

EXEMPLO DE FORMATO VÁLIDO:
## Objective
Implementar a função de soma para retornar a adição correta de dois números inteiros.

## Parameters
- Arquivo alvo: src/math_utils.py
- Linguagem: Python 3.11

## Acceptance Criteria
- **AC-1**: A função soma(a, b) deve retornar a soma matemática correta para números positivos.
- **AC-2**: A função soma(a, b) deve levantar TypeError caso receba argumentos não numéricos.

## Tests
- **AC-1**: test_ac_1_soma_positivos
- **AC-2**: test_ac_2_soma_tipo_invalido
"""

PLAN_SYSTEM_PROMPT = """Você é um Arquiteto de Software Sênior especializado em planejamento de implementação modular.
Dada a Especificação (SPEC) e a visão do projeto, você deve gerar um plano de execução estruturado em passos numerados e ordenados.

REGRAS CRÍTICAS DE FORMATO:
1. RETORNE APENAS OS PASSOS NUMERADOS. NÃO inclua nenhum texto conversacional, saudação, introdução ou explicação antes ou depois do plano. Comece imediatamente no passo 1.
2. Dependências devem vir antes dos dependentes (ex: criar funções auxiliares/módulos antes de usá-los).
3. Testes devem ser escritos antes ou junto com a implementação correspondente.
4. NÃO inclua passo de execução de testes. A verificação é feita automaticamente pelo harness ao final da implementação. Os passos devem ser apenas de implementação e de criação de testes.
5. Cada passo DEVE seguir rigorosamente este formato de linha:
   N. [Verbo no imperativo] <ação descritiva> — arquivo(s): <caminho> — cobre: AC-x
   (onde N é o número sequencial contínuo sem buracos, e AC-x é o ID correspondente da SPEC).
"""

SPEC_JSON_SYSTEM_PROMPT = """Você é um Engenheiro de Requisitos Sênior especializado em Spec-Driven Development.
Analise a tarefa e o contexto técnico do projeto e retorne ESTRITAMENTE um objeto JSON válido (sem texto fora do JSON) no formato:
{
  "objective": "objetivo claro, binário e sem ambiguidades",
  "parameters": ["parâmetro 1", "parâmetro 2"],
  "acceptance_criteria": [
    {"id": "AC-1", "text": "critério binário verificável (passa/falha)"}
  ],
  "tests": [
    {"ac": "AC-1", "name": "test_ac_1_descricao"}
  ]
}

REGRAS CRÍTICAS:
- NÃO escreva código de implementação.
- NÃO use placeholders como TBD ou N/A.
- Cada ID em `acceptance_criteria` deve seguir `AC-n` (ex.: AC-1, AC-2) sem duplicatas.
- Todo critério `AC-n` deve ter ao menos um teste correspondente em `tests` com nome no formato `test_ac_<n>_<descricao>`.
"""

PLAN_JSON_SYSTEM_PROMPT = """Você é um Arquiteto de Software Sênior especializado em planejamento de implementação modular.
Dada a Especificação (SPEC) e o contexto do projeto, retorne ESTRITAMENTE um objeto JSON válido (sem texto fora do JSON) no formato:
{
  "steps": [
    {
      "n": 1,
      "action": "Verbo no imperativo + ação descritiva",
      "files": ["src/exemplo.py"],
      "covers": ["AC-1"]
    }
  ]
}

REGRAS CRÍTICAS:
- Numere `n` de forma contínua a partir de 1 (1, 2, 3...) sem buracos.
- Dependências devem vir antes dos dependentes; testes antes ou junto da implementação.
- NÃO inclua passo de execução de testes (`run_tests` é executado automaticamente pelo harness).
- Cubra 100% dos critérios `AC-n` da SPEC e cite apenas IDs de critérios existentes na SPEC.
"""


def montar_prompt_spec(tarefa: str, contexto: str) -> List[Dict[str, str]]:
    # Monta as mensagens (system + user) para a geracao da SPEC em Markdown.
    user_content = f"""Tarefa solicitada pelo usuário:
    {tarefa}
    Contexto técnico do projeto (Visão geral):
    {contexto}
    Gere a especificação técnica formal seguindo rigorosamente o formato exigido.
    """

    return [{"role": "system", "content": SPEC_SYSTEM_PROMPT}, {"role": "user", "content": user_content}]


def montar_prompt_plano(spec_content: str, contexto: str) -> List[Dict[str, str]]:
    # Monta as mensagens para a geração do Plano em Markdown.
    user_content = f"""Especificação aprovada (SPEC):
    {spec_content}
    Contexto técnico do projeto:
    {contexto}
    Gere o plano de execução detalhado em passos numerados cobrindo todos os critérios de aceitação (AC-n).
    """

    return [{"role": "system", "content": PLAN_SYSTEM_PROMPT}, {"role": "user", "content": user_content}]


def montar_prompt_spec_json(tarefa: str, contexto: str) -> List[Dict[str, str]]:
    # Monta as mensagens (system + user) para geração da SPEC em JSON estruturado.
    user_content = (
        f"Tarefa solicitada pelo usuário:\n{tarefa}\n\n"
        f"Contexto técnico do projeto:\n{contexto}\n\n"
        "Retorne apenas o objeto JSON da SPEC conforme as regras."
    )
    return [{"role": "system", "content": SPEC_JSON_SYSTEM_PROMPT}, {"role": "user", "content": user_content}]


def montar_prompt_plano_json(
    spec_content: str,
    contexto: str,
    allowed_ac_ids: Optional[Iterable[str]] = None,
) -> List[Dict[str, str]]:
    # Monta as mensagens para geração do PLAN em JSON estruturado.
    ids_lista = sorted({str(ac).strip().upper() for ac in (allowed_ac_ids or []) if str(ac).strip()})
    bloco_ids = f"\nIDs de critérios permitidos em `covers`: {', '.join(ids_lista)}." if ids_lista else ""

    user_content = (
        f"Especificação aprovada (SPEC):\n{spec_content}\n\n"
        f"Contexto técnico do projeto:\n{contexto}{bloco_ids}\n\n"
        "Retorne apenas o objeto JSON do PLAN conforme as regras."
    )
    return [{"role": "system", "content": PLAN_JSON_SYSTEM_PROMPT}, {"role": "user", "content": user_content}]