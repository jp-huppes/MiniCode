import os
from typing import Any, Dict, List, Optional, Tuple
from src.llm.routing import ROLE_REPAIR, ROLE_TOOL_SELECTION

from src.llm.client import LLMClient
from src.spec.models import Plan, PlanInvalidoError, Spec, SpecInvalidaError
from src.spec.prompts import (
    montar_prompt_plano,
    montar_prompt_plano_json,
    montar_prompt_spec,
    montar_prompt_spec_json,
)
from src.spec.validators import (
    extrair_plano_json,
    extrair_spec_estruturada,
    extrair_spec_json,
    plan_json_schema,
    remover_preambulo_plano,
    remover_preambulo_spec,
    spec_json_schema,
    validar_plano,
    validar_plano_json,
    validar_spec,
    validar_spec_json,
)

MAX_SPEC_ATTEMPTS = 3
MAX_PLAN_ATTEMPTS = 3


def resolver_modo_json_spec_plan(use_json: Optional[bool] = None) -> bool:
    """
    Resolve o sinalizador de geração JSON vs. Markdown para SPEC e PLAN.
    Permite desligar o JSON via parâmetro ou variável de ambiente (ex.: 'markdown' / '0' / 'false').
    """
    if use_json is not None:
        return bool(use_json)

    raw_fmt = (os.getenv("MINICODE_SPEC_PLAN_FORMAT") or "").strip().lower()
    if raw_fmt in {"md", "markdown", "legacy"}:
        return False
    if raw_fmt in {"json"}:
        return True

    raw_flag = (os.getenv("MINICODE_SPEC_JSON") or "1").strip().lower()
    return raw_flag not in {"0", "false", "no", "off"}


def _texto(resposta: Any) -> str:
    # aceita LLMResponse (produção) ou str (testes e clientes simples)
    if isinstance(resposta, str):
        return resposta
    return getattr(resposta, "content", None) or ""


def _chamar_llm(
    llm_client: LLMClient,
    mensagens: List[Dict[str, Any]],
    response_schema: Optional[Dict[str, Any]] = None,
    role: Optional[str] = None,
) -> str:
    """
    Caminho único de chamada ao LLMClient para SPEC e PLAN,
    aceitando `response_schema` opcional quando o modo estruturado estiver ativo.
    """
    try:
        if role is not None:
            resposta = llm_client.chat(mensagens, response_schema=response_schema, role=role)
        else:
            resposta = llm_client.chat(mensagens, response_schema=response_schema)
    except TypeError:
        try:
            resposta = llm_client.chat(mensagens, response_schema=response_schema)
        except TypeError:
            resposta = llm_client.chat(mensagens)
    return _texto(resposta)


def gerar_spec_com_retry(
    tarefa: str,
    contexto: str,
    llm_client: LLMClient,
    max_attempts: int = MAX_SPEC_ATTEMPTS,
    response_schema: Optional[Dict[str, Any]] = None,
    use_json: Optional[bool] = None,
) -> Tuple[Spec, str]:
    """
    Gera a SPEC com retry limitado e feedback direcionado em caso de falha.
    Em modo JSON (padrão), valida via JSON + regras V1-V7 e renderiza Markdown determinístico.
    Retorna a tupla (Spec estruturada, conteúdo markdown limpo).
    """
    modo_json = resolver_modo_json_spec_plan(use_json)
    schema_ativo = response_schema if response_schema is not None else (spec_json_schema() if modo_json else None)
    mensagens = montar_prompt_spec_json(tarefa, contexto) if modo_json else montar_prompt_spec(tarefa, contexto)

    tentativa = 0
    ultimo_resultado = None

    while tentativa < max_attempts:
        bruto = _chamar_llm(llm_client, mensagens, response_schema=schema_ativo)

        if modo_json:
            resultado = validar_spec_json(bruto)
            if resultado.ok:
                spec_obj = extrair_spec_json(bruto)
                return spec_obj, spec_obj.raw_content
            conteudo_historico = bruto
        else:
            texto_resposta = remover_preambulo_spec(bruto)
            resultado = validar_spec(texto_resposta)
            if resultado.ok:
                return extrair_spec_estruturada(texto_resposta), texto_resposta
            conteudo_historico = texto_resposta

        ultimo_resultado = resultado
        erros_formatados = "\n".join(f"- {e}" for e in resultado.errors)

        mensagens.append({"role": "assistant", "content": conteudo_historico})
        mensagens.append({
            "role": "user",
            "content": (
                f"A SPEC gerada falhou na validação programática com os seguintes erros:\n{erros_formatados}\n\n"
                "Corrija estritamente a saída mantendo o formato exigido."
            ),
        })
        tentativa += 1

    erros_finais = ultimo_resultado.errors if ultimo_resultado else ["Tentativas esgotadas sem validação."]
    raise SpecInvalidaError(erros_finais)


def gerar_plano_com_retry(
    spec: Spec,
    contexto: str,
    llm_client: LLMClient,
    max_attempts: int = MAX_PLAN_ATTEMPTS,
    response_schema: Optional[Dict[str, Any]] = None,
    use_json: Optional[bool] = None,
) -> str:
    """
    Gera o PLAN com retry limitado e validação de cobertura (AC-n).
    Em modo JSON (padrão), usa schema dinâmico com enum dos IDs da SPEC, valida P1-P5
    e renderiza o plano em Markdown determinístico.
    """
    modo_json = resolver_modo_json_spec_plan(use_json)
    ac_ids = sorted(spec.acceptance_criteria.keys())

    if modo_json:
        schema_ativo = response_schema if response_schema is not None else plan_json_schema(spec)
        mensagens = montar_prompt_plano_json(spec.raw_content, contexto, allowed_ac_ids=ac_ids)
    else:
        schema_ativo = response_schema
        mensagens = montar_prompt_plano(spec.raw_content, contexto)

    tentativa = 0
    ultimo_resultado = None

    while tentativa < max_attempts:
        bruto = _chamar_llm(llm_client, mensagens, response_schema=schema_ativo)

        if modo_json:
            resultado = validar_plano_json(bruto, spec)
            if resultado.ok:
                plano_obj: Plan = extrair_plano_json(bruto)
                return plano_obj.raw_content
            conteudo_historico = bruto
        else:
            texto_resposta = remover_preambulo_plano(bruto)
            resultado = validar_plano(texto_resposta, spec)
            if resultado.ok:
                return texto_resposta
            conteudo_historico = texto_resposta

        ultimo_resultado = resultado
        erros_formatados = "\n".join(f"- {e}" for e in resultado.errors)

        mensagens.append({"role": "assistant", "content": conteudo_historico})
        mensagens.append({
            "role": "user",
            "content": (
                f"O PLAN gerado falhou na validação programática com os seguintes erros:\n{erros_formatados}\n\n"
                "Corrija o plano garantindo a numeração correta e a cobertura absoluta de todos os critérios de aceitação (AC-n)."
            ),
        })
        tentativa += 1

    erros_finais = ultimo_resultado.errors if ultimo_resultado else ["Tentativas esgotadas sem validação de plano."]
    raise PlanInvalidoError(erros_finais)