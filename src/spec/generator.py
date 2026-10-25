from typing import Tuple
from src.llm.client import LLMClient
from src.spec.models import Spec, SpecInvalidaError, PlanInvalidoError
from src.spec.prompts import montar_prompt_spec, montar_prompt_plano
from src.spec.validators import validar_spec, extrair_spec_estruturada, validar_plano

MAX_SPEC_ATTEMPTS = 3
MAX_PLAN_ATTEMPTS = 3

def gerar_spec_com_retry(tarefa: str, contexto: str, llm_client: LLMClient, max_attempts: int = MAX_SPEC_ATTEMPTS) -> Tuple[Spec, str]:
    """
    Gera a especificacao (SPEC) com loop de retry limitado e feedback direcionado em caso de falha.
        Retorna a tupla (Spec estruturada, conteudo markdown cru).
    """
    mensagens = montar_prompt_spec(tarefa, contexto)
    tentativa = 0; ultimo_resultado = None

    while tentativa < max_attempts:
        texto_resposta = llm_client.chat(mensagens)
        resultado = validar_spec(texto_resposta)
        
        if resultado.ok:
            spec_obj = extrair_spec_estruturada(texto_resposta)
            return spec_obj, texto_resposta

        ultimo_resultado = resultado
        erros_formatados = "\n".join(f"- {e}" for e in resultado.errors)
        
        # adiciona historico de chat e feedback especfico para correcao na proxima tentativa
        mensagens.append({"role": "assistant", "content": texto_resposta})
        mensagens.append({
            "role": "user", 
            "content": f"A SPEC gerada falhou na validação programática com os seguintes erros:\n{erros_formatados}\n\nCorrija estritamente o documento mantendo o formato exigido."
        })
        tentativa += 1

    erros_finais = ultimo_resultado.errors if ultimo_resultado else ["Tentativas esgotadas sem validação."]
    raise SpecInvalidaError(erros_finais)

def gerar_plano_com_retry(spec: Spec, contexto: str, llm_client: LLMClient, max_attempts: int = MAX_PLAN_ATTEMPTS) -> str:
    """
    Gera o Plano de execucao (PLAN) com loop de retry limitado e validacao de cobertura (AC-n).
        Retorna o conteudo markdown cru do plano aprovado.
    """
    mensagens = montar_prompt_plano(spec.raw_content, contexto)
    tentativa = 0; ultimo_resultado = None

    while tentativa < max_attempts:
        texto_resposta = llm_client.chat(mensagens); resultado = validar_plano(texto_resposta, spec)
        
        if resultado.ok: return texto_resposta

        ultimo_resultado = resultado
        erros_formatados = "\n".join(f"- {e}" for e in resultado.errors)
        
        mensagens.append({"role": "assistant", "content": texto_resposta})
        mensagens.append({
            "role": "user",
            "content": f"O PLAN gerado falhou na validação programática com os seguintes erros:\n{erros_formatados}\n\nCorrija o plano garantindo a numeração correta e a cobertura absoluta de todos os critérios de aceitação (AC-n)."
        })
        tentativa += 1

    erros_finais = ultimo_resultado.errors if ultimo_resultado else ["Tentativas esgotadas sem validação de plano."]
    raise PlanInvalidoError(erros_finais)