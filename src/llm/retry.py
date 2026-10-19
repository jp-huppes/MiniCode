import time
import random
import logging
from openai import APIError, APITimeoutError, RateLimitError
from src.llm.errors import LLMError,LLMTimeoutError,LLMRateLimitError,LLMServerError,LLMBadRequestError


logger = logging.getLogger(__name__)

def calcular_backoff(tentativa: int, t_base: float = 1.0, t_max: float = 30.0, jitter: float = 1.0) -> float:
    #calcula o tempo de espera com backoff exponencial e ruido (jitter).
    expo = min(t_max, t_base * (2 ** tentativa))
    ruido = random.uniform(0, jitter)
    return expo + ruido

def executar_com_retry(func, max_tentativas: int = 3, t_base: float = 1.0, t_max: float = 30.0):
    #executa uma função de chamada HTTP com política de retry baseada no tipo de erro.
    for k in range(max_tentativas):
        try:
            return func()
        except APITimeoutError as e:
            if k == max_tentativas - 1:
                raise LLMTimeoutError(f"Timeout esgotado após {k+1} tentativas: {e}", "timeout", k+1)
            espera = calcular_backoff(k, t_base, t_max)
            logger.warning(f"[Retry {k+1}/{max_tentativas}] Timeout detectado. Aguardando {espera:.2f}s...")
            time.sleep(espera)
            
        except RateLimitError as e:
            if k == max_tentativas - 1:
                raise LLMRateLimitError(f"Rate limit excedido após {k+1} tentativas: {e}", "rate_limit", k+1)
            #aguardar um pouco mais em rate limit
            espera = calcular_backoff(k, t_base, t_max) * 2
            logger.warning(f"[Retry {k+1}/{max_tentativas}] Rate limit (429). Aguardando {espera:.2f}s...")
            time.sleep(espera)

        except APIError as e:
            status_code = getattr(e, "status_code", 500)
            if status_code >= 500: # erros de Servidor
                if k == max_tentativas - 1:
                    raise LLMServerError(f"Erro do servidor {status_code} após {k+1} tentativas: {e}", "server_error", k+1)
                espera = calcular_backoff(k, t_base, t_max)
                logger.warning(f"[Retry {k+1}/{max_tentativas}] Erro 5xx ({status_code}). Aguardando {espera:.2f}s...")
                time.sleep(espera)
            else: # erros de cliente (nao deve ter retry)
                raise LLMBadRequestError(f"Erro de requisição inválida ({status_code}): {e}", "bad_request", k+1)
        except Exception as e:
            raise LLMError(f"Erro inesperado na chamada LLM: {e}", "unknown", k+1)
            
    raise LLMError("Número máximo de tentativas excedido.", "max_retries", max_tentativas)