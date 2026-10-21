import time
import random
import logging
from openai import APIError, APITimeoutError, RateLimitError
from src.llm.errors import LLMError, LLMTimeoutError, LLMRateLimitError, LLMServerError, LLMBadRequestError, LLMRetryError
from typing import Optional

logger = logging.getLogger(__name__)

def calcular_backoff(tentativa: int, t_base: float = 1.0, t_max: float = 30.0, jitter: float = 1.0) -> float:
    # calcula o tempo de espera com backoff exponencial e ruido (jitter).
    expo = min(t_max, t_base * (2 ** tentativa))
    ruido = random.uniform(0, jitter)
    return expo + ruido

def executar_com_retry(func, max_retries: int = 3, base_delay: float = 1.0, t_max: float = 30.0, max_tentativas: Optional[int] = None, t_base: Optional[float] = None):
    # executa uma função de chamada HTTP com política de retry baseada no tipo de erro.
    if max_tentativas is not None: max_retries = max_tentativas
    if t_base is not None: base_delay = t_base

    for k in range(max_retries):
        try: return func()
        except APITimeoutError as e:
            if k == max_retries - 1:
                raise LLMTimeoutError(f"Timeout esgotado após {k+1} tentativas: {e}", "timeout", k+1)
            espera = calcular_backoff(k, base_delay, t_max)
            logger.warning(f"[Retry {k+1}/{max_retries}] Timeout detectado. Aguardando {espera:.2f}s...")
            time.sleep(espera)
        except RateLimitError as e:
            if k == max_retries - 1:
                raise LLMRateLimitError(f"Rate limit excedido após {k+1} tentativas: {e}", "rate_limit", k+1)
            espera = calcular_backoff(k, base_delay, t_max) * 2
            logger.warning(f"[Retry {k+1}/{max_retries}] Rate limit (429). Aguardando {espera:.2f}s...")
            time.sleep(espera)
        except APIError as e:
            status_code = getattr(e, "status_code", 500)
            if status_code >= 500: # erros de servidor
                if k == max_retries - 1:
                    raise LLMServerError(f"Erro do servidor {status_code} após {k+1} tentativas: {e}", "server_error", k+1)
                espera = calcular_backoff(k, base_delay, t_max)
                logger.warning(f"[Retry {k+1}/{max_retries}] Erro 5xx ({status_code}). Aguardando {espera:.2f}s...")
                time.sleep(espera)
            else: # erros de cliente (nao deve ter retry)
                raise LLMBadRequestError(f"Erro de requisição inválida ({status_code}): {e}", "bad_request", k+1)
        except Exception as e:
            if isinstance(e, LLMError):
                raise e
            if k == max_retries - 1:
                raise LLMRetryError(f"Erro inesperado na chamada LLM após {k+1} tentativas: {e}")
            espera = calcular_backoff(k, base_delay, t_max)
            logger.warning(f"[Retry {k+1}/{max_retries}] Erro inesperado: {e}. Aguardando {espera:.2f}s...")
            time.sleep(espera)

    raise LLMRetryError(f"Número máximo de tentativas ({max_retries}) excedido.")