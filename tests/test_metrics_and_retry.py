import pytest
from src.llm.metrics import LLMMetricsManager
from src.llm.errors import LLMRetryError
from src.llm.retry import executar_com_retry

def test_metrics_manager_aggregation():
    # Verifica se o LLMMetricsManager acumula tokens, latencia e contadores corretamente
    manager = LLMMetricsManager()
    manager.registrar(prompt_tokens=100, completion_tokens=50, latency_seconds=1.2, retries=0, finish_reason="stop", success=True)
    manager.registrar(prompt_tokens=200, completion_tokens=80, latency_seconds=2.0, retries=1, finish_reason="stop", success=True)
    manager.registrar(prompt_tokens=0, completion_tokens=0, latency_seconds=0.0, retries=0, finish_reason="stop", success=False, motivo="reformulacao")

    summary = manager.summary()
    assert summary["total_calls"] == 3
    assert summary["total_prompt_tokens"] == 300
    assert summary["total_completion_tokens"] == 130
    assert summary["success_count"] == 2
    assert summary["failure_count"] == 1

def test_retry_success_after_transient_failure():
    # Valida se o mecanismo de retry recupera de falhas temporárias antes de esgotar o limite
    tentativas = 0
    def operacao_instavel():
        nonlocal tentativas
        tentativas += 1
        if tentativas < 3: raise ConnectionError("Falha de rede temporária")
        return "sucesso_obtido"

    resultado = executar_com_retry(operacao_instavel, max_retries=3, base_delay=0.01)
    assert resultado == "sucesso_obtido"
    assert tentativas == 3

def test_retry_exhaustion_raises_llm_retry_error():
    # Valida que o esgotamento das tentativas eleva LLMRetryError
    def operacao_sempre_falha():
        raise TimeoutError("Timeout persistente no servidor")

    with pytest.raises(LLMRetryError): executar_com_retry(operacao_sempre_falha, max_retries=2, base_delay=0.01)