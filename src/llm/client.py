import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from openai import OpenAI

from src.llm.routing import AgentRoutingConfig, attach_routing_to_client
from src.llm.action_contract import interpretar_acao
from src.llm.errors import LLMFormatError
from src.llm.metrics import LLMMetricsManager
from src.llm.retry import executar_com_retry

logger = logging.getLogger(__name__)

VALID_CONSTRAINED_MODES = {"none", "v-a", "v-b", "v-c", "v-d"}


def resolve_constrained_mode(explicit_mode: Optional[str] = None) -> str:
    raw = (explicit_mode or os.getenv("MINICODE_CONSTRAINED_MODE") or "none").strip().lower()
    return raw if raw in VALID_CONSTRAINED_MODES else "none"


@dataclass
class LLMResponse:
    content: Optional[str]
    tool_calls: Optional[List[Dict[str, Any]]] = None
    finish_reason: Optional[str] = None
    latency_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLMClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        tool_mode: str = "json",
        constrained_mode: Optional[str] = None,
        default_temperature: float = 0.1,
        routing_config: Optional[AgentRoutingConfig] = None,
    ):
        self.api_key = api_key or os.getenv("MINICODE_API_KEY")
        self.base_url = base_url or os.getenv("MINICODE_API_BASE")
        self.model_name = model_name or os.getenv("MINICODE_MODEL_NAME", "Qwen/Qwen2.5-14B-Instruct-AWQ")
        self.timeout = timeout
        self.max_retries = max_retries
        self.tool_mode = tool_mode
        self.constrained_mode = resolve_constrained_mode(constrained_mode)
        self.default_temperature = default_temperature

        if not self.api_key or not self.base_url:
            raise ValueError("As credenciais MINICODE_API_KEY e MINICODE_API_BASE sao obrigatorias.")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        self.metrics = LLMMetricsManager()

        cfg = routing_config or AgentRoutingConfig(
            teacher_model=self.model_name,
            tool_model=self.model_name,
            temperature=0.0,
            max_tokens=512,
        )
        attach_routing_to_client(self, cfg)

    def _constricao_ativa(self, response_schema: Optional[Dict[str, Any]]) -> bool:
        if self.constrained_mode == "none":
            return False
        if self.constrained_mode == "v-d":
            return True
        return response_schema is not None

    def _aplicar_constricao(
        self,
        payload: Dict[str, Any],
        response_schema: Optional[Dict[str, Any]],
    ) -> None:
        if not self._constricao_ativa(response_schema):
            return

        if self.constrained_mode == "v-d":
            payload["response_format"] = {"type": "json_object"}
            return

        if response_schema is None:
            return

        if self.constrained_mode == "v-a":
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_response",
                    "schema": response_schema,
                    "strict": True,
                },
            }
        elif self.constrained_mode == "v-b":
            extra = dict(payload.get("extra_body") or {})
            extra["guided_json"] = response_schema
            payload["extra_body"] = extra
        elif self.constrained_mode == "v-c":
            extra = dict(payload.get("extra_body") or {})
            extra["structured_outputs"] = {"json": response_schema}
            payload["extra_body"] = extra

    def _traduzir_e_injetar_catalogo(
        self,
        messages: List[Dict[str, Any]],
        tools_schema: List[Dict[str, Any]],
        catalogo_enxuto: bool = False,
    ) -> List[Dict[str, Any]]:
        ferramentas_info = []
        for t in tools_schema:
            func = t["function"]
            nome = func["name"]
            desc = func.get("description", "")
            if catalogo_enxuto:
                ferramentas_info.append(f"- {nome}: {desc}")
            else:
                ferramentas_info.append(
                    f"- Nome: {nome}\n"
                    f"  Descricao: {desc}\n"
                    f"  Parâmetros: {func.get('parameters', {})}"
                )

        if catalogo_enxuto:
            catalogo_str = (
                "\n\n[FERRAMENTAS DISPONÍVEIS]\n"
                + "\n".join(ferramentas_info)
                + "\n\nRetorne estritamente um JSON conforme o schema solicitado."
            )
        else:
            catalogo_str = (
                "\n\n[CATALOGO DE FERRAMENTAS DISPONÍVEIS]\n"
                + "\n".join(ferramentas_info)
                + "\n\nINSTRUCAO DE FORMATO:\n"
                "Responda estritamente com um objeto JSON contendo:\n"
                "{\n"
                '  "thought": "seu raciocínio breve",\n'
                '  "tool": "nome_da_ferramenta",\n'
                '  "args": { ... }\n'
                "}"
            )

        novas_mensagens: List[Dict[str, Any]] = []
        system_injetado = False

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")

            if role == "system":
                novas_mensagens.append({"role": "system", "content": content + catalogo_str})
                system_injetado = True
            elif role == "tool":
                tool_name = msg.get("name", "ferramenta")
                novas_mensagens.append({
                    "role": "user",
                    "content": f"[Resultado da ferramenta '{tool_name}']\n{content}",
                })
            else:
                novas_mensagens.append(dict(msg))

        if not system_injetado:
            novas_mensagens.insert(
                0,
                {"role": "system", "content": "Voce é um agente de programacao autonomo." + catalogo_str},
            )

        return novas_mensagens

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools_schema: Optional[List[Dict[str, Any]]] = None,
        response_schema: Optional[Dict[str, Any]] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        max_reformulacoes: int = 2,
    ) -> LLMResponse:
        mensagens_para_envio = [dict(m) for m in messages]
        constricao_ativa = self._constricao_ativa(response_schema)

        if tools_schema and self.tool_mode == "json":
            mensagens_para_envio = self._traduzir_e_injetar_catalogo(
                mensagens_para_envio,
                tools_schema,
                catalogo_enxuto=constricao_ativa,
            )

        temp_efetiva = temperature if temperature is not None else self.default_temperature
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": mensagens_para_envio,
            "temperature": temp_efetiva,
            "max_tokens": max_tokens,
        }

        if tools_schema and self.tool_mode == "native":
            payload["tools"] = tools_schema
            payload["tool_choice"] = "auto"

        self._aplicar_constricao(payload, response_schema)

        tentativa_reformulacao = 0
        while tentativa_reformulacao <= max_reformulacoes:
            def _requisicao():
                return self.client.chat.completions.create(**payload)

            start_time = time.time()
            response = executar_com_retry(_requisicao, max_retries=self.max_retries)
            latency = time.time() - start_time

            choice = response.choices[0]
            msg = choice.message
            content = msg.content or ""
            finish_reason = choice.finish_reason

            usage = getattr(response, "usage", None)
            prompt_tok = usage.prompt_tokens if usage else 0
            completion_tok = usage.completion_tokens if usage else 0

            tool_calls_parsed = None

            if finish_reason == "tool_calls" and getattr(msg, "tool_calls", None):
                tool_calls_parsed = [
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                    for tc in msg.tool_calls
                ]
                self.metrics.registrar(
                    prompt_tok, completion_tok, latency, retries=0, finish_reason=finish_reason, success=True
                )
                return LLMResponse(
                    content=content,
                    tool_calls=tool_calls_parsed,
                    finish_reason=finish_reason,
                    latency_seconds=latency,
                    prompt_tokens=prompt_tok,
                    completion_tokens=completion_tok,
                )

            # Camada 2 roda sempre para validar o envelope mesmo quando a camada 1 está ativa
            if tools_schema and self.tool_mode == "json":
                avaliacao = interpretar_acao(content, finish_reason=finish_reason)
                if avaliacao.ok and avaliacao.action is not None:
                    self.metrics.registrar(
                        prompt_tok, completion_tok, latency, retries=0, finish_reason=finish_reason, success=True
                    )
                    tool_calls_parsed = [
                        {
                            "id": f"call_{uuid.uuid4().hex[:8]}",
                            "name": avaliacao.action.tool,
                            "arguments": avaliacao.action.args,
                        }
                    ]
                    return LLMResponse(
                        content=content,
                        tool_calls=tool_calls_parsed,
                        finish_reason=finish_reason,
                        latency_seconds=latency,
                        prompt_tokens=prompt_tok,
                        completion_tokens=completion_tok,
                    )

                categoria = avaliacao.diagnostic.category.value if avaliacao.diagnostic else "nao_json"
                motivo_curto = (
                    avaliacao.diagnostic.format_retry_message()
                    if avaliacao.diagnostic
                    else "JSON inválido."
                )
                self.metrics.registrar(
                    prompt_tok,
                    completion_tok,
                    latency,
                    retries=0,
                    finish_reason=finish_reason,
                    success=False,
                    motivo=categoria,
                )

                if tentativa_reformulacao < max_reformulacoes:
                    tentativa_reformulacao += 1
                    logger.warning(
                        f"Contrato de ação inválido [{categoria}] ({motivo_curto}). "
                        f"Reformulação ({tentativa_reformulacao}/{max_reformulacoes})..."
                    )
                    mensagens_para_envio.append({"role": "assistant", "content": content})
                    mensagens_para_envio.append({
                        "role": "user",
                        "content": f"Erro de formato: {motivo_curto} Responda apenas com JSON válido.",
                    })
                    payload["messages"] = mensagens_para_envio
                    continue

                raise LLMFormatError(f"Número máximo de reformulações atingido [{categoria}]: {motivo_curto}")

            # Chamada sem tools_schema (ex.: geração de SPEC/PLAN)
            self.metrics.registrar(
                prompt_tok, completion_tok, latency, retries=0, finish_reason=finish_reason, success=True
            )
            return LLMResponse(
                content=content,
                tool_calls=None,
                finish_reason=finish_reason,
                latency_seconds=latency,
                prompt_tokens=prompt_tok,
                completion_tokens=completion_tok,
            )

        raise LLMFormatError("Falha crítica ao obter resposta formatada da LLM.")


def interpretar_com_retry(
    client: Any,
    messages: List[Dict[str, Any]],
    max_reformulacoes: int = 2,
    metrics: Any = None,
    **chat_kwargs: Any,
) -> Any:
    """
    Executa chamadas ao LLM e aplica retry dirigido até `max_reformulacoes`
    quando a resposta viola o contrato de ação, sem reenviar o schema inteiro.
    """
    from src.llm.action_contract import InterpretedAction, interpretar_acao

    conversa = [dict(m) for m in messages]
    ultimo_resultado: Optional[InterpretedAction] = None

    for tentativa in range(max_reformulacoes + 1):
        resposta = client.chat(conversa, **chat_kwargs)
        conteudo = resposta.content if hasattr(resposta, "content") else str(resposta)
        finish_reason = getattr(resposta, "finish_reason", None)

        avaliacao = interpretar_acao(conteudo or "", finish_reason=finish_reason)
        ultimo_resultado = avaliacao

        if avaliacao.ok:
            if metrics is not None and hasattr(metrics, "record_action_result"):
                metrics.record_action_result(success=True, category=None, attempt=tentativa)
            return avaliacao

        categoria = avaliacao.diagnostic.category.value if avaliacao.diagnostic else "nao_json"
        if metrics is not None and hasattr(metrics, "record_action_result"):
            metrics.record_action_result(success=False, category=categoria, attempt=tentativa)

        if tentativa < max_reformulacoes and avaliacao.diagnostic is not None:
            conversa.append({"role": "assistant", "content": conteudo or ""})
            conversa.append({
                "role": "user",
                "content": (
                    f"Formato inválido ({categoria}): {avaliacao.diagnostic.format_retry_message()} "
                    "Responda apenas com um objeto JSON válido contendo 'thought', 'tool' e 'args'."
                ),
            })

    assert ultimo_resultado is not None
    return ultimo_resultado