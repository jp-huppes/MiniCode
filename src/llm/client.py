import os
import time
import uuid
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from openai import OpenAI

from src.llm.metrics import LLMMetricsManager
from src.llm.retry import executar_com_retry
from src.llm.action_parser import parse_acao_json
from src.llm.errors import LLMFormatError, ParseError

logger = logging.getLogger(__name__)

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
    ):
        self.api_key = api_key or os.getenv("MINICODE_API_KEY")
        self.base_url = base_url or os.getenv("MINICODE_API_BASE")
        self.model_name = model_name or os.getenv("MINICODE_MODEL_NAME", "Qwen/Qwen2.5-14B-Instruct-AWQ")
        self.timeout = timeout
        self.max_retries = max_retries
        self.tool_mode = tool_mode

        if not self.api_key or not self.base_url: raise ValueError("As credenciais MINICODE_API_KEY e MINICODE_API_BASE sao obrigatorias.")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        self.metrics = LLMMetricsManager()

    def _traduzir_e_injetar_catalogo(self, messages: List[Dict[str, Any]], tools_schema: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Traducao de mensagens para o modo JSON (fallback):
            injeta o catalogo de ferramentas no System Prompt.
            converte roles de 'tool' para 'user' com prefixo amigavel para o Qwen.
        """

        ferramentas_info = []
        for t in tools_schema:
            func = t["function"]
            ferramentas_info.append(f"- Nome: {func['name']}\n  Descricao: {func['description']}\n  Parâmetros: {func['parameters']}")

        catalogo_str = "\n\n[CATALOGO DE FERRAMENTAS DISPONÍVEIS]\n" + "\n".join(ferramentas_info) + "\n\n" "INSTRUCAO DE FORMATO:\n" "Se precisar usar uma ferramenta, responda estritamente com um objeto JSON contendo:\n" "{\n  \"thought\": \"seu raciocínio breve opcional\",\n  \"action\": \"nome_da_ferramenta\",\n  \"args\": { ... }\n}"

        novas_mensagens = []; system_injetado = False

        for msg in messages:
            role = msg.get("role"); content = msg.get("content", "")

            if role == "system":
                novas_mensagens.append({"role": "system", "content": content + catalogo_str})
                system_injetado = True
            elif role == "tool":
                # traduz resposta da ferramenta para formato de texto aceito pelo Qwen
                tool_name = msg.get("name", "ferramenta")
                conteudo_traduzido = f"[Resultado da ferramenta '{tool_name}']\n{content}"
                novas_mensagens.append({"role": "user", "content": conteudo_traduzido})
            else: novas_mensagens.append(msg)

        # caso nao havia system prompt, criamos um com o catálogo
        if not system_injetado: novas_mensagens.insert(0, {"role": "system", "content": "Voce é um agente de programacao autonomo." + catalogo_str})

        return novas_mensagens

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools_schema: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        max_reformulacoes: int = 2,
    ) -> LLMResponse:
        
        # prepara as mensagens conforme o modo de ferramenta selecionado
        mensagens_para_envio = messages.copy(); ferramentas_nomes = []

        if tools_schema:
            ferramentas_nomes = [t["function"]["name"] for t in tools_schema]
            if self.tool_mode == "json":
                mensagens_para_envio = self._traduzir_e_injetar_catalogo(mensagens_para_envio, tools_schema)

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": mensagens_para_envio,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # se for modo nativo e houver suporte, podemos passar tools nativas
        if tools_schema and self.tool_mode == "native":
            payload["tools"] = tools_schema
            payload["tool_choice"] = "auto"

        tentativa_reformulacao = 0
        while tentativa_reformulacao <= max_reformulacoes:
            def _requisicao():
                return self.client.chat.completions.create(**payload)

            try:
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

                # registra metrica da chamada bruta
                self.metrics.registrar(prompt_tok, completion_tok, latency, retries=0, finish_reason=finish_reason, success=True)

                tool_calls_parsed = None

                # verificacao de tool calls (nativo vs fallback)
                if finish_reason == "tool_calls" and getattr(msg, "tool_calls", None):
                    tool_calls_parsed = [{
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    } for tc in msg.tool_calls]
                elif tools_schema and self.tool_mode == "json" and content.strip():
                    try:
                        parsed_action = parse_acao_json(content, ferramentas_nomes)
                        synthetic_id = f"call_{uuid.uuid4().hex[:8]}"
                        tool_calls_parsed = [{
                            "id": synthetic_id,
                            "name": parsed_action["action"],
                            "arguments": parsed_action["args"]
                        }]
                    except ParseError as pe:
                        if tentativa_reformulacao < max_reformulacoes:
                            tentativa_reformulacao += 1
                            # registra reformulacao adicional nas metricas com motivo especificos
                            self.metrics.registrar(0, 0, 0.0, 0, finish_reason, success=False, motivo="reformulacao")
                            logger.warning(f"JSON inválido capturado ({pe}). Solicitando reformulacao ({tentativa_reformulacao}/{max_reformulacoes})...")
                            mensagens_para_envio.append({"role": "assistant", "content": content})
                            mensagens_para_envio.append({"role": "user", "content": f"Erro no formato JSON: {pe}. Por favor, corrija e responda estritamente no formato JSON exigido."})
                            payload["messages"] = mensagens_para_envio
                            continue
                        
                        else: raise LLMFormatError(f"Número máximo de reformulacoes de JSON atingido: {pe}")

                return LLMResponse(
                    content=content,
                    tool_calls=tool_calls_parsed,
                    finish_reason=finish_reason,
                    latency_seconds=latency,
                    prompt_tokens=prompt_tok,
                    completion_tokens=completion_tok,
                )

            except Exception as e: raise e

        raise LLMFormatError("Falha crítica ao obter resposta formatada da LLM.")