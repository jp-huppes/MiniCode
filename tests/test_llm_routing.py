"""Testes unitários para roteamento por papel e flags no LLMClient."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.llm.routing import (
    ROLE_PLAN,
    ROLE_REPAIR,
    ROLE_SPEC,
    ROLE_TOOL_SELECTION,
    AgentRoutingConfig,
    attach_routing_to_client,
)


@dataclass
class DummyResponse:
    content: Optional[str]
    model_used: str = ""


class DummyClient:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-14B-Instruct-AWQ") -> None:
        self.model_name = model_name
        self.calls: List[Dict[str, Any]] = []

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> DummyResponse:
        self.calls.append(
            {
                "model": self.model_name,
                "messages": messages,
                "tools": tools,
                "kwargs": kwargs,
            }
        )
        return DummyResponse(content="ok", model_used=self.model_name)


def test_role_routing_tool_selection_to_1_5b_and_others_to_14b() -> None:
    client = DummyClient(model_name="Qwen/Qwen2.5-14B-Instruct-AWQ")
    cfg = AgentRoutingConfig(
        teacher_model="Qwen/Qwen2.5-14B-Instruct-AWQ",
        tool_model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        temperature=0.0,
        max_tokens=512,
    )
    attach_routing_to_client(client, cfg)

    msgs = [{"role": "user", "content": "escolha a proxima acao"}]
    client.chat(msgs, role=ROLE_TOOL_SELECTION)
    client.chat(msgs, role=ROLE_SPEC)
    client.chat(msgs, role=ROLE_PLAN)
    client.chat(msgs, role=ROLE_REPAIR)

    assert client.calls[0]["model"] == "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    assert client.calls[1]["model"] == "Qwen/Qwen2.5-14B-Instruct-AWQ"
    assert client.calls[2]["model"] == "Qwen/Qwen2.5-14B-Instruct-AWQ"
    assert client.calls[3]["model"] == "Qwen/Qwen2.5-14B-Instruct-AWQ"
    assert client.calls[0]["kwargs"]["temperature"] == 0.0
    assert client.calls[0]["kwargs"]["max_tokens"] == 512
    # Restaura model_name padrão após a chamada roteada
    assert client.model_name == "Qwen/Qwen2.5-14B-Instruct-AWQ"


def test_role_chat_handler_override() -> None:
    client = DummyClient(model_name="Qwen/Qwen2.5-14B-Instruct-AWQ")
    seen: List[str] = []

    def local_15b_handler(messages: List[Dict[str, Any]], tools: Any = None, model: str = "", **kwargs: Any) -> DummyResponse:
        seen.append(f"{model}:{messages[-1]['content']}")
        return DummyResponse(content='{"tool": "list_files", "args": {}}', model_used=model)

    cfg = AgentRoutingConfig(
        tool_model="Qwen/Qwen2.5-Coder-1.5B-Instruct-4bit",
        role_chat_handlers={ROLE_TOOL_SELECTION: local_15b_handler},
    )
    attach_routing_to_client(client, cfg)

    resp = client.chat([{"role": "user", "content": "listar arquivos"}], role=ROLE_TOOL_SELECTION)
    assert resp.model_used == "Qwen/Qwen2.5-Coder-1.5B-Instruct-4bit"
    assert seen == ["Qwen/Qwen2.5-Coder-1.5B-Instruct-4bit:listar arquivos"]
    assert len(client.calls) == 0
