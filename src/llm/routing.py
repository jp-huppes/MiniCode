"""Estrutura de roteamento por papel/etapa e flags de ablação para o LLMClient."""

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

ROLE_TOOL_SELECTION = "tool_selection"
ROLE_SPEC = "spec"
ROLE_PLAN = "plan"
ROLE_REPAIR = "repair"

DEFAULT_TEACHER_MODEL = "Qwen/Qwen2.5-14B-Instruct-AWQ"
DEFAULT_TOOL_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


@dataclass
class AgentRoutingConfig:
    """Configuração de modelos por função e flags para ablação."""

    teacher_model: str = DEFAULT_TEACHER_MODEL
    tool_model: Optional[str] = None

    def __post_init__(self) -> None:
        if self.tool_model is None:
            self.tool_model = self.teacher_model or DEFAULT_TOOL_MODEL
    spec_model: Optional[str] = None
    plan_model: Optional[str] = None
    repair_model: Optional[str] = None

    no_spec: bool = False
    no_plan: bool = False
    no_gates: bool = False
    no_repair: bool = False
    constrained_mode: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

    role_chat_handlers: Dict[str, Callable[..., Any]] = field(default_factory=dict)

    def resolve_model_for_role(self, role: Optional[str], fallback_model: str) -> str:
        normalized = (role or ROLE_TOOL_SELECTION).strip().lower()
        if normalized == ROLE_TOOL_SELECTION:
            return self.tool_model or fallback_model
        if normalized == ROLE_SPEC:
            return self.spec_model or self.teacher_model or fallback_model
        if normalized == ROLE_PLAN:
            return self.plan_model or self.teacher_model or fallback_model
        if normalized == ROLE_REPAIR:
            return self.repair_model or self.teacher_model or fallback_model
        return fallback_model


def _call_original_chat(
    original_chat: Callable[..., Any],
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    **kwargs: Any,
) -> Any:
    call_kwargs = dict(kwargs)
    try:
        sig = inspect.signature(original_chat)
        accepts_var_kw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        if tools is not None and ("tools" in sig.parameters or accepts_var_kw):
            call_kwargs["tools"] = tools
    except (ValueError, TypeError):
        if tools is not None:
            call_kwargs["tools"] = tools

    try:
        return original_chat(messages, **call_kwargs)
    except TypeError:
        for key in ("tools", "max_tokens", "response_schema", "temperature"):
            if key in call_kwargs:
                trimmed = dict(call_kwargs)
                trimmed.pop(key, None)
                try:
                    return original_chat(messages, **trimmed)
                except TypeError:
                    call_kwargs = trimmed
        return original_chat(messages)


def attach_routing_to_client(client: Any, routing_config: Optional[AgentRoutingConfig] = None) -> Any:
    """Anexa roteamento por papel e flags de ablação a uma instância de LLMClient."""
    cfg = routing_config or AgentRoutingConfig(
        teacher_model=getattr(client, "model_name", DEFAULT_TEACHER_MODEL)
    )
    client.routing_config = cfg

    if not hasattr(client, "_original_chat"):
        client._original_chat = client.chat

    def routed_chat(
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        role: Optional[str] = None,
        model_override: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        active_role = (role or ROLE_TOOL_SELECTION).strip().lower()

        if cfg.temperature is not None and "temperature" not in kwargs:
            kwargs["temperature"] = cfg.temperature
        if cfg.max_tokens is not None and "max_tokens" not in kwargs:
            kwargs["max_tokens"] = cfg.max_tokens

        handler = cfg.role_chat_handlers.get(active_role)
        if handler is not None:
            target_model = model_override or cfg.resolve_model_for_role(
                active_role, getattr(client, "model_name", DEFAULT_TEACHER_MODEL)
            )
            return handler(messages=messages, tools=tools, model=target_model, role=active_role, **kwargs)

        target_model = model_override or cfg.resolve_model_for_role(
            active_role, getattr(client, "model_name", DEFAULT_TEACHER_MODEL)
        )
        previous_model = getattr(client, "model_name", None)
        try:
            if target_model and hasattr(client, "model_name"):
                client.model_name = target_model
            return _call_original_chat(client._original_chat, messages, tools=tools, **kwargs)
        finally:
            if previous_model is not None and hasattr(client, "model_name"):
                client.model_name = previous_model

    client.chat = routed_chat
    return client