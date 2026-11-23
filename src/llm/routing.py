"""Estrutura de roteamento por papel/etapa e flags de abla??o para o LLMClient."""

import inspect
import os
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
    """Configura??o de modelos por fun??o, endpoint pr?prio do 1.5B e flags de abla??o."""

    teacher_model: str = DEFAULT_TEACHER_MODEL
    tool_model: str = DEFAULT_TOOL_MODEL
    tool_base_url: Optional[str] = None
    tool_api_key: Optional[str] = None
    tool_timeout: Optional[float] = None

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

    def __post_init__(self) -> None:
        if not self.tool_model:
            self.tool_model = DEFAULT_TOOL_MODEL
        if not self.tool_base_url:
            env_base = (os.getenv("MINICODE_TOOL_API_BASE") or "").strip()
            if env_base:
                self.tool_base_url = env_base
        if not self.tool_api_key:
            env_key = (os.getenv("MINICODE_TOOL_API_KEY") or "").strip()
            if env_key:
                self.tool_api_key = env_key

    def resolve_model_for_role(self, role: Optional[str], fallback_model: str) -> str:
        normalized = (role or ROLE_TOOL_SELECTION).strip().lower()
        if normalized == ROLE_TOOL_SELECTION:
            return self.tool_model or DEFAULT_TOOL_MODEL or fallback_model
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


def _build_tool_endpoint_client(cfg: AgentRoutingConfig, fallback_client: Any) -> Any:
    """Cria cliente OpenAI dedicado ao endpoint pr?prio do modelo de ferramentas (1.5B)."""
    from openai import OpenAI

    api_key = (
        cfg.tool_api_key
        or os.getenv("MINICODE_TOOL_API_KEY")
        or getattr(fallback_client, "api_key", None)
        or os.getenv("MINICODE_API_KEY")
        or "local-tool-key"
    )
    timeout = cfg.tool_timeout or getattr(fallback_client, "timeout", 60.0)
    return OpenAI(api_key=api_key, base_url=cfg.tool_base_url, timeout=timeout)


def attach_routing_to_client(client: Any, routing_config: Optional[AgentRoutingConfig] = None) -> Any:
    """Anexa roteamento por papel, endpoint pr?prio do 1.5B e flags de abla??o a uma inst?ncia de LLMClient."""
    cfg = routing_config or AgentRoutingConfig(
        teacher_model=getattr(client, "model_name", DEFAULT_TEACHER_MODEL),
        tool_model=DEFAULT_TOOL_MODEL,
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
        target_model = model_override or cfg.resolve_model_for_role(
            active_role, getattr(client, "model_name", DEFAULT_TEACHER_MODEL)
        )
        if handler is not None:
            return handler(messages=messages, tools=tools, model=target_model, role=active_role, **kwargs)

        previous_model = getattr(client, "model_name", None)
        previous_sdk_client = getattr(client, "client", None)
        previous_base_url = getattr(client, "base_url", None)

        use_dedicated_tool_endpoint = (
            active_role == ROLE_TOOL_SELECTION
            and bool(cfg.tool_base_url)
            and hasattr(client, "client")
        )

        try:
            if target_model and hasattr(client, "model_name"):
                client.model_name = target_model
            if use_dedicated_tool_endpoint:
                client.client = _build_tool_endpoint_client(cfg, client)
                if hasattr(client, "base_url"):
                    client.base_url = cfg.tool_base_url
            return _call_original_chat(client._original_chat, messages, tools=tools, **kwargs)
        finally:
            if previous_model is not None and hasattr(client, "model_name"):
                client.model_name = previous_model
            if use_dedicated_tool_endpoint:
                if previous_sdk_client is not None:
                    client.client = previous_sdk_client
                if previous_base_url is not None and hasattr(client, "base_url"):
                    client.base_url = previous_base_url

    client.chat = routed_chat
    return client
