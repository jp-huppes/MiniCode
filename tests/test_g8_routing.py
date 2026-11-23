from src.llm.routing import (
    DEFAULT_TEACHER_MODEL,
    DEFAULT_TOOL_MODEL,
    ROLE_PLAN,
    ROLE_REPAIR,
    ROLE_SPEC,
    ROLE_TOOL_SELECTION,
    AgentRoutingConfig,
    attach_routing_to_client,
)


class _FakeClient:
    def __init__(self):
        self.model_name = DEFAULT_TEACHER_MODEL
        self.base_url = "http://teacher-host:8000/v1"
        self.api_key = "teacher-key"
        self.timeout = 30.0
        self.client = object()
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append({
            "model": self.model_name,
            "base_url": self.base_url,
            "client": self.client,
            "messages": messages,
            "kwargs": kwargs,
        })
        return f"ok:{self.model_name}"


def test_g8_routing_tool_1_5b_and_teacher_roles(monkeypatch):
    fake_dedicated = object()
    monkeypatch.setattr(
        "src.llm.routing._build_tool_endpoint_client",
        lambda cfg, client: fake_dedicated,
    )

    cfg = AgentRoutingConfig(
        teacher_model=DEFAULT_TEACHER_MODEL,
        tool_model=DEFAULT_TOOL_MODEL,
        tool_base_url="http://tool-15b-host:8001/v1",
        tool_api_key="tool-key",
        temperature=0.0,
        max_tokens=256,
    )
    client = _FakeClient()
    attach_routing_to_client(client, cfg)

    msgs = [{"role": "user", "content": "ler arquivo"}]
    out_tool = client.chat(msgs, role=ROLE_TOOL_SELECTION)
    out_spec = client.chat(msgs, role=ROLE_SPEC)
    out_plan = client.chat(msgs, role=ROLE_PLAN)
    out_repair = client.chat(msgs, role=ROLE_REPAIR)

    assert out_tool == f"ok:{DEFAULT_TOOL_MODEL}"
    assert out_spec == f"ok:{DEFAULT_TEACHER_MODEL}"
    assert out_plan == f"ok:{DEFAULT_TEACHER_MODEL}"
    assert out_repair == f"ok:{DEFAULT_TEACHER_MODEL}"

    assert client.calls[0]["model"] == DEFAULT_TOOL_MODEL
    assert client.calls[0]["base_url"] == "http://tool-15b-host:8001/v1"
    assert client.calls[0]["client"] is fake_dedicated
    assert client.calls[0]["kwargs"]["temperature"] == 0.0
    assert client.calls[0]["kwargs"]["max_tokens"] == 256

    for call in client.calls[1:]:
        assert call["model"] == DEFAULT_TEACHER_MODEL
        assert call["base_url"] == "http://teacher-host:8000/v1"

    assert client.model_name == DEFAULT_TEACHER_MODEL
    assert client.base_url == "http://teacher-host:8000/v1"
