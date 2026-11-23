from contextlib import contextmanager
from types import SimpleNamespace

import torch

from src.dataset.build_sft_dataset import DEFAULT_SYSTEM_PROMPT as TRAINING_SYSTEM_PROMPT_V2
from src.llm.local_peft_handler import (
    DEFAULT_LOCAL_TOOL_MAX_TOKENS,
    MEASURED_Y_MAX_TOKENS,
    MEASURED_Y_P95_TOKENS,
    LocalPeftToolHandler,
    register_local_peft_handler,
)
from src.llm.routing import AgentRoutingConfig, ROLE_TOOL_SELECTION, attach_routing_to_client


class _FakeTokenizer:
    pad_token = "<|endoftext|>"
    eos_token = "<|im_end|>"
    pad_token_id = 0
    eos_token_id = 2

    def __init__(self) -> None:
        self.last_messages = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.last_messages = messages
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"

    def __call__(self, prompt_text, return_tensors="pt"):
        return {
            "input_ids": torch.tensor([[10, 11, 12, 13]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
        }

    def decode(self, token_ids, skip_special_tokens=True):
        return '```json\n{"thought": "ok", "tool": "done", "args": {}}\n```'


class _FakePeftModel:
    def __init__(self, finish_with_eos: bool = True) -> None:
        self._param = torch.nn.Parameter(torch.zeros(1))
        self.adapter_disabled = False
        self.adapter_states_on_generate = []
        self.last_gen_kwargs = None
        self.finish_with_eos = finish_with_eos

    def parameters(self):
        return iter([self._param])

    @contextmanager
    def disable_adapter(self):
        prev = self.adapter_disabled
        self.adapter_disabled = True
        try:
            yield
        finally:
            self.adapter_disabled = prev

    def generate(self, **kwargs):
        self.adapter_states_on_generate.append(self.adapter_disabled)
        self.last_gen_kwargs = dict(kwargs)
        input_ids = kwargs["input_ids"]
        if self.finish_with_eos:
            new_tokens = torch.tensor([[21, 22, 2]], dtype=torch.long)
        else:
            max_new = kwargs["max_new_tokens"]
            new_tokens = torch.full((1, max_new), 21, dtype=torch.long)
        return torch.cat([input_ids, new_tokens], dim=-1)


def test_handler_contract_prompt_v2_and_disable_adapter():
    tok = _FakeTokenizer()
    model = _FakePeftModel(finish_with_eos=True)
    handler = LocalPeftToolHandler(model=model, tokenizer=tok, use_adapter=True)

    client = SimpleNamespace(model_name="Qwen/Qwen2.5-14B-Instruct-AWQ", chat=lambda *a, **k: None)
    attach_routing_to_client(client, AgentRoutingConfig(max_tokens=512))
    register_local_peft_handler(client, handler=handler)

    assert DEFAULT_LOCAL_TOOL_MAX_TOKENS > MEASURED_Y_P95_TOKENS
    assert DEFAULT_LOCAL_TOOL_MAX_TOKENS > MEASURED_Y_MAX_TOKENS
    assert DEFAULT_LOCAL_TOOL_MAX_TOKENS > 512

    # 1) Modo LoRA (celula C)
    resp_c = client.chat(
        [{"role": "system", "content": "prompt antigo"}, {"role": "user", "content": "Implemente"}],
        role=ROLE_TOOL_SELECTION,
    )
    assert resp_c.content.startswith("```json")
    assert resp_c.finish_reason == "stop"
    assert resp_c.prompt_tokens == 4
    assert resp_c.completion_tokens == 3
    assert tok.last_messages[0]["content"] == TRAINING_SYSTEM_PROMPT_V2
    assert model.adapter_states_on_generate[-1] is False
    assert model.last_gen_kwargs["do_sample"] is False
    assert model.last_gen_kwargs["max_new_tokens"] == DEFAULT_LOCAL_TOOL_MAX_TOKENS

    # 2) Alterna para Base (celula B) via disable_adapter() sem recarregar modelo
    handler.set_use_adapter(False)
    resp_b = client.chat(
        [{"role": "user", "content": "Implemente"}],
        role=ROLE_TOOL_SELECTION,
    )
    assert resp_b.finish_reason == "stop"
    assert model.adapter_states_on_generate[-1] is True


def test_handler_reports_length_finish_reason_when_truncated():
    tok = _FakeTokenizer()
    model = _FakePeftModel(finish_with_eos=False)
    handler = LocalPeftToolHandler(model=model, tokenizer=tok, max_tokens=16)

    resp = handler([{"role": "user", "content": "Crie arquivo longo"}], max_tokens=16)
    assert resp.finish_reason == "length"
    assert resp.completion_tokens == 16