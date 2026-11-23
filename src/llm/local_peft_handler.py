"""Handler in-process (PEFT) para servir Qwen2.5-Coder-1.5B-Instruct (base e LoRA) no roteamento do MiniCode."""

from __future__ import annotations

import logging
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.dataset.build_sft_dataset import DEFAULT_SYSTEM_PROMPT as TRAINING_SYSTEM_PROMPT_V2
from src.llm.client import LLMResponse
from src.llm.routing import ROLE_TOOL_SELECTION

logger = logging.getLogger(__name__)

DEFAULT_BASE_MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DEFAULT_ADAPTER_DIR = "checkpoints/qwen25_coder_1_5b_lora/final_adapter"

# Medicoes empiricas do tamanho de y (tokens reais no tokenizador Qwen2.5-Coder):
#   corpus_filtered/dataset.jsonl (n=211): p50=51 | p95=136 | max=257
#   datasets/sft_jsonl/train.jsonl (n=132): p50=58 | p95=141 | max=257
# O padrao legado de 512 no roteamento cortava write_file longos em tarefas E2E
# (gerando JSON_TRUNCADO). Usamos 1024 como orcamento seguro acima do p95 e do max
# para preservar acoes write_file completas sem truncamento.
MEASURED_Y_P95_TOKENS = 141
MEASURED_Y_MAX_TOKENS = 257
DEFAULT_LOCAL_TOOL_MAX_TOKENS = 1024

_HANDLER_CACHE: Dict[Tuple[str, str, bool], "LocalPeftToolHandler"] = {}


def _normalizar_mensagens_para_treino(
    messages: List[Dict[str, Any]],
    system_prompt_v2: str = TRAINING_SYSTEM_PROMPT_V2,
    enforce_system_prompt: bool = True,
) -> List[Dict[str, str]]:
    """Garante o mesmo formato de mensagens (system v2 + user/assistant) usado no treino SFT."""
    normalizadas: List[Dict[str, str]] = []
    has_system = False

    for msg in messages:
        role = str(msg.get("role", "user"))
        content = str(msg.get("content", ""))
        if role == "system":
            if not has_system:
                normalizadas.append(
                    {
                        "role": "system",
                        "content": system_prompt_v2 if enforce_system_prompt else content,
                    }
                )
                has_system = True
        elif role == "tool":
            tool_name = msg.get("name", "ferramenta")
            normalizadas.append(
                {
                    "role": "user",
                    "content": f"[Resultado da ferramenta '{tool_name}']\n{content}",
                }
            )
        else:
            normalizadas.append({"role": role, "content": content})

    if not has_system:
        normalizadas.insert(0, {"role": "system", "content": system_prompt_v2})

    return normalizadas


class LocalPeftToolHandler:
    """
    Handler in-process para ROLE_TOOL_SELECTION usando Qwen2.5-Coder-1.5B + LoRA (PEFT).

    Carrega o modelo base + adapter LoRA uma unica vez em memoria/VRAM e permite
    alternar instantaneamente entre o modelo base (celula B) e o modelo com LoRA
    (celula C) via `disable_adapter()`, garantindo exatamente os mesmos pesos base.
    """

    def __init__(
        self,
        base_model_id: str = DEFAULT_BASE_MODEL_ID,
        adapter_dir: str | Path = DEFAULT_ADAPTER_DIR,
        max_tokens: int = DEFAULT_LOCAL_TOOL_MAX_TOKENS,
        use_adapter: bool = True,
        enforce_system_prompt: bool = True,
        system_prompt_v2: str = TRAINING_SYSTEM_PROMPT_V2,
        load_in_4bit: bool = False,
        model: Any = None,
        tokenizer: Any = None,
        metrics: Any = None,
    ) -> None:
        self.base_model_id = base_model_id
        self.adapter_dir = Path(adapter_dir)
        self.max_tokens = int(max_tokens)
        self.use_adapter = bool(use_adapter)
        self.enforce_system_prompt = bool(enforce_system_prompt)
        self.system_prompt_v2 = system_prompt_v2
        self.load_in_4bit = bool(load_in_4bit)
        self.metrics = metrics

        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
        else:
            self.model, self.tokenizer = self._carregar_modelo_e_tokenizer()

    def _carregar_modelo_e_tokenizer(self) -> Tuple[Any, Any]:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not self.adapter_dir.exists():
            raise FileNotFoundError(
                f"Diretorio do adapter LoRA nao encontrado: {self.adapter_dir.resolve()}"
            )

        tokenizer = AutoTokenizer.from_pretrained(str(self.adapter_dir), use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        has_cuda = torch.cuda.is_available()
        if has_cuda:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32

        model_kwargs: Dict[str, Any] = {
            "torch_dtype": dtype,
            "device_map": "auto" if has_cuda else None,
        }

        if self.load_in_4bit and has_cuda:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )

        logger.info(
            "Carregando modelo base '%s' + adapter PEFT '%s' (cuda=%s, dtype=%s, 4bit=%s)...",
            self.base_model_id,
            self.adapter_dir,
            has_cuda,
            dtype,
            self.load_in_4bit,
        )
        base_model = AutoModelForCausalLM.from_pretrained(self.base_model_id, **model_kwargs)
        peft_model = PeftModel.from_pretrained(base_model, str(self.adapter_dir))
        peft_model.eval()
        return peft_model, tokenizer

    @property
    def active_variant(self) -> str:
        return "lora" if self.use_adapter else "base"

    def set_use_adapter(self, enabled: bool) -> None:
        """Alterna entre modelo com LoRA (`True`, celula C) e modelo base (`False`, celula B)."""
        self.use_adapter = bool(enabled)

    def _adapter_context(self):
        if self.use_adapter:
            return nullcontext()
        if hasattr(self.model, "disable_adapter"):
            return self.model.disable_adapter()
        raise RuntimeError("O modelo carregado nao suporta disable_adapter().")

    def _resolver_max_new_tokens(self, requested_max_tokens: Optional[int]) -> int:
        if requested_max_tokens is None:
            return self.max_tokens
        if int(requested_max_tokens) == 512 and self.max_tokens > 512:
            return self.max_tokens
        return max(1, int(requested_max_tokens))

    def __call__(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        role: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        import torch

        t0 = time.perf_counter()
        msgs_treino = _normalizar_mensagens_para_treino(
            messages,
            system_prompt_v2=self.system_prompt_v2,
            enforce_system_prompt=self.enforce_system_prompt,
        )

        prompt_text = self.tokenizer.apply_chat_template(
            msgs_treino,
            tokenize=False,
            add_generation_prompt=True,
        )

        encoded = self.tokenizer(prompt_text, return_tensors="pt")
        device = next(self.model.parameters()).device
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        prompt_tokens = int(input_ids.shape[-1])
        max_new_tokens = self._resolver_max_new_tokens(kwargs.get("max_tokens"))
        temperature = float(kwargs.get("temperature", 0.0) or 0.0)

        gen_kwargs: Dict[str, Any] = {
            "input_ids": input_ids,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask

        if temperature <= 0.0:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature

        with torch.inference_mode(), self._adapter_context():
            output_ids = self.model.generate(**gen_kwargs)

        generated_ids = output_ids[0][prompt_tokens:]
        completion_tokens = int(generated_ids.shape[-1])
        content = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        eos_id = self.tokenizer.eos_token_id
        terminou_por_eos = (
            completion_tokens > 0
            and eos_id is not None
            and int(generated_ids[-1].item()) == int(eos_id)
        )
        finish_reason = (
            "length"
            if (completion_tokens >= max_new_tokens and not terminou_por_eos)
            else "stop"
        )
        latency = time.perf_counter() - t0

        if self.metrics is not None and hasattr(self.metrics, "registrar"):
            self.metrics.registrar(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency=latency,
                retries=0,
                finish_reason=finish_reason,
                success=True,
                motivo=f"local_peft_{self.active_variant}",
            )

        return LLMResponse(
            content=content,
            tool_calls=None,
            finish_reason=finish_reason,
            latency_seconds=latency,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


def get_or_create_peft_handler(
    base_model_id: str = DEFAULT_BASE_MODEL_ID,
    adapter_dir: str | Path = DEFAULT_ADAPTER_DIR,
    max_tokens: int = DEFAULT_LOCAL_TOOL_MAX_TOKENS,
    use_adapter: bool = True,
    enforce_system_prompt: bool = True,
    load_in_4bit: bool = False,
    metrics: Any = None,
) -> LocalPeftToolHandler:
    """Retorna instancia cacheada do handler PEFT para evitar recarregar pesos na GPU."""
    resolved_adapter = str(Path(adapter_dir).resolve())
    cache_key = (base_model_id, resolved_adapter, bool(load_in_4bit))
    handler = _HANDLER_CACHE.get(cache_key)
    if handler is None:
        handler = LocalPeftToolHandler(
            base_model_id=base_model_id,
            adapter_dir=adapter_dir,
            max_tokens=max_tokens,
            use_adapter=use_adapter,
            enforce_system_prompt=enforce_system_prompt,
            load_in_4bit=load_in_4bit,
            metrics=metrics,
        )
        _HANDLER_CACHE[cache_key] = handler
    else:
        handler.max_tokens = int(max_tokens)
        handler.set_use_adapter(use_adapter)
        handler.enforce_system_prompt = bool(enforce_system_prompt)
        if metrics is not None:
            handler.metrics = metrics
    return handler


def register_local_peft_handler(
    llm_client: Any,
    handler: Optional[LocalPeftToolHandler] = None,
    *,
    base_model_id: str = DEFAULT_BASE_MODEL_ID,
    adapter_dir: str | Path = DEFAULT_ADAPTER_DIR,
    max_tokens: int = DEFAULT_LOCAL_TOOL_MAX_TOKENS,
    use_adapter: bool = True,
    enforce_system_prompt: bool = True,
    load_in_4bit: bool = False,
) -> LocalPeftToolHandler:
    """
    Registra o handler PEFT in-process em `routing_config.role_chat_handlers[ROLE_TOOL_SELECTION]`
    e ajusta `temperature=0.0` e `max_tokens` no roteamento do `llm_client`.
    """
    if handler is None:
        handler = get_or_create_peft_handler(
            base_model_id=base_model_id,
            adapter_dir=adapter_dir,
            max_tokens=max_tokens,
            use_adapter=use_adapter,
            enforce_system_prompt=enforce_system_prompt,
            load_in_4bit=load_in_4bit,
            metrics=getattr(llm_client, "metrics", None),
        )
    elif getattr(llm_client, "metrics", None) is not None and handler.metrics is None:
        handler.metrics = llm_client.metrics

    cfg = getattr(llm_client, "routing_config", None)
    if cfg is None:
        from src.llm.routing import AgentRoutingConfig, attach_routing_to_client

        cfg = AgentRoutingConfig(
            teacher_model=getattr(llm_client, "model_name", "Qwen/Qwen2.5-14B-Instruct-AWQ"),
            tool_model=base_model_id,
            temperature=0.0,
            max_tokens=max_tokens,
        )
        attach_routing_to_client(llm_client, cfg)

    cfg.tool_model = base_model_id
    cfg.temperature = 0.0
    cfg.max_tokens = int(max_tokens)
    cfg.role_chat_handlers[ROLE_TOOL_SELECTION] = handler
    return handler