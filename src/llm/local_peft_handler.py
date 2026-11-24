"""Handler in-process (PEFT) para servir Qwen2.5-Coder-1.5B-Instruct (base e LoRA) no roteamento do MiniCode."""

from __future__ import annotations

import gc
import json
import logging
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.dataset.build_sft_dataset import DEFAULT_SYSTEM_PROMPT as TRAINING_SYSTEM_PROMPT_V2
from src.llm.client import LLMResponse
from src.llm.routing import ROLE_TOOL_SELECTION
from src.tools.catalog import CATALOG_BY_NAME, ENVELOPE_ADAPTER, LLM_TOOLS

logger = logging.getLogger(__name__)

DEFAULT_BASE_MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DEFAULT_ADAPTER_DIR = "checkpoints/qwen25_coder_1_5b_lora/final_adapter"

MEASURED_Y_P95_TOKENS = 141
MEASURED_Y_MAX_TOKENS = 257
DEFAULT_LOCAL_TOOL_MAX_TOKENS = 1024

_HANDLER_CACHE: Dict[Tuple[str, str, bool], "LocalPeftToolHandler"] = {}

_ROOT_ALLOWED_KEYS: Set[str] = {"thought", "tool", "args"}
_ALLOWED_LLM_TOOLS: Set[str] = set(LLM_TOOLS)
_ARGS_BY_TOOL: Dict[str, Set[str]] = {
    t_name: set(CATALOG_BY_NAME[t_name].args_model.model_fields.keys())
    for t_name in _ALLOWED_LLM_TOOLS
    if t_name in CATALOG_BY_NAME
}
_ALL_ALLOWED_ARG_KEYS: Set[str] = set().union(*_ARGS_BY_TOOL.values()) if _ARGS_BY_TOOL else set()


def _limpar_cache_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


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


def validar_prefixo_envelope_json(texto: str) -> Tuple[bool, bool]:
    """
    Valida se `texto` e um prefixo valido de um objeto JSON conforme `AcaoEnvelope` (LLM_TOOLS).
    Retorna `(prefixo_valido, objeto_completo_e_valido)`.
    """
    s = texto.lstrip(" \t\r\n")
    if not s:
        return True, False
    if s[0] != "{":
        return False, False

    in_string = False
    escape = False
    depth = 0
    str_buf: List[str] = []
    # Pilha de estado por nivel de objeto '{':
    # ("obj", expecting_key: bool, seen_keys: set[str], current_key: Optional[str])
    # ou ("arr",)
    stack: List[Any] = []
    known_tool: Optional[str] = None
    in_args_obj = False

    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if in_string:
            if escape:
                if ch not in {'"', "\\", "/", "b", "f", "n", "r", "t", "u"}:
                    return False, False
                str_buf.append(ch)
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
                completed_str = "".join(str_buf)
                str_buf = []
                if stack and stack[-1][0] == "obj":
                    _, expecting_key, seen_keys, current_key = stack[-1]
                    if expecting_key:
                        if len(stack) == 1:
                            if completed_str not in _ROOT_ALLOWED_KEYS or completed_str in seen_keys:
                                return False, False
                        elif len(stack) == 2 and in_args_obj:
                            allowed_args = (
                                _ARGS_BY_TOOL.get(known_tool, set())
                                if known_tool
                                else _ALL_ALLOWED_ARG_KEYS
                            )
                            if completed_str not in allowed_args or completed_str in seen_keys:
                                return False, False
                        seen_keys.add(completed_str)
                        stack[-1] = ("obj", False, seen_keys, completed_str)
                    else:
                        if len(stack) == 1 and current_key == "tool":
                            if completed_str not in _ALLOWED_LLM_TOOLS:
                                return False, False
                            known_tool = completed_str
                        stack[-1] = ("obj", False, seen_keys, None)
            else:
                if ord(ch) < 0x20:
                    return False, False
                str_buf.append(ch)
            i += 1
            continue

        # Fora de string
        if ch in " \t\r\n":
            i += 1
            continue

        if ch == '"':
            if stack and stack[-1][0] == "obj":
                _, expecting_key, _, current_key = stack[-1]
                if not expecting_key and current_key is None and len(stack) in (1, 2):
                    # Em objeto, apos valor completo so pode vir ',' ou '}'
                    return False, False
            in_string = True
            str_buf = []
            i += 1
            continue

        if ch == "{":
            if depth == 0:
                stack.append(("obj", True, set(), None))
            else:
                if stack and stack[-1][0] == "obj":
                    _, expecting_key, _, current_key = stack[-1]
                    if expecting_key:
                        return False, False
                    if len(stack) == 1:
                        if current_key != "args":
                            return False, False
                        in_args_obj = True
                stack.append(("obj", True, set(), None))
            depth += 1
            i += 1
            continue

        if ch == "}":
            if depth <= 0 or not stack or stack[-1][0] != "obj":
                return False, False
            _, expecting_key, seen_keys, current_key = stack[-1]
            # Nao permite '}' logo apos ':' sem valor
            if not expecting_key and current_key is not None:
                return False, False
            if len(stack) == 2 and in_args_obj:
                in_args_obj = False
            stack.pop()
            depth -= 1
            if depth == 0:
                if s[i + 1 :].strip():
                    return False, False
                try:
                    obj = json.loads(s[: i + 1])
                    if not isinstance(obj, dict) or obj.get("tool") not in _ALLOWED_LLM_TOOLS:
                        return False, False
                    ENVELOPE_ADAPTER.validate_python(obj)
                    return True, True
                except Exception:
                    return False, False
            if stack and stack[-1][0] == "obj":
                _, _, p_seen, _ = stack[-1]
                stack[-1] = ("obj", False, p_seen, None)
            i += 1
            continue

        if ch == "[":
            if depth == 0:
                return False, False
            if stack and stack[-1][0] == "obj":
                _, expecting_key, _, current_key = stack[-1]
                if expecting_key or current_key is None:
                    return False, False
                if len(stack) == 1:
                    return False, False
            stack.append(("arr",))
            depth += 1
            i += 1
            continue

        if ch == "]":
            if depth <= 0 or not stack or stack[-1][0] != "arr":
                return False, False
            stack.pop()
            depth -= 1
            if stack and stack[-1][0] == "obj":
                _, _, p_seen, _ = stack[-1]
                stack[-1] = ("obj", False, p_seen, None)
            i += 1
            continue

        if ch == ":":
            if not stack or stack[-1][0] != "obj":
                return False, False
            _, expecting_key, seen_keys, current_key = stack[-1]
            if expecting_key or current_key is None:
                return False, False
            i += 1
            continue

        if ch == ",":
            if not stack:
                return False, False
            if stack[-1][0] == "obj":
                _, expecting_key, seen_keys, current_key = stack[-1]
                if expecting_key or current_key is not None:
                    return False, False
                stack[-1] = ("obj", True, seen_keys, None)
            i += 1
            continue

        # Literais JSON (numeros, true, false, null)
        if stack and stack[-1][0] == "obj":
            _, expecting_key, seen_keys, current_key = stack[-1]
            if expecting_key or current_key is None:
                return False, False
            if len(stack) == 1 and current_key in {"tool", "args"}:
                return False, False

        if ch in "-0123456789":
            j = i + 1
            while j < n and s[j] in "0123456789.eE+-":
                j += 1
            if j < n:
                try:
                    float(s[i:j])
                except ValueError:
                    return False, False
            if stack and stack[-1][0] == "obj":
                _, _, p_seen, _ = stack[-1]
                stack[-1] = ("obj", False, p_seen, None)
            i = j
            continue

        matched_lit = False
        for lit in ("true", "false", "null"):
            if s.startswith(lit, i):
                if stack and stack[-1][0] == "obj":
                    _, _, p_seen, _ = stack[-1]
                    stack[-1] = ("obj", False, p_seen, None)
                i += len(lit)
                matched_lit = True
                break
            if lit.startswith(s[i:]):
                return True, False
        if matched_lit:
            continue

        return False, False

    return True, False


def _criar_logits_processor_envelope(tokenizer: Any, prompt_len: int, do_sample: bool = False):
    import torch
    from transformers import LogitsProcessor

    eos_id = int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else -1
    special_ids = set(int(x) for x in getattr(tokenizer, "all_special_ids", []) if int(x) != eos_id)
    token_str_cache: Dict[int, str] = {}

    def _decode_tok(tid: int) -> str:
        if tid not in token_str_cache:
            token_str_cache[tid] = tokenizer.decode([tid], skip_special_tokens=False)
        return token_str_cache[tid]

    class MiniCodeJsonEnvelopeLogitsProcessor(LogitsProcessor):
        def __init__(self) -> None:
            self._completed = False

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
            gen_ids = input_ids[0][prompt_len:].tolist()
            prefix_text = tokenizer.decode(gen_ids, skip_special_tokens=True) if gen_ids else ""

            if prefix_text:
                _, done_now = validar_prefixo_envelope_json(prefix_text)
                if done_now:
                    self._completed = True

            if self._completed and eos_id >= 0:
                scores[0, :] = float("-inf")
                scores[0, eos_id] = 0.0
                return scores

            sorted_ids = torch.argsort(scores[0], descending=True).tolist()
            valid_ids: List[int] = []
            max_candidates = 32 if do_sample else 1

            for tid in sorted_ids:
                if tid in special_ids:
                    continue
                if tid == eos_id:
                    if self._completed:
                        valid_ids.append(tid)
                        if len(valid_ids) >= max_candidates:
                            break
                    continue
                cand_piece = _decode_tok(tid)
                if not cand_piece:
                    continue
                ok, _ = validar_prefixo_envelope_json(prefix_text + cand_piece)
                if ok:
                    valid_ids.append(tid)
                    if len(valid_ids) >= max_candidates:
                        break

            if valid_ids:
                kept = scores[0, valid_ids].clone()
                scores[0, :] = float("-inf")
                scores[0, valid_ids] = kept
            return scores

    return MiniCodeJsonEnvelopeLogitsProcessor()


class LocalPeftToolHandler:
    """
    Handler in-process para ROLE_TOOL_SELECTION usando Qwen2.5-Coder-1.5B + LoRA (PEFT).

    Carrega o modelo base + adapter LoRA uma unica vez em memoria/VRAM e permite
    alternar instantaneamente entre o modelo base (celula B) e o modelo com LoRA
    (celula C e celula E) via disable_adapter() e constrained_mode.
    """

    def __init__(
        self,
        base_model_id: str = DEFAULT_BASE_MODEL_ID,
        adapter_dir: str | Path = DEFAULT_ADAPTER_DIR,
        max_tokens: int = DEFAULT_LOCAL_TOOL_MAX_TOKENS,
        use_adapter: bool = True,
        constrained_mode: bool = False,
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
        self.constrained_mode = bool(constrained_mode)
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
        base_tag = "lora" if self.use_adapter else "base"
        return f"{base_tag}_constrained" if self.constrained_mode else base_tag

    def set_use_adapter(self, enabled: bool) -> None:
        """Alterna entre modelo com LoRA (`True`, celula C/E) e modelo base (`False`, celula B)."""
        self.use_adapter = bool(enabled)

    def set_constrained_mode(self, enabled: bool) -> None:
        """Alterna decodificacao restrita por JSON Schema (`True`, celula E)."""
        self.constrained_mode = bool(enabled)

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

    def _executar_geracao_unica(
        self,
        prompt_text: str,
        max_new_tokens: int,
        temperature: float,
    ) -> Tuple[str, int, int, str]:
        import torch
        from transformers import LogitsProcessorList

        input_ids = None
        attention_mask = None
        output_ids = None
        generated_ids = None
        try:
            encoded = self.tokenizer(prompt_text, return_tensors="pt")
            device = next(self.model.parameters()).device
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            prompt_tokens = int(input_ids.shape[-1])
            do_sample = temperature > 0.0

            gen_kwargs: Dict[str, Any] = {
                "input_ids": input_ids,
                "max_new_tokens": max_new_tokens,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            if attention_mask is not None:
                gen_kwargs["attention_mask"] = attention_mask

            if not do_sample:
                gen_kwargs["do_sample"] = False
            else:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = temperature

            if self.constrained_mode:
                proc = _criar_logits_processor_envelope(
                    self.tokenizer, prompt_len=prompt_tokens, do_sample=do_sample
                )
                gen_kwargs["logits_processor"] = LogitsProcessorList([proc])

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
            return content, prompt_tokens, completion_tokens, finish_reason
        finally:
            del input_ids, attention_mask, output_ids, generated_ids
            _limpar_cache_cuda()

    def __call__(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        role: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
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

        max_new_tokens = self._resolver_max_new_tokens(kwargs.get("max_tokens"))
        temperature = float(kwargs.get("temperature", 0.0) or 0.0)

        try:
            content, prompt_tokens, completion_tokens, finish_reason = self._executar_geracao_unica(
                prompt_text=prompt_text,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                logger.warning("CUDA OOM detectado no 1.5B; limpando cache e tentando novamente...")
                _limpar_cache_cuda()
                content, prompt_tokens, completion_tokens, finish_reason = self._executar_geracao_unica(
                    prompt_text=prompt_text,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
            else:
                raise

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
    constrained_mode: bool = False,
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
            constrained_mode=constrained_mode,
            enforce_system_prompt=enforce_system_prompt,
            load_in_4bit=load_in_4bit,
            metrics=metrics,
        )
        _HANDLER_CACHE[cache_key] = handler
    else:
        handler.max_tokens = int(max_tokens)
        handler.set_use_adapter(use_adapter)
        handler.set_constrained_mode(constrained_mode)
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
    constrained_mode: bool = False,
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
            constrained_mode=constrained_mode,
            enforce_system_prompt=enforce_system_prompt,
            load_in_4bit=load_in_4bit,
            metrics=getattr(llm_client, "metrics", None),
        )
    else:
        handler.set_use_adapter(use_adapter)
        handler.set_constrained_mode(constrained_mode)
        if getattr(llm_client, "metrics", None) is not None and handler.metrics is None:
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
