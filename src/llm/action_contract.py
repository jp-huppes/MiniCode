from dataclasses import dataclass, field
from enum import Enum
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

from src.agent.schemas import ActionRequest
from src.tools.catalog import (
    ALLOWED_TOOLS,
    CATALOG_BY_NAME,
    ENVELOPE_ADAPTER,
)


class ActionErrorCategory(str, Enum):
    NAO_JSON = "nao_json"
    JSON_TRUNCADO = "json_truncado"
    TOOL_INVALIDA = "tool_invalida"
    CAMPO_EXTRA = "campo_extra"
    CAMPO_AUSENTE = "campo_ausente"
    TIPO_ERRADO = "tipo_errado"
    ARGS_DE_OUTRA_TOOL = "args_de_outra_tool"


@dataclass
class ActionDiagnostic:
    category: ActionErrorCategory
    field_path: str
    message: str
    allowed_values: List[str] = field(default_factory=list)

    def format_retry_message(self) -> str:
        """Mensagem curta e dirigida para retry (sem reenviar o schema inteiro)."""
        allowed_part = ""
        if self.allowed_values:
            allowed_part = f" Valores/campos válidos: {self.allowed_values}."
        loc_part = f"[{self.field_path}] " if self.field_path else ""
        return f"{loc_part}{self.message}{allowed_part}".strip()


@dataclass
class InterpretedAction:
    action: Optional[ActionRequest] = None
    diagnostic: Optional[ActionDiagnostic] = None

    @property
    def ok(self) -> bool:
        return self.action is not None and self.diagnostic is None


_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def _extract_balanced_object(text: str) -> Optional[str]:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for idx in range(start, len(text)):
        ch = text[idx]
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]

    # Se abriu '{' mas não fechou todas as chaves, retorna do '{' até o fim para ser classificado
    return text[start:]


def extrair_candidato(texto: str) -> Optional[str]:
    """
    Extração tolerante (camadas 1-3):
    1. Bloco ```json ... ``` ou ``` ... ```
    2. Objeto JSON balanceado (ou início de objeto se truncado)
    3. Texto puro / tolerância a prefixo 'json'
    """
    if not texto or not texto.strip():
        return None

    stripped = texto.strip()

    fence_match = _FENCE_RE.search(stripped)
    if fence_match:
        inner = fence_match.group(1).strip()
        if inner:
            return inner

    balanced = _extract_balanced_object(stripped)
    if balanced is not None:
        return balanced.strip()

    # Tolerância a prefixo 'json' sem cerca (ex.: "json\n{\"tool\": ...}")
    no_prefix = re.sub(r"^json\s*[:\n\r]+", "", stripped, flags=re.IGNORECASE).strip()
    if no_prefix:
        return no_prefix

    return None


def _looks_truncated(raw_candidate: str, finish_reason: Optional[str] = None) -> bool:
    if finish_reason == "length":
        return True
    s = raw_candidate.strip()
    if not s:
        return False
    open_braces = s.count("{") - s.count("}")
    open_brackets = s.count("[") - s.count("]")
    quotes = s.count('"') - s.count('\\"')
    return open_braces > 0 or open_brackets > 0 or (quotes % 2 == 1)


def _detect_other_tool_args(tool_name: str, args: Any) -> Optional[str]:
    if not isinstance(args, dict) or tool_name not in CATALOG_BY_NAME:
        return None
    current_fields = set(CATALOG_BY_NAME[tool_name].args_model.model_fields.keys())
    args_keys = set(args.keys())
    if not args_keys or args_keys.issubset(current_fields):
        return None

    for other_name, other_spec in CATALOG_BY_NAME.items():
        if other_name == tool_name:
            continue
        other_fields = set(other_spec.args_model.model_fields.keys())
        if other_fields and args_keys.issubset(other_fields) and not args_keys.issubset(current_fields):
            return other_name
    return None


def _classify_validation_error(obj: Dict[str, Any], exc: ValidationError) -> ActionDiagnostic:
    raw_tool = obj.get("tool") if "tool" in obj else obj.get("action")
    raw_args = obj.get("args")

    if raw_tool is None or raw_tool not in ALLOWED_TOOLS:
        return ActionDiagnostic(
            category=ActionErrorCategory.TOOL_INVALIDA,
            field_path="tool",
            message=f"Ferramenta '{raw_tool}' inválida ou ausente.",
            allowed_values=sorted(ALLOWED_TOOLS),
        )

    other_tool = _detect_other_tool_args(raw_tool, raw_args)
    if other_tool:
        valid_fields = list(CATALOG_BY_NAME[raw_tool].args_model.model_fields.keys())
        return ActionDiagnostic(
            category=ActionErrorCategory.ARGS_DE_OUTRA_TOOL,
            field_path="args",
            message=f"Argumentos parecem pertencer a '{other_tool}', não a '{raw_tool}'.",
            allowed_values=valid_fields,
        )

    errors = exc.errors()
    first_err = errors[0] if errors else {}
    err_type = first_err.get("type", "")
    loc = [str(p) for p in first_err.get("loc", []) if str(p) != "args"]

    # Normaliza caminho para args.<campo> quando aplicável
    if isinstance(raw_args, dict) and raw_tool in CATALOG_BY_NAME:
        spec = CATALOG_BY_NAME[raw_tool]
        valid_fields = list(spec.args_model.model_fields.keys())
        extra_keys = [k for k in raw_args.keys() if k not in spec.args_model.model_fields]
        if extra_keys:
            return ActionDiagnostic(
                category=ActionErrorCategory.CAMPO_EXTRA,
                field_path=f"args.{extra_keys[0]}",
                message=f"Campo extra '{extra_keys[0]}' não permitido em '{raw_tool}'.",
                allowed_values=valid_fields,
            )
        try:
            spec.args_model.model_validate(raw_args)
        except ValidationError as arg_exc:
            a_err = arg_exc.errors()[0]
            a_type = a_err.get("type", "")
            a_loc = ".".join(str(p) for p in a_err.get("loc", []))
            field_path = f"args.{a_loc}" if a_loc else "args"
            if a_type == "missing":
                return ActionDiagnostic(
                    category=ActionErrorCategory.CAMPO_AUSENTE,
                    field_path=field_path,
                    message=f"Campo obrigatório ausente em '{raw_tool}'.",
                    allowed_values=valid_fields,
                )
            return ActionDiagnostic(
                category=ActionErrorCategory.TIPO_ERRADO,
                field_path=field_path,
                message=f"Tipo inválido em '{field_path}': {a_err.get('msg', 'valor inválido')}.",
                allowed_values=valid_fields,
            )

    field_path = ".".join(loc) if loc else "root"
    if err_type == "extra_forbidden":
        return ActionDiagnostic(
            category=ActionErrorCategory.CAMPO_EXTRA,
            field_path=field_path,
            message=f"Campo extra não permitido em '{field_path}'.",
        )
    if err_type == "missing":
        return ActionDiagnostic(
            category=ActionErrorCategory.CAMPO_AUSENTE,
            field_path=field_path,
            message=f"Campo obrigatório ausente em '{field_path}'.",
        )

    return ActionDiagnostic(
        category=ActionErrorCategory.TIPO_ERRADO,
        field_path=field_path,
        message=f"Estrutura ou tipo inválido em '{field_path}': {first_err.get('msg', 'erro de validação')}.",
    )


def interpretar_acao(texto: str, finish_reason: Optional[str] = None) -> InterpretedAction:
    candidato = extrair_candidato(texto)
    if not candidato:
        cat = (
            ActionErrorCategory.JSON_TRUNCADO
            if finish_reason == "length"
            else ActionErrorCategory.NAO_JSON
        )
        return InterpretedAction(
            diagnostic=ActionDiagnostic(
                category=cat,
                field_path="root",
                message="Nenhum objeto JSON encontrado na resposta.",
                allowed_values=sorted(ALLOWED_TOOLS),
            )
        )

    try:
        obj = json.loads(candidato)
    except json.JSONDecodeError as exc:
        is_trunc = _looks_truncated(candidato, finish_reason=finish_reason)
        cat = (
            ActionErrorCategory.JSON_TRUNCADO
            if is_trunc
            else ActionErrorCategory.NAO_JSON
        )
        return InterpretedAction(
            diagnostic=ActionDiagnostic(
                category=cat,
                field_path="root",
                message=f"JSON inválido ({exc.msg}).",
            )
        )

    if not isinstance(obj, dict):
        return InterpretedAction(
            diagnostic=ActionDiagnostic(
                category=ActionErrorCategory.TIPO_ERRADO,
                field_path="root",
                message="A resposta JSON deve ser um objeto.",
            )
        )

    # Normaliza 'action' -> 'tool' se vier com chave alternativa antes do Pydantic
    normalized_obj: Dict[str, Any] = dict(obj)
    if "tool" not in normalized_obj and "action" in normalized_obj:
        normalized_obj["tool"] = normalized_obj.pop("action")
    if "args" not in normalized_obj:
        normalized_obj["args"] = {}

    try:
        envelope = ENVELOPE_ADAPTER.validate_python(normalized_obj)
    except ValidationError as exc:
        return InterpretedAction(diagnostic=_classify_validation_error(normalized_obj, exc))

    args_dict = (
        envelope.args.model_dump(exclude_none=False)
        if hasattr(envelope.args, "model_dump")
        else dict(envelope.args)
    )
    return InterpretedAction(action=ActionRequest(tool=envelope.tool, args=args_dict))