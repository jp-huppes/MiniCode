import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

from src.llm.action_contract import interpretar_acao


@dataclass
class TargetValidationResult:
    ok: bool
    errors: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)


def _procurar_chave_proibida(obj: Any, chave_proibida: str, caminho: str = "") -> List[str]:
    encontrados: List[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            novo_caminho = f"{caminho}.{k}" if caminho else str(k)
            if k == chave_proibida:
                encontrados.append(novo_caminho)
            encontrados.extend(_procurar_chave_proibida(v, chave_proibida, novo_caminho))
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            novo_caminho = f"{caminho}[{idx}]"
            encontrados.extend(_procurar_chave_proibida(item, chave_proibida, novo_caminho))
    return encontrados


def validar_exemplo_alvo(exemplo: Union[Dict[str, Any], str]) -> TargetValidationResult:
    """
    Valida um exemplo-alvo do dataset usando a validação da F4.
    Retorna ok=True quando válido ou ok=False com a lista de erros descritivos.
    Exige 'tool' e proíbe 'file_path' em qualquer nível do alvo.
    """
    errors: List[str] = []
    categories: List[str] = []

    if isinstance(exemplo, dict):
        caminhos_file_path = _procurar_chave_proibida(exemplo, "file_path")
        if caminhos_file_path:
            categories.append("campo_proibido_file_path")
            for c in caminhos_file_path:
                errors.append(f"Uso proibido de 'file_path' em '{c}'; use 'path' dentro de 'args'.")

        if "tool" not in exemplo or not exemplo.get("tool"):
            categories.append("tool_ausente")
            errors.append("O exemplo-alvo deve conter a chave 'tool' (nunca 'file_path' ou 'action').")

        texto_para_f4 = json.dumps(exemplo, ensure_ascii=False)
    elif isinstance(exemplo, str):
        texto_para_f4 = exemplo.strip()
        if not texto_para_f4:
            return TargetValidationResult(
                ok=False,
                errors=["O exemplo-alvo está vazio."],
                categories=["nao_json"],
            )
        try:
            parsed_raw = json.loads(texto_para_f4)
            caminhos_file_path = _procurar_chave_proibida(parsed_raw, "file_path")
            if caminhos_file_path:
                categories.append("campo_proibido_file_path")
                for c in caminhos_file_path:
                    errors.append(f"Uso proibido de 'file_path' em '{c}'; use 'path' dentro de 'args'.")
            if isinstance(parsed_raw, dict) and ("tool" not in parsed_raw or not parsed_raw.get("tool")):
                categories.append("tool_ausente")
                errors.append("O exemplo-alvo deve conter a chave 'tool'.")
        except json.JSONDecodeError:
            pass
    else:
        return TargetValidationResult(
            ok=False,
            errors=[f"Tipo de exemplo-alvo não suportado: {type(exemplo).__name__}."],
            categories=["tipo_invalido"],
        )

    avaliacao = interpretar_acao(texto_para_f4)
    if not avaliacao.ok:
        if avaliacao.diagnostic is not None:
            cat = avaliacao.diagnostic.category.value
            if cat not in categories:
                categories.append(cat)
            msg = avaliacao.diagnostic.format_retry_message()
            if msg not in errors:
                errors.append(msg)
        else:
            if "validacao_f4" not in categories:
                categories.append("validacao_f4")
            errors.append("Exemplo-alvo falhou na validação do contrato de ação (F4).")

    return TargetValidationResult(
        ok=len(errors) == 0,
        errors=errors,
        categories=categories,
    )