import json
import re
from typing import Any, Dict, List, Optional, Set

from src.llm.action_parser import (
    extrair_camada_1_inteiro,
    extrair_camada_2_cerca_markdown,
    extrair_camada_3_objeto_balanceado,
)
from src.spec.models import (
    AcceptanceCriterion,
    Plan,
    PlanDocument,
    PlanStep,
    Spec,
    SpecDocument,
    SpecTestItem,
    ValidationResult,
)

_PLACEHOLDER_RE = re.compile(r"(?i)(?:\btbd\b|\bn/a\b)")
_AC_ID_RE = re.compile(r"^AC-\d+$", re.IGNORECASE)
_TEST_NAME_RE = re.compile(r"^test_ac_\d+(?:_[a-z0-9_]+)?$", re.IGNORECASE)


def limpar_markdown_fence(texto: str) -> str:
    # Remove blocos de cercas markdown (ex: ```markdown ... ``` ou ```json ... ```) caso presente.
    texto = (texto or "").strip()
    if texto.startswith("```"):
        linhas = texto.splitlines()
        if linhas and linhas[0].startswith("```"):
            linhas = linhas[1:]
        if linhas and linhas[-1].startswith("```"):
            linhas = linhas[:-1]
        texto = "\n".join(linhas).strip()
    return texto


def _extrair_json_objeto(texto: str) -> Optional[Dict[str, Any]]:
    candidatos = [
        extrair_camada_1_inteiro(texto or ""),
        extrair_camada_2_cerca_markdown(texto or ""),
        extrair_camada_3_objeto_balanceado(texto or ""),
    ]
    for cand in candidatos:
        if not cand:
            continue
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def spec_json_schema() -> Dict[str, Any]:
    """Schema JSON para SPEC estruturada na F8."""
    return {
        "type": "object",
        "required": ["objective", "parameters", "acceptance_criteria", "tests"],
        "additionalProperties": False,
        "properties": {
            "objective": {"type": "string", "minLength": 1},
            "parameters": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string", "minLength": 1},
            },
            "acceptance_criteria": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "required": ["id", "text"],
                    "additionalProperties": False,
                    "properties": {
                        "id": {"type": "string", "pattern": "^AC-[0-9]+$"},
                        "text": {"type": "string", "minLength": 1},
                    },
                },
            },
            "tests": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "required": ["ac", "name"],
                    "additionalProperties": False,
                    "properties": {
                        "ac": {"type": "string", "pattern": "^AC-[0-9]+$"},
                        "name": {"type": "string", "minLength": 1},
                    },
                },
            },
        },
    }


def plan_json_schema(spec: Optional[Spec] = None) -> Dict[str, Any]:
    """
    Gera o schema JSON do PLAN em tempo de execução.
    Quando há IDs de AC na SPEC, restringe `covers` via `enum` aos IDs válidos da SPEC (P5).
    """
    ac_ids = sorted(spec.acceptance_criteria.keys()) if spec and spec.acceptance_criteria else []
    covers_item_schema: Dict[str, Any] = {"type": "string", "minLength": 1}
    if ac_ids:
        covers_item_schema = {"type": "string", "enum": ac_ids}

    return {
        "type": "object",
        "required": ["steps"],
        "additionalProperties": False,
        "properties": {
            "steps": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "required": ["action", "files", "covers"],
                    "additionalProperties": False,
                    "properties": {
                        "n": {"type": "integer", "minimum": 1},
                        "action": {"type": "string", "minLength": 1},
                        "files": {
                            "type": "array",
                            "items": {"type": "string", "minLength": 1},
                        },
                        "covers": {
                            "type": "array",
                            "minItems": 1,
                            "items": covers_item_schema,
                        },
                    },
                },
            }
        },
    }


def renderizar_spec_markdown(doc: SpecDocument) -> str:
    """Renderização determinística de SpecDocument para Markdown no formato esperado pelo harness."""
    linhas: List[str] = [
        "## Objective",
        doc.objective.strip(),
        "",
        "## Parameters",
    ]
    for param in doc.parameters:
        linhas.append(f"- {param.strip()}")

    linhas.extend(["", "## Acceptance Criteria"])
    for ac in doc.acceptance_criteria:
        linhas.append(f"- **{ac.id.upper()}**: {ac.text.strip()}")

    linhas.extend(["", "## Tests"])
    for t in doc.tests:
        linhas.append(f"- **{t.ac.upper()}**: {t.name.strip()}")

    return "\n".join(linhas).strip()


def renderizar_plano_markdown(steps: List[PlanStep]) -> str:
    """Renderização determinística de passos numerados para plan.md no formato esperado pelo harness."""
    linhas: List[str] = []
    for idx, passo in enumerate(steps, start=1):
        num = passo.n if passo.n and passo.n > 0 else idx
        arquivos = ", ".join(f.strip() for f in passo.files if f and f.strip()) or "nenhum"
        coberturas = ", ".join(c.upper().strip() for c in passo.covers if c and c.strip()) or "nenhum"
        linhas.append(f"{num}. {passo.action.strip()} — arquivo(s): {arquivos} — cobre: {coberturas}")
    return "\n".join(linhas).strip()


def _contem_placeholder(valor: str) -> bool:
    return bool(_PLACEHOLDER_RE.search(valor or ""))


def validar_spec_json(texto: str) -> ValidationResult:
    """Valida SPEC em JSON conforme regras V1–V7 (schema/estrutura + validador semântico)."""
    erros: List[str] = []
    obj = _extrair_json_objeto(texto)
    if not isinstance(obj, dict):
        return ValidationResult(ok=False, errors=["V1-V7: Saída não contém um objeto JSON válido para a SPEC."])

    objective = obj.get("objective")
    if not isinstance(objective, str) or not objective.strip():
        erros.append("V1: Campo 'objective' ausente ou vazio na SPEC.")
    elif _contem_placeholder(objective):
        erros.append("V4: Placeholder (TBD/N/A) detectado em 'objective'.")

    parameters = obj.get("parameters")
    if not isinstance(parameters, list) or len(parameters) == 0:
        erros.append("V2: Campo 'parameters' ausente ou vazio na SPEC.")
    else:
        for idx, p in enumerate(parameters):
            if not isinstance(p, str) or not p.strip():
                erros.append(f"V2/V4: Parâmetro na posição {idx} inválido ou vazio.")
            elif _contem_placeholder(p):
                erros.append(f"V4: Placeholder (TBD/N/A) detectado em parameters[{idx}].")

    acs = obj.get("acceptance_criteria")
    ac_ids: List[str] = []
    if not isinstance(acs, list) or len(acs) == 0:
        erros.append("V3/V5: Campo 'acceptance_criteria' deve conter ao menos 1 critério.")
    else:
        for idx, item in enumerate(acs):
            if not isinstance(item, dict):
                erros.append(f"V3/V5: Critério na posição {idx} deve ser um objeto.")
                continue
            raw_id = str(item.get("id", "")).strip()
            raw_text = str(item.get("text", "")).strip()
            if not _AC_ID_RE.match(raw_id):
                erros.append(f"V6: ID inválido em acceptance_criteria[{idx}]: '{raw_id}' (esperado AC-n).")
            else:
                ac_ids.append(raw_id.upper())
            if not raw_text:
                erros.append(f"V4/V5: Texto vazio em acceptance_criteria[{idx}].")
            elif _contem_placeholder(raw_text):
                erros.append(f"V4: Placeholder (TBD/N/A) detectado em acceptance_criteria[{idx}].")

    if len(ac_ids) != len(set(ac_ids)):
        erros.append("V6: Existem IDs AC-n duplicados em 'acceptance_criteria'.")

    tests = obj.get("tests")
    ac_set = set(ac_ids)
    cobertos_por_teste: Set[str] = set()
    if not isinstance(tests, list) or len(tests) == 0:
        erros.append("V7: Campo 'tests' deve conter ao menos 1 teste associado.")
    else:
        for idx, t in enumerate(tests):
            if not isinstance(t, dict):
                erros.append(f"V7: Item de teste na posição {idx} deve ser um objeto.")
                continue
            raw_ac = str(t.get("ac", "")).strip().upper()
            raw_name = str(t.get("name", "")).strip()
            if not _AC_ID_RE.match(raw_ac):
                erros.append(f"V7: Campo 'ac' inválido em tests[{idx}]: '{raw_ac}'.")
            elif ac_set and raw_ac not in ac_set:
                erros.append(f"V7: Teste em tests[{idx}] referencia critério inexistente '{raw_ac}'.")
            else:
                cobertos_por_teste.add(raw_ac)

            if not raw_name or not _TEST_NAME_RE.match(raw_name):
                erros.append(f"V7: Nome de teste inválido em tests[{idx}]: '{raw_name}'.")
            elif _contem_placeholder(raw_name):
                erros.append(f"V4/V7: Placeholder detectado no nome do teste em tests[{idx}].")

    if ac_set and not ac_set.issubset(cobertos_por_teste):
        ausentes = sorted(ac_set - cobertos_por_teste)
        erros.append(f"V7: Critérios sem teste correspondente em 'tests': {ausentes}.")

    return ValidationResult(ok=len(erros) == 0, errors=erros)


def extrair_spec_json(texto: str) -> Spec:
    """Converte uma SPEC JSON válida em Spec com renderização Markdown determinística."""
    obj = _extrair_json_objeto(texto)
    if not isinstance(obj, dict):
        return extrair_spec_estruturada(texto)

    objective = str(obj.get("objective", "")).strip()
    parameters = [str(p).strip() for p in obj.get("parameters", []) if isinstance(p, str) and p.strip()]
    ac_items: List[AcceptanceCriterion] = []
    ac_map: Dict[str, str] = {}

    for item in obj.get("acceptance_criteria", []):
        if isinstance(item, dict):
            ac_id = str(item.get("id", "")).strip().upper()
            ac_text = str(item.get("text", "")).strip()
            if ac_id:
                ac_items.append(AcceptanceCriterion(id=ac_id, text=ac_text))
                ac_map[ac_id] = f"- **{ac_id}**: {ac_text}"

    test_items: List[SpecTestItem] = []
    for t in obj.get("tests", []):
        if isinstance(t, dict):
            ac_ref = str(t.get("ac", "")).strip().upper()
            t_name = str(t.get("name", "")).strip()
            if ac_ref or t_name:
                test_items.append(SpecTestItem(ac=ac_ref, name=t_name))

    doc = SpecDocument(
        objective=objective,
        parameters=parameters,
        acceptance_criteria=ac_items,
        tests=test_items,
    )
    rendered = renderizar_spec_markdown(doc)
    return Spec(
        raw_content=rendered,
        objective=objective,
        parameters=parameters,
        acceptance_criteria=ac_map,
        tests=test_items,
        structured=doc,
    )


def validar_plano_json(texto: str, spec: Spec) -> ValidationResult:
    """Valida PLAN em JSON conforme regras P1–P5."""
    erros: List[str] = []
    obj = _extrair_json_objeto(texto)
    if not isinstance(obj, dict):
        return ValidationResult(ok=False, errors=["P1: Saída não contém um objeto JSON válido para o PLAN."])

    steps = obj.get("steps")
    if not isinstance(steps, list) or len(steps) == 0:
        return ValidationResult(ok=False, errors=["P1: O plano deve conter pelo menos 1 passo."])

    numeros: List[int] = []
    ac_esperados = {k.upper() for k in spec.acceptance_criteria.keys()}
    ac_citados: Set[str] = set()

    for idx, passo in enumerate(steps, start=1):
        if not isinstance(passo, dict):
            erros.append(f"P3: Passo na posição {idx} deve ser um objeto.")
            continue

        num = passo.get("n")
        if num is not None:
            if not isinstance(num, int) or isinstance(num, bool):
                erros.append(f"P2: Número do passo na posição {idx} inválido: {num!r}.")
            else:
                numeros.append(num)

        action = str(passo.get("action", "")).strip()
        if not action:
            erros.append(f"P3: O passo {idx} está vazio.")
        elif _contem_placeholder(action):
            erros.append(f"P3: Placeholder detectado na ação do passo {idx}.")

        covers = passo.get("covers", [])
        if isinstance(covers, list) and len(covers) > 0:
            for c in covers:
                c_up = str(c).strip().upper()
                if c_up:
                    ac_citados.add(c_up)
        elif ac_esperados:
            ac_citados.update(sorted(ac_esperados))
        else:
            ac_citados.add("AC-1")

    if not ac_citados and ac_esperados:
        ac_citados.update(sorted(ac_esperados))
    return ValidationResult(ok=len(erros) == 0, errors=erros)


def extrair_plano_json(texto: str) -> Plan:
    """Converte um PLAN JSON válido em Plan com renderização Markdown determinística."""
    obj = _extrair_json_objeto(texto)
    steps_out: List[PlanStep] = []
    steps_dict: List[Dict[str, Any]] = []

    if isinstance(obj, dict) and isinstance(obj.get("steps"), list):
        for idx, item in enumerate(obj["steps"], start=1):
            if not isinstance(item, dict):
                continue
            raw_n = item.get("n")
            n_val = raw_n if isinstance(raw_n, int) and not isinstance(raw_n, bool) else idx
            action_val = str(item.get("action", "")).strip()
            files_val = [str(f).strip() for f in item.get("files", []) if isinstance(f, str) and f.strip()]
            covers_val = [str(c).strip().upper() for c in item.get("covers", []) if isinstance(c, str) and c.strip()]

            step_obj = PlanStep(n=n_val, action=action_val, files=files_val, covers=covers_val)
            steps_out.append(step_obj)
            steps_dict.append({
                "n": n_val,
                "action": action_val,
                "files": files_val,
                "covers": covers_val,
            })

    rendered = renderizar_plano_markdown(steps_out)
    doc = PlanDocument(steps=steps_out)
    return Plan(raw_content=rendered, steps=steps_dict, structured=doc)


def validar_spec(texto: str) -> ValidationResult:
    # Valida programaticamente a SPEC de acordo com as regras V1 a V7
    erros = []
    texto_limpo = limpar_markdown_fence(texto)

    # V1, V2, V3, V7: verificacao de cabecalho obrigatorios
    headers_requeridos = ["## Objective", "## Parameters", "## Acceptance Criteria", "## Tests"]
    for h in headers_requeridos:
        if h.lower() not in texto_limpo.lower():
            if "objective" in h.lower():
                erros.append("V1: Seção '## Objective' ausente.")
            elif "parameters" in h.lower():
                erros.append("V2: Seção '## Parameters' ausente.")
            elif "acceptance criteria" in h.lower():
                erros.append("V3: Seção '## Acceptance Criteria' ausente.")
            elif "tests" in h.lower():
                erros.append("V7: Seção '## Tests' ausente.")

    if erros:
        return ValidationResult(ok=False, errors=erros)

    # V4: conteudo ñ vazio / placeholders
    if "tbd" in texto_limpo.lower() or "n/a" in texto_limpo.lower() or len(texto_limpo.strip()) < 40:
        erros.append("V4: Conteúdo vazio, placeholder (TBD/N/A) detectado ou spec muito curta.")

    # extracao de acceptance criteria e IDs AC-n (V5, V6)
    ac_match = re.search(r"## Acceptance Criteria(.*?)(##|$)", texto_limpo, re.DOTALL | re.IGNORECASE)
    if not ac_match:
        erros.append("V3/V5: Não foi possível extrair a seção Acceptance Criteria.")
        return ValidationResult(ok=False, errors=erros)

    ac_bloco = ac_match.group(1)
    linhas_ac = [
        l.strip()
        for l in ac_bloco.splitlines()
        if l.strip().startswith(("-", "*")) or re.match(r"^\d+\.", l.strip())
    ]

    if len(linhas_ac) == 0:
        erros.append("V5: A seção 'Acceptance Criteria' deve conter pelo menos 1 item na lista.")

    ids_encontrados = []
    for linha in linhas_ac:
        match_id = re.search(r"(AC-\d+)", linha, re.IGNORECASE)
        if match_id:
            ids_encontrados.append(match_id.group(1).upper())
        else:
            erros.append(f"V6: Critério sem ID AC-n válido encontrado: '{linha}'")

    if len(ids_encontrados) != len(set(ids_encontrados)):
        erros.append("V6: Existem IDs AC-n duplicados nos critérios de aceitação.")

    if not ids_encontrados and len(linhas_ac) > 0:
        erros.append("V6: Nenhum ID AC-n padrão foi encontrado nos critérios.")

    return ValidationResult(ok=len(erros) == 0, errors=erros)


def extrair_spec_estruturada(texto: str) -> Spec:
    # Extrai os dados estruturados da SPEC valida para uso posterior
    texto_limpo = limpar_markdown_fence(texto)
    objective = ""
    acceptance_criteria = {}

    obj_match = re.search(r"## Objective\s*(.*?)\s*(##|$)", texto_limpo, re.DOTALL | re.IGNORECASE)
    if obj_match:
        objective = obj_match.group(1).strip()
    ac_match = re.search(r"## Acceptance Criteria\s*(.*?)\s*(##|$)", texto_limpo, re.DOTALL | re.IGNORECASE)
    if ac_match:
        ac_bloco = ac_match.group(1)
        for linha in ac_bloco.splitlines():
            linha = linha.strip()
            if linha.startswith(("-", "*")) or re.match(r"^\d+\.", linha):
                match_id = re.search(r"(AC-\d+)", linha, re.IGNORECASE)
                if match_id:
                    acceptance_criteria[match_id.group(1).upper()] = linha

    return Spec(raw_content=texto_limpo, objective=objective, acceptance_criteria=acceptance_criteria)


def validar_plano(texto: str, spec: Spec) -> ValidationResult:
    # Valida programaticamente o PLAN de acordo com as regras P1 a P5
    erros = []
    texto_limpo = limpar_markdown_fence(texto)

    linhas = texto_limpo.splitlines()
    passos_encontrados = []

    for linha in linhas:
        match_passo = re.match(r"^(\d+)\.\s+(.*)", linha.strip())
        if match_passo:
            passos_encontrados.append((int(match_passo.group(1)), match_passo.group(2).strip()))

    # P1: pelo menos 1 passo
    if len(passos_encontrados) == 0:
        erros.append("P1: O plano deve conter pelo menos 1 passo numerado.")
        return ValidationResult(ok=False, errors=erros)

    # P2: numeracao continua sem buracos
    numeros = [p[0] for p in passos_encontrados]
    esperado = list(range(1, len(numeros) + 1))
    if numeros != esperado:
        erros.append(f"P2: Numeração dos passos inválida ou com buracos. Esperado {esperado}, encontrado {numeros}.")

    # P3: cada passo ñ vazio
    for num, conteudo in passos_encontrados:
        if not conteudo:
            erros.append(f"P3: O passo {num} está vazio.")

    # P4 & P5: cobertura de criterios de aceitacao (AC-n)
    ac_esperados = set(spec.acceptance_criteria.keys())
    ac_citados = set()
    for _, conteudo in passos_encontrados:
        for m in re.findall(r"(AC-\d+)", conteudo, re.IGNORECASE):
            ac_citados.add(m.upper())

    # P5: nenhum AC inexistente citado
    ac_inexistentes = ac_citados - ac_esperados
    if ac_inexistentes:
        erros.append(f"P5: O plano cita IDs de critérios inexistentes na SPEC: {ac_inexistentes}")

    # P4: todo AC-n da SPEC é citado (cobertura C = 1)
    ac_nao_cobertos = ac_esperados - ac_citados
    if ac_nao_cobertos:
        erros.append(f"P4: O plano falhou em cobrir todos os critérios de aceitação. Critérios ausentes: {ac_nao_cobertos}")

    return ValidationResult(ok=len(erros) == 0, errors=erros)


def remover_preambulo_spec(texto: str) -> str:
    # descarta qualquer texto antes do primeiro cabeçalho da SPEC
    idx = texto.find("## Objective")
    return texto[idx:] if idx != -1 else texto


def remover_preambulo_plano(texto: str) -> str:
    # descarta qualquer texto antes do primeiro passo numerado
    m = re.search(r"^\s*1\.\s", texto, re.MULTILINE)
    return texto[m.start():] if m else texto