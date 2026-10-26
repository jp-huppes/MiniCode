import re
from typing import List, Set
from src.spec.models import ValidationResult, Spec

def limpar_markdown_fence(texto: str) -> str:
    # Remove blocos de cercas markdown (ex: ```markdown ... ```) caso presente.
    texto = texto.strip()
    if texto.startswith("```"):
        linhas = texto.splitlines()
        if linhas[0].startswith("```"): linhas = linhas[1:]
        if linhas and linhas[-1].startswith("```"): linhas = linhas[:-1]
        texto = "\n".join(linhas).strip()
    return texto

def validar_spec(texto: str) -> ValidationResult:
    # Valida programaticamente a SPEC de acordo com as regras V1 a V7
    erros = []; texto_limpo = limpar_markdown_fence(texto)
    
    # V1, V2, V3, V7: verificacao de cabecalho obrigatorios
    headers_requeridos = ["## Objective", "## Parameters", "## Acceptance Criteria", "## Tests"]
    for h in headers_requeridos:
        if h.lower() not in texto_limpo.lower():
            if "objective" in h.lower(): erros.append("V1: Seção '## Objective' ausente.")
            elif "parameters" in h.lower(): erros.append("V2: Seção '## Parameters' ausente.")
            elif "acceptance criteria" in h.lower(): erros.append("V3: Seção '## Acceptance Criteria' ausente.")
            elif "tests" in h.lower(): erros.append("V7: Seção '## Tests' ausente.")

    if erros: return ValidationResult(ok=False, errors=erros)

    # V4: conteudo ñ vazio / placeholders
    if "tbd" in texto_limpo.lower() or "n/a" in texto_limpo.lower() or len(texto_limpo.strip()) < 40:
        erros.append("V4: Conteúdo vazio, placeholder (TBD/N/A) detectado ou spec muito curta.")

    # extracao de acceptance criteria e IDs AC-n (V5, V6)
    ac_match = re.search(r"## Acceptance Criteria(.*?)(##|$)", texto_limpo, re.DOTALL | re.IGNORECASE)
    if not ac_match:
        erros.append("V3/V5: Não foi possível extrair a seção Acceptance Criteria.")
        return ValidationResult(ok=False, errors=erros)
    
    ac_bloco = ac_match.group(1)
    linhas_ac = [l.strip() for l in ac_bloco.splitlines() if l.strip().startswith(("-", "*")) or re.match(r"^\d+\.", l.strip())]
    
    if len(linhas_ac) == 0: erros.append("V5: A seção 'Acceptance Criteria' deve conter pelo menos 1 item na lista.")
    
    ids_encontrados = []
    for linha in linhas_ac:
        match_id = re.search(r"(AC-\d+)", linha, re.IGNORECASE)
        if match_id: ids_encontrados.append(match_id.group(1).upper())
        else: erros.append(f"V6: Critério sem ID AC-n válido encontrado: '{linha}'")
    
    if len(ids_encontrados) != len(set(ids_encontrados)):
        erros.append("V6: Existem IDs AC-n duplicados nos critérios de aceitação.")

    if not ids_encontrados and len(linhas_ac) > 0:
        erros.append("V6: Nenhum ID AC-n padrão foi encontrado nos critérios.")

    return ValidationResult(ok=len(erros) == 0, errors=erros)

def extrair_spec_estruturada(texto: str) -> Spec:
    # Extrai os dados estruturados da SPEC valida para uso posterior
    texto_limpo = limpar_markdown_fence(texto)
    objective = ""; acceptance_criteria = {}

    obj_match = re.search(r"## Objective\s*(.*?)\s*(##|$)", texto_limpo, re.DOTALL | re.IGNORECASE)
    if obj_match: objective = obj_match.group(1).strip()
    ac_match = re.search(r"## Acceptance Criteria\s*(.*?)\s*(##|$)", texto_limpo, re.DOTALL | re.IGNORECASE)
    if ac_match:
        ac_bloco = ac_match.group(1)
        for linha in ac_bloco.splitlines():
            linha = linha.strip()
            if linha.startswith(("-", "*")) or re.match(r"^\d+\.", linha):
                match_id = re.search(r"(AC-\d+)", linha, re.IGNORECASE)
                if match_id: acceptance_criteria[match_id.group(1).upper()] = linha

    return Spec(raw_content=texto_limpo, objective=objective, acceptance_criteria=acceptance_criteria)

def validar_plano(texto: str, spec: Spec) -> ValidationResult:
    # Valida programaticamente o PLAN de acordo com as regras P1 a P5
    erros = []; texto_limpo = limpar_markdown_fence(texto)
    
    linhas = texto_limpo.splitlines(); passos_encontrados = []
    
    for linha in linhas:
        match_passo = re.match(r"^(\d+)\.\s+(.*)", linha.strip())
        if match_passo: passos_encontrados.append((int(match_passo.group(1)), match_passo.group(2).strip()))

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
        if not conteudo: erros.append(f"P3: O passo {num} está vazio.")

    # P4 & P5: cobertura de criterios de aceitacao (AC-n)
    ac_esperados = set(spec.acceptance_criteria.keys()); ac_citados = set()
    for _, conteudo in passos_encontrados:
        for m in re.findall(r"(AC-\d+)", conteudo, re.IGNORECASE):
            ac_citados.add(m.upper())

    # P5: nenhum AC inexistente citado
    ac_inexistentes = ac_citados - ac_esperados
    if ac_inexistentes: erros.append(f"P5: O plano cita IDs de critérios inexistentes na SPEC: {ac_inexistentes}")

    # P4: todo AC-n da SPEC é citado (cobertura C = 1)
    ac_nao_cobertos = ac_esperados - ac_citados
    if ac_nao_cobertos: erros.append(f"P4: O plano falhou em cobrir todos os critérios de aceitação. Critérios ausentes: {ac_nao_cobertos}")

    return ValidationResult(ok=len(erros) == 0, errors=erros)

def remover_preambulo_spec(texto: str) -> str:
    # descarta qualquer texto antes do primeiro cabeçalho da SPEC
    idx = texto.find("## Objective")
    return texto[idx:] if idx != -1 else texto

def remover_preambulo_plano(texto: str) -> str:
    # descarta qualquer texto antes do primeiro passo numerado
    m = re.search(r"^\s*1\.\s", texto, re.MULTILINE)
    return texto[m.start():] if m else texto