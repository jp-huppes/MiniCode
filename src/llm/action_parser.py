import json
import re
from typing import Dict, Any, List
from src.llm.errors import ParseError

def extrair_camada_1_inteiro(texto: str) -> str:
    return texto.strip()

def extrair_camada_2_cerca_markdown(texto: str) -> str:
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", texto, re.IGNORECASE)
    if match: return match.group(1).strip()
    return ""

def extrair_camada_3_objeto_balanceado(texto: str) -> str:
    # Varre do primeiro '{' contando chaves para isolar o objeto JSON
    start = texto.find('{')
    if start == -1: return ""
    
    depth = 0; in_string = False; escape = False
    
    for i in range(start, len(texto)):
        char = texto[i]
        if escape:
            escape = False
            continue
        if char == '\\':
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if not in_string:
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    return texto[start:i+1]
    return ""

def parse_acao_json(texto: str, ferramentas_disponiveis: List[str]) -> Dict[str, Any]:
    """
    Executa o parser em camadas para extrair e validar o JSON de acao do agente, permitindo opcionalmente chaves de raciocinios (como 'thought') e validando o contrato contra
    as ferramentas disponiveis.
    """
    candidatos = [extrair_camada_1_inteiro(texto), extrair_camada_2_cerca_markdown(texto), extrair_camada_3_objeto_balanceado(texto)]

    ultimo_erro = None
    for cand in candidatos:
        if not cand: continue
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict) and "action" in obj and "args" in obj:
                acao = obj.get("action")
                if acao in ferramentas_disponiveis: return obj
                else: raise ParseError(f"Ação '{acao}' não está nas ferramentas disponíveis: {ferramentas_disponiveis}")
        except json.JSONDecodeError as e:
            ultimo_erro = e
            continue
        except ParseError as pe: raise pe

    raise ParseError(f"Falha ao extrair JSON válido de ação. Último erro: {ultimo_erro}. Texto bruto: {texto[:200]}...")