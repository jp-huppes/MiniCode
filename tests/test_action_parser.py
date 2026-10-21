import pytest
from src.llm.action_parser import parse_acao_json
from src.llm.errors import ParseError

FERRAMENTAS_DISPONIVEIS = ["read_file", "write_file", "list_files", "run_tests"]

def test_parser_pure_json():
    # Valida a extração de JSON puro (Camada 1)
    texto = '{"action": "read_file", "args": {"path": "main.py"}}'
    res = parse_acao_json(texto, FERRAMENTAS_DISPONIVEIS)
    assert res["action"] == "read_file"
    assert res["args"]["path"] == "main.py"

def test_parser_markdown_fence_json():
    # Valida a extracao dentro de cercas markdown com lang json (Camada 2)
    texto = 'Aqui está a ação solicitada:\n```json\n{"action": "write_file", "args": {"path": "test.py", "content": "print(1)"}}\n```'
    res = parse_acao_json(texto, FERRAMENTAS_DISPONIVEIS)
    assert res["action"] == "write_file"
    assert res["args"]["path"] == "test.py"

def test_parser_markdown_fence_no_lang():
    # Valida a extracao dentro de cercas markdown sem linguagem especificada
    texto = '```\n{"action": "list_files", "args": {}}\n```'
    res = parse_acao_json(texto, FERRAMENTAS_DISPONIVEIS)
    assert res["action"] == "list_files"

def test_parser_text_before_and_after():
    # Valida a extracao de objeto JSON balanceado rodeado por texto livre (Camada 3)
    texto = 'Estou analisando o projeto...\n{"action": "run_tests", "args": {"path": "tests/"}}\nAguarde a execução.'
    res = parse_acao_json(texto, FERRAMENTAS_DISPONIVEIS)
    assert res["action"] == "run_tests"
    assert res["args"]["path"] == "tests/"

def test_parser_thought_key_included():
    #Valida se aceita e preserva opcionalmente a chave 'thought' de raciociono do Qwen
    texto = '{\n  "thought": "Preciso ler o arquivo de configuração",\n  "action": "read_file",\n  "args": {"path": "pyproject.toml"}\n}'
    res = parse_acao_json(texto, FERRAMENTAS_DISPONIVEIS)
    assert res["action"] == "read_file"
    assert res["thought"] == "Preciso ler o arquivo de configuração"

def test_parser_invalid_tool_raises_parse_error():
    # Valida que acoes fora das ferramentas registradas geram ParseError
    texto = '{"action": "ferramenta_fantasma", "args": {}}'
    with pytest.raises(ParseError) as exc_info: parse_acao_json(texto, FERRAMENTAS_DISPONIVEIS)
    assert "não está nas ferramentas disponíveis" in str(exc_info.value)

def test_parser_missing_fields_raises_parse_error():
    # Valida que JSONs sem os campos obrigatorio 'action' ou 'args' falham
    texto = '{"action": "read_file"}'  # faltando args
    with pytest.raises(ParseError): parse_acao_json(texto, FERRAMENTAS_DISPONIVEIS)

def test_parser_no_json_at_all_raises_parse_error():
    # Valida que textos sem nenhum JSON válido geram ParseError
    texto = 'Olá, sou um modelo de linguagem e decidi responder apenas em texto corrido.'
    with pytest.raises(ParseError): parse_acao_json(texto, FERRAMENTAS_DISPONIVEIS)