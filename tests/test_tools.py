import pytest
from pathlib import Path
from src.tools.registry import ToolRegistry
from src.tools.sandbox import resolve_path
from src.tools.base import ToolResult

@pytest.fixture
def workspace(tmp_path):
    # Cria um diret temp isolado para servir como workspace nos testes
    return tmp_path

@pytest.fixture
def registry(workspace):
    # Instancia o ToolRegistry apontando para o workspace temporário
    return ToolRegistry(workspace)


def test_sandbox_security_escape(workspace):
    # Garante que tentativas de path traversal (fuga do workspace) bloqueiam o acesso
    with pytest.raises(Exception): resolve_path(workspace, "../fora_do_workspace.txt")


def test_write_and_read_roundtrip(registry):
    # Verifica se write_file grava corretamente e read_file recupera o conteúdo idêntico
    
    # escreve arquivo
    res_write = registry.dispatch("write_file", {
        "path": "src/exemplo.py",
        "content": "def soma(a, b):\n    return a + b\n"
    })
    assert res_write.ok is True
    assert res_write.meta.get("modifies_files") is True

    # le arquivo
    res_read = registry.dispatch("read_file", {"path": "src/exemplo.py"})
    assert res_read.ok is True
    assert "def soma(a, b):" in res_read.output
    assert res_read.meta.get("total_lines") == 2


def test_list_files(registry):
    # Verifica se a listagem de arq funciona e mapeia diret criados
    registry.dispatch("write_file", {"path": "modulo/arquivo1.py", "content": "# 1"})
    registry.dispatch("write_file", {"path": "modulo/arquivo2.py", "content": "# 2"})

    res = registry.dispatch("list_files", {"path": "."})
    assert res.ok is True
    assert "modulo/arquivo1.py" in res.output
    assert "modulo/arquivo2.py" in res.output


def test_dispatch_never_propagates_exceptions(registry):
    #Garante que o dispatch encapsula qualquer erro e NUNCA propaga excecoes brutas

    # 1. tool inexistente
    res_inexistente = registry.dispatch("ferramenta_fantasma", {})
    assert res_inexistente.ok is False
    assert isinstance(res_inexistente, ToolResult)
    assert "desconhecida" in res_inexistente.error

    # 2. arg invalidos / tipo errado ou arquivo inexistente via tool
    res_erro_arquivo = registry.dispatch("read_file", {"path": "arquivo_que_nao_existe.txt"})
    assert res_erro_arquivo.ok is False
    assert isinstance(res_erro_arquivo, ToolResult)


def test_export_formats(registry):
    # Verifica se os formatos exportados para a LLM estão integros e compativeis
    openai_tools = registry.export_openai_tools()
    assert isinstance(openai_tools, list)
    assert len(openai_tools) >= 4
    
    # valida estrutura OpenAI
    for t in openai_tools:
        assert t["type"] == "function"
        assert "name" in t["function"]
        assert "parameters" in t["function"]

    prompt_text = registry.export_prompt_text()
    assert isinstance(prompt_text, str)
    assert "read_file" in prompt_text
    assert "write_file" in prompt_text
    assert "run_tests" in prompt_text
    assert "list_files" in prompt_text