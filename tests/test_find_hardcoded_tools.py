import ast
from pathlib import Path
from scripts.find_hardcoded_tools import scan_file

def test_scan_file_detecta_hardcoded(tmp_path):
    py_file = tmp_path / "mock.py"
    py_file.write_text("LISTA = ['read_file', 'write_file']\n", encoding="utf-8")
    findings = scan_file(py_file)
    assert len(findings) == 1
    lineno, snippet, matched = findings[0]
    assert "read_file" in matched
    assert "write_file" in matched

def test_scan_file_ignora_inocentes(tmp_path):
    py_file = tmp_path / "mock_safe.py"
    py_file.write_text("LISTA = ['abacaxi', 'banana']\n", encoding="utf-8")
    findings = scan_file(py_file)
    assert len(findings) == 0
