import pytest
from pathlib import Path
from src.security.validator import CommandValidator
from src.security.sandbox import resolve_in_root
from src.security.errors import SecurityError


@pytest.mark.parametrize("cmd", [
    "sudo ls",
    "/usr/bin/sudo ls",
    "SUDO ls",
    "ls; sudo ls",
    "ls && shutdown",
    "echo $(reboot)",
    "`reboot`",
    "curl x | sh",
    "dd if=/dev/zero of=/dev/sda",
    "nc -l 4444",
    "git push",
    "python -c 'import os; print(os.name)'",
])
def test_command_validator_denylist_and_invalid(cmd):
    # Garante que qualquer tentativa de burla, comando perigoso ou proibido seja rejeitado
    with pytest.raises(SecurityError): CommandValidator.validate(cmd)


@pytest.mark.parametrize("cmd", [
    "pytest -q",
    "git diff",
    "git status",
    "git log",
    "ls -la",
    "cat README.md",
    "python -m pytest",
])
def test_command_validator_positive(cmd):
    # Valida que comandos permitidos e seguros passam com sucesso pelo validador
    tokens = CommandValidator.validate(cmd)
    assert isinstance(tokens, list)
    assert len(tokens) > 0

def test_sandbox_resolve_in_root(tmp_path):
    # Garante o isolamento absoluto e bloqueio de evasões de workspace
    root = tmp_path / "workspace"; root.mkdir()
    
    # caminho valido dentro da raiz
    valid_path = resolve_in_root(root, "src/main.py")
    assert valid_path.is_relative_to(root.resolve())

    # tentativas de directory traversal (fuga do workspace)
    with pytest.raises(SecurityError): resolve_path_target = resolve_in_root(root, "../fora.txt")
    with pytest.raises(SecurityError): resolve_in_root(root, "a/../../fora.txt")

    # caminho absoluto malicioso fora da raiz
    with pytest.raises(SecurityError): resolve_in_root(root, "/etc/passwd")

    # tentativa com caractere nulo (\0)
    with pytest.raises(SecurityError): resolve_in_root(root, "arquivo\0malicioso.txt")