import sys
import json
from pathlib import Path
import pytest

from src.cli.main import main, parse_args, RunResult, run_task
from src.cli.renderer import JsonRenderer, QuietRenderer, HumanRenderer
from src.agent.state import State

def test_parse_args_defaults():
    args = parse_args(["Criar calculadora"])
    assert args.task == "Criar calculadora"
    assert args.project == "."
    assert args.max_iterations == 25
    assert args.max_repair_attempts == 3
    assert args.test_cmd == "pytest"
    assert not args.json
    assert not args.quiet
    assert not args.verbose
    assert not args.no_color

def test_parse_args_custom_flags():
    args = parse_args([
        "Refatorar código",
        "--project", "meu_projeto",
        "--max-iterations", "10",
        "--max-repair-attempts", "2",
        "--test-cmd", "pytest -q",
        "--json",
        "--verbose",
        "--no-color"
    ])
    assert args.task == "Refatorar código"
    assert args.project == "meu_projeto"
    assert args.max_iterations == 10
    assert args.max_repair_attempts == 2
    assert args.test_cmd == "pytest -q"
    assert args.json is True
    assert args.verbose is True
    assert args.no_color is True

def test_main_usage_error_empty_task(tmp_path):
    # Tarefa preenchida apenas com espaços em branco deve retornar exit code 2 (erro de uso)
    code = main(["   ", "--project", str(tmp_path)])
    assert code == 2

def test_main_json_without_task_raises_error(tmp_path):
    # Flag --json exigida sem tarefa no modo direto deve retornar exit code 2
    code = main(["--json", "--project", str(tmp_path)])
    assert code == 2

def test_main_invalid_project_dir_file(tmp_path):
    # Passar um arquivo comum como --project em vez de diretorio deve retornar exit code 2
    arquivo_falso = tmp_path / "arquivo.txt"
    arquivo_falso.write_text("conteudo")
    code = main(["Minha tarefa", "--project", str(arquivo_falso)])
    assert code == 2

def test_json_renderer_output(capsys, tmp_path):
    renderer = JsonRenderer()
    result = RunResult(
        final_state="DONE",
        success=True,
        run_dir=tmp_path / ".minicode/runs/run_abc123",
        error=None
    )
    renderer.finish(result)
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["final_state"] == "DONE"
    assert data["success"] is True
    assert "run_abc123" in data["run_dir"]

def test_quiet_renderer_output(capsys, tmp_path):
    renderer = QuietRenderer()
    result = RunResult(
        final_state="DONE",
        success=True,
        run_dir=tmp_path
    )
    renderer.finish(result)
    captured = capsys.readouterr()
    assert "DONE" in captured.out.strip()

def test_human_renderer_events(capsys):
    renderer = HumanRenderer(verbose=True, no_color=True)
    renderer.handle("state_changed", {"old": "RECEIVED", "new": "SPECIFIED"})
    renderer.handle("spec_ready", {"path": "specification.md", "content": "# Spec"})
    renderer.handle("action", {"tool": "write_file", "args": {"path": "calc.py"}, "error": None})
    renderer.handle("tests_run", {"command": "pytest", "passed": True, "output": ""})
    renderer.handle("repair_attempt", {"k": 1, "K": 3})
    
    captured = capsys.readouterr()
    assert "[ESTADO]" in captured.out
    assert "Especificação gerada" in captured.out
    assert "write_file" in captured.out
    assert "Testes" in captured.out

def test_repl_exit_command(monkeypatch):
    # Simula o user digitando '/exit' no REPL
    inputs = iter(["/exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    
    from src.cli.main import repl
    config = {"project_dir": Path(".")}
    renderer = QuietRenderer()
    code = repl(config, renderer)
    assert code == 0

def test_repl_help_command_and_empty_lines(monkeypatch):
    # Simula entrada vazia, comando /help e em seguida /exit
    inputs = iter(["   ", "/help", "/exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    
    from src.cli.main import repl
    config = {"project_dir": Path(".")}
    renderer = QuietRenderer()
    code = repl(config, renderer)
    assert code == 0

def test_repl_keyboard_interrupt(monkeypatch):
    # Simula interrupção Ctrl+C (KeyboardInterrupt) no prompt do REPL
    def mock_input(prompt=""):
        raise KeyboardInterrupt()
    monkeypatch.setattr("builtins.input", mock_input)
    
    from src.cli.main import repl
    config = {"project_dir": Path(".")}
    renderer = QuietRenderer()
    code = repl(config, renderer)
    assert code == 130