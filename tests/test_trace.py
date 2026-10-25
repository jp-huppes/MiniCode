import json
from pathlib import Path
from src.trace.logger import RunLogger
from src.trace.state import RunState

def test_run_logger_and_state(tmp_path):
    # Testa a criação da run, atalho 'latest', actions.jsonl e escrita atômica do state.json
    logger = RunLogger(tmp_path, "Tarefa de teste")
    assert logger.run_dir.exists()
    assert (tmp_path / ".minicode" / "latest").exists()

    # testa log de acao com truncamento e sanitização
    logger.log_action(
        state="IMPLEMENTING",
        tool="write_file",
        args={"path": "calc.py", "content": "print('hello')"},
        result="ok",
        duration_ms=45
    )
    
    actions_content = logger.actions_path.read_text(encoding="utf-8")
    assert "write_file" in actions_content
    assert "sha256" in actions_content  # verifica sanitização de conteudo em write_file
    logger.close()

    # Testa RunState e persistencia
    state = RunState(logger.run_id, "Tarefa de teste", logger.run_dir)
    state.record_transition("RECEIVED", "SPECIFIED")
    state.record_llm_call(prompt_tokens=100, completion_tokens=50)
    state.record_file_changed("calc.py")
    state.finalize("DONE", failure_reason=None)

    assert state.state_path.exists()
    state_data = json.loads(state.state_path.read_text(encoding="utf-8"))
    
    assert state_data["run_id"] == logger.run_id
    assert state_data["final_state"] == "DONE"
    assert state_data["llm_calls"] == 1
    assert state_data["tokens"]["total"] == 150
    assert "calc.py" in state_data["files_changed"]
    assert len(state_data["transitions"]) == 1