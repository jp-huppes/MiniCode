import ast
import json
from pathlib import Path
from types import SimpleNamespace

from scripts import run_eval
from scripts.eval_baselines import DEFAULT_SYSTEM_PROMPT as EVAL_SYSTEM_PROMPT
from src.agent.orchestrator import AgentOrchestrator, montar_system_prompt
from src.dataset.build_sft_dataset import DEFAULT_SYSTEM_PROMPT as SFT_SYSTEM_PROMPT


def test_prompt_parity_enxuto():
    esperado = montar_system_prompt(Path("."), catalogo_enxuto=True)
    assert esperado == SFT_SYSTEM_PROMPT
    assert esperado == EVAL_SYSTEM_PROMPT


def test_notebook_train_config_prompt_parity():
    nb_path = Path("notebooks/train.ipynb")
    if not nb_path.exists():
        return
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    cell5_src = "".join(nb["cells"][5]["source"])
    tree = ast.parse(cell5_src)
    found_prompt = None
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "system_prompt":
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                found_prompt = node.value.value
    assert found_prompt == montar_system_prompt(Path("."), catalogo_enxuto=True)


def test_orchestrator_catalogo_enxuto_override_and_default(tmp_path):
    c_none = SimpleNamespace(constrained_mode="none")
    c_json = SimpleNamespace(constrained_mode="json_schema")

    # Padrão None preserva comportamento original baseado em constrained_mode
    assert AgentOrchestrator(tmp_path, tmp_path, c_none).system_prompt == montar_system_prompt(tmp_path, catalogo_enxuto=False)
    assert AgentOrchestrator(tmp_path, tmp_path, c_json).system_prompt == montar_system_prompt(tmp_path, catalogo_enxuto=True)

    # Override explícito True força catálogo enxuto mesmo com constrained_mode="none"
    assert AgentOrchestrator(tmp_path, tmp_path, c_none, catalogo_enxuto=True).system_prompt == SFT_SYSTEM_PROMPT


def test_run_eval_e2e_repassa_catalogo_enxuto(tmp_path, monkeypatch):
    prompts_capturados = []

    class DummyClient:
        constrained_mode = "none"

    monkeypatch.setattr(run_eval, "_construir_cliente", lambda temperature, constrained_mode: DummyClient())

    def fake_run(self, task, plan):
        prompts_capturados.append(self.system_prompt)
        return "DONE"

    monkeypatch.setattr(AgentOrchestrator, "run", fake_run)

    tasks = [{"task_id": "t1", "task": "Implementar soma"}]
    run_eval.rodar_celula(
        cell_name="c1",
        tasks=tasks,
        repeats=1,
        base_dir=tmp_path,
        temperature=0.0,
        constrained_mode="none",
        catalogo_enxuto=True,
    )
    assert len(prompts_capturados) == 1
    assert prompts_capturados[0] == SFT_SYSTEM_PROMPT
