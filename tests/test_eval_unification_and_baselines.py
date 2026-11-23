import json

from scripts.eval_baselines import (
    build_few_shot_prefix,
    evaluate_predictions as eval_baseline_predictions,
    normalizar_args,
    paired_bootstrap_delta_ci,
    run_llm_baseline,
    run_synthetic_smoke_test,
    select_stratified_few_shot_examples,
)
from src.agent.context_manager import estimar_tokens
from src.eval.eval_level1 import (
    evaluate_predictions as eval_l1_predictions,
    is_schema_valid,
    normalize_args as l1_normalize_args,
)


def test_removeprefix_preserva_dotfiles():
    assert normalizar_args({"path": "./.gitignore"})["path"] == ".gitignore"
    assert normalizar_args({"path": "./src/app.py"})["path"] == "src/app.py"
    assert l1_normalize_args({"path": "./.gitignore"})["path"] == ".gitignore"
    assert l1_normalize_args({"path": "./src/app.py"})["path"] == "src/app.py"


def test_eval_level1_e_baselines_alinhados_ao_catalogo():
    # replace_text usa old_str/new_str (não old_text/new_text) e done não aceita summary
    ok_replace, parsed_replace = is_schema_valid(
        '{"thought": "ajustar", "tool": "replace_text", "args": {"path": "a.py", "old_str": "x=1", "new_str": "x=2"}}'
    )
    assert ok_replace is True
    assert parsed_replace == {"tool": "replace_text", "args": {"path": "a.py", "old_str": "x=1", "new_str": "x=2"}}

    bad_replace, _ = is_schema_valid(
        '{"tool": "replace_text", "args": {"path": "a.py", "old_text": "x=1", "new_text": "x=2"}}'
    )
    assert bad_replace is False

    ok_done, parsed_done = is_schema_valid('{"thought": "fim", "tool": "done", "args": {}}')
    assert ok_done is True
    assert parsed_done == {"tool": "done", "args": {}}

    bad_done, _ = is_schema_valid('{"tool": "done", "args": {"summary": "ok"}}')
    assert bad_done is False

    rows = [
        {
            "target": {"tool": "done", "args": {}},
            "prediction": '{"tool": "done", "args": {"summary": "invalido"}}',
        },
        {
            "target": {"tool": "replace_text", "args": {"path": "./.env.example", "old_str": "A", "new_str": "B"}},
            "prediction": '{"thought": "ok", "tool": "replace_text", "args": {"path": ".env.example", "old_str": "A", "new_str": "B"}}',
        },
    ]
    rep_l1 = eval_l1_predictions(rows)
    assert rep_l1["SchemaValid"]["count"] == 1
    assert rep_l1["Acc_args"]["count"] == 1

    base_ex = [
        {"task_id": "t1", "y": {"tool": "done", "args": {}}},
        {"task_id": "t2", "y": {"tool": "replace_text", "args": {"path": "./.env.example", "old_str": "A", "new_str": "B"}}},
    ]
    base_preds = [
        '{"tool": "done", "args": {"summary": "invalido"}}',
        '{"thought": "ok", "tool": "replace_text", "args": {"path": ".env.example", "old_str": "A", "new_str": "B"}}',
    ]
    rep_base = eval_baseline_predictions(base_ex, base_preds)
    assert rep_base["json_validity"] == 0.5
    assert rep_base["exact_action_acc"] == 0.5


def test_paired_bootstrap_agrupa_por_task_id():
    # Mesmo run_id, mas task_ids distintos -> bootstrap deve agrupar por task_id
    examples = [
        {"task_id": "task_1", "run_id": "same_run"},
        {"task_id": "task_2", "run_id": "same_run"},
        {"task_id": "task_3", "run_id": "same_run"},
        {"task_id": "task_4", "run_id": "same_run"},
    ]
    # Em 3 das 4 tarefas A vence B (1 vs 0), em 1 tarefa empatam (1 vs 1)
    correct_a = [1, 1, 1, 1]
    correct_b = [0, 0, 0, 1]
    res = paired_bootstrap_delta_ci(examples, correct_a, correct_b, n_resamples=500, seed=42)
    assert res["delta_mean"] > 0.5
    assert res["ci95_low"] > 0.0


def test_few_shot_estratificado_e_limite_tokens():
    train = [
        {"task_id": "t1", "x": "p1", "y": {"tool": "read_file", "args": {"path": "a.py"}}},
        {"task_id": "t1", "x": "p2", "y": {"tool": "read_file", "args": {"path": "b.py"}}},
        {"task_id": "t2", "x": "p3", "y": {"tool": "write_file", "args": {"path": "c.py", "content": "1"}}},
        {"task_id": "t3", "x": "p4", "y": {"tool": "done", "args": {}}},
    ]
    escolhidos = select_stratified_few_shot_examples(train, k=3, seed=42)
    tools = {ex["y"]["tool"] for ex in escolhidos}
    tasks = {ex["task_id"] for ex in escolhidos}
    assert tools == {"read_file", "write_file", "done"}
    assert tasks == {"t1", "t2", "t3"}

    # Limite apertado de tokens impede estouro da janela de contexto
    train_longo = [
        {"task_id": "t1", "x": "A" * 400, "y": {"tool": "read_file", "args": {"path": "a.py"}}},
        {"task_id": "t2", "x": "B" * 2000, "y": {"tool": "write_file", "args": {"path": "b.py", "content": "C" * 2000}}},
    ]
    prefixo = build_few_shot_prefix(train_longo, k=2, max_prompt_tokens=200, seed=42)
    assert estimar_tokens(prefixo) <= 200
    assert "A" * 400 in prefixo
    assert "B" * 2000 not in prefixo

    # run_llm_baseline também ajusta o prefixo se o exemplo de teste for grande
    msgs_vistas = []
    def fake_chat(msgs, temp):
        msgs_vistas.append(msgs)
        return '{"tool": "done", "args": {}}'

    run_llm_baseline(
        [{"task_id": "t_big", "x": "X" * 600}],
        chat_fn=fake_chat,
        few_shot_prefix=prefixo,
        system_prompt="SYS",
        max_input_tokens=220,
    )
    total_tokens = sum(estimar_tokens(m["content"]) for m in msgs_vistas[0])
    assert total_tokens <= 220


def test_synthetic_smoke_test_ok():
    res = run_synthetic_smoke_test(few_shot_k=2, resamples=100, seed=42)
    assert res["few_shot"]["json_validity"] == 1.0
    assert res["few_shot"]["exact_action_acc"] == 1.0
