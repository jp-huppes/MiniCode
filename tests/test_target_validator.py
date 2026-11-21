import pytest

from src.dataset.target_validator import TargetValidationResult, validar_exemplo_alvo


def test_exemplo_alvo_valido_com_thought_tool_e_args():
    alvo = {
        "thought": "Criar módulo de soma.",
        "tool": "write_file",
        "args": {
            "path": "src/calc.py",
            "content": "def soma(a: int, b: int) -> int:\n    return a + b\n",
        },
    }
    res = validar_exemplo_alvo(alvo)
    assert isinstance(res, TargetValidationResult)
    assert res.ok is True
    assert res.errors == []


def test_exemplo_alvo_valido_em_json_string():
    raw = '{"thought": "Ler arquivo", "tool": "read_file", "args": {"path": "README.md"}}'
    res = validar_exemplo_alvo(raw)
    assert res.ok is True
    assert res.errors == []


def test_rejeita_file_path_em_args_ou_raiz():
    alvo_em_args = {
        "thought": "Escrever arquivo",
        "tool": "write_file",
        "args": {
            "file_path": "src/calc.py",
            "content": "def soma(a, b): return a + b",
        },
    }
    res_args = validar_exemplo_alvo(alvo_em_args)
    assert res_args.ok is False
    assert any("file_path" in err for err in res_args.errors)

    alvo_na_raiz = {
        "file_path": "src/calc.py",
        "tool": "write_file",
        "args": {
            "path": "src/calc.py",
            "content": "def soma(a, b): return a + b",
        },
    }
    res_raiz = validar_exemplo_alvo(alvo_na_raiz)
    assert res_raiz.ok is False
    assert any("file_path" in err for err in res_raiz.errors)


def test_rejeita_quando_falta_tool_ou_usa_action():
    alvo_sem_tool = {
        "thought": "Chave errada",
        "action": "write_file",
        "args": {"path": "a.py", "content": "x = 1"},
    }
    res = validar_exemplo_alvo(alvo_sem_tool)
    assert res.ok is False
    assert len(res.errors) >= 1


@pytest.mark.parametrize(
    "alvo_invalido",
    [
        "",
        "texto livre sem json",
        {"thought": "x", "tool": "ferramenta_inexistente", "args": {}},
        {"thought": "x", "tool": "write_file", "args": {"path": "a.py", "content": 999}},
        {"thought": "x", "tool": "read_file", "args": {"content": "sem path"}},
    ],
)
def test_exemplos_invalidos_retornam_ok_false_e_lista_de_erros(alvo_invalido):
    res = validar_exemplo_alvo(alvo_invalido)
    assert res.ok is False
    assert isinstance(res.errors, list)
    assert len(res.errors) > 0


def test_passo_4_filtra_e_corrige_exemplos_alvo():
    exemplos_brutos = [
        {
            "thought": "Corrigir file_path para path",
            "tool": "write_file",
            "args": {"file_path": "src/ops.py", "content": "def dobro(x): return x * 2\n"},
        },
        {
            "thought": "Ferramenta inválida",
            "tool": "exec_shell",
            "args": {"cmd": "rm -rf /"},
        },
        {
            "thought": "Exemplo já válido",
            "tool": "read_file",
            "args": {"path": "src/ops.py"},
        },
    ]

    dataset_final = []
    descartados = []

    for ex in exemplos_brutos:
        check = validar_exemplo_alvo(ex)
        if check.ok:
            dataset_final.append(ex)
            continue

        # Tentativa simples de correção quando o único problema foi file_path em args
        if "args" in ex and isinstance(ex["args"], dict) and "file_path" in ex["args"]:
            corrigido = {
                **ex,
                "args": {
                    **{k: v for k, v in ex["args"].items() if k != "file_path"},
                    "path": ex["args"]["file_path"],
                },
            }
            if validar_exemplo_alvo(corrigido).ok:
                dataset_final.append(corrigido)
                continue

        descartados.append((ex, check.errors))

    assert len(dataset_final) == 2
    assert dataset_final[0]["args"]["path"] == "src/ops.py"
    assert "file_path" not in dataset_final[0]["args"]
    assert len(descartados) == 1