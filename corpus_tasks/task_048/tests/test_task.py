from src.buggy_ops import contar_ocorrencias

def test_contar_ocorrencias():
    fn = contar_ocorrencias
    assert fn([1, 2, 1, 3, 1], 1) == 3 and fn([], 5) == 0
