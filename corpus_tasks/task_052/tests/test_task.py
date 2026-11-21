from src.buggy_ops import filtrar_pares

def test_filtrar_pares():
    fn = filtrar_pares
    assert fn([1, 2, 3, 4, 5, 6]) == [2, 4, 6]
