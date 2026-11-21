from src.buggy_ops import contar_maiores

def test_contar_maiores():
    fn = contar_maiores
    assert fn([1, 5, 5, 8], 5) == 1
