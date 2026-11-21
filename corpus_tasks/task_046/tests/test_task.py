from src.buggy_ops import soma_lista

def test_soma_lista():
    fn = soma_lista
    assert fn([1, 2, 3]) == 6 and fn([]) == 0
