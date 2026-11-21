from src.buggy_ops import maior_elemento

def test_maior_elemento():
    fn = maior_elemento
    assert fn([-10, -3, -25]) == -3 and fn([4, 9, 1]) == 9
