from src.buggy_ops import dobrar_valores

def test_dobrar_valores():
    fn = dobrar_valores
    assert fn([1, 3, 5]) == [2, 6, 10]
