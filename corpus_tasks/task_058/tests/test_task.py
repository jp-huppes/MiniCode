from src.buggy_ops import absoluto_lista

def test_absoluto_lista():
    fn = absoluto_lista
    assert fn([-4, 2, -9]) == [4, 2, 9]
