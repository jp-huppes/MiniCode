from src.pure_ops import clamp

def test_clamp():
    fn = clamp
    assert fn(5, 0, 10) == 5 and fn(-3, 0, 10) == 0 and fn(15, 0, 10) == 10
