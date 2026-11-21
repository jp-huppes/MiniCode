from src.pure_ops import factorial

def test_factorial():
    fn = factorial
    assert fn(0) == 1 and fn(1) == 1 and fn(5) == 120
