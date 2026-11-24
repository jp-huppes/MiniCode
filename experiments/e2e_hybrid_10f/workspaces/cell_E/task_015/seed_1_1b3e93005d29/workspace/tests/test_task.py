from src.pure_ops import safe_div

def test_safe_div():
    fn = safe_div
    assert fn(10, 2) == 5.0 and fn(10, 0, -1.0) == -1.0
