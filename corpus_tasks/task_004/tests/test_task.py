from src.pure_ops import gcd_val

def test_gcd_val():
    fn = gcd_val
    assert fn(12, 18) == 6 and fn(7, 5) == 1 and fn(0, 9) == 9
