from src.pure_ops import lcm_val

def test_lcm_val():
    fn = lcm_val
    assert fn(4, 6) == 12 and fn(3, 5) == 15 and fn(0, 5) == 0
