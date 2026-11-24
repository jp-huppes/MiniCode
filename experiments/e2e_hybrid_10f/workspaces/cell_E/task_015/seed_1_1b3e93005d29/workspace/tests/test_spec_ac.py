from src.pure_ops import safe_div
def test_ac_1_division_by_non_zero():
    assert safe_div(10, 2) == 5

def test_ac_2_division_by_zero_returns_default():
    assert safe_div(10, 0, default=1.0) == 1.0