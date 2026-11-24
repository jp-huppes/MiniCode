from src.pure_ops import safe_div

def test_ac_1():
    assert safe_div(10, 2) == 5.0
    assert safe_div(10, 0) is None
from src.pure_ops import safe_div

def test_ac_1():
    assert safe_div(10, 2) == 5.0

def test_ac_2():
    assert safe_div(10, 0) is None