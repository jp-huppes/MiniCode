from src.pure_ops import fahrenheit_to_celsius
def test_ac_1():
    assert fahrenheit_to_celsius(32) == 0
    assert fahrenheit_to_celsius(212) == 100