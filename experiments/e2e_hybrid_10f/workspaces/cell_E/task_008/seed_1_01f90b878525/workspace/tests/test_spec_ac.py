from src.pure_ops import fahrenheit_to_celsius
def test_ac_1_fahrenheit_to_celsius_returns_zero_for_32():
    assert abs(fahrenheit_to_celsius(32) - 0.0) < 1e-9