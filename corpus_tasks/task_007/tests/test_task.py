from src.pure_ops import celsius_to_fahrenheit

def test_celsius_to_fahrenheit():
    fn = celsius_to_fahrenheit
    assert abs(fn(0) - 32.0) < 1e-9 and abs(fn(100) - 212.0) < 1e-9
