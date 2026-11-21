from src.pure_ops import fahrenheit_to_celsius

def test_fahrenheit_to_celsius():
    fn = fahrenheit_to_celsius
    assert abs(fn(32) - 0.0) < 1e-9 and abs(fn(212) - 100.0) < 1e-9
