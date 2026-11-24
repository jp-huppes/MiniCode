from src.pure_ops import is_even

def test_ac_1_is_even_returns_true_for_even_numbers():
    assert is_even(2) is True
    assert is_even(4) is True
    assert is_even(6) is True