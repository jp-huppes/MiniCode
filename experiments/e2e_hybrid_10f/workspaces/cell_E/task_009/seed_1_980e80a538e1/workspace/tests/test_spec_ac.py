from src.pure_ops import count_vowels

def test_ac_1_count_vowels_returns_correct_number_of_vowels():
    assert count_vowels('Abacate') == 4
    assert count_vowels('xyz') == 0