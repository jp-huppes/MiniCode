from src.pure_ops import count_vowels

def test_count_vowels():
    fn = count_vowels
    assert fn('Abacate') == 4 and fn('xyz') == 0
