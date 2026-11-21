from src.pure_ops import reverse_words

def test_reverse_words():
    fn = reverse_words
    assert fn('um dois tres') == 'tres dois um' and fn('ola') == 'ola'
