from src.buggy_ops import limitar_tamanho

def test_limitar_tamanho():
    fn = limitar_tamanho
    assert fn('abcdef', 3) == 'abc' and fn('hi', 5) == 'hi'
