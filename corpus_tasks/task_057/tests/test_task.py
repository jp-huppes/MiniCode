from src.buggy_ops import juntar_nao_vazios

def test_juntar_nao_vazios():
    fn = juntar_nao_vazios
    assert fn(['a', '', 'b', ''], '-') == 'a-b'
