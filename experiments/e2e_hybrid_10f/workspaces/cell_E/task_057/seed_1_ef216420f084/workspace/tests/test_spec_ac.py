

def test_ac_1_ignore_empty_strings():
    from src.buggy_ops import juntar_nao_vazios
    assert juntar_nao_vazios([], ',') == ''
    assert juntar_nao_vazios(['a', '', 'b'], ',') == 'a,b'


def test_ac_1_ignore_empty_strings():
    from src.buggy_ops import juntar_nao_vazios
    assert juntar_nao_vazios([], ',') == ''
    assert juntar_nao_vazios(['a', '', 'b'], ',') == 'a,b'
