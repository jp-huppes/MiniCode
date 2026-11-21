from src.buggy_ops import indice_alvo

def test_indice_alvo():
    fn = indice_alvo
    assert fn(['x', 'y', 'z'], 'y') == 1 and fn(['x', 'y'], 'w') == -1
