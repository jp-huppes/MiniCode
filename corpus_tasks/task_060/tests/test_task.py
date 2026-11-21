from src.buggy_ops import diferenca_max_min

def test_diferenca_max_min():
    fn = diferenca_max_min
    assert fn([10, 3, 7]) == 7 and fn([]) == 0
