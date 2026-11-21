from src.buggy_ops import eh_positivo

def test_eh_positivo():
    fn = eh_positivo
    assert fn(5) is True and fn(0) is False and fn(-2) is False
