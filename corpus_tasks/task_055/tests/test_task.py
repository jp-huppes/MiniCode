from src.buggy_ops import inverter_lista

def test_inverter_lista():
    fn = inverter_lista
    orig = [1, 2, 3]; assert fn(orig) == [3, 2, 1] and orig == [1, 2, 3]
