from src.pure_ops import dot_product

def test_dot_product():
    fn = dot_product
    assert fn([1, 2, 3], [4, 5, 6]) == 32 and fn([], []) == 0
