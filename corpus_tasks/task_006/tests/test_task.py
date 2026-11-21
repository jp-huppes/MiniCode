from src.pure_ops import is_prime

def test_is_prime():
    fn = is_prime
    assert fn(2) is True and fn(13) is True and fn(1) is False and fn(9) is False
