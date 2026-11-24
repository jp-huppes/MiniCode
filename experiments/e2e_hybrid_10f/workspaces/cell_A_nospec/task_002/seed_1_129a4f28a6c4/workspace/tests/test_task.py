from src.pure_ops import is_even

def test_is_even():
    fn = is_even
    assert fn(2) is True and fn(3) is False and fn(0) is True
