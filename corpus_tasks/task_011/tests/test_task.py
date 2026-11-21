from src.pure_ops import is_palindrome

def test_is_palindrome():
    fn = is_palindrome
    assert fn('A man a plan a canal Panama') is True and fn('python') is False
