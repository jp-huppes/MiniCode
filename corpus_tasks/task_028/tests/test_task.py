from src.parsers import parse_simple_math

def test_parse_simple_math():
    fn = parse_simple_math
    assert fn('12 + 8') == 20 and fn('10 - 3') == 7
