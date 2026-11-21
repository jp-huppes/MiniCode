from src.parsers import parse_semicolon_pairs

def test_parse_semicolon_pairs():
    fn = parse_semicolon_pairs
    assert fn('a=1;b=2;') == {'a': '1', 'b': '2'}
