from src.parsers import parse_query_string

def test_parse_query_string():
    fn = parse_query_string
    assert fn('?a=1&b=2') == {'a': '1', 'b': '2'} and fn('k=v') == {'k': 'v'}
