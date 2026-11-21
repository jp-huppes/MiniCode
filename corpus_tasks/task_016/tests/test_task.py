from src.parsers import parse_kv_line

def test_parse_kv_line():
    fn = parse_kv_line
    assert fn('host=localhost port=8080') == {'host': 'localhost', 'port': '8080'}
