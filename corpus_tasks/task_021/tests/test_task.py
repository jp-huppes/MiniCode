from src.parsers import parse_bearer_token

def test_parse_bearer_token():
    fn = parse_bearer_token
    assert fn('Bearer abc123') == 'abc123' and fn('Basic xyz') is None
