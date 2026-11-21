from src.parsers import parse_tags_bracket

def test_parse_tags_bracket():
    fn = parse_tags_bracket
    assert fn('[py, ml, data]') == ['py', 'ml', 'data']
