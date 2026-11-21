from src.parsers import parse_path_parts

def test_parse_path_parts():
    fn = parse_path_parts
    assert fn('a/b/c.py') == ['a', 'b', 'c.py'] and fn('/x/y/') == ['x', 'y']
