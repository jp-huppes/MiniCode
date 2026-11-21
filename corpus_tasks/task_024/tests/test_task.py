from src.parsers import parse_bool_flag

def test_parse_bool_flag():
    fn = parse_bool_flag
    assert fn('TRUE') is True and fn('off') is False
