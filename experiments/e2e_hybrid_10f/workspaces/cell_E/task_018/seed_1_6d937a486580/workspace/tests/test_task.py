from src.parsers import parse_int_list

def test_parse_int_list():
    fn = parse_int_list
    assert fn('10, -2, 30') == [10, -2, 30] and fn('') == []
