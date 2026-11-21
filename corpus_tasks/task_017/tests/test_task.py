from src.parsers import parse_csv_row

def test_parse_csv_row():
    fn = parse_csv_row
    assert fn(' a , b , c ') == ['a', 'b', 'c'] and fn('') == []
