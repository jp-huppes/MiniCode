from src.parsers import parse_time_hhmm

def test_parse_time_hhmm():
    fn = parse_time_hhmm
    assert fn('02:30') == (2, 30) and fn('14:05') == (14, 5)
