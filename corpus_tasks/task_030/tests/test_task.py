from src.parsers import parse_score_line

def test_parse_score_line():
    fn = parse_score_line
    assert fn('ana:42') == ('ana', 42)
