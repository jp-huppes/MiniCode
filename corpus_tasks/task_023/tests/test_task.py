from src.parsers import parse_rgb_hex

def test_parse_rgb_hex():
    fn = parse_rgb_hex
    assert fn('#FF8800') == (255, 136, 0) and fn('000000') == (0, 0, 0)
