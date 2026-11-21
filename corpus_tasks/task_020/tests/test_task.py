from src.parsers import parse_log_level

def test_parse_log_level():
    fn = parse_log_level
    assert fn('[INFO] iniciado') == 'INFO' and fn('[WARN] disco cheio') == 'WARN'
