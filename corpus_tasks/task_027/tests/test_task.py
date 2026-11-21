from src.parsers import parse_env_line

def test_parse_env_line():
    fn = parse_env_line
    assert fn('PORT="8000"') == ('PORT', '8000') and fn("NAME='app'") == ('NAME', 'app')
