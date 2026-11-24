from src.parsers import parse_kv_line
def test_ac_1_converts_string_to_dict():
    assert parse_kv_line('key1=value1 key2=value2') == {'key1': 'value1', 'key2': 'value2'}

def test_ac_2_ignores_tokens_without_equal_sign():
    assert parse_kv_line('key1 value1 key2 value2') == {'key1': 'value1', 'key2': 'value2'}

def test_ac_3_returns_empty_dict_for_empty_string():
    assert parse_kv_line('') == {}

def test_ac_1_converts_string_to_dict():
    from src.parsers import parse_kv_line
    assert parse_kv_line('key=value') == {'key': 'value'}

def test_ac_2_ignores_tokens_without_equal_sign():
    from src.parsers import parse_kv_line
    assert parse_kv_line('key value') == {}

def test_ac_3_returns_empty_dict_for_empty_string():
    from src.parsers import parse_kv_line
    assert parse_kv_line('') == {}

def test_ac_1_converts_string_to_dict():
    from src.parsers import parse_kv_line
    assert parse_kv_line('key=value') == {'key': 'value'}

def test_ac_2_ignores_tokens_without_equal_sign():
    from src.parsers import parse_kv_line
    assert parse_kv_line('key value') == {}

def test_ac_3_returns_empty_dict_for_empty_string():
    from src.parsers import parse_kv_line
    assert parse_kv_line('') == {}

def test_ac_1_converts_string_to_dict():
    from src.parsers import parse_kv_line
    assert parse_kv_line('key1=value1 key2=value2') == {'key1': 'value1', 'key2': 'value2'}

def test_ac_2_ignores_tokens_without_equal_sign():
    from src.parsers import parse_kv_line
    assert parse_kv_line('key1 value1 key2 value2 key3') == {'key1': 'value1', 'key2': 'value2'}

def test_ac_3_returns_empty_dict_for_empty_string():
    from src.parsers import parse_kv_line
    assert parse_kv_line('') == {}