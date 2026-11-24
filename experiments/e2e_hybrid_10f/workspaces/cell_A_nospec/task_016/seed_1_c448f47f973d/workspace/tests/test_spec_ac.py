from src.parsers import parse_kv_line

def test_ac_1():
    assert parse_kv_line('key=value') == {'key': 'value'}

def test_ac_2():
    assert parse_kv_line('key1=value1 key2=value2') == {'key1': 'value1', 'key2': 'value2'}

def test_ac_3():
    assert parse_kv_line('key1=value1 key2=value2 key3=value3') == {'key1': 'value1', 'key2': 'value2', 'key3': 'value3'}