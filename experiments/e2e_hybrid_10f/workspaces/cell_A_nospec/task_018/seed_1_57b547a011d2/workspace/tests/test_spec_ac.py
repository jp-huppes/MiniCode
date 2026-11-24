from src.parsers import parse_int_list
def test_ac_1():
    assert parse_int_list('1 2 3') == [1, 2, 3]

def test_ac_2():
    assert parse_int_list('4 5 6') == [4, 5, 6]

def test_ac_3():
    assert parse_int_list('7 8 9') == [7, 8, 9]
from src.parsers import parse_int_list
def test_parse_int_list():
    assert parse_int_list('1 2 3') == [1, 2, 3]
    assert parse_int_list('4 5 6') == [4, 5, 6]