from src.structures import UniqueSet

def test_uniqueset():
    cls = UniqueSet
    s = cls(); assert s.add(1) is True and s.add(1) is False and s.contains(1) and s.len_items() == 1
