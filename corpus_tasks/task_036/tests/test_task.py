from src.structures import SimpleCache

def test_simplecache():
    cls = SimpleCache
    c = cls(2); c.put('a', 1); c.put('b', 2); c.put('c', 3); assert c.get('a') is None and c.get('b') == 2 and c.get('c') == 3
