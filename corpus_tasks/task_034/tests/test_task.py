from src.structures import CounterMap

def test_countermap():
    cls = CounterMap
    c = cls(); c.add('a', 2); c.add('b', 5); assert c.count('a') == 2; assert c.top_key() == 'b'
