from src.structures import MinTracker

def test_mintracker():
    cls = MinTracker
    m = cls(); m.push(5); m.push(2); m.push(8); assert m.get_min() == 2; m.pop(); assert m.get_min() == 2; m.pop(); assert m.get_min() == 5
