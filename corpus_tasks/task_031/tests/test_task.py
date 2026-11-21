from src.structures import Stack

def test_stack():
    cls = Stack
    s = cls(); assert s.is_empty(); s.push(1); s.push(2); assert s.peek() == 2; assert s.pop() == 2; assert s.pop() == 1; assert s.pop() is None
