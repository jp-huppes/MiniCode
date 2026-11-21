from src.structures import IntRange

def test_intrange():
    cls = IntRange
    ir = cls(2, 5); assert ir.contains(2) and ir.contains(5) and not ir.contains(6) and ir.length() == 4
