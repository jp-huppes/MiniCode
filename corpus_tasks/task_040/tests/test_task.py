from src.structures import WeightedBag

def test_weightedbag():
    cls = WeightedBag
    wb = cls(); wb.add('x', 3); wb.add('y', 4); assert wb.total_weight() == 7
