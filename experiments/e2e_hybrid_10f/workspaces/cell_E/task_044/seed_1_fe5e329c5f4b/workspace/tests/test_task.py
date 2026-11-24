from src.structures import BoundedList

def test_boundedlist():
    cls = BoundedList
    bl = cls(2); bl.append(10); bl.append(20); bl.append(30); assert bl.to_list() == [10, 20]
