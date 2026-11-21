from src.structures import HistoryList

def test_historylist():
    cls = HistoryList
    h = cls(); h.record('x'); h.record('y'); assert h.items() == ['x', 'y']; h.clear(); assert h.items() == []
